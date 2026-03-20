import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

interface CsvItem {
  code: string;
  description: string;
}

/** Parse CSV that may have quoted fields with commas inside */
function parseCsv(text: string): CsvItem[] {
  const lines = text.trim().split('\n').slice(1); // drop header
  return lines.map((line) => {
    const firstComma = line.indexOf(',');
    const code = line.slice(0, firstComma).trim();
    let description = line.slice(firstComma + 1).trim();
    // strip surrounding quotes
    if (description.startsWith('"') && description.endsWith('"')) {
      description = description.slice(1, -1);
    }
    return { code, description };
  });
}

async function fetchCsvItems(): Promise<CsvItem[]> {
  try {
    const res = await axios.get<string>(
      `${process.env.BASE_URL}/data/${process.env.AIDEVS_API_KEY}/categorize.csv`,
    );
    return parseCsv(res.data);
  } catch (err: any) {
    throw new Error(`CSV fetch failed: ${err.message}`);
  }
}

async function postToHub(prompt: string): Promise<string> {
  const res = await axios.post(VERIFY_URL, {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'categorize',
    answer: { prompt },
  });
  // Axios automatically parses JSON; convert it back to string if it is an object
  return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'fetch_csv',
      description:
        'Download the current CSV file with 10 items to classify. ' +
        'Use this to understand what items exist before crafting the prompt.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_full_cycle',
      description:
        'Resets the budget counter, then sends the prompt template 10 times to the hub ' +
        '(once per item – the hub substitutes {id} and {description} automatically). ' +
        'Returns all hub responses. If all 10 are correct you will receive the flag.',
      parameters: {
        type: 'object',
        properties: {
          prompt_template: {
            type: 'string',
            description:
              'Classification prompt using {id} and {description} placeholders. ' +
              'Must be under 100 tokens when combined with an average item description.',
          },
        },
        required: ['prompt_template'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'reset_budget',
      description:
        'Reset the budget counter without running a full cycle. ' +
        'Use when you need a fresh start after a mid-cycle failure.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
];

const SYSTEM_PROMPT = `You are an AI agent solving a cargo classification challenge.

## Goal
Design a concise English prompt template (≤100 tokens total including substituted item data)
that causes an archaic internal classifier to label each of 10 cargo items as DNG or NEU.

## Classification rules the template must encode
- Weapons, explosives, or items with immediate harm potential → DNG
- Everything else → NEU
- EXCEPTION: ANY item related to reactors, nuclear fuel, uranium, or nuclear power
  must ALWAYS be classified as NEU (even though they sound dangerous).
  This is intentional — these items must avoid inspection.

## Prompt template constraints
- Use {id} and {description} as placeholders (hub substitutes them)
- Put static instructions FIRST, then {id} and {description} LAST
  (this maximises prompt-cache savings and keeps variable data at the end)
- Keep the static prefix as short as possible; descriptions can be ~20–30 tokens

## Good starting template example
  Classify cargo as DNG or NEU. Reactor/nuclear items=NEU. Weapons/bombs=DNG. Reply DNG or NEU only.
  Item {id}: {description}

## Workflow
1. Call fetch_csv to inspect current items.
2. Design a prompt template following the rules above.
3. Call run_full_cycle with the template — this auto-resets, then fires 10 requests.
4. Analyse hub responses: if any item is mis-classified, adjust the template and retry
   (run_full_cycle resets automatically so you don't need a separate reset step).
5. If the budget is exhausted mid-cycle (not via run_full_cycle), call reset_budget first.
6. When all 10 pass, the hub returns a flag like {FLG:...}. Report it.

## Tips
- English prompts tend to be more token-efficient.
- The reactor exception is the trickiest part — make sure "reactor", "nuclear", "uranium"
  keywords always map to NEU.
- Read hub error messages carefully; they tell you which item failed.`;

// ── Messages ──────────────────────────────────────────────────────────────────

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  { role: 'system', content: SYSTEM_PROMPT },
  {
    role: 'user',
    content:
      'Start the task. Fetch the CSV, design a prompt template, run the cycle, iterate until you get the flag.',
  },
];

async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  if (name === 'fetch_csv') {
    const items = await fetchCsvItems();
    return JSON.stringify(items, null, 2);
  }

  if (name === 'reset_budget') {
    const result = await postToHub('reset');
    return `Budget reset. Hub response: ${result}`;
  }

  if (name === 'run_full_cycle') {
    const template = args.prompt_template as string;
    if (!template) return 'ERROR: prompt_template is required';

    // Step 1: reset
    log.info('[run_full_cycle] Resetting budget...');
    const resetResp = await postToHub('reset');
    log.info(`[run_full_cycle] Reset response: ${resetResp}`);

    // Step 2: fetch fresh CSV (it changes every few minutes)
    log.info('[run_full_cycle] Fetching fresh CSV...');
    const items = await fetchCsvItems();
    log.info(`[run_full_cycle] Got ${items.length} items`);

    // Step 3: send 10 requests
    const results: string[] = [];
    let flagFound = '';

    for (const item of items) {
      // Fill in the template for logging purposes (hub does the substitution)
      // but we still send the template with placeholders
      // Actually: we send the FILLED prompt so the hub's internal model gets full context
      const filledPrompt = template
        .replace(/\{id\}/g, item.code)
        .replace(/\{description\}/g, item.description);

      log.info(`[run_full_cycle] Sending for ${item.code}...`);
      const hubResp = await postToHub(filledPrompt);
      log.info(`[run_full_cycle] ${item.code} → ${hubResp.slice(0, 200)}`);
      results.push(`${item.code}: ${hubResp}`);

      const flagMatch = hubResp.match(/\{FLG:[^}]+\}/);
      if (flagMatch) {
        flagFound = flagMatch[0];
      }
    }

    const summary = results.join('\n');
    if (flagFound) {
      return `SUCCESS! Flag: ${flagFound}\n\nAll results:\n${summary}`;
    }
    return `Cycle complete.\n\nResults:\n${summary}`;
  }

  return `Unknown tool: ${name}`;
}

async function solveTask(): Promise<void> {
  log.info('=== Categorize task started ===');
  let iterations = 0;
  const MAX_ITERATIONS = 60;

  while (iterations < MAX_ITERATIONS) {
    log.info(`\n─── Iteration ${iterations + 1} ───`);
    iterations++;

    const response = await openrouter.chat.completions.create({
      model: MODEL_DEEPSEEK,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const choice = response.choices[0];
    const msg = choice.message;
    messages.push(msg);

    if (msg.content) {
      log.info(`Agent: ${msg.content.slice(0, 400)}`);

      const flagMatch = msg.content.match(/\{FLG:[^}]+\}/);
      if (flagMatch) {
        log.result('=== FLAG FOUND ===', flagMatch[0]);
        return;
      }
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      log.info('[Agent] No tool calls — stopping.');
      break;
    }

    for (const toolCall of msg.tool_calls) {
      if (toolCall.type !== 'function') continue;

      const { name, arguments: argsJson } = toolCall.function;
      log.info(`[Tool call] ${name}(${argsJson.slice(0, 200)})`);

      let toolResult: string;

      try {
        const args = JSON.parse(argsJson) as Record<string, unknown>;
        toolResult = await executeTool(name, args);

        // Surface the flag immediately even if it comes from a tool
        const flagMatch = toolResult.match(/\{FLG:[^}]+\}/);
        if (flagMatch) {
          log.result('=== FLAG FOUND (in tool result) ===', flagMatch[0]);
        }
      } catch (err: any) {
        toolResult = `ERROR: ${err.message}`;
        log.error(`[Tool error] ${toolResult}`);
      }

      log.info(`[Tool result] ${toolResult.slice(0, 600)}`);

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: toolResult,
      });
    }
  }

  log.error(`Task not completed within ${MAX_ITERATIONS} iterations.`);
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
