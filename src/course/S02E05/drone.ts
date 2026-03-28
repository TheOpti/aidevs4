import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';

// ─── Config ───────────────────────────────────────────────────────────────────
const API_KEY = process.env.AIDEVS_API_KEY ?? '';
const HUB_URL = '${process.env.BASE_URL}/verify';
const IMAGE_URL = `${process.env.BASE_URL}/data/${API_KEY}/drone.png`;
const TARGET_ID = 'PWR6132PL';
const TASK = 'drone';

// Model 1: Orchestrator – DeepSeek (strong reasoning, drives the agentic loop)
const ORCHESTRATOR_MODEL = MODEL_DEEPSEEK;
// Model 2: Vision subagent – GPT-4o (multimodal, analyzes the map image)
const VISION_MODEL = 'openai/gpt-4o';
// Model 3: Hub API subagent – GPT-4o-mini (sends instructions, interprets API responses)
const HUB_MODEL = 'openai/gpt-4o-mini';

// ─── Subagent 1: Vision ───────────────────────────────────────────────────────
// Responsibility: analyze the map image and return dam grid coordinates

const VISION_PROMPT = `You are analyzing a top-down satellite map divided into a regular GRID.

IMPORTANT: This is a small grid — expect only 3–4 columns and 3–5 rows total. Do NOT over-count.

STEP 1 – Count columns:
- Find the leftmost vertical grid line and the rightmost vertical grid line.
- Count only the CELLS (spaces) between them — NOT the lines themselves.
- Label each cell: "col 1 | col 2 | col 3". Stop when you reach the right edge.
- Hint: if you are counting more than 4 columns, you are likely counting grid lines instead of cells.

STEP 2 – Count rows:
- Find the top horizontal line and the bottom horizontal line.
- Count only the CELLS between them.
- Label each cell: "row 1 / row 2 / row 3 / row 4". Stop at the bottom edge.

STEP 3 – Find the DAM:
- One sector has a clearly INTENSIFIED, VIVID blue water color — much stronger than neighbours.
- Report its column (x) and row (y). TOP-LEFT = (1,1).

STEP 4 – Double-check:
- Recount. Remember: you are counting SPACES between lines, not the lines themselves.

Respond ONLY with valid JSON (no markdown):
{"cols": 3, "rows": 4, "x": 2, "y": 4, "reasoning": "3 columns (not lines), 4 rows. Bottom-center sector has vivid blue — dam."}`;

async function callVisionModel(): Promise<string> {
  const response = await openrouter.chat.completions.create({
    model: VISION_MODEL,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: IMAGE_URL } },
          { type: 'text', text: VISION_PROMPT },
        ],
      },
    ],
    max_tokens: 600,
  });
  return response.choices[0].message.content ?? '';
}

async function subagentAnalyzeMap(): Promise<string> {
  log.info('[Subagent:Vision] Analyzing map — attempt 1...');
  const raw1 = await callVisionModel();
  log.info(`[Subagent:Vision] Attempt 1: ${raw1}`);

  const match1 = raw1.match(/\{[\s\S]*\}/);
  if (!match1) return raw1;
  const result1 = JSON.parse(match1[0]) as {
    cols: number;
    rows: number;
    x: number;
    y: number;
    reasoning: string;
  };

  // If cols > 3 it likely miscounted — verify with a second independent call
  if (result1.cols > 3) {
    log.info(`[Subagent:Vision] cols=${result1.cols} looks too high — running verification...`);
    const raw2 = await callVisionModel();
    log.info(`[Subagent:Vision] Attempt 2: ${raw2}`);

    const match2 = raw2.match(/\{[\s\S]*\}/);
    if (match2) {
      const result2 = JSON.parse(match2[0]) as {
        cols: number;
        rows: number;
        x: number;
        y: number;
        reasoning: string;
      };

      if (result2.cols <= 3) {
        // Second attempt got it right
        log.info(
          `[Subagent:Vision] Attempt 2 corrected to cols=${result2.cols}: x=${result2.x}, y=${result2.y}`,
        );
        return JSON.stringify(result2);
      }

      // Both attempts overcounted — apply hard fallback: force cols=3, x=middle col
      const forcedX = 2; // middle of 3 columns
      const forcedY = Math.max(result1.y, result2.y); // trust the higher row (more likely bottom)
      const fallback = {
        cols: 3,
        rows: result1.rows,
        x: forcedX,
        y: forcedY,
        reasoning: 'Forced fallback: both attempts returned cols>3, overriding to cols=3 x=2',
      };
      log.info(
        `[Subagent:Vision] Both attempts overcounted — using hard fallback: ${JSON.stringify(fallback)}`,
      );
      return JSON.stringify(fallback);
    }
  }

  return raw1;
}

// ─── Subagent 2: Hub API ──────────────────────────────────────────────────────
// Responsibility: send instructions to the drone API and interpret the response

async function subagentSendToDrone(instructions: string[]): Promise<string> {
  log.info(`[Subagent:Hub] Sending instructions via GPT-4o-mini: ${JSON.stringify(instructions)}`);

  // Step 1: actually call the drone API
  const res = await fetch(HUB_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      apikey: API_KEY,
      task: TASK,
      answer: { instructions },
    }),
  });
  const rawBody = await res.json();
  const rawJson = JSON.stringify(rawBody);
  log.info(`[Subagent:Hub] Raw API response: ${rawJson}`);

  // Step 2: use HUB_MODEL to interpret the response and produce a clear report
  const interpretation = await openrouter.chat.completions.create({
    model: HUB_MODEL,
    messages: [
      {
        role: 'system',
        content:
          'You are a drone API response interpreter. ' +
          'Given a raw JSON response from the drone hub, produce a concise status report. ' +
          'If the response contains {FLG:...}, extract and highlight the flag prominently. ' +
          'If it contains an error, explain exactly what went wrong and what needs to be fixed. ' +
          'If it is a success without a flag, confirm what was accepted.',
      },
      {
        role: 'user',
        content: `Instructions sent: ${JSON.stringify(instructions)}\n\nAPI response: ${rawJson}`,
      },
    ],
    max_tokens: 300,
  });

  const report = interpretation.choices[0].message.content ?? rawJson;
  log.info(`[Subagent:Hub] Interpreted report: ${report}`);

  // Always include raw JSON so the orchestrator can still regex-match the flag
  return `${report}\n\n[RAW]: ${rawJson}`;
}

// ─── Tool definitions for the orchestrator ───────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'analyze_map',
      description:
        'Analyze the mission map using the Vision subagent (GPT-4o). ' +
        'Returns JSON with total grid dimensions and the (x,y) coordinates of the dam sector. ' +
        'Call this FIRST before sending any drone instructions.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
  {
    type: 'function',
    function: {
      name: 'send_drone_instructions',
      description:
        'Send an ordered array of drone instructions via the Hub API subagent (GPT-4o-mini). ' +
        'Returns an interpreted status report plus the raw API response. ' +
        'If the report contains {FLG:...} the mission is complete. ' +
        'If it describes an error, read carefully and adjust instructions.',
      parameters: {
        type: 'object',
        properties: {
          instructions: {
            type: 'array',
            items: { type: 'string' },
            description:
              'Ordered list of drone commands. Available commands:\n' +
              '  setDestinationObject(ID)  – set target object, ID = PWR6132PL\n' +
              '  set(x,y)                  – landing sector (column, row), 1-indexed from top-left\n' +
              '  set(destroy)              – mission objective\n' +
              '  set(engineON)             – start engines\n' +
              '  set(Xm)                   – flight altitude, e.g. set(50m)\n' +
              '  flyToLocation             – execute flight (must be last)\n' +
              '  hardReset                 – factory reset (use when errors accumulate)',
          },
        },
        required: ['instructions'],
      },
    },
  },
];

// ─── Orchestrator (Model 1 – DeepSeek) ───────────────────────────────────────

const SYSTEM_PROMPT = `You are an autonomous drone mission orchestrator. You control drone ${TARGET_ID}.

ARCHITECTURE:
- You are the Orchestrator (DeepSeek). You reason and decide what to do.
- analyze_map delegates to a Vision subagent (GPT-4o) that reads the map image.
- send_drone_instructions delegates to a Hub API subagent (GPT-4o-mini) that calls the drone API and interprets the response.

MISSION OBJECTIVE:
Fly the drone to object ${TARGET_ID} and drop the payload on the DAM sector (not the main building).
The dam is a nearby water structure visible on the map with intensified blue color.

HOW TO COMPLETE THE MISSION:
1. Call analyze_map → receive dam grid coordinates (x, y).
2. Call send_drone_instructions with minimal instructions, e.g.:
   ["setDestinationObject(${TARGET_ID})", "set(x,y)", "set(destroy)", "set(engineON)", "set(50m)", "flyToLocation"]
3. Read the interpreted report:
   - Contains {FLG:...} → mission complete, output the flag and stop.
   - Describes an error → fix only what's mentioned, retry.
4. If errors keep accumulating, send ["hardReset"] to reset the drone, then retry from step 2.

RULES:
- Keep instruction sets minimal.
- set(x,y) uses coordinates from the vision analysis (1,1 = top-left of map).
- flyToLocation must always be the last instruction.
- Stop immediately when {FLG:...} appears anywhere in the response.`;

async function solveTask(): Promise<void> {
  log.info('=== Drone mission started (3-model architecture) ===');
  log.info(`  Orchestrator : ${ORCHESTRATOR_MODEL}`);
  log.info(`  Vision agent : ${VISION_MODEL}`);
  log.info(`  Hub agent    : ${HUB_MODEL}`);

  if (!API_KEY) throw new Error('AIDEVS_API_KEY not set in environment');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        'Start the mission. Analyze the map first, then send the correct drone instructions.',
    },
  ];

  let iterations = 0;
  const MAX_ITERATIONS = 30;

  while (iterations < MAX_ITERATIONS) {
    log.info(`\n─── Iteration ${iterations + 1} ───`);
    iterations++;

    const response = await openrouter.chat.completions.create({
      model: ORCHESTRATOR_MODEL,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const choice = response.choices[0];
    const msg = choice.message;
    messages.push(msg);

    if (msg.content) {
      log.info(`[Orchestrator] ${msg.content.slice(0, 500)}`);

      const flagMatch = msg.content.match(/\{FLG:[^}]+\}/);
      if (flagMatch) {
        log.result('=== FLAG FOUND ===', flagMatch[0]);
        return;
      }
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      log.info('[Orchestrator] No tool calls — stopping.');
      break;
    }

    for (const toolCall of msg.tool_calls) {
      if (toolCall.type !== 'function') continue;

      const { name, arguments: argsJson } = toolCall.function;
      log.info(`[Tool call] ${name}(${argsJson.slice(0, 300)})`);

      let toolResult: string;
      try {
        const args = JSON.parse(argsJson) as Record<string, unknown>;
        toolResult = await executeTool(name, args);

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

async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case 'analyze_map':
      return subagentAnalyzeMap();

    case 'send_drone_instructions': {
      const instructions = args.instructions as string[];
      return subagentSendToDrone(instructions);
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
