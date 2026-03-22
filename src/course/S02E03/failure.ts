import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

let cachedLines: string[] | null = null;

async function getLogLines(): Promise<string[]> {
  if (cachedLines) return cachedLines;

  const apiKey = process.env.AIDEVS_API_KEY!;
  const url = `${process.env.BASE_URL}/data/${apiKey}/failure.log`;
  log.info(`[fetch] Downloading logs from ${url}`);

  const res = await axios.get<string>(url, { responseType: 'text' });
  cachedLines = res.data.split('\n').filter((l) => l.trim().length > 0);
  log.info(`[fetch] Downloaded ${cachedLines.length} lines`);
  return cachedLines;
}

/** Conservative token estimate: ceil(chars / 3.5) */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

async function postToHub(logs: string): Promise<string> {
  try {
    const res = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'failure',
      answer: { logs },
    });
    return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
  } catch (err: any) {
    // Axios wraps 4xx/5xx as errors — but the response body contains the
    // technicians' feedback we need. Return it instead of throwing.
    if (err.response) {
      const body = err.response.data;
      return `HTTP ${err.response.status}: ${typeof body === 'string' ? body : JSON.stringify(body)}`;
    }
    throw err;
  }
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'fetch_logs_stats',
      description:
        'Download the log file (cached after first call) and return stats: total line count, file size in bytes, and a preview of the first N lines. Use this first to understand the scope of the data.',
      parameters: {
        type: 'object',
        properties: {
          preview_lines: {
            type: 'number',
            description: 'Number of lines to preview (default 20)',
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_logs',
      description:
        'Search the cached log file. Filter by log level and/or a case-insensitive substring/regex pattern. Returns matching lines (up to max_results). Use multiple focused searches to discover all relevant component IDs.',
      parameters: {
        type: 'object',
        properties: {
          level: {
            type: 'string',
            description: 'Filter by log level: CRIT, ERRO, WARN, INFO (omit for all levels)',
          },
          pattern: {
            type: 'string',
            description:
              'Case-insensitive substring or JS regex pattern to match in the line (e.g. "pump", "coolant", "PWR", "reactor")',
          },
          max_results: {
            type: 'number',
            description: 'Maximum number of matching lines to return (default 100)',
          },
          offset: {
            type: 'number',
            description: 'Skip the first N matches (for pagination, default 0)',
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'list_component_ids',
      description:
        'Scan the entire log file and return every unique component ID found, optionally filtered by log level. ' +
        'Call this early to get a complete inventory of all components and their event counts. ' +
        'Use it after each failed submission to check if you are missing any component IDs.',
      parameters: {
        type: 'object',
        properties: {
          level: {
            type: 'string',
            description:
              'Restrict scan to lines with this level: CRIT, ERRO, WARN, INFO. Omit for all levels.',
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'count_tokens',
      description:
        'Estimate the token count of a text string using a conservative approximation (chars / 3.5). Always call this before submitting to ensure you are under 1500 tokens.',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'The text whose tokens to count' },
        },
        required: ['text'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'submit_logs',
      description:
        'Submit the condensed log string to the verification endpoint and return the technicians feedback. ' +
        'Call count_tokens first — if over 1500, do NOT submit, compress further instead. ' +
        'IMPORTANT: even a 400 response contains valuable technician feedback — read it carefully and use it to fix missing or unclear components.',
      parameters: {
        type: 'object',
        properties: {
          logs: {
            type: 'string',
            description:
              'Newline-separated condensed log string. Each line: [YYYY-MM-DD HH:MM] [LEVEL] COMPONENT description',
          },
        },
        required: ['logs'],
      },
    },
  },
];

// ─── Tool executor ────────────────────────────────────────────────────────────

/**
 * Extract the component ID from a log line.
 * Lines look like: [2026-03-21 06:04:13] [CRIT] ECCS8 reported runaway ...
 * The component ID is the first whitespace-free token after the level tag.
 */
function extractComponentId(line: string): string | null {
  const m = line.match(/\]\s+\[(?:CRIT|ERRO|WARN|INFO)\]\s+(\S+)/);
  return m ? m[1] : null;
}

async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  // ── fetch_logs_stats ──────────────────────────────────────────────────────
  if (name === 'fetch_logs_stats') {
    const lines = await getLogLines();
    const preview = Number(args.preview_lines ?? 20);
    const totalChars = lines.join('\n').length;
    const previewText = lines.slice(0, preview).join('\n');
    return (
      `Total lines: ${lines.length}\n` +
      `Total chars: ${totalChars}\n` +
      `Estimated tokens (full file): ${estimateTokens(lines.join('\n'))}\n\n` +
      `--- First ${preview} lines ---\n${previewText}`
    );
  }

  // ── search_logs ───────────────────────────────────────────────────────────
  if (name === 'search_logs') {
    const lines = await getLogLines();
    const level = (args.level as string | undefined)?.toUpperCase();
    const pattern = args.pattern as string | undefined;
    const maxResults = Number(args.max_results ?? 100);
    const offset = Number(args.offset ?? 0);

    let regex: RegExp | null = null;
    if (pattern) {
      try {
        regex = new RegExp(pattern, 'i');
      } catch {
        regex = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
      }
    }

    const matched: string[] = [];
    for (const line of lines) {
      const levelMatch = !level || line.includes(`[${level}]`);
      const patternMatch = !regex || regex.test(line);
      if (levelMatch && patternMatch) matched.push(line);
    }

    const page = matched.slice(offset, offset + maxResults);
    return (
      `Matched: ${matched.length} lines (showing ${offset}–${offset + page.length})\n` +
      page.join('\n')
    );
  }

  // ── list_component_ids ────────────────────────────────────────────────────
  if (name === 'list_component_ids') {
    const lines = await getLogLines();
    const level = (args.level as string | undefined)?.toUpperCase();

    const counts = new Map<string, number>();
    for (const line of lines) {
      if (level && !line.includes(`[${level}]`)) continue;
      const id = extractComponentId(line);
      if (id) counts.set(id, (counts.get(id) ?? 0) + 1);
    }

    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const label = level ? `at level ${level}` : 'across all levels';
    const rows = sorted.map(([id, n]) => `${id.padEnd(20)} ${n}`).join('\n');
    return `Unique component IDs (${sorted.length}) ${label}:\n${'COMPONENT_ID'.padEnd(20)} COUNT\n${'-'.repeat(30)}\n${rows}`;
  }

  // ── count_tokens ──────────────────────────────────────────────────────────
  if (name === 'count_tokens') {
    const text = args.text as string;
    const tokens = estimateTokens(text);
    const lines = text.split('\n').filter((l) => l.trim()).length;
    return (
      `Estimated tokens: ${tokens}\nLines: ${lines}\nChars: ${text.length}\n` +
      `Status: ${tokens <= 1500 ? '✅ Under limit' : '❌ OVER 1500 — compress further before submitting'}`
    );
  }

  // ── submit_logs ───────────────────────────────────────────────────────────
  if (name === 'submit_logs') {
    const logsText = args.logs as string;
    const tokens = estimateTokens(logsText);
    if (tokens > 1500) {
      return `BLOCKED: Estimated ${tokens} tokens — exceeds 1500 limit. Compress further and try again.`;
    }
    log.info(
      `[submit] Sending ${logsText.split('\n').filter(Boolean).length} lines (~${tokens} tokens)`,
    );
    const result = await postToHub(logsText);
    return result;
  }

  return `Unknown tool: ${name}`;
}

const SYSTEM_PROMPT = `You are an agent analysing power-plant system logs after a failure incident.

## Goal
Produce a condensed log string (max 1500 tokens) containing only events relevant to the failure analysis:
power supply, cooling systems, water pumps, reactor, control software, generators, transformers,
and any other plant components.

## Strategy
1. Start with fetch_logs_stats to understand the file size.
2. Call list_component_ids (no level filter) to get a full inventory of every component ID in the file.
   Then call it again with level=CRIT and level=ERRO to see which components had critical events.
3. Use search_logs to fetch the actual CRIT/ERRO/WARN lines for each relevant component.
4. Skip INFO logs and events unrelated to the plant (e.g. user logins, admin systems).
5. ALWAYS call count_tokens before submitting — if > 1500, shorten descriptions further.
6. Submit with submit_logs. Whether the response is a success or an HTTP error, it contains
   technician feedback — read it carefully. It tells you exactly which components are missing
   or unclear. Add those components and resubmit.
7. Iterate until you receive the flag {FLG:...}.

## Line format
[YYYY-MM-DD HH:MM] [LEVEL] COMPONENT_ID short description of the event

## Compression rules
- Shorten descriptions while keeping: timestamp, level, component ID, and the essence of the event.
- Merge repeated events for the same component into one line where possible.
- Priority order: CRIT > ERRO > WARN. Skip INFO entirely.`;

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  {
    role: 'system',
    content: SYSTEM_PROMPT,
  },
  {
    role: 'user',
    content:
      'Download the failure logs, filter out relevant events, compress them to under 1500 tokens, and submit for verification. Iterate based on technician feedback until you receive the flag.',
  },
];

async function solveTask(): Promise<void> {
  log.info('=== Failure log task started ===');
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

        const flagMatch = toolResult.match(/\{FLG:[^}]+\}/);
        if (flagMatch) {
          log.result('=== FLAG FOUND (in tool result) ===', flagMatch[0]);
        }
      } catch (err: any) {
        toolResult = `ERROR: ${err.message}`;
        log.error(`[Tool error] ${toolResult}`);
      }

      // Show more of the result in logs so we can see technician feedback
      log.info(`[Tool result] ${toolResult.slice(0, 1200)}`);

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
