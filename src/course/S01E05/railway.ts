import axios, { AxiosResponseHeaders, RawAxiosResponseHeaders } from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Calculates how many ms to wait based on rate-limit headers.
 * Supports Retry-After (seconds or date), X-RateLimit-Reset (Unix timestamp or seconds).
 * Defaults to 60s if headers are unreadable.
 */
function parseRateLimitWait(headers: RawAxiosResponseHeaders | AxiosResponseHeaders): number {
  const retryAfter = headers['retry-after'] as string | undefined;
  if (retryAfter) {
    const asSeconds = parseInt(retryAfter, 10);
    if (!isNaN(asSeconds)) return asSeconds * 1000;
    const asDate = new Date(retryAfter).getTime();
    if (!isNaN(asDate)) return Math.max(1000, asDate - Date.now());
  }

  const resetRaw =
    (headers['x-ratelimit-reset'] as string | undefined) ??
    (headers['x-rate-limit-reset'] as string | undefined) ??
    (headers['ratelimit-reset'] as string | undefined);

  if (resetRaw) {
    const val = parseInt(resetRaw, 10);
    if (!isNaN(val)) {
      // Unix timestamp (>1e9) vs. "seconds until reset"
      if (val > 1_000_000_000) return Math.max(1000, val * 1000 - Date.now());
      return val * 1000;
    }
  }

  log.info('[Railway] No usable rate-limit header found, defaulting to 60s wait.');
  return 60_000;
}

/**
 * Sends action to Railway API.
 * Automatically retries on 503 (backoff) and 429 (waits for limit reset).
 * Throws only when MAX_RETRIES is exhausted.
 */
async function callRailwayAPI(action: Record<string, unknown>): Promise<{
  data: unknown;
  rateLimitInfo: string;
}> {
  const payload = {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'railway',
    answer: action,
  };

  const MAX_RETRIES = 30;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    log.info(`[Railway] Attempt ${attempt}/${MAX_RETRIES} — action: ${JSON.stringify(action)}`);

    try {
      const response = await axios.post(VERIFY_URL, payload, {
        validateStatus: (status) => status < 500,
      });

      const headers = response.headers;

      // Collect limit info to pass to agent
      const remaining =
        headers['x-ratelimit-remaining'] ??
        headers['x-rate-limit-remaining'] ??
        headers['ratelimit-remaining'] ??
        'unknown';

      const rateLimitInfo = `Remaining requests: ${remaining}`;

      log.info(
        `[Railway] Status: ${response.status} | ${rateLimitInfo} | Body: ${JSON.stringify(response.data)}`,
      );

      if (response.status === 429) {
        const waitMs = parseRateLimitWait(headers);
        log.info(`[Railway] Rate limited (429). Waiting ${waitMs}ms...`);
        await sleep(waitMs);
        continue;
      }

      // Success (2xx, 4xx other than 429 — e.g. validation errors we want to see)
      return { data: response.data, rateLimitInfo };
    } catch (err: any) {
      // axios throws only on 5xx if validateStatus didn't catch it
      if (err.response?.status === 503) {
        const backoff = Math.min(5000 * attempt, 30_000);
        log.info(`[Railway] 503 (overload). Waiting ${backoff}ms before retry...`);
        await sleep(backoff);
        continue;
      }

      if (err.response?.status === 429) {
        const waitMs = parseRateLimitWait(err.response.headers ?? {});
        log.info(`[Railway] Rate limited (429 via catch). Waiting ${waitMs}ms...`);
        await sleep(waitMs);
        continue;
      }

      log.error(`[Railway] Unexpected error: ${err.message}`);
      throw err;
    }
  }

  throw new Error(`[Railway] Exceeded ${MAX_RETRIES} retries without success.`);
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'callRailwayAction',
      description: `Calls the Railway API with the given action object.
Automatically retries on 503 (server overload) and 429 (rate limit) errors.
Returns the API response body and rate limit info.
Always start with {"action": "help"} to get the API documentation.
Watch for a flag in the format {FLG:...} in the response — that means the task is complete.`,
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'object',
            description:
              'The action object to send as the "answer" field. Must always include "action" key. Use exactly the parameter names and values from the API documentation.',
            additionalProperties: true,
          },
        },
        required: ['action'],
      },
    },
  },
];

const SYSTEM_PROMPT = `You are an AI agent solving a railway API task.

Your goal: activate railway route X-01 by calling a self-documenting API step by step.

RULES:
1. Always start with {"action": "help"} to read the full API documentation.
2. Follow the API documentation EXACTLY — use the exact action names and parameter names it specifies.
3. The API is intentionally overloaded (503) and rate-limited. Your tool handles retries automatically, so just call it normally.
4. After each API call, check if the response contains a flag in the format {FLG:...}. If it does, report it immediately.
5. If a call fails with a validation error, read the error message carefully — it will tell you what went wrong.
6. Do NOT guess parameter names or action names. Use only what the documentation says.
7. Do NOT make unnecessary API calls — rate limits are very strict.

When you find the flag {FLG:...}, output it clearly and stop.`;

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  { role: 'system', content: SYSTEM_PROMPT },
  {
    role: 'user',
    content:
      'Please activate railway route X-01. Start by calling the help action to read the API documentation, then follow the steps exactly.',
  },
];

async function solveTask(): Promise<void> {
  log.info('=== Railway task started ===');
  let iterations = 0;
  const MAX_ITERATIONS = 50;

  while (iterations < MAX_ITERATIONS) {
    log.info(`Iteration ${iterations + 1}...`);
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
      log.info(`Agent: ${msg.content.slice(0, 300)}`);

      const flagMatch = msg.content.match(/\{FLG:[^}]+\}/);
      if (flagMatch) {
        log.result('=== FLAG FOUND ===', flagMatch[0]);
        return;
      }
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      log.info('[Agent] No tool calls — agent finished reasoning.');
      break;
    }

    for (const toolCall of msg.tool_calls) {
      if (toolCall.type !== 'function') continue;

      const { name, arguments: argsJson } = toolCall.function;
      log.info(`[Tool] ${name}(${argsJson})`);

      let toolResult: string;

      try {
        const args = JSON.parse(argsJson) as { action: Record<string, unknown> };

        if (name === 'callRailwayAction') {
          const { data, rateLimitInfo } = await callRailwayAPI(args.action);
          const dataStr = JSON.stringify(data);
          toolResult = `API Response: ${dataStr}\n${rateLimitInfo}`;

          const flagMatch = dataStr.match(/\{FLG:[^}]+\}/);
          if (flagMatch) {
            log.result('Flag found: ', flagMatch[0]);
            messages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: toolResult,
            });

            return;
          }
        } else {
          toolResult = `Unknown tool: ${name}`;
        }
      } catch (err: any) {
        toolResult = `ERROR: ${err.message}`;
        log.error(`[Tool error] ${toolResult}`);
      }

      log.info(`[Tool result] ${toolResult.slice(0, 500)}`);

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
