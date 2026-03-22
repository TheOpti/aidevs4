import 'dotenv/config';
import OpenAI from 'openai';

// ============================================================
// Model constants
// ============================================================

/** Local Qwen model served via LM Studio */
export const MODEL_QWEN = 'qwen3.5-9b';

/** OpenAI-compatible OSS model (e.g. served via proxy / LM Studio) */
export const MODEL_GPT_OSS = 'openai/gpt-oss-20b';

/** Google Gemma 3 with vision capabilities */
export const MODEL_GEMMA = 'google/gemma-3-12b';

/** DeepSeek v 3.2 */
export const MODEL_DEEPSEEK = 'deepseek/deepseek-v3.2';

/** Google Gemini Flash 1.5 with vision capabilities */
export const MODEL_GEMINI_VISION = 'google/gemini-2.0-flash-001';

// ============================================================
// OpenAI client factory
// ============================================================

/**
 * Creates and returns an OpenAI client pointed at the local
 * LM Studio instance (http://localhost:1234/v1).
 *
 * Reuse this everywhere instead of duplicating `new OpenAI(...)`.
 */
export function createOpenAIClient(): OpenAI {
  return new OpenAI({
    baseURL: 'http://localhost:1234/v1',
    apiKey: 'lm-studio',
  });
}

export function createOpenRouterAiClient(): OpenAI {
  return new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY,
  });
}

/** Pre-built default client — import this when a single shared instance is enough. */
export const openai = createOpenAIClient();
export const openrouter = createOpenRouterAiClient();

// ============================================================
// Unified logger
// ============================================================

/**
 * Unified logging utility for all tasks.
 *
 * Formats:
 *   log.step(1, 'Fetch data')       →  [Step 1] Fetch data
 *   log.info('Done')                →  [Info] Done
 *   log.tool('getLocations', args)  →  [Tool] getLocations | {"name":"Jan"}
 *   log.api('check', payload)       →  [API] check | {"packageid":"PKG1"}
 *   log.result('Answer submitted')  →  [Result] Answer submitted
 *   log.error('Failed', err)        →  [Error] Failed
 */
export const log = {
  step: (n: number, description: string) => console.log(`\n[Step ${n}] ${description}`),

  info: (message: string, data?: unknown) =>
    data !== undefined ? console.log(`[Info] ${message}`, data) : console.log(`[Info] ${message}`),

  tool: (name: string, data?: unknown) =>
    data !== undefined
      ? console.log(`[Tool] ${name} |`, JSON.stringify(data))
      : console.log(`[Tool] ${name}`),

  api: (action: string, data?: unknown) =>
    data !== undefined
      ? console.log(`[API] ${action} |`, JSON.stringify(data))
      : console.log(`[API] ${action}`),

  result: (message: string, data?: unknown) =>
    data !== undefined
      ? console.log(`[Result] ${message}`, data)
      : console.log(`[Result] ${message}`),

  error: (message: string, err?: unknown) => console.error(`[Error] ${message}`, err ?? ''),
};
