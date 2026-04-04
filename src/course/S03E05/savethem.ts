import axios from 'axios';
import 'dotenv/config';
import fs from 'fs';
import OpenAI from 'openai';
import path from 'path';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { S03E05 } from 'src/shared/api';
import { runPlanner, ScoutFindings } from './planer';

const API_KEY = process.env.AIDEVS_API_KEY!;
const BASE_URL = process.env.BASE_URL!;
const CACHE_FILE = path.resolve('src/data/savethem_data.json');

// ─── Cache helpers ────────────────────────────────────────────────────────────

function loadCache(): ScoutFindings | null {
  try {
    if (fs.existsSync(CACHE_FILE)) {
      const raw = fs.readFileSync(CACHE_FILE, 'utf-8');
      const parsed = JSON.parse(raw) as ScoutFindings;
      // Basic sanity check
      if (parsed.map && parsed.vehicles && parsed.legend) {
        log.info(`[Cache] Loaded findings from ${CACHE_FILE} — skipping scout.`);
        return parsed;
      }
    }
  } catch {
    log.info('[Cache] Could not read savethem_data.json — will run scout.');
  }
  return null;
}

function saveCache(findings: ScoutFindings): void {
  try {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(findings, null, 2), 'utf-8');
    log.info(`[Cache] Findings saved to ${CACHE_FILE}`);
  } catch (err) {
    log.info('[Cache] Could not write savethem_data.json:', (err as Error).message);
  }
}

// ─── Shared HTTP helpers ──────────────────────────────────────────────────────

async function postTool(url: string, query: string): Promise<unknown> {
  try {
    const { data } = await axios.post(url, { apikey: API_KEY, query });
    return data;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      return err.response.data; // error bodies always contain useful info
    }
    throw err;
  }
}

async function searchTools(query: string): Promise<unknown> {
  return postTool(S03E05.TOOL_SEARCH, query);
}

async function callToolByPath(path: string, query: string): Promise<unknown> {
  return postTool(`${BASE_URL}${path}`, query);
}

// ─── Scout tools ──────────────────────────────────────────────────────────────

const scoutTools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'search_tools',
      description:
        'Search the tool registry to discover available API endpoints. ' +
        'Returns up to 3 best-matching tools. ' +
        'YOU MAY CALL THIS MULTIPLE TIMES IN A SINGLE TURN to search in parallel.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Short ENGLISH keyword query' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'call_tool',
      description:
        'Call an API tool discovered via search_tools. ' +
        'YOU MAY CALL THIS MULTIPLE TIMES IN A SINGLE TURN to fetch data in parallel. ' +
        'Even 4xx responses contain useful hints — always read the body.',
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'URL path from search_tools, e.g. "/api/maps"' },
          query: { type: 'string', description: 'ENGLISH query to send to the tool' },
        },
        required: ['url', 'query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'report_findings',
      description:
        'Call this ONLY when you have ALL three pieces of data:\n' +
        '  1. The full 10x10 map grid\n' +
        '  2. Stats for EVERY vehicle (fuel/food per move, terrain restrictions)\n' +
        '  3. The complete terrain legend (symbol → meaning + passability)\n' +
        'Pack everything into structured JSON.',
      parameters: {
        type: 'object',
        properties: {
          map: {
            type: 'array',
            description: '10x10 grid — array of 10 rows, each row is array of 10 symbol strings',
            items: { type: 'array', items: { type: 'string' } },
          },
          vehicles: {
            type: 'array',
            description:
              'One object per vehicle: { name, fuel_per_move, food_per_move, cannot_cross[] }',
            items: { type: 'object' },
          },
          legend: {
            type: 'object',
            description: 'Map from symbol to { meaning, passable_by: string[] }',
          },
          extra_notes: {
            type: 'string',
            description: 'Any other relevant facts (dismount rules, special tiles, etc.)',
          },
        },
        required: ['map', 'vehicles', 'legend'],
      },
    },
  },
];

// ─── Scout agent ──────────────────────────────────────────────────────────────

const SCOUT_SYSTEM = `
You are a SCOUT agent. Collect three pieces of intelligence as fast as possible.
A messenger must travel from base (S) to Skolwin (G) on a 10x10 grid.

=== WHAT YOU NEED ===
  A) MAP      — the full 10x10 grid (query maps tool with the destination city name)
  B) VEHICLES — fuel_per_move + food_per_move + terrain restrictions for EVERY vehicle
  C) LEGEND   — what each map symbol means and whether it blocks movement

=== HOW TO BE FAST ===
  • In your FIRST turn: emit MULTIPLE search_tools calls simultaneously —
    one for "map terrain", one for "vehicle fuel consumption", one for "legend notes books".
  • In your SECOND turn: once you have the URLs, emit ALL data-fetching call_tool
    calls at once — map, all vehicles, legend — in a single response.
  • The legend lives in a non-obvious tool (not the maps tool). Search for
    "books", "notes", "codex", "glossary", "knowledge" to find it.
  • Only call report_findings when you genuinely have A + B + C.

=== RULES ===
  - All queries must be in ENGLISH.
  - Never repeat an identical (url + query) pair.
  - A 4xx response body often reveals the correct query format — read it.
`.trim();

async function runScout(): Promise<ScoutFindings> {
  log.info('\n══════════ SCOUT AGENT STARTED ══════════');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SCOUT_SYSTEM },
    {
      role: 'user',
      content:
        'Go. In this first response emit THREE parallel search_tools calls: ' +
        '"map terrain", "vehicle fuel consumption", "legend notes books". ' +
        'Then in the next turn fire all data calls at once.',
    },
  ];

  let iterations = 0;
  const MAX = 15; // tight budget forces parallelism

  while (iterations < MAX) {
    log.info(`\n── Scout iteration ${++iterations} ──`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_DEEPSEEK,
      messages,
      tools: scoutTools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);
    if (msg.content) log.info(`[Scout] ${msg.content.slice(0, 300)}`);

    if (!msg.tool_calls?.length) {
      log.info('[Scout] No tool calls — nudging…');
      messages.push({
        role: 'user',
        content:
          'You must call a tool. If A+B+C are complete → call report_findings. ' +
          'If legend is missing → search "books notes codex". ' +
          'Batch multiple calls in one response to save iterations.',
      });
      continue;
    }

    // ── Execute ALL tool calls in this turn in parallel ───────────────────────
    const pending = msg.tool_calls
      .filter((tc) => tc.type === 'function')
      .map(async (tc) => {
        const { name, arguments: argsJson } = tc.function;
        const args = JSON.parse(argsJson) as Record<string, unknown>;
        log.info(`[Scout tool] ${name}(${argsJson.slice(0, 200)})`);

        if (name === 'report_findings') {
          return { tc, result: '__REPORT__', findings: args as unknown as ScoutFindings };
        }

        let result: unknown;
        if (name === 'search_tools') {
          result = await searchTools(args.query as string);
        } else if (name === 'call_tool') {
          result = await callToolByPath(args.url as string, args.query as string);
        } else {
          result = { error: `Unknown tool: ${name}` };
        }

        const resultStr = JSON.stringify(result);
        log.info(`[Scout result] ${name} → ${resultStr.slice(0, 400)}`);
        return { tc, result: resultStr, findings: null };
      });

    const results = await Promise.all(pending);

    // Check for report_findings first
    for (const { result, findings } of results) {
      if (result === '__REPORT__' && findings) {
        log.info('[Scout] report_findings received — handing off to planner.');
        return findings;
      }
    }

    // Push all tool results back into message history
    for (const { tc, result } of results) {
      if (result !== '__REPORT__') {
        messages.push({ role: 'tool', tool_call_id: tc.id, content: result as string });
      }
    }
  }

  throw new Error(`Scout did not complete within ${MAX} iterations.`);
}

// ─── Orchestrator ─────────────────────────────────────────────────────────────

async function solveTask(): Promise<void> {
  log.info('=== Savethem multi-agent task started ===');

  // Try to load cached scout findings first
  let findings = loadCache();

  if (!findings) {
    findings = await runScout();
    saveCache(findings);
  }

  await runPlanner(findings);
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
