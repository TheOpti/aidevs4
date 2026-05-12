import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import * as readline from 'readline';
import { log, MODEL_GEMMA4_26B, openrouter } from 'src/shared/agents';
import { S05E05, VERIFY_URL } from 'src/shared/api';

// ── PWR Protection Table — fetched from docs at startup ──────────────────────
let PWR_TABLE: Record<number, number> = {};

async function loadPWRTable(): Promise<void> {
  log.info('[PWR] Fetching protection table from documentation…');
  const { data: md } = await axios.get<string>(S05E05.DOCS_URL);

  // Each table row looks like: | 1500 | 03 | 1501 | 03 | … (up to 10 pairs per row)
  const cellPair = /\|\s*(\d{4})\s*\|\s*(\d+)\s*/g;
  let match: RegExpExecArray | null;
  let count = 0;
  while ((match = cellPair.exec(md)) !== null) {
    PWR_TABLE[parseInt(match[1], 10)] = parseInt(match[2], 10);
    count++;
  }

  if (count === 0) throw new Error('PWR table parse failed — no entries found');
  log.info(
    `[PWR] Loaded ${count} entries (years ${Math.min(...Object.keys(PWR_TABLE).map(Number))}–${Math.max(...Object.keys(PWR_TABLE).map(Number))})`,
  );
}

// ── Helper calculations ───────────────────────────────────────────────────────

function calculateSyncRatio(day: number, month: number, year: number): string {
  const raw = (day * 8 + month * 12 + year * 7) % 101;
  return (raw / 100).toFixed(2);
}

function getPWRForYear(year: number): number {
  return PWR_TABLE[year] ?? -1;
}

function getRequiredInternalMode(year: number): number {
  if (year < 2000) return 1;
  if (year <= 2150) return 2;
  if (year <= 2300) return 3;
  return 4;
}

// ── Readline for operator interaction ─────────────────────────────────────────

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function askOperator(question: string): Promise<string> {
  return new Promise((resolve) => rl.question(question, resolve));
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function callAPI(answer: Record<string, unknown>): Promise<unknown> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'timetravel',
      answer,
    });
    return data;
  } catch (error: any) {
    const body = error.response?.data;
    if (body) return body;
    return { error: error.message };
  }
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'calculate_sync_ratio',
      description: `Calculate the syncRatio for a target date and return the EXACT value to send to the API.
Formula: raw = (day×8 + month×12 + year×7) mod 101. Then syncRatio = raw / 100.
CRITICAL: The returned "syncRatioValue" is the final number to pass to api_action configure syncRatio.
Do NOT recompute it. Do NOT divide again. Use the returned value directly.
Examples: raw=82 → syncRatioValue=0.82 | raw=8 → syncRatioValue=0.08 | raw=100 → syncRatioValue=1.00`,
      parameters: {
        type: 'object',
        properties: {
          day: { type: 'number', description: 'Day of target date (1-31)' },
          month: { type: 'number', description: 'Month of target date (1-12)' },
          year: { type: 'number', description: 'Year of target date (1500-2499)' },
        },
        required: ['day', 'month', 'year'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_pwr_for_year',
      description:
        'Look up the required PWR protection level for a given target year from the CHRONOS-P1 protection table.',
      parameters: {
        type: 'object',
        properties: {
          year: { type: 'number', description: 'Target year (1500-2499)' },
        },
        required: ['year'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_required_internal_mode',
      description: `Get the internalMode value required for a given target year.
Mode 1: years below 2000
Mode 2: years 2000-2150
Mode 3: years 2151-2300
Mode 4: years 2301 and above`,
      parameters: {
        type: 'object',
        properties: {
          year: { type: 'number', description: 'Target year' },
        },
        required: ['year'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'api_action',
      description: `Call the CHRONOS-P1 machine API.
Available actions:
- "help": Get help and available commands
- "getConfig": Get current device configuration and status (also returns stabilization hints after date is set)
- "reset": Reset the device to defaults
- "configure": Set a parameter value (requires param and value)

Configurable params (only when device is in standby):
- "day" (number 1-31)
- "month" (number 1-12)  
- "year" (number 1500-2499)
- "syncRatio" (decimal 0.00-1.00)
- "stabilization" (number, read hint from getConfig response)`,
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['help', 'getConfig', 'reset', 'configure'],
          },
          param: {
            type: 'string',
            enum: ['day', 'month', 'year', 'syncRatio', 'stabilization'],
            description: 'Required only for "configure" action',
          },
          value: {
            type: 'number',
            description: 'Required only for "configure" action',
          },
        },
        required: ['action'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'instruct_operator',
      description: `Display instructions to the human operator and wait for their confirmation.
Use this to tell the operator what to do MANUALLY in the web UI preview (PT-A, PT-B, PWR slider, standby/active toggle).
IMPORTANT: PT-A, PT-B, PWR, and standby/active can ONLY be changed manually in the web UI — NOT via API.
Also use this to tell the operator what internalMode to wait for, or when to activate and jump.`,
      parameters: {
        type: 'object',
        properties: {
          instructions: {
            type: 'string',
            description: 'Clear, specific instructions for the operator',
          },
          wait_for_confirmation: {
            type: 'boolean',
            description: 'If true, wait for operator to press Enter before continuing',
            default: true,
          },
        },
        required: ['instructions'],
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `You are the CHRONOS-P1 time machine assistant. Your role is to guide a human operator through 3 jumps step by step, handling API configuration and telling the operator what to do manually.

## MISSION
Three jumps are required:
1. JUMP 1 (future): Travel to November 5, 2238 → pick up new battery pack
2. JUMP 2 (back to present): Return to May 12, 2026 (today's date)
3. JUMP 3 (time tunnel to past): Open a time tunnel to November 12, 2024 — the day before Rafał was found in the cave

## DOCUMENTATION

### syncRatio formula
syncRatio = ((day × 8) + (month × 12) + (year × 7)) mod 101 ÷ 100
Result must be 2 decimal places: 82 → 0.82, 0 → 0.00, 100 → 1.00

### internalMode (auto-changes every few seconds, cannot be set manually)
- Mode 1: year < 2000
- Mode 2: year 2000–2150
- Mode 3: year 2151–2300
- Mode 4: year 2301+

### Switches
- PT-A only (ON): Travel to the PAST
- PT-B only (ON): Travel to the FUTURE
- PT-A + PT-B both ON: Open a TIME TUNNEL (requires ≥60% battery)

### API parameters (only configurable when device is in STANDBY mode)
day, month, year, syncRatio, stabilization

### Manual UI controls only (NOT via API)
PT-A switch, PT-B switch, PWR slider, standby/active toggle

### Flux density
Calculated automatically — must reach 100% before jump is possible.
It increases as more parameters are configured correctly.

### Stabilization
After setting day, month, year → call getConfig to get the stabilization hint → set it via API.

### Jumping
Only possible when: flux density = 100%, device is in ACTIVE mode, internalMode matches target year range.

## PROCEDURE FOR EACH JUMP

For each jump, follow this exact sequence:
1. Call api_action("getConfig") to check current state
2. Make sure device is in STANDBY — instruct operator if not
3. Use calculate_sync_ratio to compute syncRatio for target date
4. Use get_pwr_for_year to find PWR level
5. Use get_required_internal_mode to find required internalMode
6. Configure via API: day, month, year, syncRatio (in standby)
7. Call api_action("getConfig") to read stabilization hint
8. Configure stabilization via API
9. Use instruct_operator to tell operator:
   - What PWR value to set on the slider
   - Which switches to enable (PT-A, PT-B, or both)
   - To switch to ACTIVE mode
10. Use instruct_operator to tell operator: "Wait until internalMode shows X, then click the pulsing sphere"
11. After jump success, proceed to next jump

## IMPORTANT NOTES
- Device must be in STANDBY before any API configure call
- For JUMP 1 (to 2238): PT-B only (future)
- For JUMP 2 (back to 2026): PT-A only (past)
- For JUMP 3 (tunnel to 2024): PT-A + PT-B both ON (tunnel mode)
- Battery recharge happens after arriving in 2238
- Always call getConfig after each configuration step to verify

Start by calling api_action("help") to see the initial state, then guide the operator step by step through all 3 jumps.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== CHRONOS-P1 Time Machine Assistant ===\n');
  log.info('This assistant will guide you through 3 time jumps.');
  log.info(`Please have the web UI open at: ${S05E05.TIMETRAVEL_PREVIEW}\n`);

  await loadPWRTable();

  // Track the last day/month/year set via API so we can auto-calculate syncRatio
  const configuredDate = { day: 0, month: 0, year: 0 };

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content:
        'Start the time travel mission. Guide me through all 3 jumps: first to 2238-11-05 to get batteries, then back to 2026-05-12, and finally open a tunnel to 2024-11-12.',
    },
  ];

  let iteration = 0;
  const MAX = 60;
  let taskDone = false;

  while (iteration++ < MAX && !taskDone) {
    log.info(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_GEMMA4_26B,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.content) {
      log.info(`\n🤖 Assistant: ${msg.content}`);
    }

    if (!msg.tool_calls?.length) {
      log.info('\nNo tool calls — agent has finished.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;
      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw || '{}');

      let result: string;
      try {
        if (name === 'calculate_sync_ratio') {
          const { day, month, year } = args;
          const ratio = calculateSyncRatio(day, month, year);
          const rawVal = (day * 8 + month * 12 + year * 7) % 101;
          result = JSON.stringify({
            syncRatioValue: parseFloat(ratio),
            readyToSend: ratio,
            calculation: `(${day}×8 + ${month}×12 + ${year}×7) mod 101 = ${rawVal}, then ${rawVal}/100 = ${ratio}`,
            WARNING: `Use syncRatioValue=${ratio} directly in api_action. Do NOT divide again.`,
          });
          log.info(`[calculate_sync_ratio] ${day}/${month}/${year} → syncRatioValue=${ratio}`);
        } else if (name === 'get_pwr_for_year') {
          const pwr = getPWRForYear(args.year);
          result = JSON.stringify({ year: args.year, pwr, found: pwr !== -1 });
          log.info(`[get_pwr_for_year] ${args.year} → PWR=${pwr}`);
        } else if (name === 'get_required_internal_mode') {
          const mode = getRequiredInternalMode(args.year);
          result = JSON.stringify({ year: args.year, requiredInternalMode: mode });
          log.info(`[get_required_internal_mode] ${args.year} → mode=${mode}`);
        } else if (name === 'api_action') {
          const { action, param } = args;
          let { value } = args;

          // Track configured date so we can auto-calculate syncRatio
          if (action === 'configure') {
            if (param === 'day') configuredDate.day = value;
            if (param === 'month') configuredDate.month = value;
            if (param === 'year') configuredDate.year = value;
          }

          // Override syncRatio: always compute from the tracked date, never trust LLM's value
          let syncRatioError = false;
          if (action === 'configure' && param === 'syncRatio') {
            const { day, month, year } = configuredDate;
            if (!day || !month || !year) {
              result = JSON.stringify({
                error: 'Set day, month, and year before configuring syncRatio.',
              });
              syncRatioError = true;
            } else {
              value = parseFloat(calculateSyncRatio(day, month, year));
              log.info(
                `[api_action] syncRatio auto-calculated from ${day}/${month}/${year} → ${value}`,
              );
            }
          }

          if (!syncRatioError) {
            let payload: Record<string, unknown> = { action };
            if (action === 'configure' && param !== undefined) {
              payload = { action, param, value };
            }
            log.info(`[api_action] Calling: ${JSON.stringify(payload)}`);
            const apiResult = await callAPI(payload);
            result = JSON.stringify(apiResult);
            log.info(`[api_action] Response: ${JSON.stringify(apiResult).slice(0, 300)}`);

            // Check if we got a flag in the response
            const resultStr = JSON.stringify(apiResult);
            if (resultStr.includes('FLG:') || resultStr.includes('{{FLG')) {
              log.info('\n🎉 FLAG FOUND! Task complete!');
              log.info(resultStr);
              taskDone = true;
            }
          }
        } else if (name === 'instruct_operator') {
          const { instructions, wait_for_confirmation = true } = args;
          log.info('\n' + '═'.repeat(60));
          log.info('👤 OPERATOR INSTRUCTIONS:');
          log.info('═'.repeat(60));
          log.info(instructions);
          log.info('═'.repeat(60));
          if (wait_for_confirmation) {
            await askOperator('\n✅ Press Enter when done...');
            result = 'Operator confirmed action completed.';
          } else {
            result = 'Instructions displayed to operator.';
          }
        } else {
          result = `Unknown tool: ${name}`;
        }
      } catch (err: unknown) {
        result = `Tool error: ${String(err)}`;
        log.error(`[Error] ${name}: ${err}`);
      }

      messages.push({ role: 'tool', tool_call_id: call.id, content: result! });
    }
  }

  if (iteration >= MAX) log.error(`\nMax iterations (${MAX}) reached without completing task.`);

  rl.close();
}

solveTask().catch((err) => {
  log.error('Fatal:', err);
  rl.close();
  process.exit(1);
});
