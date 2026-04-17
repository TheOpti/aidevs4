import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_GEMMA4_26B, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

// ── API helpers ───────────────────────────────────────────────────────────────

async function runCmd(cmd: string): Promise<string> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'shellaccess',
      answer: { cmd },
    });
    return JSON.stringify(data);
  } catch (error: any) {
    return JSON.stringify(error.response?.data ?? { error: error.message });
  }
}

// Submit builds the echo command itself — no risk of agent mangling the JSON
async function submitAnswer(params: {
  date: string;
  city: string;
  latitude: number;
  longitude: number;
}): Promise<string> {
  const json = JSON.stringify(params);
  const cmd = `echo '${json}'`;
  log.info(`[Submit] ${cmd}`);
  return runCmd(cmd);
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'shell',
      description:
        'Run a shell command on the remote server. ' +
        'Available: ls, find, cat, grep, sed, tr, echo, date, jq. ' +
        'NOT available: awk, python3, perl.',
      parameters: {
        type: 'object',
        required: ['cmd'],
        properties: {
          cmd: { type: 'string' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'submit',
      description:
        'Submit the final answer once you have confirmed all four values. ' +
        'This builds and sends the required JSON to the verification server. ' +
        'Call this ONLY after verifying the data from the log files — not before.',
      parameters: {
        type: 'object',
        required: ['date', 'city', 'latitude', 'longitude'],
        properties: {
          date: {
            type: 'string',
            description: 'The date ONE DAY BEFORE Rafał was found. Format: YYYY-MM-DD.',
          },
          city: {
            type: 'string',
            description: 'City name exactly as it appears in the log files.',
          },
          latitude: {
            type: 'number',
            description: 'Latitude from gps.json for the matching entry_id.',
          },
          longitude: {
            type: 'number',
            description: 'Longitude from gps.json for the matching entry_id.',
          },
        },
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `You are a shell-based investigator. Your goal: find when and where Rafał's body was discovered, then submit an answer dated ONE DAY EARLIER.

## DATA SOURCES
- /data/time_logs.csv  — events with format: DATE;DESCRIPTION;LOCATION_ID;ENTRY_ID
- /data/gps.json       — GPS coordinates with entry_id, latitude, longitude

## INVESTIGATION STEPS

### Step 1 — Explore
  ls /data
  cat /data/time_logs.csv | head -5   # understand CSV format
  cat /data/gps.json | head -30       # understand JSON structure

### Step 2 — Find the body discovery event
  grep -i "ciało\|zwłoki\|znaleziono\|odnaleziono\|body\|found" /data/time_logs.csv

  This will return a line like:
    2024-11-13;W jaskini znaleziono ciało...;LOCATION_ID;ENTRY_ID

  Extract: DATE, LOCATION_ID, ENTRY_ID

### Step 3 — Find the city name for that LOCATION_ID
  grep ";LOCATION_ID;" /data/time_logs.csv | head -10
  # Look for a line that names the city explicitly

  Or search a locations/cities file if it exists:
  ls /data
  grep "LOCATION_ID" /data/*.json 2>/dev/null | head -5

### Step 4 — Find GPS coordinates for that ENTRY_ID
  Use ONLY this reliable command (awk is not available):
  cat /data/gps.json | tr '\\n' ' ' | sed 's/}, {/}\\n{/g' | grep '"entry_id": ENTRY_ID'

  This prints the full JSON object for that entry_id on one line.
  Then extract lat/lon with a second grep or sed on that single line.

### Step 5 — Compute date minus 1 day
  date -d "FOUND_DATE - 1 day" +%Y-%m-%d

### Step 6 — Submit
  Call the submit() tool with the four values.
  Do NOT manually write echo — use submit() so the format is guaranteed correct.

## COMMON MISTAKES TO AVOID
- Do NOT use awk or python3 (not installed)
- Do NOT grep for just the entry_id number — it may appear in multiple entries; always grep the full '"entry_id": NUMBER' pattern
- Do NOT submit until you have confirmed the city name from the logs
- The answer date must be the day BEFORE discovery, not the discovery date itself`;

// ── Agent loop ────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== Shellaccess task starting ===\n');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content:
        "Start by exploring /data, then find the event where Rafał's body was discovered. " +
        'Extract the date, city, and GPS coordinates. Then call submit() with date set to ONE DAY BEFORE the discovery.',
    },
  ];

  let iteration = 0;
  const MAX = 40;
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

    if (msg.content) log.info(`Agent: ${msg.content.slice(0, 800)}`);
    if (!msg.tool_calls?.length) {
      log.info('No tool calls — done.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;
      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw || '{}');

      let result: string;
      try {
        if (name === 'shell') {
          log.info(`[Shell] $ ${args.cmd}`);
          result = await runCmd(args.cmd);
        } else if (name === 'submit') {
          log.info(
            `[Submit] date=${args.date} city=${args.city} lat=${args.latitude} lon=${args.longitude}`,
          );
          result = await submitAnswer(args);
          try {
            const parsed = JSON.parse(result);
            if (parsed?.flag || JSON.stringify(parsed).match(/\{\{|\}\}/)) {
              log.info('\n✅ FLAG:', result);
              taskDone = true;
            } else if (parsed?.code === -850) {
              log.info('⚠️  Verification rejected — wrong values, keep investigating');
            }
          } catch {}
        } else {
          result = `Unknown tool: ${name}`;
        }
      } catch (err: unknown) {
        result = `Tool error: ${String(err)}`;
        log.error(`[Error] ${name}: ${err}`);
      }

      log.info(`[Output] ${result!.slice(0, 600)}`);
      messages.push({ role: 'tool', tool_call_id: call.id, content: result! });
    }
  }

  if (iteration >= MAX) log.error(`Failed within ${MAX} iterations.`);
}

solveTask().catch((err) => {
  log.error('Fatal:', err);
  process.exit(1);
});
