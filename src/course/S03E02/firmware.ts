/**
 * Firmware task — agentic loop
 *
 * Tools:
 *  - shell(cmd)   → POST to shell API
 *  - verify(code) → POST to VERIFY_URL
 */

import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { S03E02, VERIFY_URL } from 'src/shared/api';

const MODEL = 'anthropic/claude-sonnet-4-6'; // explicitly required by task hints

// ── Shell helper ──────────────────────────────────────────────────────────────

async function runShell(cmd: string): Promise<string> {
  try {
    const res = await axios.post(
      S03E02.SHELL_URL,
      { apikey: process.env.AIDEVS_API_KEY, cmd },
      { timeout: 15_000 },
    );

    console.debug('Shell response:', res.data);

    const data = res.data;
    if (typeof data === 'string') return data;

    // Shell returns { code, message, data } — the actual result is in .data
    // e.g. pwd → { code:125, message:"Current working directory.", data:"/" }
    // e.g. ls  → { code:125, message:"Directory listing.", data:[...] }
    if (data.data !== undefined) {
      const payload =
        typeof data.data === 'string' ? data.data : JSON.stringify(data.data, null, 2);
      return payload || data.message || '(empty)';
    }

    if (data.output !== undefined) return String(data.output);
    if (data.message !== undefined) return String(data.message);
    if (data.error) return `[API ERROR] ${data.error}`;

    return JSON.stringify(data);
  } catch (err: any) {
    const status = err.response?.status;
    const body = err.response?.data;

    if (status === 429) {
      console.warn('[RATE LIMIT] Waiting 5s...');
      await new Promise((r) => setTimeout(r, 5_000));
      return '[RATE LIMIT] Waited 5s. Please retry the same command.';
    }
    if (status === 503) {
      console.warn('[503] Waiting 3s...');
      await new Promise((r) => setTimeout(r, 3_000));
      return '[503] Service unavailable, waited 3s. Please retry.';
    }
    if (status === 403) {
      // Ban duration is in the response — parse it and actually wait
      const banMsg = JSON.stringify(body);
      const seconds = banMsg.match(/(\d+)\s*sec/i)?.[1];
      const waitMs = seconds ? parseInt(seconds) * 1_000 + 1_000 : 30_000;
      console.warn(`[BAN] Waiting ${waitMs / 1000}s before continuing...`);
      await new Promise((r) => setTimeout(r, waitMs));
      return `[BAN] Served ${waitMs / 1000}s ban. ${banMsg} — You may now retry.`;
    }

    return `[HTTP ${status}] ${JSON.stringify(body) ?? err.message}`;
  }
}

// ── Verify helper ─────────────────────────────────────────────────────────────

async function submitAnswer(code: string): Promise<string> {
  try {
    const res = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'firmware',
      answer: { confirmation: code },
    });
    return JSON.stringify(res.data);
  } catch (err: any) {
    return `[VERIFY ERROR] ${JSON.stringify(err.response?.data) ?? err.message}`;
  }
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'shell',
      description: `Execute a single shell command on the remote virtual machine.
IMPORTANT: Start with "help" — this is a NON-STANDARD shell, not regular Linux.
Do not assume standard Linux commands work. Read "help" output carefully first.

Rules (violations = timed ban + VM reset):
- Never access /etc, /root, /proc/
- Check for .gitignore in any directory before touching files; never touch listed paths.
- You are a regular (non-root) user.

If you get [BAN], stop issuing commands — the code already waited out the ban for you.
Returns the command output as a string.`,
      parameters: {
        type: 'object',
        properties: {
          cmd: { type: 'string', description: 'Shell command to execute' },
        },
        required: ['cmd'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'verify',
      description: 'Submit the ECCS-... code to Centrala /verify to complete the task.',
      parameters: {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description:
              'The code in format ECCS-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (40 chars after dash)',
          },
        },
        required: ['code'],
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `You are an agent operating on a restricted virtual machine to run firmware software.

You are FULLY AUTONOMOUS. Never ask for permission. Never say "Would you like me to...". 
Always call the shell tool immediately with the next logical command. Keep going until the task is done.

Execute this plan step by step — always use the shell tool for the next step, no pausing:
1. Run "help" — this is a non-standard shell with limited commands (ls, cat, cd, rm, reboot, pwd, editline, help, find, whoami).
2. Explore the filesystem: ls /, then navigate to find password files and settings.
3. Find the password for /opt/firmware/cooler/cooler.bin — it is stored in multiple places. Search thoroughly.
4. Read /opt/firmware/cooler/settings.ini to understand what needs to be fixed.
5. Fix settings.ini using "editline" (NOT standard editors — use the editline command from help).
6. Run /opt/firmware/cooler/cooler.bin with password and correct parameters.
7. When you see ECCS-[exactly 40 alphanumeric chars] in any output, immediately call verify tool.

Strict rules:
- NEVER access /etc, /root, /proc/ — instant ban.
- In each new directory, check for .gitignore first; never touch listed files.
- After [BAN]: wait is already done in code, issue one careful next command.
- After [RATE LIMIT] or [503]: retry the same command.
- ALWAYS issue the next shell command immediately after reading output. Do not stop to explain.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

const openrouter = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY!,
});

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  { role: 'system', content: SYSTEM },
  {
    role: 'user',
    content: 'Start the firmware task. Run "help" first.',
  },
];

async function run() {
  console.log(`=== Firmware agent started (model: ${MODEL}) ===\n`);
  let iteration = 0;
  const MAX = 80; // increased from 60

  while (iteration++ < MAX) {
    console.log(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.content) {
      console.log(`Agent: ${msg.content.slice(0, 500)}`);
    }

    // ECCS code: exactly 40 alphanumeric chars after dash
    const eccsMatch = msg.content?.match(/ECCS-[a-zA-Z0-9]{40}/);
    if (eccsMatch && !msg.tool_calls?.length) {
      console.log('\n=== ECCS code found in text, submitting... ===');
      const result = await submitAnswer(eccsMatch[0]);
      console.log('Verify result:', result);
      return;
    }

    if (!msg.tool_calls?.length) {
      console.log('[Agent] No tool calls — stopping.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;

      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw);

      console.log(`\n[Tool] ${name}(${argsRaw.slice(0, 200)})`);

      let result: string;

      if (name === 'shell') {
        await new Promise((r) => setTimeout(r, 400)); // gentle rate-limit buffer
        result = await runShell(args.cmd);
      } else if (name === 'verify') {
        result = await submitAnswer(args.code);
        console.log('\n=== VERIFY RESULT ===\n', result);
      } else {
        result = `Unknown tool: ${name}`;
      }

      console.log(`[Result] ${result.slice(0, 600)}`);

      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: result,
      });
    }
  }

  console.error(`\nTask not completed within ${MAX} iterations.`);
}

run().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
