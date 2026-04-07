import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_CLAUDE_SONNET, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Block {
  col: number;
  top_row: number;
  bottom_row: number;
  direction: 'up' | 'down';
}

// ── API helper ────────────────────────────────────────────────────────────────

async function sendCommand(command: string): Promise<any> {
  const res = await axios.post(VERIFY_URL, {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'reactor',
    answer: { command },
  });
  return res.data;
}

// ── Physics (pure, no API calls) ──────────────────────────────────────────────

function stepBlock(b: Block): Block {
  let { col, top_row, bottom_row, direction } = b;
  if (direction === 'down') {
    top_row++;
    bottom_row++;
    if (bottom_row >= 5) direction = 'up';
  } else {
    top_row--;
    bottom_row--;
    if (top_row <= 1) direction = 'down';
  }
  return { col, top_row, bottom_row, direction };
}

function isCrushed(blocks: Block[], col: number): boolean {
  return blocks.some((b) => b.col === col && b.bottom_row === 5);
}

// ── Lookahead table generator ─────────────────────────────────────────────────

function buildLookahead(blocks: Block[], lookaheadSteps = 6): string {
  const COLS = 7;
  let state = [...blocks];
  const rows: string[] = [];

  rows.push('Steps ahead | ' + Array.from({ length: COLS }, (_, i) => `col${i + 1}`).join(' | '));
  rows.push('------------|' + '---------|'.repeat(COLS));

  for (let step = 1; step <= lookaheadSteps; step++) {
    state = state.map(stepBlock);
    const cells = Array.from({ length: COLS }, (_, i) =>
      isCrushed(state, i + 1) ? ' DEADLY ' : '  safe  ',
    );
    rows.push(`  step +${step}    | ` + cells.join(' | '));
  }

  return rows.join('\n');
}

// ── Message builder ───────────────────────────────────────────────────────────

function buildUserMessage(stateJson: string, blocks: Block[]): string {
  const lookahead = buildLookahead(blocks, 6);
  return (
    `Here is the current board state:\n\n${stateJson}\n\n` +
    `PRE-COMPUTED DANGER TABLE (do NOT re-calculate this yourself — trust it completely):\n\`\`\`\n${lookahead}\n\`\`\`\n\n` +
    `"step +1" = what happens after your NEXT command.\n` +
    `"step +2" = what happens after the command after that.\n` +
    `A column marked DEADLY at step +1 will crush the robot if it is there after your next command.\n\n` +
    `Choose your next command using the table above. Think briefly, then call send_command.`
  );
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'send_command',
      description:
        'Send a movement command to the robot. Every command (including "wait") advances all blocks by one step. Returns the full new board state.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            enum: ['right', 'left', 'wait'],
            description:
              '"right" moves the robot one column to the right. "left" moves it one column to the left. "wait" keeps the robot in place but still advances all blocks by one step.',
          },
        },
        required: ['command'],
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `\
You are an AI controlling a robot navigating a 7×5 nuclear reactor grid.

═══ OBJECTIVE ═══
Move the robot from col 1, row 5 to col 7, row 5.
The robot always stays on row 5. Commands: "right", "left", "wait".
Every command advances all blocks by one step.

═══ MAP SYMBOLS ═══
P = robot  |  G = goal  |  B = reactor block  |  . = empty

═══ YOUR ONLY JOB ═══
You will receive a PRE-COMPUTED DANGER TABLE before each decision.
Trust it completely — NEVER try to recalculate block positions yourself.
The table already accounts for block physics, bouncing, and direction reversals.

═══ DECISION RULES ═══
1. Check the danger table at "step +1" — this is what happens after your next command.
2. "right"  → the robot moves to (current col + 1). That column must be safe at step +1.
3. "wait"   → the robot stays in (current col).    That column must be safe at step +1.
4. "left"   → the robot moves to (current col - 1). That column must be safe at step +1.

PRIORITY ORDER:
- Prefer "right" when safe (make forward progress).
- Use "wait" to let a block pass if moving right is deadly.
- Use "left" only if both "right" and "wait" are deadly.
- Also check "step +2" to avoid trapping yourself with no escape next turn.

═══ WINNING ═══
Keep calling send_command until reached_goal is true.
Think briefly before each move, then call send_command.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== Reactor agent starting ===\n');

  const startState = await sendCommand('start');
  log.info('Initial state:\n' + JSON.stringify(startState, null, 2));

  if (startState.reached_goal) {
    log.info('Already at goal!');
    log.info(`\n🏁 FLAG: ${startState.message}`);
    return;
  }

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content: buildUserMessage(JSON.stringify(startState, null, 2), startState.blocks as Block[]),
    },
  ];

  let iteration = 0;
  const MAX = 80;

  while (iteration++ < MAX) {
    log.info(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_CLAUDE_SONNET,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.content) {
      log.info(`Agent reasoning: ${msg.content.slice(0, 800)}`);
    }

    if (!msg.tool_calls?.length) {
      log.info('[Agent] No tool calls — stopping loop.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;

      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw);

      log.info(`\n[Tool] ${name}(${argsRaw})`);

      let result: string;

      if (name === 'send_command') {
        try {
          const data = await sendCommand(args.command);
          result = JSON.stringify(data);

          // ── Goal reached — flag is in the message, no board to render ──
          if (data.code === 0 || data.reached_goal) {
            log.info('\n🎉 GOAL REACHED!');
            log.info(`\n🏁 FLAG: ${data.message}`);
            messages.push({ role: 'tool', tool_call_id: call.id, content: result });
            return;
          }

          // ── Normal step — board is present ──
          const boardStr = (data.board as string[][]).map((row) => row.join(' ')).join('\n');
          log.info(`Board after "${args.command}":\n${boardStr}`);
          log.info(`Player: col ${data.player?.col}  |  reached_goal: ${data.reached_goal}`);

          messages.push({
            role: 'user',
            content: buildUserMessage(JSON.stringify(data, null, 2), data.blocks as Block[]),
          });
        } catch (err: any) {
          const errMsg = err.response?.data ? JSON.stringify(err.response.data) : err.message;
          result = `API error: ${errMsg}`;
          log.error('[Tool error]', result);
        }
      } else {
        result = `Unknown tool: ${name}`;
        messages.push({ role: 'tool', tool_call_id: call.id, content: result });
      }

      log.info(`[Tool result preview] ${result.slice(0, 400)}`);
    }
  }

  log.error(`\nFailed to reach goal within ${MAX} iterations.`);
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
