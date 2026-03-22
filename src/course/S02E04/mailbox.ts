import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { S02E04, VERIFY_URL } from 'src/shared/api';

async function zmailRequest(body: Record<string, unknown>): Promise<string> {
  const res = await axios.post(S02E04.MAILBOX, { apikey: process.env.AIDEVS_API_KEY, ...body });
  return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'zmail_raw',
      description:
        'Call the zmail API with ANY action and parameters. ' +
        'Use this to read the full body of a message once you know the correct action name ' +
        'from the API documentation. Pass all fields in "params".',
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            description: 'Exact action name from the API docs, e.g. "getEmail", "readMessage".',
          },
          params: {
            type: 'object',
            description: 'Additional key-value parameters, e.g. { "id": "abc123" }',
            additionalProperties: true,
          },
        },
        required: ['action'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'zmail_get_inbox',
      description:
        'List inbox messages (metadata + snippet only, no full body). Supports pagination.',
      parameters: {
        type: 'object',
        properties: {
          page: { type: 'number', description: 'Page number (default 1)' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'zmail_search',
      description:
        'Search the mailbox with Gmail-style operators: from:, to:, subject:, OR, AND. ' +
        'Returns metadata + snippets only — NO full body. ' +
        'Always fetch full content with zmail_raw before drawing conclusions.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description:
              'Search query, e.g. "from:proton.me" or "subject:hasło OR subject:password"',
          },
          page: { type: 'number', description: 'Page number (default 1)' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'submit_answer',
      description:
        'Submit all three values to the verification hub. ' +
        'Call only when you have found all three. ' +
        'Read the hub response carefully — it says which fields are still wrong.',
      parameters: {
        type: 'object',
        properties: {
          password: { type: 'string', description: 'Employee system password' },
          date: { type: 'string', description: 'Attack date (YYYY-MM-DD)' },
          confirmation_code: {
            type: 'string',
            description: 'Security ticket code: SEC- + 32 chars = 36 chars total',
          },
        },
        required: ['password', 'date', 'confirmation_code'],
      },
    },
  },
];

async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case 'zmail_raw': {
      const { action, params = {} } = args as { action: string; params?: Record<string, unknown> };
      return zmailRequest({ action, ...params });
    }

    case 'zmail_get_inbox':
      return zmailRequest({ action: 'getInbox', page: args.page ?? 1 });

    case 'zmail_search':
      return zmailRequest({ action: 'search', query: args.query, page: args.page ?? 1 });

    case 'submit_answer': {
      const res = await axios.post(VERIFY_URL, {
        apikey: process.env.AIDEVS_API_KEY,
        task: 'mailbox',
        answer: {
          password: args.password,
          date: args.date,
          confirmation_code: args.confirmation_code,
        },
      });
      return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

const SYSTEM_PROMPT = `You are an autonomous security-research agent.
Your goal: extract three specific values from a live mailbox and submit them.

## Target values

| Field             | Description                                              | Format                          |
|-------------------|----------------------------------------------------------|---------------------------------|
| date              | Date the security dept plans to attack our power plant   | YYYY-MM-DD                      |
| password          | Employee system password still in the mailbox            | plain string                    |
| confirmation_code | Ticket confirmation code from the security dept          | SEC-<32 chars> = 36 chars total |

## Known facts

- Wiktor (informant) emailed FROM a @proton.me address.
- Gmail-style operators work: from:, to:, subject:, OR, AND.
- The mailbox is LIVE — new messages may arrive. Retry if something is missing.

## Critical workflow

### Step 0 — read API documentation
The API documentation has been provided to you at the start of the conversation.
Find the EXACT action name used to read a full message body.
It is NOT "getMessage" — that action does not exist. Use what the docs say.

### Step 1 — find the password
Search: subject:hasło OR subject:password
Fetch the full body of matching messages with zmail_raw using the correct action + id parameter.

### Step 2 — find the attack date
Search: from:proton.me   (Wiktor's report email)
Also try: subject:atak OR subject:attack OR subject:plan
Fetch full bodies.

### Step 3 — find the confirmation code
Search: subject:SEC- OR subject:potwierdzenie OR subject:ticket
The code is SEC- followed by exactly 32 alphanumeric characters.
Fetch full bodies of security ticket messages.

## Rules

- Snippets are truncated — ALWAYS fetch the full message before extracting values.
- Never call zmail_raw with action="getMessage" — it does not work.
- If the hub rejects your answer, keep searching based on its feedback.
- Continue until you receive {FLG:...}.`;

// ─── Bootstrap: pre-fetch API help ───────────────────────────────────────────
// Inject the real API documentation upfront so the agent knows the correct
// action names from iteration 1 and does not waste turns guessing.

async function fetchHelpContext(): Promise<string> {
  try {
    const [p1, p2] = await Promise.all([
      zmailRequest({ action: 'help', page: 1 }),
      zmailRequest({ action: 'help', page: 2 }),
    ]);
    return (
      'zmail API documentation (retrieved at startup):\n\n' +
      `=== Page 1 ===\n${p1}\n\n` +
      `=== Page 2 ===\n${p2}`
    );
  } catch (err: any) {
    return `Failed to pre-fetch API docs: ${err.message}`;
  }
}

async function solveTask(): Promise<void> {
  log.info('=== Mailbox task started ===');

  log.info('Pre-fetching zmail API documentation...');
  const helpContext = await fetchHelpContext();
  log.info(`[Help]\n${helpContext.slice(0, 600)}`);

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        `${helpContext}\n\n` +
        'Start the investigation. ' +
        'Use the correct action names from the documentation above when calling zmail_raw. ' +
        'Find the password, planned attack date, and confirmation code, then submit.',
    },
  ];

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
