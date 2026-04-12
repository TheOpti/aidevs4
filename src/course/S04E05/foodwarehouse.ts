import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { MODEL_CLAUDE_SONNET, openrouter } from 'src/shared/agents';
import { S04E05, VERIFY_URL } from 'src/shared/api';

// ── API helper ────────────────────────────────────────────────────────────────
async function callAPI(answer: Record<string, unknown>): Promise<unknown> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'foodwarehouse',
      answer,
    });
    return data;
  } catch (error: any) {
    const body = error.response?.data;
    if (body) return body;
    return { error: error.message };
  }
}

// ── Tool implementations ──────────────────────────────────────────────────────
async function toolHelp(): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'help' }));
}

async function toolReset(): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'reset' }));
}

async function toolDatabase(query: string): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'database', query }));
}

async function toolSignature(args: Record<string, unknown>): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'signatureGenerator', ...args }));
}

async function toolOrders(action: string, params: Record<string, unknown> = {}): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'orders', action, ...params }));
}

async function toolDone(): Promise<string> {
  return JSON.stringify(await callAPI({ tool: 'done' }));
}

// ── Tool definitions ──────────────────────────────────────────────────────────
const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'help',
      description: 'Fetch full API documentation for the foodwarehouse task.',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'reset',
      description: 'Reset all orders back to initial state.',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'database',
      description:
        'Execute a read-only SQL query against the SQLite database (SELECT or "show tables").',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'SQL query, e.g. "show tables" or "SELECT * FROM users"',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'signatureGenerator',
      description:
        'Generate a SHA1 security signature required when creating an order. Pass user fields from the database as arguments.',
      parameters: {
        type: 'object',
        properties: {
          args: {
            type: 'object',
            description: 'Key-value pairs from the database row used to generate the signature.',
          },
        },
        required: ['args'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'orders',
      description: 'Manage orders: get list, create a new order, append items, or delete an order.',
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['get', 'create', 'append', 'delete'],
            description: 'Operation to perform.',
          },
          params: {
            type: 'object',
            description: `Extra fields depending on action:
- create: { title, creatorID, destination, signature }
- append: { id, items: { goodName: quantity, ... } }  (batch mode — always use object form)
- delete: { id }`,
          },
        },
        required: ['action'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'done',
      description:
        'Submit all orders for final verification. Call only when every city order is complete.',
      parameters: { type: 'object', properties: {} },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────
const SYSTEM = `
You are an agent managing a food-and-tools warehouse. Your job is to create one delivery order per city listed in the food4cities.json data, then submit them for verification.

## Workflow

1. Call 'help' to understand the full API if needed.
2. Call 'database' with "show tables" to discover the schema.
3. Inspect every table to understand all columns — pay special attention to any role, permission, or responsibility fields.
4. Find the ONE user who is responsible for transport. Do NOT assume any user_id — query explicitly for a role, flag, or column that marks someone as the transport responsible person. Use that user's id as creatorID and their fields (login, birthday, etc.) for the signature.
5. Find destination codes for each city (city name → destination value).
6. For each city in the JSON data:
   a. Call 'signatureGenerator' with { action: "generate", login, birthday, destination } to get a SHA1 hash.
   b. Call 'orders' action=create with title, creatorID, destination, and the hash as signature.
   c. Call 'orders' action=append with the order id and ALL items as a single batch object.
7. After all orders are created and filled, call 'done'.

## Rules
- NEVER guess or arbitrarily pick a creatorID — the database tells you exactly which user is the transport responsible person.
- Never guess destination codes or signatures — always derive them from the database.
- Always use batch append (items as an object) instead of one append call per product.
- Create exactly as many orders as there are cities in the JSON — no more, no less.
- Items must match the JSON exactly: no missing goods, no extra goods, correct quantities.
- If something goes wrong, call 'reset' and start over from step 2.
`.trim();

// ── Agent loop ────────────────────────────────────────────────────────────────
async function solveTask() {
  console.log('=== Foodwarehouse task starting ===\n');

  // ── Download food4cities.json ─────────────────────────────────────────────
  console.log('Fetching food4cities.json...');
  const { data: cityNeeds } = await axios.get(S04E05.FOOD_URL);
  console.log('Fetched. Cities:', Object.keys(cityNeeds).join(', '));

  // ── Reset to clean state ──────────────────────────────────────────────────
  console.log('Resetting orders...');
  console.log(await toolReset());

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content: `Here is the food4cities.json data — the exact goods and quantities each city needs:\n\n${JSON.stringify(cityNeeds, null, 2)}\n\nNow explore the database, generate signatures, create one order per city, fill each order with its items, then call done.`,
    },
  ];

  let iteration = 0;
  const MAX = 60;
  let taskDone = false;

  while (iteration++ < MAX && !taskDone) {
    console.log(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_CLAUDE_SONNET,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.content) console.log(`Agent: ${msg.content.slice(0, 800)}`);
    if (!msg.tool_calls?.length) {
      console.log('No tool calls — stopping.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;
      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw || '{}');
      console.log(`[Tool] ${name}(${argsRaw?.slice(0, 600)})`);

      let result: string;
      try {
        switch (name) {
          case 'help':
            result = await toolHelp();
            break;
          case 'reset':
            result = await toolReset();
            break;
          case 'database':
            result = await toolDatabase(args.query);
            break;
          case 'signatureGenerator':
            result = await toolSignature(args.args);
            break;
          case 'orders':
            result = await toolOrders(args.action, args.params);
            break;
          case 'done':
            result = await toolDone();
            taskDone = true;
            break;
          default:
            result = `Unknown tool: ${name}`;
        }
      } catch (err: unknown) {
        result = `Tool error: ${String(err)}`;
        console.error(`[Tool error] ${name}: ${err}`);
      }

      console.log(`[Result] ${result!.slice(0, 300)}`);
      messages.push({ role: 'tool', tool_call_id: call.id, content: result! });
    }
  }

  if (iteration >= MAX) console.error(`Agent did not finish within ${MAX} iterations.`);
}

solveTask().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
