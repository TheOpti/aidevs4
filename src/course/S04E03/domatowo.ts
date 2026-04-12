import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { MODEL_CLAUDE_SONNET, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

const log = {
  info: (...a: unknown[]) => console.log('[INFO]', ...a),
  error: (...a: unknown[]) => console.error('[ERR]', ...a),
};

// ── API helper ────────────────────────────────────────────────────────────────

async function callAPI(answer: Record<string, unknown>): Promise<unknown> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'domatowo',
      answer,
    });
    return data;
  } catch (error: any) {
    // Return the response body even on 4xx/5xx — it contains error details
    const body = error.response?.data;
    if (body) return body;
    return { error: error.message };
  }
}

// ── Tool implementations ──────────────────────────────────────────────────────
// CORRECT API PARAMS: "object" = unit hash, "where" = destination cell

async function apiHelp(): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'help' }));
}

async function apiGetMap(symbols?: string[]): Promise<string> {
  const answer: Record<string, unknown> = { action: 'getMap' };
  if (symbols?.length) answer.symbols = symbols;
  return JSON.stringify(await callAPI(answer));
}

async function apiGetLogs(): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'getLogs' }));
}

async function apiCreate(type: 'transporter' | 'scout', passengers?: number): Promise<string> {
  const answer: Record<string, unknown> = { action: 'create', type };
  if (type === 'transporter' && passengers !== undefined) answer.passengers = passengers;
  return JSON.stringify(await callAPI(answer));
}

async function apiMove(object: string, where: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'move', object, where }));
}

async function apiInspect(object: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'inspect', object }));
}

async function apiDropScouts(object: string): Promise<string> {
  // "unload" is the likely action name from help docs
  return JSON.stringify(await callAPI({ action: 'unload', object }));
}

async function apiCallHelicopter(destination: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'callHelicopter', destination }));
}

// Escape hatch: pass any raw JSON answer object to API
async function apiRawAction(answer: Record<string, unknown>): Promise<string> {
  return JSON.stringify(await callAPI(answer));
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'help',
      description: 'Get full API docs — exact action names, param names and types. Call FIRST.',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getMap',
      description: 'Get 11x11 map. Optional symbols filter e.g. ["B3","UL"].',
      parameters: {
        type: 'object',
        properties: {
          symbols: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getLogs',
      description: 'Get logs: unit positions, inspect results, action_points_left.',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'create',
      description:
        'Create unit. Returns JSON with "object" field = unit hash. SAVE this hash. ' +
        'Scout cost=5. Transporter cost=5+5*passengers.',
      parameters: {
        type: 'object',
        required: ['type'],
        properties: {
          type: { type: 'string', enum: ['scout', 'transporter'] },
          passengers: { type: 'number' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'move',
      description:
        'Move unit. Use "object" (hash from create) and "where" (cell like "D6"). ' +
        'Transporter: UL cells only, 1pt/cell. Scout: any cell, 7pt/cell.',
      parameters: {
        type: 'object',
        required: ['object', 'where'],
        properties: {
          object: { type: 'string', description: 'Unit hash from create response.' },
          where: { type: 'string', description: 'Target cell e.g. "D6", "F1", "B10".' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'inspect',
      description: 'Inspect current cell of unit. Cost 1pt. Returns whether partisan is here.',
      parameters: {
        type: 'object',
        required: ['object'],
        properties: {
          object: { type: 'string', description: 'Unit hash.' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'dropScouts',
      description: 'Unload scouts from transporter at current location. Cost 0pt.',
      parameters: {
        type: 'object',
        required: ['object'],
        properties: {
          object: { type: 'string', description: 'Transporter hash.' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'callHelicopter',
      description:
        'Call rescue helicopter to confirmed partisan location. Only after inspect confirms partisan.',
      parameters: {
        type: 'object',
        required: ['destination'],
        properties: {
          destination: { type: 'string', description: 'Cell confirmed by inspect e.g. "F1".' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'rawAction',
      description:
        'Send any raw "answer" JSON to the API. Use for debugging or trying exact param names from help docs.',
      parameters: {
        type: 'object',
        required: ['answer'],
        properties: {
          answer: {
            type: 'object',
            description:
              'Exact answer object, e.g. {"action":"move","object":"abc123","where":"F6"}',
            additionalProperties: true,
          },
        },
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `Jesteś dowódcą operacji ewakuacyjnej w mieście Domatowo. Budżet: 300 punktów akcji.

## MAPA 11x11
\`\`\`
   A    B    C    D    E    F    G    H    I    J    K
1  DR   UL   UL   UL   --   B3   B3   DR   --   PK   PK
2  DR   DR   --   UL   UL   B3   B3   DR   UL   PK   PK
3  --   --   --   UL   PK   --   --   DR   UL   --   --
4  B1   B1   --   UL   PK   SZ   SZ   SZ   UL   BS   BS
5  B1   B1   --   UL   PK   SZ   SZ   SZ   UL   BS   BS
6  UL   UL   UL   UL   UL   UL   UL   UL   UL   UL   --
7  B2   B2   --   UL   --   KS   KS   KS   --   DR   --
8  B2   B2   --   UL   --   KS   KS   KS   --   DR   --
9  --   UL   UL   UL   UL   UL   UL   UL   UL   UL   --
10 B3   B3   B3   --   DR   --   --   B3   B3   DR   --
11 B3   B3   B3   --   DR   --   --   B3   B3   DR   --
\`\`\`
UL=droga(tylko transportery), B3=Blok3p=NAJWYŻSZY, inne=dowolne dla zwiadowców

## CEL: Przeszukaj TYLKO komórki B3
- Góra:      F1, G1, F2, G2
- Dół-lewo:  A10, B10, C10, A11, B11, C11
- Dół-prawo: H10, I10, H11, I11

## FORMAT API (KRYTYCZNE)
Zwróć uwagę na poprawne nazwy pól:
- create → zwraca { "object": "<hash>", "spawn": "<cell>", ... }
- move   → { action:"move", object:"<hash>", where:"<cell>" }
- inspect→ { action:"inspect", object:"<hash>" }
- unload → { action:"unload", object:"<hash>" }  ← wysadzenie z transportera

Zawsze zapamiętaj pole "object" z odpowiedzi create — to jest ID jednostki.

## PLAN (optymalny koszt ~140 pkt)
Krok 1: help() → potwierdź nazwy akcji
Krok 2: create transporter 2 pasażerów (15 pkt) → zapisz hash T, spawny A6/B6/C6
Krok 3: move T: A6→D6 (3 pola=3 pkt), D6→D1 (5 pól=5 pkt) — tylko UL!
Krok 4: unload T na D1 (0 pkt) → zwiadowcy S1,S2 na D1
Krok 5: move S1: D1→E1→F1 (2 pola=14 pkt), inspect F1 (1 pkt)
         move S2: D1→E1→G1 (2+1=21 pkt), inspect G1 (1 pkt)  ← lub E2→F2, G2
Krok 6: move T: D1→D6→D9→B9 (12 pkt)
Krok 7: create scout (5 pkt), move do B10 (B9→B10 =7 pkt), inspect (1 pkt)
Krok 8: powtarzaj dla pozostałych B3
Krok 9: przy znalezieniu → callHelicopter(cell)

Jeśli inspect zwróci informację o człowieku/partyzancie → NATYCHMIAST callHelicopter.
Po każdej grupie akcji: getLogs() by sprawdzić punkty i pozycje.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== Domatowo task starting ===\n');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content:
        'Zacznij od help() żeby potwierdzić dokładne nazwy parametrów. ' +
        'Potem utwórz transporter z 2 zwiadowcami i przeszukaj B3. ' +
        'WAŻNE: po create zapamiętaj pole "object" z odpowiedzi — to hash jednostki. ' +
        'Do move używaj: object=<hash>, where=<cell>.',
    },
  ];

  let iteration = 0;
  const MAX = 120;
  let taskDone = false;

  while (iteration++ < MAX && !taskDone) {
    log.info(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_CLAUDE_SONNET,
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
      log.info(`[Tool] ${name}(${argsRaw?.slice(0, 400)})`);

      let result: string;
      try {
        switch (name) {
          case 'help':
            result = await apiHelp();
            break;
          case 'getMap':
            result = await apiGetMap(args.symbols);
            break;
          case 'getLogs':
            result = await apiGetLogs();
            break;
          case 'create':
            result = await apiCreate(args.type, args.passengers);
            break;
          case 'move':
            result = await apiMove(args.object, args.where);
            break;
          case 'inspect':
            result = await apiInspect(args.object);
            break;
          case 'dropScouts':
            result = await apiDropScouts(args.object);
            break;
          case 'rawAction':
            result = await apiRawAction(args.answer);
            break;
          case 'callHelicopter':
            result = await apiCallHelicopter(args.destination);
            log.info('🚁 HELICOPTER:', result);
            try {
              const p = JSON.parse(result);
              if (p?.flag || p?.code === 0 || JSON.stringify(p).includes('flag')) {
                log.info('✅ FLAG OBTAINED:', result);
                taskDone = true;
              }
            } catch {}
            break;
          default:
            result = `Unknown tool: ${name}`;
        }
      } catch (err: unknown) {
        result = `Tool error: ${String(err)}`;
        log.error(`[Tool error] ${name}: ${err}`);
      }

      log.info(`[Result] ${result?.slice(0, 600)}`);
      messages.push({ role: 'tool', tool_call_id: call.id, content: result! });
    }
  }

  if (iteration >= MAX) log.error(`Failed within ${MAX} iterations.`);
}

solveTask().catch((err) => {
  log.error('Fatal:', err);
  process.exit(1);
});
