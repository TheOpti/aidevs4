import axios from 'axios';
import fs from 'fs';
import OpenAI from 'openai';
import path from 'path';
import { log, MODEL_DEEPSEEK, MODEL_GEMMA, openai, openrouter } from 'src/shared/agents';
import { sendResult } from 'src/shared/api';

const DATA_DIR = path.join(process.cwd(), 'src', 'data');
const DECLARATION_PATH = path.join(DATA_DIR, 'declaration.md');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Remove old declaration if exists
if (fs.existsSync(DECLARATION_PATH)) {
  fs.unlinkSync(DECLARATION_PATH);
}

const SYSTEM_PROMPT = `You are an expert in SPK transport declaration documents.

Follow these steps IN ORDER. Do not skip any step. Do not call submitAnswer until step 3.

STEP 1 — Download ALL documentation at once:
  Call fetchTextFile("${process.env.BASE_URL}/dane/doc/index.md")
  This single call will return the index AND every included file combined.
  If there is an image file (e.g. .png, .jpg) in the documentation, call fetchImage tool to get its content.
  Some files say you have no right to access them - ignore them.
  There's also "WDP" key - you need to calculate how many additional wagons we might need.

STEP 2 — Read all the returned content carefully.
  Find the file that contains the declaration form template and use it as the EXACT format (file including "SYSTEM PRZESYŁEK KONDUKTORSKICH").
  Fill in the following values:
    - Data: ${new Date().toLocaleDateString('pl-PL', { day: '2-digit', month: '2-digit', year: 'numeric' })}
    - Nadawca (identyfikator): 450202122
    - Punkt nadawczy: Gdańsk
    - Punkt docelowy: Żarnowiec
    - Waga: 2800 kg
    - Budżet: 0 PP
    - Zawartość: kasety z paliwem do reaktora
    - Uwagi specjalne: (leave empty / brak)
  Produce the complete, formatted declaration text exactly as the template specifies.

STEP 3 — Once you have complete form, send it using submitAnswer tool. If there's an error, Come back to Step 2 and think which fields need to be changed.
`;

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'fetchTextFile',
      description: 'Pobiera zawartość pliku tekstowego lub markdown z podanego URL.',
      strict: true,
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'Pełny URL do pliku .md, .txt itp.',
          },
        },
        required: ['url'],
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'fetchImage',
      description:
        'Pobiera obraz z URL i analizuje jego zawartość. Użyj dla plików .png/.jpg z dokumentacji.',
      strict: true,
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'Pełny URL do pliku graficznego.',
          },
          question: {
            type: 'string',
            description:
              'Co konkretnie chcesz wiedzieć z obrazu? Np. "Przepisz całą tabelę z listą tras wyłączonych".',
          },
        },
        required: ['url', 'question'],
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'submitAnswer',
      description: 'Wysyła gotową deklarację do weryfikacji. Wywołaj dokładnie raz na końcu.',
      strict: true,
      parameters: {
        type: 'object',
        properties: {
          answer: {
            type: 'object',
            properties: {
              declaration: { type: 'string' },
            },
            required: ['declaration'],
            additionalProperties: false,
          },
        },
        required: ['answer'],
        additionalProperties: false,
      },
    },
  },
];

const fetchedUrls = new Set<string>();

async function fetchTextFile(url: string): Promise<string> {
  if (/\.(png|jpg|jpeg|gif|webp)$/i.test(url)) {
    return `ERROR: This is an image file. Use fetchImage tool instead, not fetchTextFile.`;
  }

  if (fetchedUrls.has(url)) {
    return `[ALREADY FETCHED - skipping duplicate request for ${url}]`;
  }

  fetchedUrls.add(url);
  log.info(`[fetchTextFile] ${url}`);

  const response = await axios.get<string>(url, { responseType: 'text', timeout: 15000 });
  return response.data;
}

async function fetchImage(url: string, question: string): Promise<string> {
  log.info(`[fetchImage] ${url}`);

  const response = await axios.get<ArrayBuffer>(url, {
    responseType: 'arraybuffer',
    timeout: 15000,
  });

  const contentType = response.headers['content-type'] ?? 'image/png';
  const base64 = Buffer.from(response.data).toString('base64');

  const visionResponse = await openai.chat.completions.create({
    model: MODEL_GEMMA,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: `data:${contentType};base64,${base64}` },
          },
          {
            type: 'text',
            text: question,
          },
        ],
      },
    ],
    max_tokens: 2048,
  });

  return visionResponse.choices[0].message.content ?? '';
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────
async function callTool(name: string, args: Record<string, unknown>): Promise<string> {
  try {
    switch (name) {
      case 'fetchTextFile':
        return await fetchTextFile(args.url as string);

      case 'fetchImage':
        return await fetchImage(args.url as string, args.question as string);

      case 'submitAnswer': {
        const answer = args.answer as { declaration: string };
        log.info(`Saving declaration to ${DECLARATION_PATH}...`);
        fs.writeFileSync(DECLARATION_PATH, answer.declaration);

        const result = await sendResult('sendit', answer);
        return typeof result === 'string' ? result : JSON.stringify(result);
      }

      default:
        return `Unknown tool: ${name}`;
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error(`[callTool] Error in tool ${name}: ${message}`);
    return `ERROR: ${message}`;
  }
}

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  { role: 'system', content: SYSTEM_PROMPT },
  { role: 'user', content: 'Start the investigation. Write the document.' },
];

async function solveTask() {
  let iterations = 0;

  while (iterations < 50) {
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
    }

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      log.error('Agent stopped without calling submitAnswer.');
      break;
    }

    let taskFinished = false;

    for (const toolCall of msg.tool_calls) {
      if (toolCall.type !== 'function') continue;

      const name = toolCall.function.name;
      const args = JSON.parse(toolCall.function.arguments);

      log.tool(name, args);
      const result = await callTool(name, args);

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: result,
      });

      if (name === 'submitAnswer') {
        log.result('Verification result', result);
        taskFinished = true;
      }
    }

    if (taskFinished) break;
  }
}

solveTask();
