import axios from 'axios';
import { execSync } from 'child_process';
import 'dotenv/config';
import * as fs from 'fs';
import https from 'https';
import OpenAI from 'openai';
import * as os from 'os';
import * as path from 'path';
import { MODEL_CLAUDE_SONNET, openrouter } from 'src/shared/agents';
import { S04E04, VERIFY_URL } from 'src/shared/api';

// ── Helpers ───────────────────────────────────────────────────────────────────
function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      })
      .on('error', reject);
  });
}

// ── Config ────────────────────────────────────────────────────────────────────
const ZIP_PATH = path.join(os.tmpdir(), 'natan_notes.zip');
const EXTRACT_DIR = path.join(os.tmpdir(), 'natan_notes');

// ── API helper ────────────────────────────────────────────────────────────────
async function callAPI(answer: Record<string, unknown> | unknown[]): Promise<unknown> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'filesystem',
      answer,
    });
    return data;
  } catch (error: any) {
    const body = error.response?.data;
    if (body) return body;
    return { error: error.message };
  }
}

// ── Load Natan's notes from disk ──────────────────────────────────────────────
function loadNotes(notesDir: string): string {
  const files = ['README.md', 'ogłoszenia.txt', 'rozmowy.txt', 'transakcje.txt'];
  const sections: string[] = [];
  for (const file of files) {
    const filePath = path.join(notesDir, file);
    const content = fs.readFileSync(filePath, 'utf-8');
    sections.push(`=== ${file} ===\n${content}`);
  }
  return sections.join('\n\n');
}

// ── Tool implementations ──────────────────────────────────────────────────────
async function toolReset(): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'reset' }));
}

async function toolCreateDir(dirPath: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'createDir', path: dirPath }));
}

async function toolCreateFile(filePath: string, content: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'createFile', path: filePath, content }));
}

async function toolListDir(dirPath: string): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'listDir', path: dirPath }));
}

async function toolBatch(operations: Record<string, unknown>[]): Promise<string> {
  return JSON.stringify(await callAPI(operations));
}

async function toolDone(): Promise<string> {
  return JSON.stringify(await callAPI({ action: 'done' }));
}

// ── Tool definitions ──────────────────────────────────────────────────────────
const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'createDir',
      description: 'Create a single directory.',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string', description: 'e.g. /miasta' } },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'createFile',
      description: 'Create a single file with text content.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          content: { type: 'string' },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'listDir',
      description: 'List contents of a directory to verify what was created.',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'batch',
      description:
        'Execute many filesystem operations in a single API call. STRONGLY preferred over calling createDir/createFile one-by-one.',
      parameters: {
        type: 'object',
        properties: {
          operations: {
            type: 'array',
            description:
              'Array of operation objects, each with "action", "path", and optionally "content".',
            items: { type: 'object' },
          },
        },
        required: ['operations'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'done',
      description:
        'Submit the finished filesystem to Centrala for verification. Call only when the structure is complete.',
      parameters: { type: 'object', properties: {} },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────
const SYSTEM = `
You are an agent that reads trade notes written by Natan Rams and builds a structured virtual filesystem via API tools.

## Required filesystem structure

### /miasta/<CityName>
One file per city that appears in the notes.
Content: a JSON object where each key is a good the city NEEDS and the value is the required quantity (a number — no unit strings).
Key naming: singular nominative, NO Polish diacritic characters (replace ą→a, ę→e, ó→o, ś→s, ź/ż→z, ć→c, ń→n, ł→l).
Example content: {"chleb":45,"woda":120,"mlotek":6}

### /osoby/<PersonName>
One file per person who manages trade in a city (one person per city, derived from the conversation notes).
Filename convention: FirstName_LastName with no Polish chars.
File content (two lines):
Full Name
[CityName](/miasta/CityName)

### /towary/<goodName>
One file per distinct good that appears on the SELLER side in the transactions list.
Filename: singular nominative noun, no Polish chars.
Content: one markdown link per city that sells this good, one per line:
[CityName](/miasta/CityName)
If multiple cities sell the same good, list all of them (one link per line).

## Important rules
- No Polish diacritic characters in any filename or file content.
- City names themselves contain no diacritics, so they stay unchanged.
- Quantities are plain integers — never include units like "kg", "butelek", "workow".
- Use the 'batch' tool and send ALL operations (dirs + files) in a single call instead of making many individual calls.

## Workflow
1. Carefully read and analyse all notes provided.
2. Derive every city, person, and tradeable good from the source texts.
3. Build the full list of batch operations and call batch once.
4. Verify with listDir on /miasta, /osoby, and /towary.
5. Call done.
`.trim();

// ── Agent loop ────────────────────────────────────────────────────────────────
async function solveTask() {
  console.log('=== Filesystem task starting ===\n');

  // ── 1. Download notes zip ────────────────────────────────────────────────
  if (!fs.existsSync(ZIP_PATH)) {
    console.log('Downloading natan_notes.zip...');
    await downloadFile(S04E04.FILESYSTEM_URL, ZIP_PATH);
    console.log('Downloaded.');
  } else {
    console.log('natan_notes.zip already cached.');
  }

  // ── 2. Extract ────────────────────────────────────────────────────────────
  if (!fs.existsSync(EXTRACT_DIR)) {
    console.log('Extracting...');
    execSync(`unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"`);
    console.log('Extracted.');
  }

  // ── 3. Load note contents ─────────────────────────────────────────────────
  const notesContent = loadNotes(EXTRACT_DIR);
  console.log(`Loaded notes from: ${EXTRACT_DIR}`);

  console.log('Resetting filesystem...');
  console.log(await toolReset());

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content: `Here are all of Natan Rams's notes. Read them carefully, extract every city, person, and tradeable good, then build the filesystem as described in your instructions.\n\n${notesContent}`,
    },
  ];

  let iteration = 0;
  const MAX = 30;
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
          case 'createDir':
            result = await toolCreateDir(args.path);
            break;
          case 'createFile':
            result = await toolCreateFile(args.path, args.content);
            break;
          case 'listDir':
            result = await toolListDir(args.path);
            break;
          case 'batch':
            result = await toolBatch(args.operations);
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
