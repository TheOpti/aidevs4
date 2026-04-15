import axios from 'axios';
import 'dotenv/config';
import fs from 'fs';
import Groq from 'groq-sdk';
import OpenAI from 'openai';
import os from 'os';
import path from 'path';
import { log, MODEL_GEMMA4_26B, MODEL_GEMMA4_E4B, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

// ── Groq client (for audio transcription) ────────────────────────────────────
const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ── API helper ────────────────────────────────────────────────────────────────
async function callAPI(answer: Record<string, unknown>): Promise<unknown> {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'radiomonitoring',
      answer,
    });
    return data;
  } catch (error: any) {
    const body = error.response?.data;
    if (body) return body;
    return { error: error.message };
  }
}

// ── Signal types ──────────────────────────────────────────────────────────────
interface TextSignal {
  kind: 'text';
  content: string;
}

interface ImageSignal {
  kind: 'image';
  mimeType: string;
  base64: string;
}

interface AudioSignal {
  kind: 'audio';
  mimeType: string;
  base64: string;
}

interface JsonSignal {
  kind: 'json';
  content: unknown;
}

interface NoiseSignal {
  kind: 'noise';
}

type Signal = TextSignal | ImageSignal | AudioSignal | JsonSignal | NoiseSignal;

// ── Smart signal router ───────────────────────────────────────────────────────
function routeSignal(raw: Record<string, unknown>): Signal {
  // Binary attachment takes precedence
  if (typeof raw.attachment === 'string') {
    const meta = (raw.meta as string) ?? '';
    const b64 = raw.attachment as string;

    // JSON attachment — decode locally, no LLM cost
    if (meta.includes('json')) {
      try {
        const decoded = Buffer.from(b64, 'base64').toString('utf-8');
        const parsed = JSON.parse(decoded);
        return { kind: 'json', content: parsed };
      } catch {
        return { kind: 'noise' };
      }
    }

    // Audio attachment — needs Groq/Whisper transcription
    if (meta.startsWith('audio/')) {
      return { kind: 'audio', mimeType: meta, base64: b64 };
    }

    // Image attachment — needs OCR via vision model
    if (meta.startsWith('image/')) {
      return { kind: 'image', mimeType: meta, base64: b64 };
    }

    // Text-like binary — try decoding as UTF-8
    if (meta.includes('text')) {
      try {
        const decoded = Buffer.from(b64, 'base64').toString('utf-8').trim();
        if (decoded.length >= 10) {
          return { kind: 'text', content: decoded };
        }
      } catch {
        /* ignore */
      }
    }

    // Unknown binary
    return { kind: 'noise' };
  }

  // Text transcription (pre-transcribed by the API)
  if (typeof raw.transcription === 'string') {
    const text = raw.transcription.trim();
    if (text.length < 10 || /^[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+$/.test(text)) {
      return { kind: 'noise' };
    }
    return { kind: 'text', content: text };
  }

  return { kind: 'noise' };
}

// ── Morse code decoder ────────────────────────────────────────────────────────
const MORSE_TABLE: Record<string, string> = {
  '.-': 'A',
  '-...': 'B',
  '-.-.': 'C',
  '-..': 'D',
  '.': 'E',
  '..-.': 'F',
  '--.': 'G',
  '....': 'H',
  '..': 'I',
  '.---': 'J',
  '-.-': 'K',
  '.-..': 'L',
  '--': 'M',
  '-.': 'N',
  '---': 'O',
  '.--.': 'P',
  '--.-': 'Q',
  '.-.': 'R',
  '...': 'S',
  '-': 'T',
  '..-': 'U',
  '...-': 'V',
  '.--': 'W',
  '-..-': 'X',
  '-.--': 'Y',
  '--..': 'Z',
  '.----': '1',
  '..---': '2',
  '...--': '3',
  '....-': '4',
  '.....': '5',
  '-....': '6',
  '--...': '7',
  '---..': '8',
  '----.': '9',
  '-----': '0',
  // Polish diacritical marks
  '.-.-': 'Ą',
  '-.-..': 'Ć',
  '..-..': 'Ę',
  '.-..-.': 'Ł',
  '--.--': 'Ń',
  '---.': 'Ó',
  '...-...': 'Ś',
  '--..-': 'Ź',
  '--..-.': 'Ż',
  // Prosigns/special
  '-..-.': '/',
};

function decodeMorse(tiTaText: string): string {
  const words = tiTaText.split(/\(stop\)/i);
  const decodedWords: string[] = [];

  for (const word of words) {
    const tokens = word.trim().split(/\s+/).filter(Boolean);
    let decodedWord = '';
    for (const token of tokens) {
      const morse = token.replace(/Ta/g, '-').replace(/Ti/g, '.');
      const letter = MORSE_TABLE[morse] || `[${morse}]`;
      decodedWord += letter;
    }
    decodedWords.push(decodedWord);
  }

  return decodedWords.filter(Boolean).join(' ');
}

function tryDecodeMorseInText(text: string): string | null {
  if (/\b(Ti|Ta){2,}\b/.test(text)) {
    const morseMatch = text.match(
      /(?:\*[^*]+\*\s*)?((?:(?:Ti|Ta)+\s*(?:\(stop\)\s*)?)+)(?:\s*\*[^*]+\*)?/,
    );
    if (morseMatch) {
      return decodeMorse(morseMatch[1]);
    }
  }
  return null;
}

// ── Extract meaningful fragments from noise/static texts ─────────────────────
function extractNoiseFragments(text: string): string[] {
  const fragments: string[] = [];
  const matches = text.matchAll(/\.{3}([^.]+?)\.{3}/g);
  for (const m of matches) {
    const fragment = m[1].trim();
    if (fragment.length > 3 && /[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}/.test(fragment)) {
      fragments.push(fragment);
    }
  }
  return [...new Set(fragments)];
}

// ── Resolve artifacts directory ───────────────────────────────────────────────
function resolveArtifactsDir(): string {
  const artifactsDir = path.join(
    typeof __dirname !== 'undefined' ? __dirname : process.cwd(),
    typeof __dirname !== 'undefined' ? 'debug_artifacts' : 'src/course/S05E01/debug_artifacts',
  );
  if (!fs.existsSync(artifactsDir)) {
    fs.mkdirSync(artifactsDir, { recursive: true });
  }
  return artifactsDir;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 1: Collect all signals up-front
// ═══════════════════════════════════════════════════════════════════════════════
async function collectSignals(artifactsDir: string): Promise<{
  texts: string[];
  audios: AudioSignal[];
  images: ImageSignal[];
  jsons: unknown[];
}> {
  const texts: string[] = [];
  const audios: AudioSignal[] = [];
  const images: ImageSignal[] = [];
  const jsons: unknown[] = [];

  let round = 0;

  while (true) {
    round++;
    const raw = (await callAPI({ action: 'listen' })) as Record<string, unknown>;
    log.info(`[Listen ${round}] code=${raw.code} message=${raw.message}`);

    // Session finished
    if (
      raw.code === 200 ||
      raw.code === 101 ||
      (typeof raw.message === 'string' &&
        (raw.message.toLowerCase().includes('enough') ||
          raw.message.toLowerCase().includes('dostatecznie')))
    ) {
      log.info('Session ended — enough data collected.');
      break;
    }

    const signal = routeSignal(raw);
    log.info(`  → routed as: ${signal.kind}`);

    // Save raw artifact for debugging
    try {
      const filePrefix = path.join(artifactsDir, `round_${round}`);
      if (typeof raw.attachment === 'string') {
        const b64 = raw.attachment as string;
        const meta = String(raw.meta || '');
        const buffer = Buffer.from(b64, 'base64');
        let ext = 'bin';
        if (meta.includes('json')) ext = 'json';
        else if (meta.startsWith('audio/')) ext = meta.split('/')[1] || 'mp3';
        else if (meta.startsWith('image/')) ext = meta.split('/')[1] || 'png';
        else if (meta.includes('text')) ext = 'txt';
        fs.writeFileSync(`${filePrefix}_attachment.${ext}`, buffer);
      }
      if (typeof raw.transcription === 'string') {
        fs.writeFileSync(`${filePrefix}_transcription.txt`, raw.transcription, 'utf-8');
      }
    } catch (err) {
      log.error(`Failed to save artifact for round ${round}:`, err);
    }

    switch (signal.kind) {
      case 'text':
        texts.push(signal.content);
        break;
      case 'audio':
        audios.push(signal);
        break;
      case 'image':
        images.push(signal);
        break;
      case 'json':
        jsons.push(signal.content);
        break;
      case 'noise':
        /* discard */ break;
    }

    // Safety valve
    if (round > 100) {
      log.info('Safety limit reached — stopping listen loop.');
      break;
    }
  }

  return { texts, audios, images, jsons };
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 2: Transcribe audio with Groq Whisper
// ═══════════════════════════════════════════════════════════════════════════════
async function transcribeAudio(
  audio: AudioSignal,
  index: number,
  artifactsDir: string,
): Promise<string> {
  const ext = audio.mimeType.split('/')[1] || 'mp3';
  const tempFile = path.join(os.tmpdir(), `radio_${index}_${Date.now()}.${ext}`);

  try {
    fs.writeFileSync(tempFile, Buffer.from(audio.base64, 'base64'));

    const result = await groqClient.audio.transcriptions.create({
      model: 'whisper-large-v3',
      file: fs.createReadStream(tempFile),
      language: 'pl',
    });

    const transcription = result.text;

    // Save transcription as txt file alongside other artifacts
    const txtFile = path.join(artifactsDir, `audio_${index + 1}_transcription.txt`);
    fs.writeFileSync(txtFile, transcription, 'utf-8');
    log.info(`[Audio ${index + 1}] Transcription saved to: ${txtFile}`);

    return transcription;
  } finally {
    try {
      fs.unlinkSync(tempFile);
    } catch {
      /* cleanup best-effort */
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 3: OCR images with Gemma 4 E4B (via OpenRouter)
// ═══════════════════════════════════════════════════════════════════════════════
async function ocrImage(img: ImageSignal): Promise<string> {
  const response = await openrouter.chat.completions.create({
    model: MODEL_GEMMA4_E4B,
    max_tokens: 500,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: `data:${img.mimeType};base64,${img.base64}` },
          },
          {
            type: 'text',
            text: 'Extract ALL text from this image exactly as it appears. Preserve labels and their values on the same line (e.g. "MAGAZYNY: 47"). Do not summarize or rephrase. Focus on: city names, warehouse counts ("magazyny"), phone numbers, area values, any digits. Return raw extracted text only.',
          },
        ],
      },
    ],
  });
  return response.choices[0].message.content ?? '';
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 4: Local text processing (Morse decoding, fragment extraction)
// ═══════════════════════════════════════════════════════════════════════════════
function processTexts(texts: string[]): {
  processedTexts: string[];
  morseDecoded: string[];
  noiseFragments: string[];
} {
  const processedTexts: string[] = [];
  const morseDecoded: string[] = [];
  const noiseFragments: string[] = [];

  for (const text of texts) {
    // Try Morse decoding
    const morse = tryDecodeMorseInText(text);
    if (morse) {
      log.info(`[Morse decoded]: "${morse}" (from: ${text.slice(0, 80)}...)`);
      morseDecoded.push(morse);
      processedTexts.push(`[ORIGINAL]: ${text}\n[MORSE DECODED]: ${morse}`);
      continue;
    }

    // Check if this is a garbled/static text
    const noiseMarkerCount = (text.match(/trzask|ksssh|bzzt|bzzzz|szum|pisk|kshhh/gi) || []).length;
    if (noiseMarkerCount > 5) {
      const fragments = extractNoiseFragments(text);
      if (fragments.length > 0) {
        log.info(`[Noise fragments extracted]: ${fragments.join(', ')}`);
        noiseFragments.push(...fragments);
      }
      processedTexts.push(`[GARBLED RADIO - extracted fragments]: ${fragments.join(' | ')}`);
      continue;
    }

    // Regular text
    processedTexts.push(text);
  }

  return { processedTexts, morseDecoded, noiseFragments };
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 5: Extract facts / filter noise with Gemma 4 26B (via OpenRouter)
// ═══════════════════════════════════════════════════════════════════════════════
async function extractFacts(rawMaterials: string): Promise<string> {
  const response = await openrouter.chat.completions.create({
    model: MODEL_GEMMA4_26B,
    max_tokens: 2000,
    messages: [
      {
        role: 'system',
        content: `You are a military intelligence analyst. Your job is to extract ONLY concrete, factual information from intercepted radio communications. Discard all noise, static descriptions, and irrelevant chatter.

Extract and organize facts into these categories:
- LOCATIONS: City names, codenames (especially "Syjon"), geographic clues (rivers, soil, radiation levels)
- NUMBERS: Warehouse counts (magazyny) — note carefully if they exist CURRENTLY or are planned for the FUTURE, area measurements, phone numbers, population figures, trade quantities
- DESCRIPTIONS: Key characteristics of mentioned places (water purification, cattle, farming, etc.)
- SETTLEMENT DATA: Any structured data about settlements (from JSON sources)

Be thorough — include every number, every city name, every measurable value you find. Do NOT add your own interpretations, just list the raw facts.`,
      },
      {
        role: 'user',
        content: rawMaterials,
      },
    ],
  });

  return response.choices[0].message.content ?? '';
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 6: Final reasoning with Gemma 4 26B (via OpenRouter)
// ═══════════════════════════════════════════════════════════════════════════════
interface Report {
  cityName: string;
  cityArea: string;
  warehousesCount: number;
  phoneNumber: string;
}

function extractJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    /* fall through */
  }

  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      return JSON.parse(match[0]);
    } catch {
      /* fall through */
    }
  }

  // Strip markdown fences, thinking tags
  const cleaned = text
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/```(?:json)?\s*/g, '')
    .replace(/```/g, '')
    .trim();
  const match2 = cleaned.match(/\{[\s\S]*\}/);
  if (match2) {
    try {
      return JSON.parse(match2[0]);
    } catch {
      /* fall through */
    }
  }

  throw new Error(`No valid JSON found in response:\n${text.slice(0, 500)}`);
}

async function synthesise(
  extractedFacts: string,
  imageOcrContent: string,
  previousReport?: Report,
  apiError?: string,
): Promise<Report> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    {
      role: 'system',
      content: `You are an intelligence analyst processing intercepted radio communications in a post-apocalyptic setting.

Your task is to compile a report with exactly these 4 fields:

1. **cityName** — the REAL Polish city referred to by codename "Syjon"
   - Clues: has river access, has cattle (bydło/wołowina), purifies water, called "miasto ocalałych" (city of survivors), "prawie biblijny raj" (almost biblical paradise — Zion!), fertile soil, low radiation
   - Cross-reference with the JSON settlement data (which city has both riverAccess=true AND farmAnimals=true AND fits the descriptions?)

2. **cityArea** — the occupied area of the EXACT city you chose in step 1, from the JSON data, rounded to exactly 2 decimal places.
   - Example: if the area is 10.7284, cityArea MUST BE "10.73"

3. **warehousesCount** — the CURRENT number of warehouses (magazyny) in Syjon
   - Look in the EXTRACTED FACTS (which includes audio transcriptions) and IMAGE OCR CONTENT for the word "MAGAZYNY" or mentions of warehouses.
   - BEWARE of statements about the future. If the text says they plan to build the N-th warehouse in the future (e.g., "planujemy wybudować dwunasty magazyn" -> "we plan to build the twelfth warehouse"), it means there are CURRENTLY N-1 warehouses (e.g., 11).
   - ONLY return the number of warehouses that exist CURRENTLY.

4. **phoneNumber** — the phone number found in the IMAGE OCR CONTENT. Only digits (e.g. "644122092"). Do not return empty string or "0" if there is a 9-digit number in the OCR output.

Return ONLY valid JSON — no explanation, no markdown:
{"cityName":"...","cityArea":"12.34","warehousesCount":123,"phoneNumber":"123456789"}`,
    },
    {
      role: 'user',
      content: `=== EXTRACTED FACTS ===\n${extractedFacts}\n\n=== IMAGE OCR CONTENT (RAW — DO NOT REINTERPRET) ===\n${imageOcrContent || '(no images)'}`,
    },
  ];

  if (previousReport && apiError) {
    messages.push({
      role: 'assistant',
      content: JSON.stringify(previousReport),
    });
    messages.push({
      role: 'user',
      content: `The API rejected this with: "${apiError}"

Go back to the IMAGE OCR CONTENT section above and re-read the MAGAZYNY line and phone number carefully.
Do NOT guess — only use numbers that literally appear in the OCR output.
Return the corrected JSON only.`,
    });
  }

  const response = await openrouter.chat.completions.create({
    model: MODEL_GEMMA4_26B,
    max_tokens: 800,
    messages,
  });

  const raw = (response.choices[0].message.content ?? '').replace(/```json|```/g, '').trim();
  return extractJson(raw) as Report;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main pipeline
// ═══════════════════════════════════════════════════════════════════════════════
async function solveTask() {
  // Resolve artifacts dir once and share across all steps
  const artifactsDir = resolveArtifactsDir();
  log.info(`Artifacts directory: ${artifactsDir}`);

  log.step(1, 'Starting session');
  const startResult = await callAPI({ action: 'start' });
  log.info('Session started:', JSON.stringify(startResult).slice(0, 200));

  // ── Step 1: Collect all data up-front ─────────────────────
  log.step(2, 'Collecting all signals (listen loop)');
  const { texts, audios, images, jsons } = await collectSignals(artifactsDir);
  log.info(
    `Collected: ${texts.length} texts, ${audios.length} audios, ${images.length} images, ${jsons.length} JSONs`,
  );

  // ── Step 2: Transcribe audio with Groq Whisper ────────────
  log.step(3, `Transcribing ${audios.length} audio signals with Groq Whisper`);
  const audioTranscriptions: string[] = [];
  for (const [i, audio] of audios.entries()) {
    log.info(`Transcribing audio ${i + 1}/${audios.length} (${audio.mimeType})...`);
    try {
      const transcription = await transcribeAudio(audio, i, artifactsDir);
      log.info(`[Audio ${i + 1} transcription]: ${transcription.slice(0, 150)}`);
      audioTranscriptions.push(transcription);
    } catch (err) {
      log.error(`Failed to transcribe audio ${i + 1}`, err);
    }
  }

  // Merge audio transcriptions into the text pool
  const allTexts = [...texts, ...audioTranscriptions];

  // ── Step 3: OCR images with Gemma 4 E4B ───────────────────
  log.step(4, `OCR on ${images.length} images with Gemma 4 E4B`);
  const imageDescriptions: string[] = [];
  for (const [i, img] of images.entries()) {
    log.info(`OCR image ${i + 1}/${images.length}...`);
    try {
      const desc = await ocrImage(img);
      log.info(`[Image ${i + 1} OCR]: ${desc}`);
      imageDescriptions.push(`[Image ${i + 1}]: ${desc}`);
    } catch (err) {
      log.error(`Failed to OCR image ${i + 1}`, err);
    }
  }

  // ── Step 4: Local text processing ─────────────────────────
  log.step(5, 'Processing texts locally (Morse, fragments)');
  const { processedTexts, morseDecoded, noiseFragments } = processTexts(allTexts);

  log.info('=== PROCESSED TEXTS ===');
  processedTexts.forEach((t, i) => log.info(`[Text ${i + 1}]: ${t.slice(0, 200)}`));
  log.info('=== MORSE DECODED ===');
  morseDecoded.forEach((m) => log.info(`  ${m}`));
  log.info('=== NOISE FRAGMENTS ===');
  log.info(noiseFragments.join(' | '));
  log.info('=== COLLECTED JSONs ===');
  jsons.forEach((j, i) => log.info(`[JSON ${i + 1}]: ${JSON.stringify(j, null, 2)}`));

  // ── Assemble raw materials (no image OCR — kept separate) ─
  const rawMaterials = [
    processedTexts.length ? `=== TRANSCRIPTIONS ===\n${processedTexts.join('\n---\n')}` : '',
    morseDecoded.length ? `=== MORSE CODE DECODED ===\n${morseDecoded.join('\n')}` : '',
    noiseFragments.length
      ? `=== KEY FRAGMENTS FROM NOISE ===\n${[...new Set(noiseFragments)].join('\n')}`
      : '',
    jsons.length
      ? `=== JSON DATA ===\n${jsons.map((j) => JSON.stringify(j, null, 2)).join('\n---\n')}`
      : '',
  ]
    .filter(Boolean)
    .join('\n\n');

  // Image OCR content kept raw and passed directly to synthesise
  const imageOcrContent = imageDescriptions.join('\n---\n');

  log.info(`Raw materials assembled: ${rawMaterials.length} chars`);
  log.info(`Image OCR content: ${imageOcrContent.length} chars`);

  // ── Step 5: Extract facts with Gemma 4 26B ────────────────
  // Note: image OCR is intentionally excluded here to avoid number corruption
  log.step(6, 'Extracting facts with Gemma 4 26B (filtering noise, no image OCR)');
  const facts = await extractFacts(rawMaterials);
  log.info('=== EXTRACTED FACTS ===');
  log.info(facts);

  // ── Step 6: Final reasoning + submit with retry loop ──────
  log.step(7, 'Final reasoning with Gemma 4 26B + submit');
  const MAX_RETRIES = 5;
  let report: Report | undefined;
  let lastError: string | undefined;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    log.info(`[Synthesis] Attempt ${attempt}/${MAX_RETRIES}...`);
    try {
      report = await synthesise(facts, imageOcrContent, report, lastError);
    } catch (err) {
      log.error(`Synthesis failed:`, err);
      continue;
    }
    log.info(`Report: ${JSON.stringify(report)}`);

    // Validate format before sending
    if (
      !report.cityName ||
      !report.cityArea ||
      report.warehousesCount == null ||
      !report.phoneNumber
    ) {
      log.info('Report has missing fields, retrying...');
      lastError = 'Report has missing fields — please fill all fields';
      continue;
    }

    log.info('Transmitting...');
    const result = (await callAPI({
      action: 'transmit',
      cityName: report.cityName,
      cityArea: String(report.cityArea),
      warehousesCount: Number(report.warehousesCount),
      phoneNumber: String(report.phoneNumber),
    })) as Record<string, unknown>;
    log.result('Transmit result:', JSON.stringify(result));

    if ((result.code as number) >= 0) {
      log.result('✅ Success!');
      return;
    }

    lastError = result.message as string;
    log.info(`Attempt ${attempt} rejected: ${lastError}`);
  }

  log.error('All attempts exhausted.');
}

solveTask().catch((err) => {
  log.error('Fatal:', err);
  process.exit(1);
});
