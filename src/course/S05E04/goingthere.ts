import axios from 'axios';
import crypto from 'crypto';
import 'dotenv/config';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { BASE_URL, VERIFY_URL } from 'src/shared/api';

const ENDPOINTS = {
  MESSAGE_API: `${BASE_URL}/api/getmessage`,
  FREQUENCY_SCANNER_API: `${BASE_URL}/api/frequencyScanner`,
};

// ── Types ─────────────────────────────────────────────────────────────────────

interface GameState {
  player: { row: number; col: number };
  base: { row: number; col: number };
  currentColumn: { column: number; yourRow: number; stoneRow: number; freeRows: number[] };
  message?: string;
  code?: number;
}

type Direction = 'ahead' | 'port' | 'starboard'; // ahead=same row, port=left/up, starboard=right/down
type Move = 'go' | 'left' | 'right';

// ── Crash sentinel ────────────────────────────────────────────────────────────

class GameCrashError extends Error {
  constructor(public readonly body: unknown) {
    super(`Game crashed: ${JSON.stringify(body).slice(0, 300)}`);
  }
}

const CRASH_PHRASES = ['crash', 'game over', 'destroyed', 'hit a rock', 'shot down'];

function looksLikeCrash(body: unknown): boolean {
  const s = JSON.stringify(body ?? '').toLowerCase();
  return CRASH_PHRASES.some((p) => s.includes(p));
}

// ── Retry wrapper ─────────────────────────────────────────────────────────────
// Never retries GameCrashError — that is a permanent state, not a transient one.

async function withRetry<T>(fn: () => Promise<T>, retries = 7, delayMs = 800): Promise<T> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (err instanceof GameCrashError) throw err;
      if (attempt === retries) throw err;
      log.info(`[retry ${attempt}/${retries}] ${String(err).slice(0, 120)}`);
      await new Promise((r) => setTimeout(r, delayMs * attempt)); // exponential back-off
    }
  }
  throw new Error('Unreachable');
}

// ── Game API calls ─────────────────────────────────────────────────────────────

async function sendCommand(command: string): Promise<GameState> {
  return withRetry(async () => {
    let data: unknown;
    try {
      const res = await axios.post(VERIFY_URL, {
        apikey: process.env.AIDEVS_API_KEY,
        task: 'goingthere',
        answer: { command },
      });
      data = res.data;
    } catch (err: any) {
      const body = err.response?.data;
      const status = err.response?.status;
      log.error(`HTTP ${status} error — body: ${JSON.stringify(body).slice(0, 300)}`);

      if (looksLikeCrash(body)) throw new GameCrashError(body);

      // 400 with no explicit crash message likely means invalid/expired session — restart
      if (status === 400) throw new GameCrashError(body ?? { status: 400 });

      throw err; // 5xx / network — let withRetry handle it
    }

    const d = data as any;

    // HTTP 200 but game body signals crash
    if (looksLikeCrash(d)) throw new GameCrashError(d);

    // Flag response — game is won, no player field expected
    if (!d.player && JSON.stringify(d).includes('FLG:')) {
      log.info(`FLAG RECEIVED: ${d.message}`);
      // Return a synthetic state so the loop exits
      return {
        player: { row: 99, col: 99 },
        base: { row: 99, col: 99 },
        currentColumn: { column: 99, yourRow: 99, stoneRow: 0, freeRows: [] },
        message: d.message,
      } as GameState;
    }

    if (!d.player) {
      log.info(`No player in body: ${JSON.stringify(d).slice(0, 300)}`);
      throw new Error(`Unexpected response (no player): ${JSON.stringify(d).slice(0, 200)}`);
    }

    return d as GameState;
  });
}

// ── Frequency scanner sub-agent ─────────────────────────────────────────────
//
// The scanner endpoint:
//   - Returns "it's clear" variants when safe
//   - Returns corrupted JSON with TYPO field names when tracking:
//       { "beingTrackeb": true, "frepuency": 713, "bata": { "betecti0nC0be": "sNsnxs" } }
//   - May return HTTP 502 — must retry on that specifically
//
// We use an LLM sub-agent to reliably extract frequency + detectionCode
// from any combination of typos, corruption, or encoding errors.

interface ScanResult {
  isTracked: false;
}
interface TrackedResult {
  isTracked: true;
  frequency: number;
  detectionCode: string;
}

/** Regex fallback: fuzzy-matches frequency and detectionCode despite typos */
function regexExtractScannerFields(
  raw: string,
): { frequency: number; detectionCode: string } | null {
  // Frequency: key looks like "frepuency", "frequency", "frequ3ncy" etc.
  // Value is always a plain integer.
  const freqMatch = raw.match(/["']?fr[a-z0-9_]{2,8}["']?\s*[=:]\s*([0-9]+)/i);

  // detectionCode: lives inside "bata"/"data" sub-object.
  // Key looks like "betecti0nC0be", "detectionCode", "d3tectionCode" etc.
  // Value is always a short alphanumeric string.
  const codeMatch =
    raw.match(/["']?b?[Dd][a-z0-9]{4,14}C[a-z0-9]{2,5}["']?\s*[=:]\s*["']?([\w]+)["']?/) ||
    raw.match(/["']?[Dd]etection[Cc]ode["']?\s*[=:]\s*["']?([\w]+)["']?/i);

  if (freqMatch && codeMatch) {
    return { frequency: Number(freqMatch[1]), detectionCode: codeMatch[1].trim() };
  }
  return null;
}

/**
 * LLM sub-agent: given the raw scanner text, returns extracted fields.
 * This handles arbitrary typos that regex cannot anticipate.
 */
async function llmExtractScannerFields(
  raw: string,
): Promise<{ frequency: number; detectionCode: string }> {
  const response = await openrouter.chat.completions.create({
    model: MODEL_DEEPSEEK,
    messages: [
      {
        role: 'system',
        content: `You are a data extraction agent. You receive a corrupted/garbled JSON string from a
missile-tracking frequency scanner. Field names contain deliberate typos and substitutions,
for example:
  - "beingTrackeb" instead of "beingTracked"
  - "frepuency" instead of "frequency"
  - "bata" instead of "data"
  - "betecti0nC0be" instead of "detectionCode"  (zeros replacing letters, letters swapped)
  - "weap0nType" instead of "weaponType"

Your task: extract exactly two values:
1. frequency — a number, found at the top level of the object
2. detectionCode — a short alphanumeric string, found INSIDE the nested data/bata sub-object

Respond ONLY with valid JSON, no explanation, no markdown:
{"frequency": <number>, "detectionCode": "<string>"}`,
      },
      {
        role: 'user',
        content: `Scanner response:
${raw}`,
      },
    ],
    max_tokens: 60,
  });

  const text = response.choices[0].message.content?.trim() ?? '';
  // Strip any accidental markdown fences
  const clean = text
    .replace(/^```json\s*/i, '')
    .replace(/```\s*$/, '')
    .trim();
  const parsed = JSON.parse(clean);
  return {
    frequency: Number(parsed.frequency),
    detectionCode: String(parsed.detectionCode).trim(),
  };
}

/**
 * GET the frequency scanner endpoint.
 * Retries on 502 (deliberate server errors) and on parse failures.
 */
async function checkFrequencyScanner(): Promise<ScanResult | TrackedResult> {
  // 502-aware retry: keep trying until we get a non-502 response
  let raw = '';
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      const res = await axios.get(
        `${ENDPOINTS.FREQUENCY_SCANNER_API}?key=${process.env.AIDEVS_API_KEY}`,
        {
          responseType: 'text',
          transformResponse: [(d) => d], // prevent axios from auto-parsing JSON
        },
      );
      raw = String(res.data);
      break; // success — exit retry loop
    } catch (err: any) {
      const status = err.response?.status;
      if (status === 502) {
        log.info(`502 received, retrying (${attempt}/10)...`);
        await new Promise((r) => setTimeout(r, 500 * attempt));
        continue;
      }
      throw err; // other errors bubble up
    }
  }

  log.info(`Scanner RAW: ${raw.slice(0, 250)}`);

  // ── Safe? ─────────────────────────────────────────────────────────────────
  // A "clear" response is always plain text with no JSON object.
  // The word itself may be mangled ("cLEeeEar") so don't rely on substring match —
  // instead check for the presence of a JSON object opener '{'.
  if (!raw.includes('{')) {
    log.info('Scanner: No JSON object — clear.');
    return { isTracked: false };
  }

  // ── Tracked — extract frequency + detectionCode ───────────────────────────
  log.info('Scanner: Tracking signal detected — extracting fields...');

  // 1. Try standard JSON parse (in case it happens to be valid)
  try {
    const parsed = JSON.parse(raw);
    // Fields may be under their canonical OR typo'd names
    const frequency =
      parsed.frequency ?? parsed.frepuency ?? parsed.freq ?? parsed.fre ?? parsed.frep;
    const dataObj = parsed.data ?? parsed.bata ?? parsed.dat ?? {};
    const detectionCode =
      dataObj.detectionCode ??
      dataObj.betecti0nC0be ??
      dataObj.detectioncode ??
      // also check top-level as fallback
      parsed.detectionCode ??
      parsed.betecti0nC0be;

    if (frequency !== undefined && detectionCode !== undefined) {
      log.info(`Parsed via JSON — freq=${frequency}, code=${detectionCode}`);
      return {
        isTracked: true,
        frequency: Number(frequency),
        detectionCode: String(detectionCode).trim(),
      };
    }
  } catch {
    /* corrupted — fall through */
  }

  // 2. Regex fallback
  const regexResult = regexExtractScannerFields(raw);
  if (regexResult) {
    log.info(`Parsed via regex — freq=${regexResult.frequency}, code=${regexResult.detectionCode}`);
    return { isTracked: true, ...regexResult };
  }

  // 3. LLM sub-agent (handles any typo pattern we haven't seen before)
  log.info('Scanner: Falling back to LLM extraction...');
  const llmResult = await llmExtractScannerFields(raw);
  log.info(`Parsed via LLM — freq=${llmResult.frequency}, code=${llmResult.detectionCode}`);
  return { isTracked: true, ...llmResult };
}

async function neutralizeRadar(frequency: number, detectionCode: string): Promise<void> {
  const disarmHash = crypto
    .createHash('sha1')
    .update(detectionCode + 'disarm')
    .digest('hex');
  log.info(`Radar: Disarming — freq=${frequency}, code=${detectionCode}, hash=${disarmHash}`);

  await withRetry(async () => {
    const { data } = await axios.post(ENDPOINTS.FREQUENCY_SCANNER_API, {
      apikey: process.env.AIDEVS_API_KEY,
      frequency,
      disarmHash,
    });
    log.info('Radar: Disarmed', JSON.stringify(data).slice(0, 200));
  });
}

// ── Radio hint ─────────────────────────────────────────────────────────────────

async function getHint(): Promise<string> {
  return withRetry(async () => {
    const { data } = await axios.post(ENDPOINTS.MESSAGE_API, {
      apikey: process.env.AIDEVS_API_KEY,
    });
    if (!data.hint) throw new Error(`No hint in response: ${JSON.stringify(data)}`);
    return data.hint as string;
  });
}

/**
 * Use LLM to interpret variable/nautical hint language into a simple direction.
 * Returns: 'ahead' | 'port' | 'starboard'
 *   - ahead    = rock is straight in front (same row) → must deviate
 *   - port     = rock is to port / left / above current row (higher row number)
 *   - starboard = rock is to starboard / right / below current row (lower row number)
 */
async function interpretHint(hint: string): Promise<Direction> {
  const response = await openrouter.chat.completions.create({
    model: MODEL_DEEPSEEK,
    messages: [
      {
        role: 'system',
        content: `You are a navigation assistant. Your only job is to classify a radio hint about rock position.
The rocket travels on a 3-row grid (rows 1-3, row 1 = top, row 3 = bottom).
"left"/"port" means the rock is in a higher row number (downward / below current heading).
Wait - let me be precise about coordinate system:
- The grid has rows 1 (top), 2 (middle), 3 (bottom)
- "port" / "left" in nautical = left side of vessel = row number DECREASES (toward row 1, top)
- "starboard" / "right" in nautical = right side of vessel = row number INCREASES (toward row 3, bottom)
- "straight ahead" / "bow" / "forward" = same row as current
 
Classify the hint into exactly one of these three words and nothing else:
- "ahead" — rock is directly in front (same row, must swerve)
- "port" — rock is to the left/port side (lower row number, i.e. toward row 1)
- "starboard" — rock is to the right/starboard side (higher row number, i.e. toward row 3)
 
Reply with ONLY one word: ahead, port, or starboard.`,
      },
      {
        role: 'user',
        content: `Hint: "${hint}"`,
      },
    ],
    max_tokens: 10,
  });

  const raw = response.choices[0].message.content?.trim().toLowerCase() ?? '';
  if (raw.includes('ahead')) return 'ahead';
  if (raw.includes('port')) return 'port';
  if (raw.includes('starboard')) return 'starboard';

  // Keyword fallback if LLM gives unexpected output
  const lower = hint.toLowerCase();
  if (
    lower.includes('ahead') ||
    lower.includes('straight') ||
    lower.includes('forward') ||
    lower.includes('bow')
  )
    return 'ahead';
  if (lower.includes('port') || lower.includes('left')) return 'port';
  if (lower.includes('starboard') || lower.includes('right')) return 'starboard';

  log.info(`Could not classify hint: "${hint}", defaulting to ahead`);
  return 'ahead';
}

// ── Movement decision ─────────────────────────────────────────────────────────
//
// Grid rows: 1 (top) → 3 (bottom)
//   "left"  → row - 1  (toward row 1 / top   / nautical port)
//   "right" → row + 1  (toward row 3 / bottom / nautical starboard)
//   "go"    → row unchanged
//
// Rock direction tells us which row in the NEXT column is blocked:
//   "ahead"    → currentRow      is blocked → "go" is forbidden
//   "port"     → currentRow - 1  is blocked → "left" is forbidden
//   "starboard"→ currentRow + 1  is blocked → "right" is forbidden

function decideMove(
  currentRow: number,
  rockDir: Direction,
  targetRow: number, // preferred destination row (base row) — steer toward this
  currentStoneRow: number, // stone row in the CURRENT column — must not pass through it
): Move {
  // All possible moves with their resulting row
  const candidates: { move: Move; resultRow: number }[] = [
    { move: 'go', resultRow: currentRow },
    { move: 'left', resultRow: currentRow - 1 },
    { move: 'right', resultRow: currentRow + 1 },
  ];

  // 1. Remove out-of-bounds options
  const inBounds = candidates.filter((c) => c.resultRow >= 1 && c.resultRow <= 3);

  // 2. Determine which row is blocked by the rock in the NEXT column (from hint)
  const nextColBlockedRow =
    rockDir === 'ahead'
      ? currentRow
      : rockDir === 'port'
        ? currentRow - 1
        : rockDir === 'starboard'
          ? currentRow + 1
          : -1;

  // 3. Remove moves that are unsafe:
  //    a) Landing on the rock in the next column
  //    b) Passing through the rock in the CURRENT column (left/right first change row,
  //       so moving to a row that has the current column's stone = crash)
  const safe = inBounds.filter((c) => {
    // Don't land on rock in next column
    if (c.resultRow === nextColBlockedRow) return false;
    // Don't pass through current column's stone (only affects left/right, not go)
    if (c.move !== 'go' && c.resultRow === currentStoneRow) return false;
    return true;
  });

  if (safe.length === 0) {
    log.error('⚠ No safe moves available — this should never happen!');
    return 'go';
  }

  // 4. Among safe options, prefer the move that lands closest to targetRow.
  safe.sort((a, b) => {
    const distA = Math.abs(a.resultRow - targetRow);
    const distB = Math.abs(b.resultRow - targetRow);
    return distA - distB;
  });

  const chosen = safe[0];
  log.info(
    `Move Options: ${safe.map((s) => `${s.move}→row${s.resultRow}`).join(', ')} | currentStone=row${currentStoneRow}, nextBlocked=row${nextColBlockedRow}`,
  );
  return chosen.move;
}

// ── Main solver ───────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== goingthere task starting ===');

  // ── Step 1: Start game ───────────────────────────────────────────────────
  log.step(1, 'Starting game...');
  let state = await sendCommand('start');
  log.info(`Player: col=${state.player.col}, row=${state.player.row}`);
  log.info(`Base:   col=${state.base.col},   row=${state.base.row}`);
  log.info(`Current col stone at row: ${state.currentColumn.stoneRow}`);

  let currentStoneRow = state.currentColumn.stoneRow;

  let playerRow = state.player.row;
  let playerCol = state.player.col;
  const baseCol = state.base.col;
  const baseRow = state.base.row;
  let step = 0;

  // ── Step 2: Navigate loop ─────────────────────────────────────────────────
  while (playerCol < baseCol) {
    step++;
    log.step(step + 1, `Position col=${playerCol}, row=${playerRow}`);

    // 2a. Check frequency scanner
    log.info('Scanner: Checking...');
    const scan = await checkFrequencyScanner();
    if (scan.isTracked) {
      log.info(`Scanner: TRACKED! frequency=${scan.frequency}, code=${scan.detectionCode}`);
      await neutralizeRadar(scan.frequency, scan.detectionCode);
    } else {
      log.info('Scanner: Clear.');
    }

    // 2b. Get radio hint about next column
    log.info('Hint: Requesting...');
    const hint = await getHint();
    log.info(`Hint: "${hint}"`);

    // 2c. Interpret hint
    const rockDir = await interpretHint(hint);
    log.info(`Interpreted rock direction: ${rockDir}`);

    // 2d. Decide move
    const move = decideMove(playerRow, rockDir, baseRow, currentStoneRow);
    log.info(`Executing move: ${move}`);

    // 2e. Execute move — GameCrashError signals crash or expired session
    try {
      state = await sendCommand(move);
    } catch (err) {
      if (err instanceof GameCrashError) {
        log.error(`💥 CRASHED or session expired! ${err.message}`);
        log.info('Restarting game from scratch...');
        await new Promise((r) => setTimeout(r, 1500));
        return solveTask();
      }
      throw err;
    }

    playerRow = state.player.row;
    playerCol = state.player.col;
    currentStoneRow = state.currentColumn.stoneRow;
    log.info(`New position: col=${playerCol}, row=${playerRow} | stone at row=${currentStoneRow}`);

    if (state.message) log.api(`API: ${state.message}`);

    // Check if we reached the base
    if (playerCol >= baseCol) {
      if (playerRow === baseRow) {
        log.result(`🎯 Reached Grudziądz base at row ${playerRow} — correct!`);
      } else {
        log.error(
          `⚠️ Reached col ${playerCol} but landed on row ${playerRow}, expected row ${baseRow}!`,
        );
      }
      log.result('Final Response:', state);
      break;
    }
  }

  log.info('=== Task complete ===');
}

solveTask().catch((err) => {
  log.error('Fatal', err);
  process.exit(1);
});
