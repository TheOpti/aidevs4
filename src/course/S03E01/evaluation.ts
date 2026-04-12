/**
 * Sensor Anomaly Detection — "evaluation" task
 *
 * Strategy (cost-optimized):
 * 1. Download & unzip sensors.zip
 * 2. PROGRAMMATIC: flag files with out-of-range values OR inactive sensors
 *    returning non-zero values  →  no LLM needed
 * 3. LLM (batched): only for files that passed programmatic checks.
 *    We need to know if the operator *incorrectly* claims there's a problem
 *    when data is actually fine.
 *    (Files with real anomalies are already flagged regardless of note content.)
 * 4. Deduplicate identical / near-identical notes before sending to LLM.
 * 5. Submit answer to /verify
 */

import axios from 'axios';
import { execSync } from 'child_process';
import 'dotenv/config';
import fs from 'fs';
import https from 'https';
import os from 'os';
import path from 'path';
import { log, MODEL_DEEPSEEK, openrouter } from 'src/shared/agents';
import { S03E01, VERIFY_URL } from 'src/shared/api';

// ── Config ────────────────────────────────────────────────────────────────────
const ZIP_PATH = path.join(os.tmpdir(), 'sensors.zip');
const EXTRACT_DIR = path.join(os.tmpdir(), 'sensors');

// ── Types ─────────────────────────────────────────────────────────────────────
interface SensorData {
  sensor_type: string;
  timestamp: number;
  temperature_K: number;
  pressure_bar: number;
  water_level_meters: number;
  voltage_supply_v: number;
  humidity_percent: number;
  operator_notes: string;
}

// ── Ranges & field mappings ───────────────────────────────────────────────────
type MeasurementField = keyof Omit<SensorData, 'sensor_type' | 'timestamp' | 'operator_notes'>;

const RANGES: Record<MeasurementField, [number, number]> = {
  temperature_K: [553, 873],
  pressure_bar: [60, 160],
  water_level_meters: [5.0, 15.0],
  voltage_supply_v: [229.0, 231.0],
  humidity_percent: [40.0, 80.0],
};

// Maps sensor_type token → JSON field name
const SENSOR_TO_FIELD: Record<string, MeasurementField> = {
  temperature: 'temperature_K',
  pressure: 'pressure_bar',
  water: 'water_level_meters',
  voltage: 'voltage_supply_v',
  humidity: 'humidity_percent',
};

const ALL_FIELDS = Object.values(SENSOR_TO_FIELD) as MeasurementField[];

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

// ── Step 1: Programmatic anomaly check ───────────────────────────────────────

const KNOWN_SENSOR_TYPES = new Set(Object.keys(SENSOR_TO_FIELD));

function hasProgrammaticAnomaly(data: SensorData): { anomaly: boolean; reason?: string } {
  const activeTokens = data.sensor_type.split('/').map((s) => s.trim().toLowerCase());

  // Flag files with unknown sensor type tokens
  const unknownToken = activeTokens.find((t) => !KNOWN_SENSOR_TYPES.has(t));
  if (unknownToken) {
    return { anomaly: true, reason: `Unknown sensor type token: "${unknownToken}"` };
  }

  const activeFields = new Set<MeasurementField>(
    activeTokens.map((t) => SENSOR_TO_FIELD[t]).filter(Boolean),
  );

  for (const field of ALL_FIELDS) {
    const value = data[field] as number;

    if (!activeFields.has(field)) {
      // Inactive sensor — must be exactly 0
      if (value !== 0) {
        return { anomaly: true, reason: `Inactive field "${field}" has non-zero value: ${value}` };
      }
    } else {
      // Active sensor — must be within range
      const [min, max] = RANGES[field];
      if (value < min || value > max) {
        return {
          anomaly: true,
          reason: `Field "${field}" value ${value} out of range [${min}, ${max}]`,
        };
      }
    }
  }

  return { anomaly: false };
}

// ── Step 2: LLM batch check for "false alarm" notes ─────────────────────────
// We send CLEAN files (no programmatic anomaly) to the LLM.
// Goal: detect operator notes that claim there IS a problem when data is fine.
// We batch many notes per call and ask for minimal output (just IDs).

const LLM_SYSTEM = `You are a sensor data auditor. You will receive operator notes from sensor readings that have ALREADY PASSED all numeric range checks — the measured data is perfectly fine.
 
Flag notes where the operator makes an EXPLICIT CLAIM that the readings/data/behavior is wrong, problematic, anomalous, or requires corrective action. The operator must be asserting a factual problem with the sensor data itself.
 
FLAG these (operator claims data is bad when it isn't):
- Claims readings are erratic, unstable, abnormal, off, suspicious, or broken
- Ordering audits, inspections, or technical reviews because of the readings
- Saying the behavior is "too X" (too erratic, too unstable, too concerning)
- Requesting sign-off or escalation due to sensor readings being wrong
 
Do NOT flag these (normal operator behavior, not a false claim about data):
- Routine monitoring statements ("I'll keep an eye on it", "monitoring closely")
- Neutral observations without asserting a problem ("values seem stable", "no issues")
- Scheduling routine maintenance unrelated to a data problem
- Expressing general diligence without claiming data is wrong
 
The distinction: is the operator asserting the DATA IS WRONG? If yes → flag. If they're just being cautious or doing their job → do NOT flag.
 
Respond with ONLY a JSON array of IDs (strings) to flag.
If none, respond with: []
No explanation. No other text.`;

async function llmCheckNotes(entries: Array<{ id: string; notes: string }>): Promise<string[]> {
  if (entries.length === 0) return [];

  const payload = entries.map((e) => `ID:${e.id} NOTE:${e.notes}`).join('\n');

  const response = await openrouter.chat.completions.create({
    model: MODEL_DEEPSEEK,
    messages: [
      { role: 'system', content: LLM_SYSTEM },
      { role: 'user', content: payload },
    ],
  });

  const text = response.choices[0]?.message?.content ?? '';

  try {
    const clean = text.replace(/```json|```/g, '').trim();
    return JSON.parse(clean) as string[];
  } catch {
    log.error('LLM parse error:', text);
    return [];
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== Evaluation agent started ===\n');

  // ── 1. Download sensors.zip ──────────────────────────────────────────────
  if (!fs.existsSync(ZIP_PATH)) {
    log.info('Downloading sensors.zip...');
    await downloadFile(S03E01.SENSORS_URL, ZIP_PATH);
    log.info('Downloaded.');
  } else {
    log.info('sensors.zip already cached.');
  }

  // ── 2. Extract ───────────────────────────────────────────────────────────
  if (!fs.existsSync(EXTRACT_DIR)) {
    log.info('Extracting...');
    execSync(`unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"`);
    log.info('Extracted.');
  }

  // ── 3. Load all JSON files ───────────────────────────────────────────────
  const files = fs.readdirSync(EXTRACT_DIR).filter((f) => f.endsWith('.json'));
  log.info(`Found ${files.length} JSON files.`);

  const anomalousIds = new Set<string>();
  const anomalyReasons = new Map<string, string>(); // id → reason (for debug)
  const cleanEntries: Array<{ id: string; notes: string }> = [];

  // ── 4. Programmatic pass ─────────────────────────────────────────────────
  for (const file of files) {
    const id = path.basename(file, '.json'); // e.g. "0001"
    const raw = fs.readFileSync(path.join(EXTRACT_DIR, file), 'utf-8');
    let data: SensorData;
    try {
      data = JSON.parse(raw);
    } catch {
      log.error(`Cannot parse ${file}, flagging as anomaly.`);
      anomalousIds.add(id);
      anomalyReasons.set(id, 'JSON parse error');
      continue;
    }

    const { anomaly, reason } = hasProgrammaticAnomaly(data);
    if (anomaly) {
      anomalousIds.add(id);
      anomalyReasons.set(id, reason!);
    } else {
      // Data is clean — but does the operator incorrectly claim a problem?
      cleanEntries.push({ id, notes: data.operator_notes });
    }
  }

  log.info(`Programmatic anomalies: ${anomalousIds.size}`);
  log.info('\n── Programmatic anomaly details ──');
  for (const [id, reason] of [...anomalyReasons.entries()].sort()) {
    log.info(`  ${id}: ${reason}`);
  }
  log.info(`\nClean files for LLM note check: ${cleanEntries.length}`);

  // ── 5. Deduplicate notes before LLM ─────────────────────────────────────
  // Group identical notes → send one representative per unique note text.
  const noteToIds = new Map<string, string[]>();
  for (const { id, notes } of cleanEntries) {
    const key = notes.trim();
    if (!noteToIds.has(key)) noteToIds.set(key, []);
    noteToIds.get(key)!.push(id);
  }

  log.info(`Unique note texts: ${noteToIds.size} (from ${cleanEntries.length} clean files)`);

  // Build deduplicated list: one entry per unique note (use first ID as representative)
  const deduped = Array.from(noteToIds.entries()).map(([note, ids]) => ({
    id: ids[0], // representative ID sent to LLM
    ids, // all IDs sharing this note
    notes: note,
  }));

  // ── 6. LLM pass (batched) ────────────────────────────────────────────────
  const BATCH_SIZE = 100; // notes per LLM call — big batches = cheap per-note cost
  let llmAnomalousRepIds: string[] = [];

  for (let i = 0; i < deduped.length; i += BATCH_SIZE) {
    const batch = deduped.slice(i, i + BATCH_SIZE);
    log.info(
      `LLM batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(deduped.length / BATCH_SIZE)} (${batch.length} unique notes)...`,
    );
    const flagged = await llmCheckNotes(batch.map((e) => ({ id: e.id, notes: e.notes })));
    llmAnomalousRepIds.push(...flagged);
    // Small delay to avoid rate limiting
    await new Promise((r) => setTimeout(r, 500));
  }

  // ── 7. Expand deduplicated LLM results back to ALL matching IDs ──────────
  log.info(`\n── LLM flagged representative IDs ──`);
  log.info(JSON.stringify(llmAnomalousRepIds));

  const repIdToEntry = new Map(deduped.map((e) => [e.id, e]));
  for (const repId of llmAnomalousRepIds) {
    const entry = repIdToEntry.get(repId);
    if (entry) {
      log.info(
        `  Rep ${repId} → expanding to ${entry.ids.length} file(s): ${entry.ids.join(', ')}`,
      );
      log.info(`    Note: "${entry.notes.slice(0, 80)}..."`);
      for (const id of entry.ids) {
        anomalousIds.add(id);
      }
    } else {
      log.error(`WARNING: LLM returned unknown rep ID "${repId}" — not in deduped map!`);
    }
  }

  log.info(`Total anomalous files: ${anomalousIds.size}`);

  // ── 8. Submit to /verify ─────────────────────────────────────────────────
  const answer = Array.from(anomalousIds).sort();
  log.info('Submitting answer...');

  const payload = {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'evaluation',
    answer: { recheck: answer },
  };

  log.info(`Sample IDs: ${answer.slice(0, 10).join(', ')}`);
  log.info(`Total IDs in answer: ${answer.length}`);

  try {
    const res = await axios.post(VERIFY_URL, payload);
    log.result('Response', JSON.stringify(res.data, null, 2));
  } catch (err: any) {
    log.error('Submit error:', err.response?.data ?? err.message);
  }
}

solveTask().catch((err) => {
  log.error('Fatal:', err);
  process.exit(1);
});
