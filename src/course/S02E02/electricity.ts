import axios from 'axios';
import 'dotenv/config';
import sharp from 'sharp';
import { log } from 'src/shared/agents';
import { DATA_BASE_URL, VERIFY_URL } from 'src/shared/api';

const TARGET_URL = `${process.env.BASE_URL}/i/solved_electricity.png`;

const CELL_LABELS = ['1x1', '1x2', '1x3', '2x1', '2x2', '2x3', '3x1', '3x2', '3x3'] as const;

type CellLabel = (typeof CELL_LABELS)[number];

/**
 * Cell size after upscaling. All pixel detection parameters are tuned for
 * this resolution — change only if you also re-tune PAD and DEPTH below.
 */
const CELL_PX = 300;

/**
 * Edge detection parameters (calibrated on CELL_PX = 300):
 *
 *   PAD   — pixels to skip from each edge before scanning.
 *           Skips the grid border line itself (~4–8 px raw, ~15 px upscaled).
 *   DEPTH — pixels to scan after the pad.
 *           Wide enough to catch the cable (~30 px upscaled) but not so wide
 *           that it bleeds into the cable running parallel along the inside.
 *   STRIP — fraction of the cell width used as the "centre strip" for N/S/W/E.
 *           1/3 on each side → middle third. Avoids corner ambiguity.
 *   DARK  — greyscale threshold: pixel < DARK → counts as cable.
 */
const PAD = 15;
const DEPTH = 25;
const STRIP = 1 / 3;
const DARK = 80;

type Edge = 'N' | 'S' | 'E' | 'W';
type CellEdges = Edge[];

type RotationEntry = {
  cell: CellLabel;
  rotations: number;
  current: CellEdges;
  target: CellEdges;
};

type RotationPlan = RotationEntry[];

type GridLines = {
  cols: [number, number, number, number];
  rows: [number, number, number, number];
};

const ROTATE_MAP: Record<Edge, Edge> = { N: 'E', E: 'S', S: 'W', W: 'N' };

function rotateEdges(edges: CellEdges, times: number): CellEdges {
  let result = [...edges];
  for (let i = 0; i < times % 4; i++) result = result.map((e) => ROTATE_MAP[e]);
  return result.sort() as CellEdges;
}

const edgeKey = (edges: CellEdges) => [...edges].sort().join(',');

/**
 * Returns 0–3 clockwise rotations needed to turn `current` into `target`,
 * or null if incompatible (different edge count → mis-detection).
 */
function calculateRotations(current: CellEdges, target: CellEdges): number | null {
  if (current.length !== target.length) return null;
  const tKey = edgeKey(target);
  for (let r = 0; r < 4; r++) {
    if (edgeKey(rotateEdges(current, r)) === tKey) return r;
  }
  return null;
}

/**
 * Detect which edges of a cell the cable exits through — no AI, pure pixels.
 *
 * For each of the 4 edges (N/S/W/E):
 *   1. Define a rectangle = [centre-strip] × [PAD … PAD+DEPTH] from that edge.
 *   2. If any pixel in that rectangle is darker than DARK → edge is connected.
 *
 * The PAD skips the grid border line (which is always dark and would give a
 * false positive for every edge).  The centre-strip (middle third of the cell)
 * avoids false positives from cables that run near a corner without crossing
 * the edge midpoint.
 *
 * @param cellBuffer - Raw greyscale pixel data (CELL_PX × CELL_PX, 1 channel)
 */
function detectEdgesPixel(cellBuffer: Buffer): CellEdges {
  const W = CELL_PX;
  const H = CELL_PX;
  const stripStart = Math.floor(W * STRIP);
  const stripEnd = Math.floor(W * (1 - STRIP));

  /** Returns true if any pixel in the rectangle [x0,x1) × [y0,y1) is dark. */
  const hasDark = (x0: number, x1: number, y0: number, y1: number): boolean => {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        if (cellBuffer[y * W + x] < DARK) return true;
      }
    }
    return false;
  };

  const edges: CellEdges = [];

  // N: top strip — rows [PAD … PAD+DEPTH), cols [stripStart … stripEnd)
  if (hasDark(stripStart, stripEnd, PAD, PAD + DEPTH)) edges.push('N');
  // S: bottom strip
  if (hasDark(stripStart, stripEnd, H - PAD - DEPTH, H - PAD)) edges.push('S');
  // W: left strip — cols [PAD … PAD+DEPTH), rows [stripStart … stripEnd)
  if (hasDark(PAD, PAD + DEPTH, stripStart, stripEnd)) edges.push('W');
  // E: right strip
  if (hasDark(W - PAD - DEPTH, W - PAD, stripStart, stripEnd)) edges.push('E');

  return edges;
}

/**
 * Find the 4 column and 4 row boundaries of the 3×3 puzzle grid by locating
 * the longest continuous dark runs in each row/column of the image.
 *
 * Grid border lines are the longest unbroken dark segments in the image —
 * much longer than any icon or decoration outside the grid.
 */
async function detectGridLines(buffer: Buffer): Promise<GridLines> {
  const { data, info } = await sharp(buffer)
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height } = info;

  type Run = { pos: number; length: number };

  // Longest dark run per row
  const hRuns: Run[] = [];
  for (let y = 0; y < height; y++) {
    let best = 0,
      runStart = -1;
    for (let x = 0; x <= width; x++) {
      const dark = x < width && data[y * width + x] < 100;
      if (dark && runStart === -1) runStart = x;
      if (!dark && runStart !== -1) {
        best = Math.max(best, x - runStart);
        runStart = -1;
      }
    }
    if (best > 20) hRuns.push({ pos: y, length: best });
  }

  // Longest dark run per column
  const vRuns: Run[] = [];
  for (let x = 0; x < width; x++) {
    let best = 0,
      runStart = -1;
    for (let y = 0; y <= height; y++) {
      const dark = y < height && data[y * width + x] < 100;
      if (dark && runStart === -1) runStart = y;
      if (!dark && runStart !== -1) {
        best = Math.max(best, y - runStart);
        runStart = -1;
      }
    }
    if (best > 20) vRuns.push({ pos: x, length: best });
  }

  const maxH = Math.max(...hRuns.map((r) => r.length));
  const maxV = Math.max(...vRuns.map((r) => r.length));

  const strongH = hRuns.filter((r) => r.length >= maxH * 0.5);
  const strongV = vRuns.filter((r) => r.length >= maxV * 0.5);

  // Cluster nearby parallel lines (grid lines are a few px thick)
  function cluster(runs: Run[]): number[] {
    const sorted = [...runs].sort((a, b) => a.pos - b.pos);
    const groups: number[][] = [[sorted[0].pos]];
    for (let i = 1; i < sorted.length; i++) {
      const last = groups[groups.length - 1];
      if (sorted[i].pos - last[last.length - 1] <= 10) last.push(sorted[i].pos);
      else groups.push([sorted[i].pos]);
    }
    return groups.map((g) => g[Math.floor(g.length / 2)]);
  }

  const hPos = cluster(strongH);
  const vPos = cluster(strongV);

  log.info(`[Grid] H clusters (${hPos.length}): ${hPos}`);
  log.info(`[Grid] V clusters (${vPos.length}): ${vPos}`);

  if (hPos.length < 4 || vPos.length < 4) {
    throw new Error(
      `Grid detection failed: ${hPos.length} H lines, ${vPos.length} V lines (need 4 each). ` +
        `Run debugSaveCells() to inspect.`,
    );
  }

  const rows = hPos.slice(0, 4).sort((a, b) => a - b) as [number, number, number, number];
  const cols = vPos.slice(0, 4).sort((a, b) => a - b) as [number, number, number, number];

  return { cols, rows };
}

/**
 * Slice the grid into 9 cells using exact grid-line coordinates.
 * Returns raw greyscale pixel buffers (CELL_PX × CELL_PX, 1 byte/pixel).
 */
async function extractCells(imageBuffer: Buffer, grid: GridLines): Promise<Buffer[]> {
  const cells: Buffer[] = [];

  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const left = grid.cols[col];
      const top = grid.rows[row];
      const width = grid.cols[col + 1] - left;
      const height = grid.rows[row + 1] - top;

      const cell = await sharp(imageBuffer)
        .extract({ left, top, width, height })
        .normalise() // maximise contrast so cables are clearly dark
        .resize(CELL_PX, CELL_PX)
        .greyscale() // 1 channel — matches detectEdgesPixel expectations
        .raw()
        .toBuffer();

      cells.push(cell);
    }
  }

  return cells;
}

/**
 * Save all 9 extracted cells to disk for visual inspection.
 * Uncomment the calls in solveTask once, verify the PNGs, then re-comment.
 */
export async function debugSaveCells(imageUrl: string, prefix: string): Promise<void> {
  log.info(`[Debug] Saving cells for "${prefix}"...`);

  const response = await axios.get<ArrayBuffer>(imageUrl, { responseType: 'arraybuffer' });
  const buffer = Buffer.from(response.data);
  const grid = await detectGridLines(buffer);

  log.info(`[Debug] cols=${grid.cols}  rows=${grid.rows}`);

  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const idx = row * 3 + col;
      const label = CELL_LABELS[idx];
      const left = grid.cols[col];
      const top = grid.rows[row];
      const width = grid.cols[col + 1] - left;
      const height = grid.rows[row + 1] - top;

      await sharp(buffer)
        .extract({ left, top, width, height })
        .normalise()
        .resize(CELL_PX, CELL_PX)
        .png()
        .toFile(`debug_${prefix}_${label}.png`);

      log.info(`[Debug]   debug_${prefix}_${label}.png`);
    }
  }
}

async function getRotationPlan(boardUrl: string): Promise<RotationPlan> {
  log.info('[Detect] Fetching board images...');

  const [currentResp, targetResp] = await Promise.all([
    axios.get<ArrayBuffer>(boardUrl, { responseType: 'arraybuffer' }),
    axios.get<ArrayBuffer>(TARGET_URL, { responseType: 'arraybuffer' }),
  ]);

  const currentBuf = Buffer.from(currentResp.data);
  const targetBuf = Buffer.from(targetResp.data);

  log.info('[Detect] Finding grid lines...');
  const [currentGrid, targetGrid] = await Promise.all([
    detectGridLines(currentBuf),
    detectGridLines(targetBuf),
  ]);
  log.info(`[Detect] Current — cols=${currentGrid.cols}  rows=${currentGrid.rows}`);
  log.info(`[Detect] Target  — cols=${targetGrid.cols}   rows=${targetGrid.rows}`);

  log.info('[Detect] Extracting cells...');
  const [currentCells, targetCells] = await Promise.all([
    extractCells(currentBuf, currentGrid),
    extractCells(targetBuf, targetGrid),
  ]);

  log.info('[Detect] Reading edges via pixel scan...');
  const plan: RotationPlan = [];

  for (let i = 0; i < 9; i++) {
    const label = CELL_LABELS[i];
    const cur = detectEdgesPixel(currentCells[i]);
    const tgt = detectEdgesPixel(targetCells[i]);
    const r = calculateRotations(cur, tgt);

    if (r === null) {
      log.info(`[Plan] ⚠️  ${label}: incompatible cur=[${cur}] tgt=[${tgt}] — skipping`);
      continue;
    }
    if (r === 0) {
      log.info(`[Plan] ${label}: ✓  [${cur}]`);
      continue;
    }

    log.info(`[Plan] ${label}: [${cur}] → [${tgt}]  ×${r}`);
    plan.push({ cell: label, rotations: r, current: cur, target: tgt });
  }

  const total = plan.reduce((s, p) => s + p.rotations, 0);
  log.info(`[Plan] ${plan.length} cell(s) to rotate — ${total} total click(s).`);
  return plan;
}

async function rotateCell(cell: CellLabel): Promise<string | null> {
  const response = await axios.post(VERIFY_URL, {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'electricity',
    answer: { rotate: cell },
  });
  const body = JSON.stringify(response.data);
  const flagMatch = body.match(/\{FLG:[^}]+\}/);
  log.info(`[API] rotate ${cell} → ${body}${flagMatch ? '  🚩 FLAG!' : ''}`);
  return flagMatch ? flagMatch[0] : null;
}

async function executeRotationPlan(plan: RotationPlan): Promise<string | null> {
  const total = plan.reduce((s, p) => s + p.rotations, 0);
  log.info(`[Execute] ${total} rotation(s) across ${plan.length} cell(s)...`);
  for (const entry of plan) {
    for (let i = 0; i < entry.rotations; i++) {
      log.info(`[Execute] ${entry.cell}  ${i + 1}/${entry.rotations}`);
      const flag = await rotateCell(entry.cell);
      if (flag) return flag;
    }
  }
  return null;
}

async function resetBoard(): Promise<void> {
  log.info('[Board] Resetting...');
  await axios.get(`${DATA_BASE_URL}/electricity.png?reset=1`);
  await new Promise((r) => setTimeout(r, 1500));
  log.info('[Board] Reset complete.');
}

async function solveTask(): Promise<void> {
  log.info('=== Electricity task started ===');

  // ── Uncomment ONCE to verify crops visually, then re-comment ─
  // await debugSaveCells(`${DATA_BASE_URL}/electricity.png`, 'current');
  // await debugSaveCells(TARGET_URL, 'target');
  // return;

  await resetBoard();

  const MAX_ATTEMPTS = 3;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    log.info(`\n═══ Attempt ${attempt}/${MAX_ATTEMPTS} ═══`);

    const boardUrl = `${DATA_BASE_URL}/electricity.png`;
    const plan = await getRotationPlan(boardUrl);

    if (plan.length === 0) {
      log.info('[Solve] Board already matches target.');
      return;
    }

    const flag = await executeRotationPlan(plan);
    if (flag) {
      log.result('=== FLAG FOUND ===', flag);
      return;
    }

    log.info('[Verify] Re-reading board after rotations...');
    const remaining = await getRotationPlan(boardUrl);

    if (remaining.length === 0) {
      log.info('[Verify] Board matches target — puzzle solved!');
      return;
    }

    log.info(`[Verify] ${remaining.length} cell(s) still wrong after attempt ${attempt}.`);
    if (attempt < MAX_ATTEMPTS) {
      log.info('[Verify] Resetting...');
      await resetBoard();
    }
  }

  log.error(`[Solve] Not solved after ${MAX_ATTEMPTS} attempts.`);
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
