// ─── Deterministic Planner (replaces runPlanner LLM agent) ───────────────────
//
// Uses BFS over (row, col, fuel, food) state space.
// Guarantees optimality within resource constraints and avoids LLM hallucination.

import axios from 'axios';
import { log } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

const API_KEY = process.env.AIDEVS_API_KEY!;
const MAX_FUEL = 10;
const MAX_FOOD = 10;

interface Vehicle {
  name: string;
  fuel_per_move: number;
  food_per_move: number;
  cannot_cross: string[]; // terrain symbols this vehicle cannot enter
}

export interface ScoutFindings {
  map: string[][];
  vehicles: Record<string, any>[];
  legend: Record<string, any>;
  extra_notes?: string;
}

interface ParsedLegend {
  [symbol: string]: {
    passable_by: string[];
    fuel_penalty: number;
  };
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────

type Direction = 'up' | 'down' | 'left' | 'right' | 'dismount';

const MOVES: { dir: Direction; dr: number; dc: number }[] = [
  { dir: 'up', dr: -1, dc: 0 },
  { dir: 'down', dr: 1, dc: 0 },
  { dir: 'left', dr: 0, dc: -1 },
  { dir: 'right', dr: 0, dc: 1 },
];

function findSymbol(map: string[][], symbol: string): [number, number] | null {
  for (let r = 0; r < map.length; r++) {
    for (let c = 0; c < map[r].length; c++) {
      if (map[r][c] === symbol) return [r, c];
    }
  }
  return null;
}

// ─── Passability check ────────────────────────────────────────────────────────

function canEnter(
  symbol: string,
  vehicleName: string,
  cannotCross: string[],
  legend: ParsedLegend,
): boolean {
  // Check vehicle's own cannot_cross list first
  if (cannotCross.includes(symbol)) return false;

  const entry = legend[symbol];
  if (!entry) return true; // unknown symbol — optimistically passable

  const pb = entry.passable_by;
  if (!pb || pb.length === 0) return false; // empty array means impassable to all
  if (pb.includes('none')) return false;
  if (pb.includes('all')) return true;
  return pb.includes(vehicleName);
}

// ─── BFS with resource tracking ───────────────────────────────────────────────

interface State {
  row: number;
  col: number;
  fuel: number; // remaining
  food: number; // remaining
  isWalking: boolean;
}

interface QueueItem {
  state: State;
  path: Direction[];
}

/**
 * BFS returning the shortest valid path for a given vehicle, with optional dismount.
 */
function bfs(
  map: string[][],
  legend: ParsedLegend,
  vehicle: Vehicle,
  walkVehicle: Vehicle,
  start: [number, number],
  goal: [number, number],
): Direction[] | null {
  const rows = map.length;
  const cols = map[0].length;

  // We prune states using visited[r][c][wIdx] = max(fuel+food remaining) seen so far
  const visited = Array.from({ length: rows }, () => Array.from({ length: cols }, () => [-1, -1]));

  const queue: QueueItem[] = [];

  const [sr, sc] = start;
  const [gr, gc] = goal;
  const initialState: State = {
    row: sr,
    col: sc,
    fuel: MAX_FUEL,
    food: MAX_FOOD,
    isWalking: false,
  };
  queue.push({ state: initialState, path: [] });
  visited[sr][sc][0] = MAX_FUEL + MAX_FOOD;

  while (queue.length > 0) {
    const { state, path } = queue.shift()!;
    const { row, col, fuel, food, isWalking } = state;

    if (row === gr && col === gc) {
      return path; // reached goal
    }

    // Attempt dismount at current cell (if not already walking and not initially walking)
    if (!isWalking && vehicle.name !== 'walk') {
      const wIdx = 1;
      const resources = fuel + food;
      if (resources > visited[row][col][wIdx]) {
        visited[row][col][wIdx] = resources;
        queue.push({
          state: { row, col, fuel, food, isWalking: true },
          path: [...path, 'dismount'],
        });
      }
    }

    const currentVeh = isWalking ? walkVehicle : vehicle;

    for (const { dir, dr, dc } of MOVES) {
      if (dir === 'dismount') continue;
      const nr = row + dr;
      const nc = col + dc;

      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

      const symbol = map[nr][nc];
      if (!canEnter(symbol, currentVeh.name, currentVeh.cannot_cross, legend)) continue;

      const isPowered = currentVeh.fuel_per_move > 0;
      const fuelCost =
        currentVeh.fuel_per_move + (isPowered ? legend[symbol]?.fuel_penalty || 0 : 0);
      const nFuel = fuel - fuelCost;
      const nFood = food - currentVeh.food_per_move;

      // We must avoid floating point precision issues causing false failures
      if (Math.round(nFuel * 1000) / 1000 < 0 || Math.round(nFood * 1000) / 1000 < 0) continue;

      const wIdx = isWalking ? 1 : 0;
      const resources = nFuel + nFood;
      if (resources <= visited[nr][nc][wIdx]) continue;
      visited[nr][nc][wIdx] = resources;

      queue.push({
        state: { row: nr, col: nc, fuel: nFuel, food: nFood, isWalking },
        path: [...path, dir],
      });
    }
  }

  return null; // no path found
}

// ─── Public entry point ───────────────────────────────────────────────────────

export async function runPlanner(findings: ScoutFindings): Promise<void> {
  log.info('\n══════════ PLANNER (deterministic BFS) STARTED ══════════');

  log.info('[Planner] Map:');
  findings.map.forEach((row, i) => log.info(`  Row ${i}: ${row.join(' ')}`));

  const start = findSymbol(findings.map, 'S');
  const goal = findSymbol(findings.map, 'G');

  if (!start) throw new Error('[Planner] Start symbol "S" not found on map!');
  if (!goal) throw new Error('[Planner] Goal symbol "G" not found on map!');

  log.info(`[Planner] Start: row=${start[0]} col=${start[1]}`);
  log.info(`[Planner] Goal:  row=${goal[0]} col=${goal[1]}`);

  // Normalise vehicles array
  const vehicles: Vehicle[] = (findings.vehicles as unknown[]).map((v) => {
    const raw = v as Record<string, unknown>;
    // Special handling if cannot_cross is not perfectly formatted
    let cross = (raw.cannot_cross as any) ?? [];
    if (typeof cross === 'string') cross = [cross];
    // We map custom object fields carefully
    return {
      name: (raw.name ?? raw.vehicle_name ?? raw.id ?? 'unknown') as string,
      fuel_per_move: Number(raw.fuel_per_move ?? raw.fuel ?? 1),
      food_per_move: Number(raw.food_per_move ?? raw.food ?? 1),
      cannot_cross: cross,
    };
  });

  const walkVeh = vehicles.find((v) => v.name.toLowerCase() === 'walk' || v.name === 'on foot');
  if (!walkVeh) {
    throw new Error('[Planner] "walk" vehicle not found in findings, dismounting impossible!');
  }

  log.info('[Planner] Vehicles:');
  vehicles.forEach((v) =>
    log.info(
      `  ${v.name}: fuel/move=${v.fuel_per_move} food/move=${v.food_per_move} cannot_cross=[${v.cannot_cross.join(',')}]`,
    ),
  );

  // Parse Legend into ParsedLegend for quick access
  const parsedLegend: ParsedLegend = {};
  for (const [sym, entry] of Object.entries((findings.legend || {}) as Record<string, any>)) {
    let fuelPenalty = 0;
    const desc = (entry.meaning ?? '').toLowerCase();
    // Use regex to locate extra fuel penalty dynamically e.g. "increases fuel consumption by 0.2"
    const penaltyMatch = desc.match(/increases fuel consumption by ([\d.]+)/);
    if (penaltyMatch) fuelPenalty = parseFloat(penaltyMatch[1]);

    parsedLegend[sym] = {
      passable_by: entry.passable_by || [],
      fuel_penalty: fuelPenalty,
    };
  }

  type Result = { vehicle: string; path: Direction[]; moveCost: number };
  const results: Result[] = [];

  for (const vehicle of vehicles) {
    log.info(`[Planner] Running BFS for vehicle: ${vehicle.name}…`);
    const path = bfs(findings.map, parsedLegend, vehicle, walkVeh, start, goal);

    if (!path) {
      log.info(`  → No valid path found.`);
      continue;
    }

    // We don't log resource used directly since it fluctuates on dismount and trees, but cost=length
    log.info(`  → Path found: ${path.length} moves/actions`);
    log.info(`  → Moves: ${path.join(', ')}`);

    results.push({ vehicle: vehicle.name, path, moveCost: path.length });
  }

  if (results.length === 0) {
    throw new Error('[Planner] No vehicle can reach the goal within resource constraints!');
  }

  results.sort((a, b) => a.moveCost - b.moveCost);
  const best = results[0]!;

  log.info(`\n[Planner] Best route: vehicle="${best.vehicle}" moves=${best.moveCost}`);
  log.info(`[Planner] Submitting answer…`);

  const answer = [best.vehicle, ...best.path];

  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: API_KEY,
      task: 'savethem',
      answer,
    });
    const resultStr = JSON.stringify(data);
    log.info(`[Planner] Verify response: ${resultStr}`);

    const flagMatch = resultStr.match(/\{FLG:[^}]+\}/);
    if (flagMatch) log.result('=== FLAG FOUND ===', flagMatch[0]);
  } catch (err) {
    const body =
      axios.isAxiosError(err) && err.response
        ? JSON.stringify(err.response.data)
        : `ERROR: ${(err as Error).message}`;
    log.error(`[Planner] Verify error: ${body}`);
  }
}
