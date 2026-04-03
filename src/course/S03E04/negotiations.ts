import axios from 'axios';
import 'dotenv/config';
import express, { Request, Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { log, MODEL_DEEPSEEK, openrouter } from '../../shared/agents';
import { S03E04, VERIFY_URL } from '../../shared/api';

const app = express();
app.use(express.json());

// ── Data structures ────────────────────────────────────────────────────────────
type CityEntry = { cityName: string; cityCode: string };
type InventoryEntry = { itemCode: string; cities: CityEntry[] };
type Inventory = Record<string, InventoryEntry>;

const INVENTORY_PATH = path.resolve(__dirname, '../../../src/data/inventory.json');

let inventory: Inventory = {};

// ── CSV helpers ────────────────────────────────────────────────────────────────
function parseCsv(raw: string, fields: string[]): Record<string, string>[] {
  return raw
    .split('\n')
    .slice(1) // skip header
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(',');
      return Object.fromEntries(fields.map((f, i) => [f, (parts[i] ?? '').trim()]));
    });
}

// ── Inventory builder ──────────────────────────────────────────────────────────
async function buildInventory(): Promise<Inventory> {
  log.info('Fetching CSV data from hub...');

  const [itemsRaw, connectionsRaw, citiesRaw] = await Promise.all([
    axios.get<string>(`${S03E04.DATA}/items.csv`, { responseType: 'text' }).then((r) => r.data),
    axios
      .get<string>(`${S03E04.DATA}/connections.csv`, { responseType: 'text' })
      .then((r) => r.data),
    axios.get<string>(`${S03E04.DATA}/cities.csv`, { responseType: 'text' }).then((r) => r.data),
  ]);

  const items = parseCsv(itemsRaw, ['name', 'code']);
  const connections = parseCsv(connectionsRaw, ['itemCode', 'cityCode']);
  const cities = parseCsv(citiesRaw, ['name', 'code']);

  log.result(
    'CSV data loaded',
    `${items.length} items, ${connections.length} connections, ${cities.length} cities`,
  );

  // city code → city name
  const cityByCode = new Map(cities.map((c) => [c.code, c.name]));

  // item code → all city codes (one item can be in multiple cities)
  const cityCodesByItemCode = new Map<string, string[]>();
  for (const conn of connections) {
    const list = cityCodesByItemCode.get(conn.itemCode) ?? [];
    list.push(conn.cityCode);
    cityCodesByItemCode.set(conn.itemCode, list);
  }

  const inv: Inventory = {};
  for (const item of items) {
    const cityCodes = cityCodesByItemCode.get(item.code) ?? [];
    const cityEntries: CityEntry[] = cityCodes
      .map((code) => ({ cityCode: code, cityName: cityByCode.get(code) ?? '' }))
      .filter((e) => e.cityName !== '');
    inv[item.name] = { itemCode: item.code, cities: cityEntries };
  }

  fs.mkdirSync(path.dirname(INVENTORY_PATH), { recursive: true });
  fs.writeFileSync(INVENTORY_PATH, JSON.stringify(inv, null, 2), 'utf-8');
  log.result('Inventory saved', INVENTORY_PATH);

  return inv;
}

// ── Data loader ────────────────────────────────────────────────────────────────
async function loadData(): Promise<void> {
  if (fs.existsSync(INVENTORY_PATH)) {
    log.info(`Inventory file found at ${INVENTORY_PATH}, loading from disk...`);
    inventory = JSON.parse(fs.readFileSync(INVENTORY_PATH, 'utf-8')) as Inventory;
    log.result('Loaded from inventory.json', `${Object.keys(inventory).length} items`);
  } else {
    inventory = await buildInventory();
  }
}

const SYSTEM = `You are a precise inventory lookup assistant.
 When given a list of item names and a user query, return ONLY the exact item name(s) from the list that best match the query —
  one per line. If nothing matches, respond with NONE.`;

// ── AI-powered item matching ───────────────────────────────────────────────────
// Sends all inventory keys to the LLM and asks it to return exact key(s) that
// best match the natural-language query.
async function matchItemKeys(query: string): Promise<string[]> {
  const keyList = Object.keys(inventory).join('\n');

  log.info(`[matchItem] Query: "${query}"`);

  const completion = await openrouter.chat.completions.create({
    model: MODEL_DEEPSEEK,
    messages: [
      { role: 'system', content: SYSTEM },
      {
        role: 'user',
        content: `Available items:\n${keyList}\n\nUser is looking for: "${query}"\n\nRespond with the exact matching item name(s) from the list above, one per line. If nothing matches, respond with NONE.`,
      },
    ],
  });

  const raw = (completion.choices[0].message.content ?? '').trim();
  log.info(`[matchItem] Model returned:\n${raw}`);

  if (raw.toUpperCase() === 'NONE') return [];

  const matched = raw
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line in inventory);

  log.result(`[matchItem] Matched keys`, matched.join(', ') || '(none)');
  return matched;
}

// ── Tool endpoint ──────────────────────────────────────────────────────────────
// POST /find_cities
// body:     { "params": "potrzebuję rezystor 1 ohm" }
// response: { "output": "Warszawa, Krakow, Gdansk" }
app.post('/find_cities', async (req: Request, res: Response) => {
  const query: string = req.body?.params ?? '';
  log.info(`[/find_cities] Received query: "${query}"`);

  if (!query) {
    log.error('[/find_cities] Missing params in request body');
    res.json({ output: 'ERROR: missing params' });
    return;
  }

  try {
    const keys = await matchItemKeys(query);

    if (keys.length === 0) {
      log.info('[/find_cities] No matching item found');
      res.json({ output: 'Item not found' });
      return;
    }

    // Collect unique city names from all matched keys (each item may have multiple cities)
    const cityNames = [
      ...new Set(keys.flatMap((k) => inventory[k].cities.map((c) => c.cityName)).filter(Boolean)),
    ];

    log.info(`[/find_cities] Cities: ${cityNames.join(', ') || '(none)'}`);

    if (cityNames.length === 0) {
      res.json({ output: `No cities found for: ${keys.join(', ')}` });
      return;
    }

    const result = cityNames.join(', ');
    log.result('[/find_cities] Response', result);
    res.json({ output: result.substring(0, 490) });
  } catch (err: any) {
    log.error(`[/find_cities] Internal error: ${err.message}`);
    res.json({ output: 'ERROR: internal error' });
  }
});

// ── Tool registration ─────────────────────────────────────────────────────────
async function submitTools(): Promise<void> {
  const publicUrl = process.env.PUBLIC_URL ?? 'https://YOUR_NGROK_URL';
  log.info(`[submitTools] Registering tools at ${publicUrl}`);

  const payload = {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'negotiations',
    answer: {
      tools: [
        {
          URL: `${publicUrl}/find_cities`,
          description:
            'Returns city names that sell a SPECIFIC single item described in natural language. ' +
            'Call this tool separately for each item you need. ' +
            'Pass item description in "params" (string). ' +
            'Returns comma-separated city names or "Item not found".',
        },
      ],
    },
  };

  try {
    const response = await axios.post(VERIFY_URL, payload);
    log.result('[submitTools] Hub response', JSON.stringify(response.data));
  } catch (err: any) {
    log.error(`[submitTools] Failed to register tools: ${err.message}`);
  }
}

// ── Task status polling ────────────────────────────────────────────────────────
async function checkResult(): Promise<void> {
  const payload = {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'negotiations',
    answer: { action: 'check' },
  };

  try {
    const response = await axios.post(VERIFY_URL, payload);
    log.result('[checkResult] Status: ', JSON.stringify(response.data));
  } catch (err: any) {
    log.error(`[checkResult] Failed: ${err.message}`);
  }
}

// ── Start ──────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT ?? 3000;

loadData()
  .then(() => {
    app.listen(PORT, async () => {
      log.info(`=== Server running on port ${PORT} ===`);
      await submitTools();

      setTimeout(() => {
        log.info(`=== Starting to check result ===`);
        setInterval(checkResult, 10_000);
      }, 5000);
    });
  })
  .catch((err) => {
    log.error(`Failed to load data: ${err.message}`);
    process.exit(1);
  });
