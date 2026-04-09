import axios from 'axios';
import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { log } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

const dataDir = path.join(__dirname, 'data');
const startTime = Date.now();

function logTime(msg: string) {
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  log.info(`[${elapsed}s] ${msg}`);
}

async function callAPI(answer: Record<string, unknown>) {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'windpower',
      answer,
    });
    return data as any;
  } catch (error: any) {
    if (error.response?.data) {
      return error.response.data as any;
    }
    logTime(`API Error: ${error.message}`);
    return null;
  }
}

async function startWindow() {
  logTime('Starting service window...');
  await callAPI({ action: 'start' });
}

async function saveJson(filename: string, data: any) {
  const filepath = path.join(dataDir, `${filename}.json`);
  await fs.writeFile(filepath, JSON.stringify(data, null, 2));
  logTime(`Saved ${filepath}`);
}

async function main() {
  // 1. Remove json files if there are any in the data directory
  logTime('Cleaning data directory...');
  try {
    await fs.mkdir(dataDir, { recursive: true });
    const files = await fs.readdir(dataDir);
    for (const file of files) {
      if (file.endsWith('.json')) {
        await fs.unlink(path.join(dataDir, file));
        logTime(`Removed ${file}`);
      }
    }
  } catch (e: any) {
    logTime(`Error cleaning data dir: ${e.message}`);
  }

  // 2. Starts service window by sending request with action: start
  await startWindow();

  // 3. Sends request for data
  logTime('Initiating requests for all params...');
  const paramsList = ['weather', 'turbinecheck', 'powerplantcheck'];

  const docRes = await callAPI({ action: 'get', param: 'documentation' });
  logTime('Documentation fetched.');
  await saveJson('documentation', docRes);

  await Promise.all(paramsList.map((p) => callAPI({ action: 'get', param: p })));

  // 4. Polling continuously and routing payload by sourceFunction
  logTime('Starting centralized polling for async data...');
  const collected = new Set<string>();
  while (collected.size < paramsList.length) {
    const res = await callAPI({ action: 'getResult' }); // Retrieve from queue
    const resStr =
      typeof res === 'object' ? JSON.stringify(res).toLowerCase() : String(res).toLowerCase();

    if (
      res?.code !== 11 &&
      !resStr.includes('pending') &&
      !resStr.includes('processing') &&
      !resStr.includes('not ready') &&
      !resStr.includes('available yet')
    ) {
      const func = res.sourceFunction;
      if (func && !collected.has(func)) {
        logTime(`Results ready for ${func}! Saving to file...`);
        await saveJson(func, res);
        collected.add(func);
      }
    }

    await new Promise((r) => setTimeout(r, 1000));
  }

  logTime('All data fetched and saved successfully.');

  // Analyze weather data for top 4 highest windMs values
  try {
    const weatherDataRaw = await fs.readFile(path.join(dataDir, 'weather.json'), 'utf-8');
    const weatherData = JSON.parse(weatherDataRaw);

    if (weatherData.forecast && Array.isArray(weatherData.forecast)) {
      // Sort in descending order based on windMs
      const sortedForecasts = [...weatherData.forecast].sort((a, b) => b.windMs - a.windMs);

      // Read documentation to get the pitchAngle threshold
      const docRaw = await fs.readFile(path.join(dataDir, 'documentation.json'), 'utf-8');
      const docData = JSON.parse(docRaw);
      const damageRule = docData.windPowerYieldPercent.find(
        (y: any) => y.yieldPercent === 'damage',
      );
      const damageCutoff = parseInt(damageRule.windMsRange, 10);

      // Extract top 4 timestamps, windMs, and calculate pitchAngle
      const top4 = sortedForecasts.slice(0, 4).map((entry: any) => ({
        timestamp: entry.timestamp,
        windMs: entry.windMs,
        pitchAngle: entry.windMs >= damageCutoff ? 90 : 0,
      }));

      logTime(`Top 4 entries with pitchAngle:\n${JSON.stringify(top4, null, 2)}`);

      // Request all unlock codes
      logTime(`Submitting all unlock code requests...`);
      for (const entry of top4) {
        const parts = entry.timestamp.split(' ');
        const startDate = parts[0];
        const startHour = parts[1];
        // Provide flat params as required by API error log
        const req = {
          action: 'unlockCodeGenerator',
          startDate,
          startHour,
          windMs: entry.windMs,
          pitchAngle: entry.pitchAngle,
        };
        const initRes: any = await callAPI(req);
        logTime(
          `Queued unlock for ${entry.timestamp}: ${JSON.stringify(initRes).substring(0, 100)}`,
        );
      }

      logTime(`Starting centralized polling for unlock codes...`);
      const unlockResponses: string[] = [];
      while (unlockResponses.length < 4) {
        const res: any = await callAPI({ action: 'getResult' });
        const resStr =
          typeof res === 'object' ? JSON.stringify(res).toLowerCase() : String(res).toLowerCase();

        if (
          res?.code !== 11 &&
          !resStr.includes('pending') &&
          !resStr.includes('processing') &&
          !resStr.includes('not ready') &&
          !resStr.includes('available yet')
        ) {
          const strRep = JSON.stringify(res);
          if (!unlockResponses.includes(strRep)) {
            logTime(`Received an unlock code payload: ${strRep.substring(0, 150)}`);
            unlockResponses.push(strRep);
          }
        }
        await new Promise((r) => setTimeout(r, 1000));
      }

      logTime(`RAW unlockResponses received:\n${JSON.stringify(unlockResponses, null, 2)}`);

      const configs: Record<string, any> = {};
      for (const resStr of unlockResponses) {
        const resObj = JSON.parse(resStr);
        // Match the response back to its timestamp
        const entry = top4.find((e: any) => {
          const parts = e.timestamp.split(' ');
          return resStr.includes(parts[0]) && resStr.includes(parts[1]);
        });

        if (entry) {
          // API generally returns 'unlockCode' property
          const unlockCode = resObj.unlockCode || resObj.key || resObj.data?.unlockCode;
          configs[entry.timestamp] = {
            pitchAngle: entry.pitchAngle,
            turbineMode: entry.pitchAngle === 90 ? 'idle' : 'production',
            unlockCode: unlockCode,
          };
        }
      }

      logTime(`Submitting final config array:\n${JSON.stringify({ configs }, null, 2)}`);
      const finalRes = await callAPI({ action: 'config', configs });
      logTime(`Config action response:\n${JSON.stringify(finalRes, null, 2)}`);

      logTime(`Sending 'done' action to finalize...`);
      const doneRes = await callAPI({ action: 'done' });
      logTime(`🏁 Final Done Action Response:\n${JSON.stringify(doneRes, null, 2)}`);
    } else {
      logTime('Warning: No forecast array found in weather.json');
    }
  } catch (err: any) {
    logTime(`Error reading or analyzing weather data: ${err.message}`);
  }
}

main().catch((err) => logTime(`Fatal error: ${err.message}`));
