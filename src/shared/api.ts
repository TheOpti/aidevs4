import axios from 'axios';
import 'dotenv/config';
import { log } from './agents';

// ============================================================
// Base URLs
// ============================================================
export const BASE_URL = process.env.BASE_URL as string;
export const VERIFY_URL = `${BASE_URL}/verify`;
export const DATA_BASE_URL = `${BASE_URL}/data/${process.env.AIDEVS_API_KEY}`;

// ============================================================
// Task-specific URLs grouped by episode
// ============================================================

export const S01E01 = {
  CSV_URL: `${BASE_URL}/data/${process.env.AIDEVS_API_KEY}/people.csv`,
};

export const S01E02 = {
  POWER_PLANTS_URL: `${BASE_URL}/data/${process.env.AIDEVS_API_KEY}/findhim_locations.json`,
  LOCATION_URL: `${BASE_URL}/api/location`,
  ACCESS_LEVEL_URL: `${BASE_URL}/api/accesslevel`,
};

export const S01E03 = {
  PACKAGES_URL: `${BASE_URL}/api/packages`,
};

export const S02E03 = {
  LOGS: `${BASE_URL}/data/${process.env.AIDEVS_API_KEY}/failure.log`,
};

// ============================================================
// Send result to verification endpoint
// ============================================================

/**
 * Sends a task result to the VERIFY_URL endpoint.
 * @param task  The task identifier string (e.g. 'people', 'findhim').
 * @param answer The answer payload — can be any JSON-serialisable value.
 * @returns The parsed response data from the server.
 */
export async function sendResult(task: string, answer: unknown): Promise<unknown> {
  log.info(`Sending answer for task "${task}"...`);

  const payload = {
    task,
    apikey: process.env.AIDEVS_API_KEY,
    answer,
  };

  try {
    const response = await axios.post(VERIFY_URL, payload);
    log.result(`Server response for task "${task}"`, response.data);
    return response.data;
  } catch (error: any) {
    log.error(`Failed to send result for task "${task}"`, error.response?.data ?? error.message);
    throw error;
  }
}
