import axios from 'axios';
import 'dotenv/config';
import { log } from '../../shared/agents';
import { S01E02 } from '../../shared/api';

// Re-export so existing importers keep working without change
export const POWER_PLANTS_URL = S01E02.POWER_PLANTS_URL;
export const LOCATION_URL = S01E02.LOCATION_URL;
export const ACCESS_LEVEL_URL = S01E02.ACCESS_LEVEL_URL;

export const PLANT_COORDS: Record<string, { lat: number; lon: number }> = {
  Zabrze: { lat: 50.3249, lon: 18.7857 },
  'Piotrków Trybunalski': { lat: 51.4058, lon: 19.703 },
  Grudziądz: { lat: 53.4837, lon: 18.7536 },
  Tczew: { lat: 54.0924, lon: 18.7776 },
  Radom: { lat: 51.4027, lon: 21.1471 },
  Chelmno: { lat: 53.3497, lon: 18.4283 },
  Żarnowiec: { lat: 54.65, lon: 18.1833 },
};

export async function getClosestPlantForPerson(name: string, surname: string) {
  log.tool('getClosestPlantForPerson', { name, surname });
  const locations = await getLocations(name, surname);
  if (!locations || !Array.isArray(locations) || locations.length === 0) {
    return { error: 'No valid locations found' };
  }

  const mappedLocations = locations.map((loc: any) => ({
    lat: loc.latitude || loc.lat,
    lon: loc.longitude || loc.lon,
  }));

  const plantLocations = Object.entries(PLANT_COORDS).map(([plantName, coords]) => ({
    name: plantName,
    ...coords,
  }));

  return findClosestToPlants({ name, surname }, mappedLocations, plantLocations);
}

export async function findClosestToPlants(
  user: { name: string; surname: string },
  userLocations: { lat: number; lon: number }[],
  plantLocations: { lat: number; lon: number; name?: string }[],
) {
  log.info(`Finding closest power plant for ${user.name} ${user.surname}...`);
  let minDistance = Infinity;
  let closestLocation = null;
  let closestPlantName = 'Unknown';

  if (!userLocations || !plantLocations) return { error: 'Missing locations' };

  for (const uLoc of userLocations) {
    for (const pLoc of plantLocations) {
      const dist = calculateDistance(uLoc.lat, uLoc.lon, pLoc.lat, pLoc.lon);
      if (dist < minDistance) {
        minDistance = dist;
        closestLocation = uLoc;
        if (pLoc.name) {
          closestPlantName = pLoc.name;
        }
      }
    }
  }

  return { minDistance, closestLocation, closestPlantName };
}

export function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth's radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; // Distance in km
}

export async function getLocations(name: string, surname: string) {
  log.api('location', { name, surname });

  try {
    const res = await axios.post(LOCATION_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      name,
      surname,
    });

    return res.data;
  } catch (error: any) {
    log.error('Error fetching locations', error.message);
    return;
  }
}

export async function getAccessLevel(name: string, surname: string, birthDate: string | number) {
  // Extracting year from YYYY-MM-DD format or keeping it if it's already a year
  const birthYear = typeof birthDate === 'string' ? parseInt(birthDate.split('-')[0]) : birthDate;

  log.api('accesslevel', { name, surname, birthYear });

  try {
    const res = await axios.post(ACCESS_LEVEL_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      name,
      surname,
      birthYear,
    });

    log.result(`Access level for ${name} ${surname}`, res.data);
    return res.data;
  } catch (error: any) {
    log.error('Error fetching access level', error.message);
    return;
  }
}


