import axios from 'axios';
import 'dotenv/config';

// Configuration
export const POWER_PLANTS_URL = `${process.env.BASE_URL}/data/${process.env.AIDEVS_API_KEY}/findhim_locations.json`;
export const LOCATION_URL = '${process.env.BASE_URL}/api/location';
export const ACCESS_LEVEL_URL = '${process.env.BASE_URL}/api/accesslevel';
export const VERIFY_URL = '${process.env.BASE_URL}/verify';

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
  console.log(`[Tool Call] getClosestPlantForPerson for ${name} ${surname}...`);
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
  console.log(`Finding closest to each Power Plant for ${user.name} ${user.surname}...`);
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
  // Returns list of coordinates or string
  console.log(`Fetching locations for name: ${name} surname: ${surname}...`);

  try {
    const res = await axios.post(LOCATION_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      name,
      surname,
    });

    return res.data;
  } catch (error: any) {
    console.error('Error fetching locations:', error.message, error.description);
    return;
  }
}

export async function getAccessLevel(name: string, surname: string, birthDate: string | number) {
  console.log(
    `Fetching access level for name: ${name}, surname: ${surname}, birthYear: ${birthDate}...`,
  );

  // Extracting year from YYYY-MM-DD format or keeping it if it's already a year
  const birthYear = typeof birthDate === 'string' ? parseInt(birthDate.split('-')[0]) : birthDate;

  try {
    const res = await axios.post(ACCESS_LEVEL_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      name,
      surname,
      birthYear,
    });

    console.log(`Access level for ${name} ${surname}: ${res.data}`);
    return res.data;
  } catch (error: any) {
    console.error('Error fetching access level:', error.message, error.description);
    return;
  }
}

export async function submitAnswer(answer: {
  name: string;
  surname: string;
  accessLevel: number;
  powerPlant: string;
}) {
  console.log(
    `Submitting answer for ${answer.name} ${answer.surname} [accessLevel: ${answer.accessLevel}, powerPlant: ${answer.powerPlant}]...`,
  );
  try {
    const res = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'findhim',
      answer,
    });
    console.log('Submit answer response:', res.data);
    return res.data;
  } catch (error: any) {
    console.error(
      'Error submitting answer:',
      error.message,
      error.response?.data || error.description,
    );
    return;
  }
}
