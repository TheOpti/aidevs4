import axios from 'axios';
import * as fs from 'fs';
import path from 'path';
import { log } from '../../shared/agents';
import { sendResult } from '../../shared/api';
import {
  PLANT_COORDS,
  POWER_PLANTS_URL,
  findClosestToPlants,
  getAccessLevel,
  getLocations,
} from './utils';

const powerPlantsFile = path.join(__dirname, '../../data/power_plants.json');
let suspectedPeople, powerPlantsData;

async function solveTask() {
  // 1. Load suspected people data from previous task
  log.step(1, 'Load suspected people from previous task');
  const peopleResultsFile = path.join(__dirname, '../../data/people_results.json');

  if (!fs.existsSync(peopleResultsFile)) {
    log.error('File people_results.json not found in data folder. Leaving script...');
    process.exit(1);
  }

  suspectedPeople = JSON.parse(fs.readFileSync(peopleResultsFile, 'utf-8'));
  log.info(`Loaded ${suspectedPeople.length} suspected people from people_results.json`);

  // 2. Fetch data about power plants
  log.step(2, 'Load power plants data');
  if (fs.existsSync(powerPlantsFile)) {
    log.info('Reusing local power_plants.json...');
    powerPlantsData = JSON.parse(fs.readFileSync(powerPlantsFile, 'utf-8'));
  } else {
    log.info('Fetching power plants data from remote...');

    const response = await axios.get(POWER_PLANTS_URL);
    powerPlantsData = response.data;

    fs.writeFileSync(powerPlantsFile, JSON.stringify(powerPlantsData, null, 2));
    log.info(`Saved power plants data to ${powerPlantsFile}`);
  }

  // 3. Get locations for each user
  log.step(3, 'Fetch locations for each suspect');
  const peopleWithLocations = [];
  for (const user of suspectedPeople) {
    const locations = await getLocations(user.name, user.surname);
    peopleWithLocations.push({ ...user, locations });
    log.info(`Found ${locations.length} locations for ${user.name} ${user.surname}`);
  }

  // 4. See what was the closest power plant for each user among the whole history
  log.step(4, 'Find closest power plant for each suspect');
  const plantLocations = Object.entries(PLANT_COORDS).map(([name, coords]) => ({
    name,
    ...coords,
  }));

  let closestUser: any = null;
  let minOverallDistance = Infinity;
  let closestOverallPlant = 'Unknown';

  for (const user of peopleWithLocations) {
    if (user.locations && Array.isArray(user.locations) && user.locations.length > 0) {
      // Map {latitude, longitude} to {lat, lon}
      const mappedLocations = user.locations.map((loc: any) => ({
        lat: loc.latitude || loc.lat,
        lon: loc.longitude || loc.lon,
      }));

      const result = await findClosestToPlants(
        { name: user.name, surname: user.surname },
        mappedLocations,
        plantLocations,
      );

      if ('error' in result) {
        log.error(`Could not find closest plant for ${user.name} ${user.surname}`, result.error);
        continue;
      }

      const closestPlantName = result.closestPlantName || 'Unknown';

      log.info(
        `Closest plant for ${user.name} ${user.surname}: ${closestPlantName} (${result.minDistance?.toFixed(2)} km)`,
      );

      if (result.minDistance && result.minDistance < minOverallDistance) {
        minOverallDistance = result.minDistance;
        closestUser = user;
        closestOverallPlant = closestPlantName;
      }
    } else {
      log.info(`No valid locations for ${user.name} ${user.surname}`);
    }
  }

  // 5. Get user with lowest distance to a power plant and send data to verification endpoint
  log.step(5, 'Submit answer');
  if (closestUser) {
    log.result(
      `Closest suspect: ${closestUser.name} ${closestUser.surname} | plant: ${closestOverallPlant} | distance: ${minOverallDistance.toFixed(2)} km`,
    );

    // Get access level
    const accessLevel = await getAccessLevel(
      closestUser.name,
      closestUser.surname,
      closestUser.born,
    );

    // Submit answer
    if (accessLevel) {
      const plantCode =
        powerPlantsData.power_plants[closestOverallPlant]?.code || closestOverallPlant;
      log.info(
        `Submitting answer for ${closestUser.name} ${closestUser.surname} [accessLevel: ${accessLevel.accessLevel}, powerPlant: ${plantCode}]`,
      );
      await sendResult('findhim', {
        name: closestUser.name,
        surname: closestUser.surname,
        accessLevel: accessLevel.accessLevel,
        powerPlant: plantCode,
      });
    } else {
      log.error('Failed to get access level, cannot submit answer');
    }
  } else {
    log.info('No closest user found');
  }
}

solveTask();
