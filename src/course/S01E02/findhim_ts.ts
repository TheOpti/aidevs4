import axios from 'axios';
import * as fs from 'fs';
import path from 'path';
import {
  PLANT_COORDS,
  POWER_PLANTS_URL,
  findClosestToPlants,
  getAccessLevel,
  getLocations,
  submitAnswer,
} from './utils';

const powerPlantsFile = path.join(__dirname, 'power_plants.json');
let suspectedPeople, powerPlantsData;

// Prawidłowa odp. to Wojciech Bielik -> PWR2758PL (Chelmno)

async function solveTask() {
  // 1. Load suspected people data from previous task
  console.log('=== Step 1 ===========');
  console.log('Getting list of suspected people from previous task...');
  const peopleResultsFile = path.join(__dirname, '../S01E01/people_results.json');

  if (!fs.existsSync(peopleResultsFile)) {
    console.log('File people_results.json not found in S01E01 folder. Leaving script...');
    process.exit(1);
  }

  suspectedPeople = JSON.parse(fs.readFileSync(peopleResultsFile, 'utf-8'));
  console.log(`Loaded ${suspectedPeople.length} suspected people from people_results.json.`);

  // 2. Fetch data about power plants
  console.log('=== Step 2 ===========');
  if (fs.existsSync(powerPlantsFile)) {
    console.log('Reusing local power_plants.json...');
    powerPlantsData = JSON.parse(fs.readFileSync(powerPlantsFile, 'utf-8'));
  } else {
    console.log('Getting power plants data...');

    const response = await axios.get(POWER_PLANTS_URL);
    powerPlantsData = response.data;

    fs.writeFileSync(powerPlantsFile, JSON.stringify(powerPlantsData, null, 2));
    console.log(`Saved power plants data to ${powerPlantsFile}`);
  }

  // 3. Get locations for each user
  console.log('=== Step 3 ===========');
  const peopleWithLocations = [];
  for (const user of suspectedPeople) {
    const locations = await getLocations(user.name, user.surname);
    peopleWithLocations.push({ ...user, locations });
    console.log(`Found ${locations.length} locations for ${user.name} ${user.surname}`);
  }

  // 4. See what was the closest power plant for each user among the whole history
  console.log('=== Step 4 ===========');
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
        console.log(
          `Error finding closest location for ${user.name} ${user.surname}:`,
          result.error,
        );
        continue;
      }

      const closestPlantName = result.closestPlantName || 'Unknown';

      console.log(
        `Closest plant for ${user.name} ${user.surname} is ${closestPlantName} (distance: ${result.minDistance?.toFixed(2)} km)`,
      );

      if (result.minDistance && result.minDistance < minOverallDistance) {
        minOverallDistance = result.minDistance;
        closestUser = user;
        closestOverallPlant = closestPlantName;
      }
    } else {
      console.log(`No valid locations for ${user.name} ${user.surname}`);
    }
  }

  // 5. Get user with lowest distance to a power plant and send data to verification endpoint
  console.log('=== Step 5 ===========');
  if (closestUser) {
    console.log(
      `Closest user overall is ${closestUser.name} ${closestUser.surname} (plant: ${closestOverallPlant}, distance: ${minOverallDistance.toFixed(2)} km)`,
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
      await submitAnswer({
        name: closestUser.name,
        surname: closestUser.surname,
        accessLevel: accessLevel.accessLevel,
        powerPlant: plantCode,
      });
    } else {
      console.log('Failed to get access level, cannot submit answer.');
    }
  } else {
    console.log('No closest user found.');
  }
}

solveTask();
