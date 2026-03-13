import axios from 'axios';
import 'dotenv/config';
import * as fs from 'fs';
import OpenAI from 'openai';
import * as path from 'path';
import { getAccessLevel, getClosestPlantForPerson, submitAnswer } from './utils';

const openai = new OpenAI({ baseURL: 'http://localhost:1234/v1', apiKey: 'lm-studio' });

// Configuration
const POWER_PLANTS_URL = `${process.env.BASE_URL}/data/${process.env.AIDEVS_API_KEY}/findhim_locations.json`;

/*
Strategia naprawcza w 3 krokach:
Uprość narzędzia: Zmień findClosestToPlants tak, by przyjmowało tylko name i surname. Pobieraj lokalizacje wewnątrz tej funkcji w TS, a modelowi zwróć tylko gotowy wynik (np. "Najbliższa elektrownia to X, dystans Y").
Loguj wszystko: Dodaj console.dir(message, { depth: null }), aby zobaczyć, czy model przypadkiem nie zwraca błędów w formacie JSON, których JSON.parse nie potrafi ugryźć.
Prompt Engineering: Dodaj do system prompt zdanie: "Jeśli nie masz nic więcej do dodania i posiadasz wszystkie dane, użyj funkcji submitAnswer. Nie powtarzaj się."
*/

async function solveTask() {
  const powerPlantsFile = path.join(__dirname, 'power_plants.json');
  let suspectedPeople, powerPlantsData;

  // 1. Load suspected people data from previous task
  console.log('=== Step 1 ===========');
  console.log('Getting list of suspected people from previous task...');
  const peopleResultsFile = path.join(__dirname, '../S01E01/people_results.json');

  if (!fs.existsSync(peopleResultsFile)) {
    console.log('File people_results.json not found in S01E01 folder. Leaving script...');
    return;
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

  // 3. Use model and define tools
  const messages: any[] = [
    {
      role: 'system',
      content: `
        You are an investigative agent. Your task is to find the person who was "very close" to one of the power plants.

        SUSPECTS TO CHECK:
        ${JSON.stringify(suspectedPeople, null, 2)}

        POWER PLANTS DATA (use for looking up the code):
        ${JSON.stringify(powerPlantsData, null, 2)}

        PROCEDURE — execute the following steps exactly as described:
        1. Call the function getClosestPlantForPerson() for EACH of the suspects to download the list of locations and find which power plant was the closest to any of his positions.
        2. Compare the results and select a person who was the closest to any power plant among all suspects (the one with the minimum distance).
        3. Call the getAccessLevel() function only for this specific person (chosen based on the smallest distance). Use the "born" field from SUSPECTS as the birthDate parameter.
        4. Call the submitAnswer() function exactly ONCE. As powerPlant, provide the power plant code ("code" from POWER PLANTS DATA), not its name. After calling submitAnswer(), you must stop working.
      `,
    },
    { role: 'user', content: 'Start the investigation. Who was the closest to a power plant?' },
  ];

  const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
    {
      type: 'function',
      function: {
        name: 'getClosestPlantForPerson',
        description:
          'Downloads a list of locations for a given person and finds which power plant was the closest to any of their positions.',
        strict: true,
        parameters: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            surname: { type: 'string' },
          },
          required: ['name', 'surname'],
          additionalProperties: false,
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'getAccessLevel',
        description: 'Gets the access level of a person.',
        strict: true,
        parameters: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            surname: { type: 'string' },
            birthDate: { type: 'string' },
          },
          required: ['name', 'surname', 'birthDate'],
          additionalProperties: false,
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'submitAnswer',
        description: 'Submits the final answer. Call this exactly once at the end.',
        strict: true,
        parameters: {
          type: 'object',
          properties: {
            answer: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                surname: { type: 'string' },
                accessLevel: { type: 'number' },
                powerPlant: { type: 'string' },
              },
              required: ['name', 'surname', 'accessLevel', 'powerPlant'],
              additionalProperties: false,
            },
          },
          required: ['answer'],
          additionalProperties: false,
        },
      },
    },
  ];

  // --- AGENT LOOP ---
  let iterations = 0;
  while (iterations < 50) {
    console.log(`[Iteration] ${iterations + 1}...`);
    iterations++;

    const response = await openai.chat.completions.parse({
      model: 'qwen3.5-9b',
      messages,
      tools,
    });

    const message = response.choices[0].message;
    console.log('Message:', message);

    const messageToPush = {
      role: message.role,
      content: message.content || '',
      tool_calls: message.tool_calls,
    };
    messages.push(messageToPush);

    for (const toolCall of message.tool_calls || []) {
      const args = JSON.parse(toolCall.function.arguments);
      let result;

      console.log(`[Tool Call] ${toolCall.function.name}...`);

      if (toolCall.function.name === 'getClosestPlantForPerson') {
        result = await getClosestPlantForPerson(args.name, args.surname);
      } else if (toolCall.function.name === 'getAccessLevel') {
        result = await getAccessLevel(args.name, args.surname, args.birthDate);
      } else if (toolCall.function.name === 'submitAnswer') {
        result = await submitAnswer(args.answer);
        console.log('Answer submitted, stopping the agent.');
        return;
      }

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: JSON.stringify(result ?? null),
      });
    }
  }
}

solveTask();
