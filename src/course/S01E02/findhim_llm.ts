import axios from 'axios';
import 'dotenv/config';
import * as fs from 'fs';
import type OpenAI from 'openai';
import * as path from 'path';
import { MODEL_GPT_OSS, log, openai } from '../../shared/agents';
import { S01E02, sendResult } from '../../shared/api';
import { getAccessLevel, getClosestPlantForPerson } from './utils';

const { POWER_PLANTS_URL } = S01E02;

async function solveTask() {
  const powerPlantsFile = path.join(__dirname, '../../data/power_plants.json');
  let suspectedPeople, powerPlantsData;

  // 1. Load suspected people data from previous task
  log.step(1, 'Load suspected people from previous task');
  const peopleResultsFile = path.join(__dirname, '../../data/people_results.json');

  if (!fs.existsSync(peopleResultsFile)) {
    log.error('File people_results.json not found in data folder. Leaving script...');
    return;
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
  log.step(3, 'Run agent loop');
  let iterations = 0;
  while (iterations < 50) {
    log.info(`Iteration ${iterations + 1}...`);
    iterations++;

    const response = await openai.chat.completions.parse({
      model: MODEL_GPT_OSS,
      messages,
      tools,
    });

    const message = response.choices[0].message;
    log.result('Model response', message);

    const messageToPush = {
      role: message.role,
      content: message.content || '',
      tool_calls: message.tool_calls,
    };
    messages.push(messageToPush);

    for (const toolCall of message.tool_calls || []) {
      const args = JSON.parse(toolCall.function.arguments);
      let result;

      log.tool(toolCall.function.name, args);

      if (toolCall.function.name === 'getClosestPlantForPerson') {
        result = await getClosestPlantForPerson(args.name, args.surname);
      } else if (toolCall.function.name === 'getAccessLevel') {
        result = await getAccessLevel(args.name, args.surname, args.birthDate);
      } else if (toolCall.function.name === 'submitAnswer') {
        const a = args.answer;
        log.info(
          `Submitting answer for ${a.name} ${a.surname} [accessLevel: ${a.accessLevel}, powerPlant: ${a.powerPlant}]`,
        );
        result = await sendResult('findhim', a);
        log.result('Answer submitted, stopping the agent');
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
