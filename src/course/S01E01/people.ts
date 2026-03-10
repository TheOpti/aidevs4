import axios from 'axios';
import 'dotenv/config';
import * as fs from 'fs';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import * as path from 'path';
import { z } from 'zod';

type Person = {
  id: number;
  name: string;
  surname: string;
  gender: string;
  birthDate: string;
  birthPlace: string;
  job: string;
};

const TAGGING_INSTRUCTION = `Jesteś klasyfikatorem zawodów. Przypisz tagi z listy: IT, transport, edukacja, medycyna, praca z ludźmi, praca z pojazdami, praca fizyczna. 
Opisy tagów:
- IT: programowanie, systemy, dane.
- transport: przemieszczanie towarów/osób, logistyka transportowa, spedycja.
- edukacja: nauczanie, szkolenia.
- medycyna: ochrona zdrowia, leczenie.
- praca z ludźmi: obsługa klienta, nauczanie, zarządzanie zespołem.
- praca z pojazdami: kierowanie, naprawa, obsługa maszyn transportowych.
- praca fizyczna: wysiłek fizyczny, rzemiosło.`;

// Configuration
const CSV_URL = `${process.env.BASE_URL}/data/${process.env.AIDEVS_API_KEY}/people.csv`;
const VERIFY_URL = '${process.env.BASE_URL}/verify';

const openai = new OpenAI({ baseURL: 'http://localhost:1234/v1', apiKey: 'lm-studio' });

// Schema for Structured Output
const TaggingResponse = z.object({
  results: z.array(
    z.object({
      id: z.number(),
      tags: z.array(z.string()),
    }),
  ),
});

async function solveTask() {
  try {
    const resultsFilePath = path.join(__dirname, 'people_results.json');
    let finalSelection: any;

    if (fs.existsSync(resultsFilePath)) {
      console.log('Found people_results.json, skipping LLM and reusing data...');
      finalSelection = JSON.parse(fs.readFileSync(resultsFilePath, 'utf-8'));
    }

    if (!finalSelection) {
      // 1. Fetch data
      console.log('=== Step 1 ===========');
      console.log('Fetching data...');
      const response = await axios.get(CSV_URL);
      const csvData = response.data;

      // Simple CSV parsing (assuming no commas inside quoted descriptions)
      // In production, it's better to use the 'papaparse' library
      const rows = csvData.split('\n').slice(1); // skip header
      const people: Person[] = rows
        .map((row: string, index: number) => {
          // Regex to handle fields in quotes
          const matches = row.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g);
          if (!matches) return null;

          const [name, surname, gender, birthDate, birthPlace, birthCountry, job] = matches.map(
            (m) => m.replace(/"/g, ''),
          );
          return { id: index, name, surname, gender, birthDate, birthPlace, birthCountry, job };
        })
        .filter(Boolean);

      // 2. Initial filtering (Men, Gdańsk, Age 20-40 in 2026)
      // Birth year must be between 1986 and 2006
      console.log('=== Step 2 ===========');
      console.log('Filtering people...');
      const filteredPeople = people.filter((p) => {
        const birthYear = parseInt(p.birthDate.split('-')[0]);
        return (
          p.gender === 'M' && p.birthPlace === 'Grudziądz' && birthYear >= 1986 && birthYear <= 2006
        );
      });

      console.log(`People after initial filtering: ${filteredPeople.length}`);

      if (filteredPeople.length === 0) {
        console.log('No people found after initial filtering.');
        return;
      }

      // 3. Tagging jobs through LLM (Batch tagging)
      console.log('=== Step 3 ===========');
      console.log('Tagging jobs - sending a request to local LM Studio...');

      const CHUNK_SIZE = 16;
      const taggedResults: { id: number; tags: string[] }[] = [];

      for (let i = 0; i < filteredPeople.length; i += CHUNK_SIZE) {
        const chunk = filteredPeople.slice(i, i + CHUNK_SIZE);
        const jobDescriptions = chunk.map((p) => `ID: ${p.id} | Opis: ${p.job}`).join('\n');

        console.log(`Processing chunk ${i / CHUNK_SIZE + 1}...`);

        const completion = await openai.chat.completions.parse({
          model: 'qwen3.5-9b',
          messages: [
            { role: 'system', content: TAGGING_INSTRUCTION },
            { role: 'user', content: jobDescriptions },
          ],
          response_format: zodResponseFormat(TaggingResponse, 'tagging'),
        });

        if (completion.choices[0].message.parsed) {
          taggedResults.push(...completion.choices[0].message.parsed.results);
        }
      }

      // 4. Connect data and filter only 'transport' tag
      console.log('=== Step 4 ===========');
      console.log('Connecting data and filtering...');

      finalSelection = filteredPeople
        .map((p) => {
          const tags = taggedResults?.find((t) => t.id === p.id)?.tags || [];
          return {
            name: p.name,
            surname: p.surname,
            gender: p.gender,
            born: parseInt(p.birthDate.split('-')[0]),
            city: p.birthPlace,
            tags: tags,
          };
        })
        .filter((p) => p.tags.includes('transport'));

      console.log(`Found ${finalSelection.length} people working in transport.`);

      fs.writeFileSync(resultsFilePath, JSON.stringify(finalSelection, null, 2));
      console.log(`Saved finalSelection to ${resultsFilePath}`);
    }

    // 5. Send response
    console.log('=== Step 5 ===========');
    console.log('Sending response...');

    const payload = {
      task: 'people',
      apikey: process.env.AIDEVS_API_KEY,
      answer: finalSelection,
    };

    const verifyResponse = await axios.post(VERIFY_URL, payload);
    console.log('Server response:', verifyResponse.data);
  } catch (error) {
    console.error('Error:', error);
  }
}

solveTask();
