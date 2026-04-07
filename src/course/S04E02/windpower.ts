import axios from 'axios';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_CLAUDE_SONNET, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

// ── API helper ────────────────────────────────────────────────────────────────

async function callAPI(answer: Record<string, unknown>) {
  try {
    const { data } = await axios.post(VERIFY_URL, {
      apikey: process.env.AIDEVS_API_KEY,
      task: 'windpower',
      answer,
    });
    return data;
  } catch (error: any) {
    log.error(error.message);
  }
}

// ── Tool implementations ──────────────────────────────────────────────────────

/** Single synchronous-style API call (help, start, config, done, turbinecheck) */
async function windpowerAction(
  action: string,
  extra: Record<string, unknown> = {},
): Promise<string> {
  const result = await callAPI({ action, ...extra });
  return JSON.stringify(result);
}

/**
 * Queue several async actions IN PARALLEL and return the list of taskIds.
 * Actions that return a taskId are: weatherForecast, turbineStatus,
 * powerPlantStatus, unlockCodeGenerator, etc.
 */
async function queueTasksParallel(
  actions: { action: string; params?: Record<string, unknown> }[],
): Promise<string> {
  const results = await Promise.all(
    actions.map(({ action, params = {} }) => callAPI({ action, ...params })),
  );
  // Each result should contain a taskId or similar identifier
  return JSON.stringify(results);
}

/**
 * Poll getResult repeatedly until all expected taskIds have results.
 * Returns a map of taskId → result.
 */
async function pollResults(taskIds: string[], timeoutMs = 30_000): Promise<string> {
  const collected: Record<string, unknown> = {};
  const deadline = Date.now() + timeoutMs;

  while (Object.keys(collected).length < taskIds.length && Date.now() < deadline) {
    // Fire all pending polls in parallel
    const pending = taskIds.filter((id) => !(id in collected));
    const responses = await Promise.all(
      pending.map((taskId) =>
        callAPI({ action: 'getResult', taskId }).catch((e) => ({ error: String(e), taskId })),
      ),
    );
    for (let i = 0; i < pending.length; i++) {
      const res = responses[i] as Record<string, unknown>;
      // Consider it done when the result is not "pending" / not an error about not-ready
      const resStr = JSON.stringify(res).toLowerCase();
      if (
        !resStr.includes('pending') &&
        !resStr.includes('not ready') &&
        !resStr.includes('processing')
      ) {
        collected[pending[i]] = res;
      }
    }
    if (Object.keys(collected).length < taskIds.length) {
      await new Promise((r) => setTimeout(r, 800));
    }
  }
  return JSON.stringify(collected);
}

// ── Tool definitions ──────────────────────────────────────────────────────────

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'windpower_action',
      description:
        'Call the windpower API with a single action. Use for: help, start, turbinecheck, done. ' +
        'Also use for config with a "configs" object (multiple entries) or single config fields. ' +
        'Returns the raw API response.',
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            description: 'Action name: help | start | config | turbinecheck | done | getResult',
          },
          extra: {
            type: 'object',
            description:
              'Additional fields merged into the request body. ' +
              'For config: pass { configs: { "YYYY-MM-DD HH:00:00": { pitchAngle, turbineMode, unlockCode } } }. ' +
              'For getResult: pass { taskId: "..." }.',
            additionalProperties: true,
          },
        },
        required: ['action'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'queue_tasks_parallel',
      description:
        'Queue multiple async API actions simultaneously (in parallel) and return all responses. ' +
        'Use for: weatherForecast, turbineStatus, powerPlantStatus, unlockCodeGenerator (one per config entry). ' +
        'This is the ONLY way to meet the 40-second deadline — never queue these one by one.',
      parameters: {
        type: 'object',
        properties: {
          actions: {
            type: 'array',
            description: 'List of actions to fire in parallel.',
            items: {
              type: 'object',
              properties: {
                action: { type: 'string' },
                params: {
                  type: 'object',
                  description:
                    'Extra params merged into the API body (e.g. date range, config data).',
                  additionalProperties: true,
                },
              },
              required: ['action'],
            },
          },
        },
        required: ['actions'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'poll_results',
      description:
        'Poll getResult for a list of taskIds IN PARALLEL until all return non-pending results. ' +
        'Returns a JSON object mapping each taskId to its result. ' +
        'Call this after queue_tasks_parallel to retrieve weather, turbine status, unlock codes, etc.',
      parameters: {
        type: 'object',
        properties: {
          taskIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of taskId strings received from previous queued actions.',
          },
        },
        required: ['taskIds'],
      },
    },
  },
];

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM = `You are an autonomous agent solving the "windpower" task for an AI engineer course.
You control a wind turbine that powers an electricity plant. You must configure the turbine schedule
within a 40-SECOND TIME LIMIT, so parallel execution is CRITICAL.

## Workflow (follow exactly in this order)

1. Call windpower_action(action="help") to learn all available async actions and their parameters.
2. Call windpower_action(action="start") to open the service window.
3. In ONE queue_tasks_parallel call, fire ALL of these simultaneously:
   - weatherForecast (get forecast data)
   - turbineStatus   (get turbine specs: max wind speed, etc.)
   - powerPlantStatus (get required power / energy deficit)
   Use whatever action names "help" reveals.
4. Call poll_results([...all taskIds from step 3...]) to retrieve all results at once.
5. Analyze results:
   - Find ALL forecast hours where wind > turbine's max safe wind speed → these are STORM hours.
     At storm hours: pitchAngle=90 (feathered), turbineMode="idle"
   - Find the BEST hour(s) with strong (but safe) wind to generate the required energy.
     At production hours: pitchAngle=0 (or optimal), turbineMode="production"
6. For EVERY config entry you need, call queue_tasks_parallel with unlockCodeGenerator actions
   (one per config entry), passing the datetime + pitchAngle + turbineMode for each.
   Fire them ALL in parallel in a single call.
7. poll_results([...unlock code taskIds...]) to get all unlock codes.
8. Call windpower_action(action="config", extra={ configs: { "YYYY-MM-DD HH:00:00": { pitchAngle, turbineMode, unlockCode }, ... } })
   to submit all configurations in one request.
9. Call windpower_action(action="turbinecheck") and wait for / poll its result.
10. Call windpower_action(action="done") to finalize. Report the flag from the response.

## Rules
- Datetime keys are always "YYYY-MM-DD HH:00:00" (minutes and seconds = 00).
- Storm = wind speed ABOVE the turbine's rated maximum safe wind speed.
- At storm: pitchAngle=90, turbineMode="idle" (no resistance, no production).
- At production: pitchAngle=0 (or whatever maximizes output), turbineMode="production".
- Every config entry MUST have a valid unlockCode from unlockCodeGenerator.
- NEVER call async actions sequentially — always batch them with queue_tasks_parallel.
- If any API call returns an error or unexpected response, log it and retry or adjust.
- When you have the final flag, print it clearly.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

async function solveTask() {
  log.info('=== Windpower task starting ===\n');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content:
        'Start solving the windpower task now. Remember: you have only 40 seconds total. ' +
        'Begin with help, then start, then queue ALL async data-fetching tasks in parallel immediately.',
    },
  ];

  let iteration = 0;
  const MAX = 80;

  while (iteration++ < MAX) {
    log.info(`\n─── Iteration ${iteration} ───`);

    const response = await openrouter.chat.completions.create({
      model: MODEL_CLAUDE_SONNET,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.content) {
      log.info(`Agent reasoning: ${msg.content.slice(0, 800)}`);
    }

    if (!msg.tool_calls?.length) {
      log.info('[Agent] No tool calls — task complete (or stuck). Final message above.');
      break;
    }

    for (const call of msg.tool_calls) {
      if (call.type !== 'function') continue;

      const { name, arguments: argsRaw } = call.function;
      const args = JSON.parse(argsRaw);
      log.info(`\n[Tool] ${name}(${argsRaw.slice(0, 300)})`);

      let result: string;

      try {
        if (name === 'windpower_action') {
          result = await windpowerAction(args.action, args.extra ?? {});
        } else if (name === 'queue_tasks_parallel') {
          result = await queueTasksParallel(args.actions);
        } else if (name === 'poll_results') {
          result = await pollResults(args.taskIds);
        } else {
          result = `Unknown tool: ${name}`;
        }
      } catch (err: unknown) {
        result = `Tool error: ${String(err)}`;
        log.error(`[Tool error] ${name}: ${err}`);
      }

      log.info(`[Tool result preview] ${result.slice(0, 500)}`);
      messages.push({ role: 'tool', tool_call_id: call.id, content: result });
    }
  }

  if (iteration >= MAX) {
    log.error(`\nFailed to reach goal within ${MAX} iterations.`);
  }
}

solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
