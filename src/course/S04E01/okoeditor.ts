import axios from 'axios';
import * as cheerio from 'cheerio';
import 'dotenv/config';
import OpenAI from 'openai';
import { log, MODEL_DEEPSEEK, MODEL_GPT_5_MINI, openrouter } from 'src/shared/agents';
import { VERIFY_URL } from 'src/shared/api';

// ─── Config ───────────────────────────────────────────────────────────────────
const EXECUTOR_MODEL = MODEL_DEEPSEEK;
const READER_MODEL = MODEL_GPT_5_MINI;
const MAX_ITERATIONS = 40;

// ─── Types ────────────────────────────────────────────────────────────────────
interface ExtractedLink {
  path: string; // e.g. "/incydenty/380792b2c86d9c5be670b3bde48e187b"
  id: string; // e.g. "380792b2c86d9c5be670b3bde48e187b"
  text: string; // visible row text
}

// ─── Reader Agent ─────────────────────────────────────────────────────────────
class ReaderAgent {
  private cookies = '';
  private messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    {
      role: 'system',
      content: `You are the "eyes" of an autonomous agent working inside the OKO surveillance
web panel. You receive raw HTML of panel pages and answer precise questions about
their content: IDs, field values, classification labels, task statuses, incident
entries, etc.
Rules:
- Extract exact values (IDs, strings) as they appear in the HTML.
- Never guess – if something is not visible, say so.
- Be terse: one focused answer per question, no preamble.`,
    },
  ];

  // ── Login ──────────────────────────────────────────────────────────────────
  // Fixes vs original:
  //   1. access_key must be AIDEVS_API_KEY — task says "Klucz: Twój apikey"
  //   2. maxRedirects=0 to capture Set-Cookie from the 302 before following it
  //   3. Success detected by body content, not HTTP status (200 can still be login page)
  async login(): Promise<void> {
    // Step 1 – GET root to obtain pre-session cookie
    const baseUrlInit = process.env.OKO_API!.replace(/\/+$/, '');
    const initResp = await axios.get(`${baseUrlInit}/`, {
      validateStatus: () => true,
    });
    this.mergeCookies(initResp.headers['set-cookie']);
    log.info('[Reader] Initial GET done, cookies:', this.cookies);

    // Step 2 – POST credentials; do NOT auto-follow redirects
    log.info('[Reader] Logging in as:', process.env.OKO_LOGIN);
    const baseUrl = process.env.OKO_API!.replace(/\/+$/, ''); // strip trailing slash
    const loginResp = await axios.post(
      `${baseUrl}/`,
      new URLSearchParams({
        action: 'login', // FIX: hidden form field required by server
        login: process.env.OKO_LOGIN!,
        password: process.env.OKO_PASSWORD!,
        access_key: process.env.AIDEVS_API_KEY!,
      }).toString(),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Cookie: this.cookies,
        },
        maxRedirects: 0,
        validateStatus: () => true,
      },
    );
    this.mergeCookies(loginResp.headers['set-cookie']);
    log.info(`[Reader] POST login status: ${loginResp.status}, cookies:`, this.cookies);

    // A body that still has the login form = failed login (even if status is 200)
    const isLoginPage = (body: string) =>
      body.includes('Logowanie operatora') || body.includes('name="action" value="login"');

    // Step 3 – follow redirect if present
    if (loginResp.status >= 300 && loginResp.status < 400 && loginResp.headers.location) {
      const location = loginResp.headers.location;
      const redirectUrl = location.startsWith('http')
        ? location
        : `${baseUrl}${location}`;

      log.info(`[Reader] Following redirect → ${redirectUrl}`);
      const afterRedirect = await axios.get(redirectUrl, {
        headers: { Cookie: this.cookies },
        validateStatus: () => true,
      });
      this.mergeCookies(afterRedirect.headers['set-cookie']);

      const body = typeof afterRedirect.data === 'string' ? afterRedirect.data : '';
      if (isLoginPage(body)) {
        log.error('[Reader] Login FAILED – still seeing login form after redirect');
        log.error('[Reader] Body snippet:', body.slice(0, 400));
        throw new Error('Login failed – check OKO_LOGIN, OKO_PASSWORD, AIDEVS_API_KEY');
      }
      log.info('[Reader] Logged in successfully (post-redirect)');
    } else {
      // 200 with login page body = silent failure (wrong credentials)
      const body = typeof loginResp.data === 'string' ? loginResp.data : '';
      if (isLoginPage(body)) {
        log.error('[Reader] Login FAILED – server returned login page (wrong credentials?)');
        log.error('[Reader] Body snippet:', body.slice(0, 400));
        throw new Error('Login failed – check OKO_LOGIN, OKO_PASSWORD, AIDEVS_API_KEY');
      }
      log.info('[Reader] Logged in successfully (200 with dashboard)');
    }
  }

  // ── Fetch raw HTML for any panel path ─────────────────────────────────────
  async fetchPage(path = '/'): Promise<string> {
    const base = process.env.OKO_API!.replace(/\/+$/, '');
    const resp = await axios.get(`${base}${path}`, {
      headers: { Cookie: this.cookies },
      validateStatus: () => true,
    });
    this.mergeCookies(resp.headers['set-cookie']);
    const html = typeof resp.data === 'string' ? resp.data : JSON.stringify(resp.data);
    log.info(`[Reader] fetchPage(${path}) → ${resp.status}, ${html.length} chars`);
    return html;
  }

  // ── Deterministic ID extraction ────────────────────────────────────────────
  // Uses cheerio to find every <a href="/section/<hex-id>"> link.
  // IDs are 32-char hex strings — regex was \d+ before (matched nothing).
  async extractLinks(path: string): Promise<ExtractedLink[]> {
    const html = await this.fetchPage(path);
    const $ = cheerio.load(html);
    const seen = new Set<string>();
    const items: ExtractedLink[] = [];

    $('a[href]').each((_, el) => {
      const href = $(el).attr('href') ?? '';
      // FIX: IDs are hex, not numeric — use [0-9a-f]+ instead of \d+
      const match = href.match(/^\/(incydenty|zadania|notatki)\/([0-9a-f]{32,})/i);
      if (!match || seen.has(href)) return;
      seen.add(href);

      const rowText = $(el)
        .closest('tr, li, div.item, article, .row, .entry-link')
        .text()
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 300);

      items.push({ path: href, id: match[2], text: rowText });
    });

    log.info(`[Reader] extractLinks(${path}) → ${items.length} items`);
    if (items.length === 0) {
      log.error('[Reader] WARNING: no links found – check login or page structure');
    }
    return items;
  }

  // ── LLM-based question answering ───────────────────────────────────────────
  async ask(question: string, paths: string[] = ['/'], refresh = false): Promise<string> {
    if (refresh || this.messages.length === 1) {
      const sections: string[] = [];
      for (const path of paths) {
        const html = await this.fetchPage(path);
        const bodyMatch = html.match(/<body[\s\S]*$/i);
        const body = bodyMatch ? bodyMatch[0] : html;
        sections.push(`=== PAGE: ${path} ===\n${body.slice(0, 40_000)}`);
      }
      this.messages.push({
        role: 'user',
        content: `[PAGE REFRESH]\n\n${sections.join('\n\n')}\n\n---\nQuestion: ${question}`,
      });
    } else {
      this.messages.push({ role: 'user', content: question });
    }

    const resp = await openrouter.chat.completions.create({
      model: READER_MODEL,
      messages: this.messages,
      max_tokens: 1000,
    });

    const answer = resp.choices[0].message.content ?? '';
    this.messages.push({ role: 'assistant', content: answer });
    log.info(`[Reader] ask → ${answer.slice(0, 400)}`);
    return answer;
  }

  // ── Cookie jar helper ──────────────────────────────────────────────────────
  private mergeCookies(raw?: string[]): void {
    if (!raw?.length) return;
    const jar = new Map<string, string>();
    for (const pair of this.cookies.split('; ').filter(Boolean)) {
      const i = pair.indexOf('=');
      if (i > 0) jar.set(pair.slice(0, i), pair.slice(i + 1));
    }
    for (const header of raw) {
      const kv = header.split(';')[0];
      const i = kv.indexOf('=');
      if (i > 0) jar.set(kv.slice(0, i).trim(), kv.slice(i + 1).trim());
    }
    this.cookies = [...jar.entries()].map(([k, v]) => `${k}=${v}`).join('; ');
  }
}

// ─── Singleton reader ─────────────────────────────────────────────────────────
const reader = new ReaderAgent();

// ─── Executor tools ───────────────────────────────────────────────────────────
async function callOkoApi(action: string, params: Record<string, unknown> = {}): Promise<string> {
  const payload = {
    apikey: process.env.AIDEVS_API_KEY,
    task: 'okoeditor',
    answer: { action, ...params },
  };
  log.info(`[Executor→API] action=${action}`, params);
  try {
    const resp = await axios.post(VERIFY_URL, payload);
    log.info(`[Executor←API]`, resp.data);
    return JSON.stringify(resp.data);
  } catch (err: any) {
    const detail = JSON.stringify(err.response?.data ?? err.message);
    log.error(`[Executor←API] ERROR: ${detail}`);
    return `API_ERROR: ${detail}`;
  }
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'oko_api',
      description:
        'Call the OKO editor API at /verify. ' +
        'Always call action="help" first to discover exact action names and parameters.',
      parameters: {
        type: 'object',
        properties: {
          action: { type: 'string', description: 'API action: help | update | done' },
          params: { type: 'object', description: 'Key-value params for the action' },
        },
        required: ['action'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'extract_links',
      description:
        'Deterministically scrape all 32-char hex IDs from a panel list page. ' +
        'Returns JSON array [{path, id, text}]. ' +
        'Use this FIRST on "/" (incidents), "/zadania" (tasks), "/notatki" (notes) ' +
        'to discover IDs before calling oko_api. Always use a leading slash.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'e.g. "/" or "/zadania" or "/notatki"' },
        },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_panel',
      description:
        'Ask the Reader LLM a semantic question about a panel page. ' +
        'Use for nuanced questions (e.g. current classification label on a detail page). ' +
        'Do NOT use for ID discovery — use extract_links instead. ' +
        'Always pass refresh=true after any mutation.',
      parameters: {
        type: 'object',
        properties: {
          question: { type: 'string' },
          paths: { type: 'array', items: { type: 'string' } },
          refresh: { type: 'boolean', description: 'Fetch fresh HTML first' },
        },
        required: ['question'],
      },
    },
  },
];

// ─── System prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an autonomous executor for the OKO surveillance system.
You are BLIND – all perception comes from two reader tools plus one API tool.

╔══════════════════════════════════════════════════════════════╗
║  STEP 0 – ALWAYS DO THIS BEFORE ANYTHING ELSE               ║
║  1. oko_api(action="help")    ← learn exact param names     ║
║  2. extract_links("/")        ← incidents list               ║
║  3. extract_links("/zadania") ← tasks list                   ║
╚══════════════════════════════════════════════════════════════╝

INCIDENT ENCODING REFERENCE (from /notatki):
  Codes are 6 chars: 4-letter type + 2-digit subtype.
  Codes MUST appear at the START of the incident title.

  RECO – rekonesans terenu
    01 znaleziono broń, 02 prowiant, 03 pojazd, 04 inne

  PROB – badanie zdobytej próbki
    01 próbka radiowa, 02 ruch internetowy, 03 fizyczny nośnik

  MOVE – wykryto ruch
    01 człowiek, 02 pojazd, 03 pojazd+człowiek, 04 zwierzęta

CRITICAL: Animals = MOVE04.  Humans = MOVE01.  Do NOT use MOVE00 or RECO01 for these.

══════════════════════════════════════════════════════════════
MISSION – complete ALL four steps in order
══════════════════════════════════════════════════════════════

STEP 1 – Fix Skolwin incident classification to ANIMALS
  a. extract_links("/") → find entry whose text contains "Skolwin" → note its hex id
  b. The Skolwin incident is currently classified as vehicle+human movement.
     Change it to animals. The correct code is MOVE04.
     oko_api(action="update", page="incydenty", id=<skolwin-hex-id>,
             title="MOVE04 <keep rest of original title or use a descriptive title with Skolwin>")
     IMPORTANT: title MUST start with "MOVE04" and MUST contain "Skolwin".
  c. Verify with extract_links("/") that the Skolwin entry now shows MOVE04.

STEP 2 – Complete Skolwin TASK
  a. extract_links("/zadania") → find entry whose text contains "Skolwin" → note its hex id
  b. oko_api(action="update", page="zadania", id=<skolwin-task-hex-id>,
             content="Widziano bobry w okolicach Skolwina", done="YES")
     Do NOT change the task title.
  c. Verify with read_panel(paths=["/zadania/<hex-id>"], refresh=true,
             question="Is the task marked done? What is the content?")

STEP 3 – Repurpose an existing incident for Komarowo (human movement)
  The API does NOT support creating new entries — only updating existing ones.
  Strategy: pick an existing incident whose ID is NOT the Skolwin one and overwrite it.
  a. From the extract_links("/") results, pick any incident ID that is NOT the Skolwin ID.
  b. The correct code for human movement is MOVE01.
     oko_api(action="update", page="incydenty", id=<other-hex-id>,
             title="MOVE01 Wykrycie ruchu ludzi w okolicach Komarowo",
             content="Wykryto ruch ludzi w okolicach miasta Komarowo.")
     IMPORTANT: title MUST start with "MOVE01" and MUST contain "Komarowo".
  c. Verify: extract_links("/") → confirm entry with "Komarowo" and "MOVE01" now appears

STEP 4 – Signal completion
  oko_api(action="done") → read flag from response and report it.

══════════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════════
- Use exact action/param names from help. Do not guess.
- After EVERY oko_api mutation: verify with extract_links or read_panel(refresh=true).
- If extract_links returns [] recheck you used a leading slash and are logged in.
- When the response contains {FLG:...}, report it immediately.
- NEVER use codes MOVE00 or RECO01 for animals/humans. Use MOVE04 and MOVE01 respectively.
`;

// ─── Main agentic loop ────────────────────────────────────────────────────────
async function solveTask(): Promise<void> {
  log.info('=== okoeditor task started ===');
  await reader.login();

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        'Start the mission. ' +
        'First call oko_api(action="help") to learn the API, ' +
        'then extract_links on all three list pages, ' +
        'then complete all four mission steps in order.',
    },
  ];

  let iterations = 0;

  while (iterations < MAX_ITERATIONS) {
    log.info(`\n─── Iteration ${++iterations} / ${MAX_ITERATIONS} ───`);

    const resp = await openrouter.chat.completions.create({
      model: EXECUTOR_MODEL,
      messages,
      tools,
      tool_choice: 'auto',
    });

    const choice = resp.choices[0];
    const msg = choice.message;
    messages.push(msg);

    if (msg.content) {
      log.info(`[Executor] ${msg.content.slice(0, 600)}`);
      const flag = msg.content.match(/\{FLG:[^}]+\}/);
      if (flag) {
        log.result('🚩 FLAG FOUND', flag[0]);
        return;
      }
    }

    if (!msg.tool_calls?.length) {
      log.info('[Executor] No tool calls – loop ended.');
      break;
    }

    for (const toolCall of msg.tool_calls) {
      if (toolCall.type !== 'function') continue;
      const { name, arguments: argsJson } = toolCall.function;
      log.info(`[Tool call] ${name}(${argsJson.slice(0, 400)})`);

      let result: string;
      try {
        const args = JSON.parse(argsJson) as Record<string, unknown>;

        if (name === 'oko_api') {
          result = await callOkoApi(
            args.action as string,
            (args.params as Record<string, unknown>) ?? {},
          );
        } else if (name === 'extract_links') {
          const items = await reader.extractLinks(args.path as string);
          result = JSON.stringify(items, null, 2);
        } else if (name === 'read_panel') {
          result = await reader.ask(
            args.question as string,
            (args.paths as string[]) ?? ['/'],
            (args.refresh as boolean) ?? false,
          );
        } else {
          result = `Unknown tool: ${name}`;
        }
      } catch (err: any) {
        result = `ERROR: ${err.message}`;
        log.error(`[Tool error] ${result}`);
      }

      log.info(`[Tool result] ${result.slice(0, 500)}`);
      const flag = result.match(/\{FLG:[^}]+\}/);
      if (flag) log.result('🚩 FLAG IN TOOL RESULT', flag[0]);

      messages.push({ role: 'tool', tool_call_id: toolCall.id, content: result });
    }
  }

  if (iterations >= MAX_ITERATIONS) log.error('Max iterations reached without completion.');
}

// ─── Entry point ──────────────────────────────────────────────────────────────
solveTask().catch((err) => {
  log.error('Fatal error:', err);
  process.exit(1);
});
