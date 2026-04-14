/**
 * AI 辩论服务器 — 零依赖版本（仅使用 Node.js 内建模块）
 * 启动方式: node server.js
 * 需要在同目录创建 .env 文件，写入 ANTHROPIC_API_KEY=sk-ant-...
 */
'use strict';

const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');
const url   = require('url');

// ── 读取 .env ────────────────────────────────
function loadEnv() {
  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}
loadEnv();

// ── 模型列表 ─────────────────────────────────
const MODELS = [
  { id: 'claude-opus-4-6',           label: 'Claude Opus 4.6',   provider: 'anthropic' },
  { id: 'claude-sonnet-4-6',         label: 'Claude Sonnet 4.6', provider: 'anthropic' },
  { id: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5',  provider: 'anthropic' },
  { id: 'gpt-4o',                    label: 'GPT-4o',            provider: 'openai'    },
  { id: 'gpt-4o-mini',               label: 'GPT-4o mini',       provider: 'openai'    },
];

// ── MIME 类型 ────────────────────────────────
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.ico':  'image/x-icon',
};

// ── 辅助：HTTPS 请求（返回 IncomingMessage 流）─
function httpsRequest(options, body) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, resolve);
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── SSE 工具 ─────────────────────────────────
function sseWrite(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// ── 辩论 System Prompt ───────────────────────
function buildSystem(side, topic) {
  const role   = side === 'pro' ? '甲方（支持方）' : '乙方（反对方）';
  const stance = side === 'pro'
    ? `你必须坚定支持以下观点：「${topic}」。`
    : `你必须坚定反对以下观点：「${topic}」。`;
  return `你是一场正式辩论中的 ${role}。
${stance}

规则：
1. 每次发言不超过 200 字，言简意赅、逻辑清晰。
2. 直接进行辩论，不要自我介绍，不要说"作为 AI"之类的话。
3. 针对对方上一轮的论点进行回应，并提出新论据加以强化。
4. 保持礼貌，但立场坚定、不动摇。
5. 使用中文回复。`;
}

// ── Anthropic 流式调用 ───────────────────────
async function* streamAnthropic({ modelId, systemPrompt, messages }) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY 未设置，请检查 .env 文件');

  const body = JSON.stringify({
    model: modelId,
    max_tokens: 600,
    stream: true,
    system: systemPrompt,
    messages,
  });

  const resp = await httpsRequest({
    hostname: 'api.anthropic.com',
    path: '/v1/messages',
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(body),
    },
  }, body);

  if (resp.statusCode !== 200) {
    let errBody = '';
    for await (const chunk of resp) errBody += chunk;
    let msg = `Anthropic API 错误 ${resp.statusCode}`;
    try { msg = JSON.parse(errBody).error?.message ?? msg; } catch {}
    throw new Error(msg);
  }

  let buf = '';
  for await (const chunk of resp) {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    let eventType = null;
    for (const line of lines) {
      if (line.startsWith('event: '))      { eventType = line.slice(7).trim(); }
      else if (line.startsWith('data: ')) {
        try {
          const d = JSON.parse(line.slice(6));
          if (eventType === 'content_block_delta' && d.delta?.type === 'text_delta') {
            yield d.delta.text;
          }
        } catch {}
      }
    }
  }
}

// ── OpenAI 流式调用 ──────────────────────────
async function* streamOpenAI({ modelId, systemPrompt, messages }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY 未设置，请检查 .env 文件');

  const oaiMessages = [{ role: 'system', content: systemPrompt }, ...messages];
  const body = JSON.stringify({
    model: modelId,
    max_tokens: 600,
    stream: true,
    messages: oaiMessages,
  });

  const resp = await httpsRequest({
    hostname: 'api.openai.com',
    path: '/v1/chat/completions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'content-length': Buffer.byteLength(body),
    },
  }, body);

  if (resp.statusCode !== 200) {
    let errBody = '';
    for await (const chunk of resp) errBody += chunk;
    let msg = `OpenAI API 错误 ${resp.statusCode}`;
    try { msg = JSON.parse(errBody).error?.message ?? msg; } catch {}
    throw new Error(msg);
  }

  let buf = '';
  for await (const chunk of resp) {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') continue;
      try {
        const d   = JSON.parse(raw);
        const tok = d.choices?.[0]?.delta?.content ?? '';
        if (tok) yield tok;
      } catch {}
    }
  }
}

// ── 路由辩论请求 ─────────────────────────────
async function handleDebate(req, res) {
  let body = '';
  for await (const chunk of req) body += chunk;
  const { topic, proModel, conModel, rounds = 3 } = JSON.parse(body);

  if (!topic || !proModel || !conModel) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ error: 'topic, proModel, conModel 必填' }));
  }

  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
  });

  const getStream = (modelId, systemPrompt, messages) => {
    const info = MODELS.find(m => m.id === modelId);
    if (!info) throw new Error(`未知模型: ${modelId}`);
    return info.provider === 'anthropic'
      ? streamAnthropic({ modelId, systemPrompt, messages })
      : streamOpenAI({ modelId, systemPrompt, messages });
  };

  const proHistory = [];
  const conHistory = [];
  const proSystem  = buildSystem('pro', topic);
  const conSystem  = buildSystem('con', topic);

  try {
    sseWrite(res, 'start', { topic, proModel, conModel, rounds });

    for (let r = 1; r <= rounds; r++) {
      // ── PRO ──
      sseWrite(res, 'turn_start', { side: 'pro', round: r });
      let proText = '';
      for await (const tok of getStream(proModel, proSystem, proHistory)) {
        proText += tok;
        sseWrite(res, 'token', { side: 'pro', round: r, token: tok });
      }
      sseWrite(res, 'turn_end', { side: 'pro', round: r, text: proText });
      proHistory.push({ role: 'assistant', content: proText });
      conHistory.push({ role: 'user',      content: `【甲方】${proText}` });

      // ── CON ──
      sseWrite(res, 'turn_start', { side: 'con', round: r });
      let conText = '';
      for await (const tok of getStream(conModel, conSystem, conHistory)) {
        conText += tok;
        sseWrite(res, 'token', { side: 'con', round: r, token: tok });
      }
      sseWrite(res, 'turn_end', { side: 'con', round: r, text: conText });
      conHistory.push({ role: 'assistant', content: conText });
      proHistory.push({ role: 'user',      content: `【乙方】${conText}` });
    }

    sseWrite(res, 'done', { message: '辩论结束' });
  } catch (err) {
    sseWrite(res, 'error', { message: err.message });
  } finally {
    res.end();
  }
}

// ── HTTP 服务器 ──────────────────────────────
const server = http.createServer(async (req, res) => {
  const { pathname } = url.parse(req.url);

  // CORS（方便本地开发）
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  // API 路由
  if (pathname === '/api/models' && req.method === 'GET') {
    const list = MODELS.map(m => ({
      ...m,
      available: m.provider === 'anthropic'
        ? !!process.env.ANTHROPIC_API_KEY
        : !!process.env.OPENAI_API_KEY,
    }));
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(list));
  }

  if (pathname === '/api/debate' && req.method === 'POST') {
    return handleDebate(req, res);
  }

  // 静态文件
  const filePath = path.join(__dirname, pathname === '/' ? 'index.html' : pathname);
  const ext = path.extname(filePath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      return res.end('Not found');
    }
    res.writeHead(200, { 'Content-Type': MIME[ext] ?? 'application/octet-stream' });
    res.end(data);
  });
});

const PORT = process.env.PORT ?? 3000;
server.listen(PORT, () => {
  console.log(`\n  AI 辩论场已启动 → http://localhost:${PORT}\n`);
});
