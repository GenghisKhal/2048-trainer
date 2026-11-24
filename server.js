'use strict';
// Trains a 2048 agent in background worker threads and serves a live dashboard.
//   node server.js            -> http://localhost:2048
// Env: PORT (2048), WORKERS (4), ALPHA (0.1)
const http = require('http');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { NUM_WEIGHTS, chooseMove } = require('./lib/ntuple');

const PORT = +process.env.PORT || 2048;
const NUM_WORKERS = +process.env.WORKERS || 4;
const DATA_DIR = path.join(__dirname, 'data');
const WEIGHTS_FILE = path.join(DATA_DIR, 'weights.bin');
const STATS_FILE = path.join(DATA_DIR, 'stats.json');
const BUCKET = 1000; // games per history point
const RECENT = 1000; // rolling window for "current strength"
const SAVE_EVERY_MS = 10 * 60 * 1000;

// ---------- shared weights ----------
const sab = new SharedArrayBuffer(NUM_WEIGHTS * 4);
const weights = new Float32Array(sab);

// ---------- training state ----------
let state = freshState();
function freshState() {
  return {
    alpha: +process.env.ALPHA || 0.1,
    paused: false,
    totalGames: 0,
    totalMoves: 0,
    trainSeconds: 0,
    bestScore: 0,
    bestRank: 0,
    history: [], // {g, avg, max, moves, reach: [% reaching 512..32768]}
  };
}
let bucket = newBucket();
function newBucket() { return { n: 0, sum: 0, max: 0, moves: 0, ranks: new Array(16).fill(0) }; }
const recent = { score: new Float64Array(RECENT), rank: new Uint8Array(RECENT), i: 0, n: 0 };
const rateWindow = []; // [timestamp, totalGames]

function load() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  try {
    const buf = fs.readFileSync(WEIGHTS_FILE);
    if (buf.length === sab.byteLength) {
      new Uint8Array(sab).set(buf);
      console.log('Loaded weights from', WEIGHTS_FILE);
    }
    const saved = JSON.parse(fs.readFileSync(STATS_FILE, 'utf8'));
    state = { ...freshState(), ...saved, paused: false };
    console.log(`Resuming at ${state.totalGames.toLocaleString()} games`);
  } catch { console.log('Starting fresh (no saved weights)'); }
}

let saving = null;
async function save() {
  if (saving) return saving;
  saving = (async () => {
    const t = Date.now();
    await fs.promises.writeFile(WEIGHTS_FILE + '.tmp', new Uint8Array(sab));
    await fs.promises.rename(WEIGHTS_FILE + '.tmp', WEIGHTS_FILE);
    await fs.promises.writeFile(STATS_FILE, JSON.stringify(state));
    state.lastSaved = Date.now();
    console.log(`Saved checkpoint at ${state.totalGames.toLocaleString()} games (${Date.now() - t} ms)`);
  })().finally(() => { saving = null; });
  return saving;
}

function record(score, rank, moves) {
  state.totalGames++;
  state.totalMoves += moves;
  if (score > state.bestScore) state.bestScore = score;
  if (rank > state.bestRank) state.bestRank = rank;
  recent.score[recent.i] = score; recent.rank[recent.i] = rank;
  recent.i = (recent.i + 1) % RECENT; recent.n = Math.min(recent.n + 1, RECENT);
  bucket.n++; bucket.sum += score; bucket.moves += moves; bucket.ranks[rank]++;
  if (score > bucket.max) bucket.max = score;
  if (bucket.n === BUCKET) {
    const reach = [];
    for (let r = 9; r <= 15; r++) {
      let c = 0; for (let k = r; k < 16; k++) c += bucket.ranks[k];
      reach.push(+(100 * c / bucket.n).toFixed(1));
    }
    state.history.push({ g: state.totalGames, avg: Math.round(bucket.sum / bucket.n), max: bucket.max, moves: Math.round(bucket.moves / bucket.n), reach });
    bucket = newBucket();
  }
}

// ---------- workers ----------
load(); // before workers start writing to the shared weights
const workers = [];
for (let i = 0; i < NUM_WORKERS; i++) {
  const w = new Worker(path.join(__dirname, 'lib', 'train-worker.js'), { workerData: { sab, alpha: state.alpha, paused: state.paused } });
  w.on('message', (batch) => {
    if (resetting) return;
    for (let j = 0; j < batch.length; j += 3) record(batch[j], batch[j + 1], batch[j + 2]);
  });
  w.on('error', (e) => console.error('worker error', e));
  workers.push(w);
}
const broadcast = (m) => workers.forEach((w) => w.postMessage(m));
let resetting = false;

function summary() {
  const now = Date.now();
  rateWindow.push([now, state.totalGames]);
  while (rateWindow.length > 1 && now - rateWindow[0][0] > 5000) rateWindow.shift();
  const [t0, g0] = rateWindow[0];
  const rate = now > t0 ? (state.totalGames - g0) / ((now - t0) / 1000) : 0;
  const dist = new Array(16).fill(0);
  let sum = 0;
  for (let k = 0; k < recent.n; k++) { dist[recent.rank[k]]++; sum += recent.score[k]; }
  return {
    alpha: state.alpha, paused: state.paused, workers: NUM_WORKERS,
    totalGames: state.totalGames, totalMoves: state.totalMoves, trainSeconds: state.trainSeconds,
    bestScore: state.bestScore, bestRank: state.bestRank, lastSaved: state.lastSaved || null,
    rate, recentN: recent.n, recentAvg: recent.n ? Math.round(sum / recent.n) : 0, recentDist: dist,
  };
}

// ---------- http ----------
const clients = new Set();
setInterval(() => {
  if (!state.paused) state.trainSeconds += 1;
  const s = summary();
  for (const c of clients) {
    const fresh = state.history.slice(c.sent);
    c.sent = state.history.length;
    c.res.write(`data: ${JSON.stringify({ type: 'tick', summary: s, history: fresh })}\n\n`);
  }
}, 1000);
setInterval(() => { if (!state.paused) save(); }, SAVE_EVERY_MS);

function readBody(req) {
  return new Promise((resolve, reject) => {
    let d = '';
    req.on('data', (c) => { d += c; if (d.length > 1e5) req.destroy(); });
    req.on('end', () => { try { resolve(d ? JSON.parse(d) : {}); } catch (e) { reject(e); } });
  });
}
const json = (res, code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };

const STATIC = {
  '/': ['public/index.html', 'text/html; charset=utf-8'],
  '/game.js': ['lib/game.js', 'text/javascript; charset=utf-8'],
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (req.method === 'GET' && STATIC[url.pathname]) {
      const [file, type] = STATIC[url.pathname];
      res.writeHead(200, { 'Content-Type': type, 'Cache-Control': 'no-cache' });
      return res.end(fs.readFileSync(path.join(__dirname, file)));
    }
    if (req.method === 'GET' && url.pathname === '/api/stats') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      const c = { res, sent: state.history.length };
      res.write(`data: ${JSON.stringify({ type: 'init', summary: summary(), history: state.history })}\n\n`);
      clients.add(c);
      req.on('close', () => clients.delete(c));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/api/policy') {
      const { board } = await readBody(req);
      if (!Array.isArray(board) || board.length !== 16) return json(res, 400, { error: 'board must be 16 ranks' });
      const b = Uint8Array.from(board, (v) => Math.max(0, Math.min(15, v | 0)));
      const { dir, values } = chooseMove(weights, b, true);
      return json(res, 200, { dir, values });
    }
    if (req.method === 'POST' && url.pathname === '/api/control') {
      const body = await readBody(req);
      if (body.action === 'pause' || body.action === 'resume') {
        state.paused = body.action === 'pause';
        broadcast({ paused: state.paused });
      } else if (body.action === 'alpha') {
        const a = +body.value;
        if (!(a > 0 && a <= 1)) return json(res, 400, { error: 'alpha must be in (0, 1]' });
        state.alpha = a;
        broadcast({ alpha: a });
      } else if (body.action === 'save') {
        await save();
      } else if (body.action === 'reset') {
        resetting = true;
        broadcast({ paused: true });
        await new Promise((r) => setTimeout(r, 400));
        weights.fill(0);
        const alpha = state.alpha;
        state = freshState(); state.alpha = alpha;
        bucket = newBucket(); recent.i = recent.n = 0; rateWindow.length = 0;
        for (const c of clients) { c.sent = 0; c.res.write(`data: ${JSON.stringify({ type: 'init', summary: summary(), history: [] })}\n\n`); }
        resetting = false;
        broadcast({ paused: false });
      } else return json(res, 400, { error: 'unknown action' });
      return json(res, 200, summary());
    }
    res.writeHead(404); res.end('not found');
  } catch (e) {
    console.error(e);
    if (!res.headersSent) json(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, '127.0.0.1', () => console.log(`Dashboard: http://localhost:${PORT}  (${NUM_WORKERS} training workers)`));

let stopping = false;
process.on('SIGINT', async () => {
  if (stopping) process.exit(1);
  stopping = true;
  console.log('\nSaving before exit...');
  broadcast({ paused: true });
  await save();
  process.exit(0);
});
