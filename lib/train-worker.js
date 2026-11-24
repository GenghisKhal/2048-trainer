'use strict';
// Training worker: plays games and updates the shared weights (lock-free, Hogwild-style).
const { parentPort, workerData } = require('worker_threads');
const { Learner } = require('./ntuple');

const weights = new Float32Array(workerData.sab);
const learner = new Learner(weights);
let alpha = workerData.alpha;
let paused = workerData.paused;
let batch = [];
let lastFlush = Date.now();

parentPort.on('message', (m) => {
  if (m.alpha !== undefined) alpha = m.alpha;
  if (m.paused !== undefined) paused = m.paused;
});

function flush() {
  if (batch.length) parentPort.postMessage(batch);
  batch = [];
  lastFlush = Date.now();
}

function loop() {
  if (paused) { flush(); setTimeout(loop, 200); return; }
  const end = Date.now() + 100; // yield to the message queue every ~100ms
  while (Date.now() < end) {
    const r = learner.playEpisode(alpha);
    batch.push(r.score, r.maxRank, r.moves);
  }
  if (Date.now() - lastFlush > 400) flush();
  setImmediate(loop);
}
loop();
