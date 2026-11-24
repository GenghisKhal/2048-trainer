'use strict';
// N-tuple network value function + TD(0) afterstate learning (Szubert & Jaskowski 2014,
// with backward-replayed updates as in Yeh et al.). Four 6-cell patterns, each applied under
// all 8 board symmetries, so 32 lookups per evaluation into 4 tables of 16^6 weights.
const { move, addRandomTile, newBoard, maxRank } = require('./game');

const PATTERNS = [
  [0, 1, 2, 3, 4, 5],
  [4, 5, 6, 7, 8, 9],
  [0, 1, 2, 4, 5, 6],
  [4, 5, 6, 8, 9, 10],
];
const TUPLE = 6;
const TABLE_SIZE = 1 << (4 * TUPLE); // 16^6
const NUM_WEIGHTS = PATTERNS.length * TABLE_SIZE;

// The 8 symmetries of the square, as cell-index maps.
function symmetries() {
  const rot = (i) => (i % 4) * 4 + (3 - Math.floor(i / 4));
  const flip = (i) => Math.floor(i / 4) * 4 + (3 - (i % 4));
  const maps = [];
  let m = [...Array(16).keys()];
  for (let r = 0; r < 4; r++) {
    maps.push(m);
    maps.push(m.map(flip));
    m = m.map(rot);
  }
  return maps;
}

// Flat feature table: for each of the 32 features, its table offset and 6 cell indices.
const NUM_FEATURES = PATTERNS.length * 8;
const FEAT_CELLS = new Int32Array(NUM_FEATURES * TUPLE);
const FEAT_BASE = new Int32Array(NUM_FEATURES);
{
  const syms = symmetries();
  let f = 0;
  PATTERNS.forEach((p, pi) => {
    for (const s of syms) {
      FEAT_BASE[f] = pi * TABLE_SIZE;
      for (let k = 0; k < TUPLE; k++) FEAT_CELLS[f * TUPLE + k] = s[p[k]];
      f++;
    }
  });
}

function featureIndex(b, f) {
  const o = f * TUPLE;
  return (
    FEAT_BASE[f] +
    (b[FEAT_CELLS[o]] |
      (b[FEAT_CELLS[o + 1]] << 4) |
      (b[FEAT_CELLS[o + 2]] << 8) |
      (b[FEAT_CELLS[o + 3]] << 12) |
      (b[FEAT_CELLS[o + 4]] << 16) |
      (b[FEAT_CELLS[o + 5]] << 20))
  );
}

function value(w, b) {
  let v = 0;
  for (let f = 0; f < NUM_FEATURES; f++) v += w[featureIndex(b, f)];
  return v;
}

// Adds delta (spread over all features) and returns the updated value.
function update(w, b, delta) {
  const d = delta / NUM_FEATURES;
  let v = 0;
  for (let f = 0; f < NUM_FEATURES; f++) {
    const i = featureIndex(b, f);
    w[i] += d;
    v += w[i];
  }
  return v;
}

// Greedy 1-ply choice: argmax over moves of reward + V(afterstate).
// Returns { dir, reward, after, values } where values[d] is null for illegal moves.
const scratch = new Uint8Array(16);
function chooseMove(w, board, wantValues = false) {
  let best = -1, bestVal = -Infinity, bestReward = 0;
  const values = wantValues ? [null, null, null, null] : null;
  for (let d = 0; d < 4; d++) {
    scratch.set(board);
    const r = move(scratch, d);
    if (r < 0) continue;
    const v = r + value(w, scratch);
    if (values) values[d] = v;
    if (v > bestVal) { bestVal = v; best = d; bestReward = r; }
  }
  return { dir: best, reward: bestReward, values };
}

class Learner {
  constructor(weights) {
    this.w = weights;
    this.afters = new Uint8Array(16 * 4096);
    this.rewards = new Int32Array(4096);
  }

  // Plays one full game with the greedy policy, then replays TD updates backward over it.
  playEpisode(alpha) {
    const w = this.w;
    const board = newBoard();
    let n = 0, score = 0;
    for (;;) {
      const { dir, reward } = chooseMove(w, board);
      if (dir < 0) break;
      move(board, dir);
      score += reward;
      if (n >= this.rewards.length) this.grow();
      this.afters.set(board, n * 16);
      this.rewards[n] = reward;
      n++;
      addRandomTile(board);
    }
    let target = 0;
    for (let t = n - 1; t >= 0; t--) {
      const after = this.afters.subarray(t * 16, t * 16 + 16);
      const err = target - value(w, after);
      target = this.rewards[t] + update(w, after, alpha * err);
    }
    return { score, maxRank: maxRank(board), moves: n };
  }

  grow() {
    const a = new Uint8Array(this.afters.length * 2); a.set(this.afters); this.afters = a;
    const r = new Int32Array(this.rewards.length * 2); r.set(this.rewards); this.rewards = r;
  }
}

module.exports = { NUM_WEIGHTS, value, chooseMove, Learner };
