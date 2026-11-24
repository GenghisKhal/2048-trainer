'use strict';
// 2048 engine. A board is a Uint8Array(16) of tile ranks (0 = empty, 1 = 2, 2 = 4, ... 15 = 32768),
// row-major, index = row * 4 + col.

const DIRS = ['up', 'right', 'down', 'left'];

// For each direction, 4 lines of 4 cell indices, ordered starting from the edge tiles slide toward.
const LINES = [
  [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]], // up
  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8], [15, 14, 13, 12]], // right
  [[12, 8, 4, 0], [13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3]], // down
  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], // left
];

// Precomputed slide-toward-index-0 for every 16-bit line (4 nibbles, cell 0 = lowest nibble).
const SLIDE = new Uint16Array(65536);
const SLIDE_SCORE = new Uint32Array(65536);
(function buildTables() {
  for (let key = 0; key < 65536; key++) {
    const cells = [key & 15, (key >> 4) & 15, (key >> 8) & 15, (key >> 12) & 15].filter((v) => v);
    const out = [];
    let score = 0;
    for (let i = 0; i < cells.length; i++) {
      if (i + 1 < cells.length && cells[i] === cells[i + 1] && cells[i] < 15) {
        out.push(cells[i] + 1);
        score += 1 << (cells[i] + 1);
        i++;
      } else out.push(cells[i]);
    }
    while (out.length < 4) out.push(0);
    SLIDE[key] = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12);
    SLIDE_SCORE[key] = score;
  }
})();

// Applies a move in place. Returns the merge reward, or -1 if the move changes nothing.
function move(board, dir) {
  const lines = LINES[dir];
  let reward = 0;
  let changed = false;
  for (let l = 0; l < 4; l++) {
    const [a, b, c, d] = lines[l];
    const key = board[a] | (board[b] << 4) | (board[c] << 8) | (board[d] << 12);
    const res = SLIDE[key];
    if (res !== key) {
      changed = true;
      board[a] = res & 15;
      board[b] = (res >> 4) & 15;
      board[c] = (res >> 8) & 15;
      board[d] = (res >> 12) & 15;
      reward += SLIDE_SCORE[key];
    }
  }
  return changed ? reward : -1;
}

// Places a 2 (90%) or 4 (10%) on a random empty cell. Returns the cell index, or -1 if full.
function addRandomTile(board, rng = Math.random) {
  let empty = 0;
  for (let i = 0; i < 16; i++) if (board[i] === 0) empty++;
  if (!empty) return -1;
  let pick = Math.floor(rng() * empty);
  for (let i = 0; i < 16; i++) {
    if (board[i] === 0 && pick-- === 0) {
      board[i] = rng() < 0.9 ? 1 : 2;
      return i;
    }
  }
  return -1;
}

function newBoard(rng = Math.random) {
  const b = new Uint8Array(16);
  addRandomTile(b, rng);
  addRandomTile(b, rng);
  return b;
}

function maxRank(board) {
  let m = 0;
  for (let i = 0; i < 16; i++) if (board[i] > m) m = board[i];
  return m;
}

const api = { DIRS, move, addRandomTile, newBoard, maxRank };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.Game2048 = api; // also loaded by the browser UI
