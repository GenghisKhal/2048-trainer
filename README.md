# 2048 Trainer

Teaches the computer to play 2048 by playing against itself, with a live dashboard.

```
npm start            # then open http://localhost:2048
```

Optional settings: `WORKERS=8` (training threads, default 4), `PORT`, `ALPHA` (learning rate, default 0.1).

## How it learns

- **Value function:** an n-tuple network: 4 patterns of 6 cells, each read under all 8 rotations and
  reflections of the board, giving 32 table lookups into ~67M weights (256 MB).
- **Policy:** for each move, take the merge reward plus the network's value of the board that
  results, and play the best one (before the random tile appears).
- **Learning:** TD(0) on those post-move boards. After each game, the updates are replayed
  backward from the end, which spreads the reward faster. Several worker threads update the
  same shared weights without locks.
- Expect the agent to reach 2048 in ~85% of games after about 50k games (around a minute), with
  steady gains after that. Lowering the learning rate to 0.01–0.025 once the curves flatten
  usually raises the ceiling.

## Files

- `lib/game.js`: the 2048 rules (shared with the browser).
- `lib/ntuple.js`: the network, move choice, and TD learning.
- `lib/train-worker.js`: the training loop that runs in each thread.
- `server.js`: the worker threads, stats, checkpoints, and HTTP/SSE API.
- `public/index.html`: the dashboard.

Checkpoints go to `data/` every 10 minutes, when you press **Save checkpoint**, and on Ctrl+C.
Training resumes from the last checkpoint when you restart.
