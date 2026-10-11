# GameMind

GameMind is a modular monolith for interactive games and reproducible search experiments. Python owns game rules, search, and neural inference. The React client uses the FastAPI API.

## Run locally (PowerShell)

From the repository root, start the backend in terminal 1:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Start the frontend in terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Set `VITE_API_URL` if the API is hosted somewhere other than `http://localhost:8000`.

## Tests and production build

Run backend and frontend checks from the repository root:

```powershell
cd backend
python -m pytest tests -q
cd ..\frontend
npm test -- --run
npm run build
```

## Reproduce the saved model artifacts

Run these commands from `backend`. Neural targets are realized outcomes from completed heuristic-guided self-play games, not exact game-theoretic values. Each target is `winner * side_to_move`: `+1` for a realized win, `-1` for a realized loss, and `0` for a draw. The generator uses seed 21, 100 games, and a 0.7 greedy / 0.3 random move policy. Exact and horizontally mirrored encodings are deduplicated globally before whole-game train/validation/test splits. This generated dataset is produced in memory; no generated-dataset file is required.

```powershell
cd backend
python -m app.training.train_genetic
python -m app.training.train_connect4
```

These commands write `models/genetic_weights.json`, `models/genetic_experiment.json`, `models/connect4.pt`, and `models/connect4.json`. Training overwrites the corresponding saved artifact files. The GA fitness is a one-ply successor-score proxy against fixed profiles; its separate small held-out tournament is not a measure of general tournament strength.

The included neural checkpoint was not retrained during the final audit. Its saved metadata records 100 generated games, 1,545 raw positions, 565 exact/mirror duplicates removed, and 980 retained positions. Eleven source games had no retained positions; the retained game groups split 62/13/14, with 682/129/169 positions. The checkpoint's train/validation/test MSE is 0.6397/1.0152/0.9794. Against the training-mean predictor, validation MSE is 0.9947 and test MSE is 0.9983; the handcrafted sign baseline has test MSE 0.9704. These metrics do not establish robust generalization.

## Experiments

Run from `backend`. `matches` is the number of games per evaluator pairing. Evaluators alternate the first seat in paired games using the same opening; a seeded shuffled order cycles through five legal opening lines. Node and time budgets are separate per-move limits and must not be supplied together. Time limits are soft and can be exceeded while search finishes a node. CSV files contain raw matches; JSON summaries contain aggregates, configuration, measurements, limitations, and artifact fingerprints. Aggregate outcomes are ordered wins-a / draws / wins-b.

Example equal-node experiment:

```powershell
cd backend
python -c "from app.experiments.tournament import run_tournament; print(run_tournament(matches=30, depth=6, budget_nodes=100, seed=20261027))"
```

Example equal-time experiment (run separately from node-budget experiments):

```powershell
cd backend
python -c "from app.experiments.tournament import run_tournament; print(run_tournament(matches=10, depth=6, time_limit_ms=50, seed=20261028))"
```

Both write `matches-*.csv` and `summary-*.json` under `backend/data/`. Finite deterministic samples, repeated openings, and evaluator/opponent choices limit any strength conclusions; report raw counts and avoid treating these experiments as universal rankings.

The final audit also ran a 30-game-per-pair equal-node sample at 20 nodes per move (depth cap 6, seed 20261027). In wins-a / draws / wins-b order, the results were handcrafted vs. genetic 12/0/18, handcrafted vs. neural 30/0/0, and genetic vs. neural 30/0/0. Each pairing alternated seats 15/15 and used five seeded opening lines, each repeated for three paired seat swaps. The raw matches and artifact-fingerprinted summary are `backend/data/matches-20261011T191818130789Z.csv` and `backend/data/summary-20261011T191818130789Z.json`. This small-node deterministic sample is descriptive only and does not establish broad tournament strength.

## Game and search behavior

Connect Four uses a 6×7 gravity board with horizontal, vertical, and diagonal win detection, legal move validation, draw detection, and winning-sequence highlighting. Tic-Tac-Toe uses depth-9 Minimax for optimal play. Nim uses normal play (taking the last object wins), configurable pile sizes, an XOR optimal strategy, and a Minimax strategy for cross-checking.

Search supports depth-limited Minimax and Alpha-Beta pruning, centre-first Connect Four ordering, iterative deepening, evaluator selection, node/time limits, candidate scores, node counts, elapsed time, completed depth, and prune counts. Values use player 1/Red's perspective. The neural model encodes two 6×7 planes (side-to-move pieces and opponent pieces) and uses an 84-128-64-1 MLP with ReLU hidden layers, tanh output, MSE loss, and Adam at 0.001. Inference uses the same encoding and runs inside search when the neural evaluator is selected.

## Submission artifacts

The reproducibility package includes the source generator/training code, saved neural checkpoint and metadata, GA weights and experiment metadata, and CSV/JSON experiment records under `backend/data/`. The per-game training dataset is intentionally generated deterministically in memory by `python -m app.training.train_connect4`; it is not a separate required artifact. Python packages are listed in `backend/requirements.txt`; frontend dependencies and scripts are declared in `frontend/package.json` (use the included lockfile when installing).
