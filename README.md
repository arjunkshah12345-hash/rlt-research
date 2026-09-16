# RLT Research Dossier

Independent analysis of the **Recurrent Looped Transformer** (Yifan Zhang, Sept 12 2026)
plus an original apples-to-apples benchmark spanning 1951→2026 sequence architectures.

## Contents
- `paper.pdf`, `paper.txt` — the RLT preprint (pulled from the official repo)
- `bench.py` — one-file benchmark: 6 models × 2 tasks × 3 seeds, matched ~75–84K params
- `run_all.py` — parallel sweep driver (36 runs)
- `results/*.json` — raw per-run results (params, accuracy@length, ms/step)
- `data/rlt_repo_published.json` — the repo's own published synthetic numbers
- `data/our_results.json` — aggregated means/min/max over seeds (generated)
- `make_charts.py` — generates all PNG charts from raw JSON (no hand-typed data)
- `build_vbg_site.py` — regenerates `site/index.html` from `results/` + `data/` (every number on the page is machine-derived)
- `update_site.py`, `gen_site_data.py` — legacy helpers from the first site iteration (unused by the current design)
- `charts/*.png` — bar charts + curves
- `site/index.html` — the minimalist report site (generated artifact)

Live report: https://arjunkshah12345-hash.github.io/rlt-research/site/

## Reproduce
```
python3 run_all.py          # ~40 min on a 10-core CPU (MacBook Pro)
python3 make_charts.py
python3 build_vbg_site.py   # regenerates site/index.html from the raw results
open site/index.html
```

## Models (all ~75–84K params)
| model | architecture |
|---|---|
| elman | vanilla RNN cell, full BPTT |
| lstm | LSTM cell, full BPTT |
| gru | GRU cell, full BPTT |
| transformer | 3-block causal Transformer, sinusoidal positions |
| token_merge | RLT ablation: recurrent decoder + input merging, self-attention only |
| rlt | causal encoder → global K/V memory; looped decoder (2×2 blocks) with cross-attention + state feedback, full BPTT |

## Tasks
- **parity** (chance 50): bits with dense running-parity targets (documented deviation:
  sparse final-token parity never leaves chance at this scale/budget for any model)
- **five** (chance 20): compose k∈[1,5] random permutations of S5, sparse final-state target

## Key findings (measured, 36 runs)
1. The paper's motivation replicates: at matched params/budget the parallel Transformer
   fails both tasks (chance parity, ~31% five-state).
2. But the fix is recurrence itself: a 1990 Elman RNN scores 100% on both tasks incl.
   4× length generalization. RLT-style works (89–100% five) but is slower (4×),
   seed-unstable on parity (1/3 seeds stuck at chance), and dominated by plain RNNs
   at this scale. Its machinery is a scale bet (10M+ params, L=48).
3. Task construction + curriculum dominate small-scale conclusions: sparse final-token
   parity never learns for anyone; dense running-targets let even Elman generalize 4×.
   The repo's "crash at 64 ops" did not reproduce under our generator.
4. Economics: RLT = ~4× Transformer training step time (342ms vs 85ms, batch 64),
   ~8× Elman; inference is loop-free so the tax is training-only.
