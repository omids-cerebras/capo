# Tutorial: running CAPO paper experiments end-to-end

This tutorial is written to be followed literally on a fresh machine.

## 0) Preconditions

- CUDA driver/toolkit compatible with your GPU
- Python environment with CAPO installed
- Access to a model checkpoint and a dataset supported by the vendored VERL runner

## 1) Install

From the CAPO repo root:

```bash
pip install -e ".[analysis]"
```

## 2) Fast diagnostics (A10-friendly)

These are designed to take minutes and reveal:
- import/path problems
- CUDA visibility issues
- obvious shape/mask mismatches
- exploding advantages / NaNs
- missing `metrics.jsonl` logging

```bash
bash experiments/capo_paper/scripts/a10/a10_diagnose_env.sh
bash experiments/capo_paper/scripts/a10/a10_smoke_tiny.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

If the smoke test fails, do **not** start the full suite—fix the issue first.

## 3) Run the paper-minimum suite (1-week budget)

All scripts share the same interface:

- `--model <MODEL_PATH>`
- `--train <TRAIN_FILE> --val <VAL_FILE>`
- optional: `--seeds "0 1 2"` etc.

### E1: main comparison (primary figure + primary table)

```bash
bash experiments/capo_paper/recipe/capo/scripts/E1_main_comparison.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

### E2: training dynamics

```bash
bash experiments/capo_paper/recipe/capo/scripts/E2_dynamics.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

### E3: stability sweep

```bash
bash experiments/capo_paper/recipe/capo/scripts/E3_stability_sweep.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

### E4: length-stratified evaluation

```bash
bash experiments/capo_paper/recipe/capo/scripts/E4_length_deciles.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

## 4) Build the paper figures/tables

Collect runs:

```bash
python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected
```

Generate artifacts:

```bash
python experiments/capo_paper/analysis/make_paper_artifacts.py --collected artifacts/collected --out artifacts/paper
```

## 5) Sanity check: what should exist

After step (4), you should have:

- `artifacts/paper/fig_main.pdf`
- `artifacts/paper/fig_dynamics.pdf`
- `artifacts/paper/fig_stability.pdf`
- `artifacts/paper/fig_length_deciles.pdf`
- `artifacts/paper/tab_main_accuracy.tex`
- `artifacts/paper/tab_stability_efficiency.tex`

If any of these are missing, the builder should have raised an error that tells you which metric/config key is missing.

## 6) Debugging tips

- Look at `outputs/.../.hydra/config.yaml` to confirm the resolved settings.
- Inspect the last ~200 lines of `metrics.jsonl` to see if a metric stopped being logged.
- If you see NaNs/inf in advantages, start by lowering learning rate or increasing CAPO epsilon.
- If you suspect length effects, rerun E4 with fewer bins and confirm monotonicity first.
