# Tutorial: Running CAPO paper experiments (CountDown, Qwen2.5-1.5B)

This guide is intentionally procedural: each step produces a tangible artifact
(a dataset, a run directory, or a paper-ready figure/table).

## 1) Environment

Use any environment that can run PyTorch + CUDA and HuggingFace Transformers.

Key requirement:
- Ensure the vendored VERL in `experiments/capo_paper/verl` is on `PYTHONPATH`.
  All provided scripts do this automatically.

## 2) Prepare CountDown data

Create a VERL-compatible parquet dataset (train + test) under a single folder:

```bash
python experiments/capo_paper/scripts/data/prepare_countdown_dataset.py \
  --out_dir data/countdown \
  --seed 123 \
  --test_size 0.1
```

This produces:
- `data/countdown/train.parquet`
- `data/countdown/test.parquet`

## 3) Quick diagnostics on an A10

These are designed to take minutes and fail loudly if something is wrong:

```bash
# Environment sanity check (imports, versions, CUDA visibility)
bash experiments/capo_paper/scripts/a10/a10_diagnose_env.sh

# 1-GPU tiny smoke run (very short training)
bash experiments/capo_paper/scripts/a10/a10_smoke_tiny.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train data/countdown/train.parquet \
  --val data/countdown/test.parquet \
  --steps 50 \
  --adv capo_eb_lite \
  --out outputs/smoke_tiny
```

## 4) Run the paper experiment suite

### One-command runner

If you want a single entrypoint that prepares data, runs the canonical suite,
and builds paper artifacts:

```bash
bash experiments/capo_paper/scripts/run_all.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workdir ./workdir \
  --seeds "0 1"
```

By default this runs:
- a 1-GPU smoke test
- E1 (short, 2048)
- E4 (long, 8192; includes length-decile eval)
- E3 (small CAPO k-band sweep)
- artifact build + optional copy into a paper folder

### Individual experiments

All experiment scripts share the same basic interface:

```bash
bash experiments/capo_paper/recipe/capo/scripts/E1_main_comparison.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data_dir data/countdown \
  --out_root outputs/paper \
  --steps 400 \
  --seeds "0 1"
```

Available scripts:
- `E1_main_comparison.sh`: main baseline comparison (multi-method, multi-seed)
- `E2_dynamics.sh`: smoother learning curves for a small subset of methods
- `E3_stability_sweep.sh`: CAPO EB k-band sensitivity sweep
- `E4_length_deciles.sh`: long-context runs + length-stratified evaluation

## 5) Build paper artifacts

Artifacts are derived from `metrics.jsonl` and validation generation dumps.

```bash
python experiments/capo_paper/analysis/collect_runs.py \
  --outputs_root outputs/paper \
  --out artifacts/collected

python experiments/capo_paper/analysis/make_paper_artifacts.py \
  --collected artifacts/collected \
  --out artifacts
```

This generates `artifacts/paper/` containing:
- `tab_main_accuracy.tex`
- `tab_stability_efficiency.tex`
- `fig_dynamics.pdf`
- `fig_stability.pdf`
- `fig_length_deciles.pdf`

The artifact builder is forgiving: if a run or metric is missing, it emits
compile-safe placeholders (dashes / empty plots) rather than failing.
