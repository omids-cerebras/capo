# Design: CAPO paper experiment harness

This document explains *how the experiment harness is structured*, what is vendored, what is patched,
and how the paper artifacts are generated deterministically from run logs.

## Philosophy

The harness optimizes for:
- **low cognitive load**: one YAML + a few thin scripts;
- **traceability**: every number in the paper comes from a run directory with a resolved config;
- **local reproducibility**: figures/tables can be built from `metrics.jsonl` without external services.

## Directory layout

```
experiments/capo_paper/
  verl/                         # vendored VERL runtime (lightly patched)
  recipe/capo/                  # CAPO entrypoint + config + launch scripts
  analysis/                     # run collector + paper artifact builder
  scripts/a10/                  # tiny diagnostics and smoke tests (A10-friendly)
  docs/                         # this doc + tutorial
  third_party/                  # upstream licenses/notices (do not edit/remove)
```

### `verl/` (vendored runtime)

We vendor a pinned snapshot of VERL to guarantee the experiment entrypoints and configs stay stable.
Only *minimal* patches are applied:

1. **CAPO advantage estimator registration**
   - File: `verl/trainer/ppo/core_algos.py`
   - What: register `capo`, `capo_eb_lite`, `capo_eb` in VERL's `register_adv_est` registry
   - Why: allows selecting CAPO purely via config:
     - `algorithm.adv_estimator: capo | capo_eb_lite | capo_eb`

2. **Local metrics logging**
   - File: `verl/trainer/ppo/ray_trainer.py`
   - What: if `trainer.local_metrics_path` is set, append per-step metric dicts to `metrics.jsonl`
   - Why: offline figure/table generation without WandB/MLFlow

No other changes are intended. If you need to modify vendored VERL, do so sparingly and document it here.

### `recipe/capo/` (how runs are launched)

- `config/capo_trainer.yaml` is the *single source of truth* for experiment defaults.
- `main_capo.py` is a thin Hydra entrypoint that:
  1) resolves config
  2) constructs VERL worker roles
  3) runs `RayPPOTrainer.fit()`

Thin scripts in `recipe/capo/scripts/` set a small number of overrides (model, dataset, seeds, CAPO knobs).

### `analysis/` (how paper artifacts are produced)

Two steps:

1) **Collect runs** into a normalized schema:

```bash
python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected
```

Outputs:
- `artifacts/collected/runs.json`
- `artifacts/collected/runs.csv`

2) **Build paper artifacts** (all figures/tables):

```bash
python experiments/capo_paper/analysis/make_paper_artifacts.py --collected artifacts/collected --out artifacts/paper
```

Outputs (authoritative set):
- `artifacts/paper/fig_main.pdf`
- `artifacts/paper/fig_dynamics.pdf`
- `artifacts/paper/fig_stability.pdf`
- `artifacts/paper/fig_length_deciles.pdf`
- `artifacts/paper/tab_main_accuracy.tex`
- `artifacts/paper/tab_stability_efficiency.tex`

The builder fails if required metrics/config keys are missing to prevent silent paper drift.

## Conventions

### Run directory structure
Hydra writes runs to:

```
outputs/<exp_name>/<YYYY-MM-DD>/<HH-MM-SS>/
  .hydra/config.yaml
  metrics.jsonl             # if enabled
  ...
```

The scripts set `exp_name` so that each experiment family is easy to collect.

### Metric keys
The artifact builder expects a small set of keys (documented at the top of `make_paper_artifacts.py`).
If you change metric names in the trainer, update the builder accordingly.

## Extending the paper

Add a new figure/table by:
1) logging the required metric(s) into `metrics.jsonl`,
2) adding an artifact function in `analysis/make_paper_artifacts.py`,
3) updating `recipe/capo/README.md` with the mapping to a script (or add a new script).
