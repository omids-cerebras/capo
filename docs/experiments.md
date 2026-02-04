# Experiments

This section documents how to reproduce the CAPO paper experiments.

## Quick Start

```bash
# 1. Prepare CountDown dataset
python -m capo.experiments.scripts.data.prepare_countdown_dataset \
    --out_dir data/countdown

# 2. Run experiments
python -m capo.experiments.recipe.capo.main_capo \
    algorithm.adv_estimator=capo_eb
```

## Experiment Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `E1_main_comparison.sh` | CAPO vs baselines | `tab_main_accuracy.tex` |
| `E2_dynamics.sh` | Learning curves | `fig_dynamics.pdf` |
| `E3_stability_sweep.sh` | Sensitivity analysis | `fig_stability.pdf` |
| `E4_length_deciles.sh` | Length-stratified eval | `fig_length_deciles.pdf` |

## Building Paper Artifacts

```bash
# Collect runs
python -m capo.experiments.analysis.collect_runs \
    --runs_dir outputs --out artifacts/collected

# Generate figures/tables
python -m capo.experiments.analysis.make_paper_artifacts \
    --collected artifacts/collected --out artifacts
```

## Configuration

See `src/capo/experiments/recipe/capo/config/` for Hydra configs.

Key config options:

```yaml
algorithm:
  adv_estimator: capo_eb  # capo | capo_eb_lite | capo_eb

capo:
  epsilon: 1e-8
  eb_lite:
    max_iters: 20
    tol: 1e-4
```

For detailed instructions, see the [Tutorial](https://github.com/omids/capo/blob/main/src/capo/experiments/docs/TUTORIAL.md).
