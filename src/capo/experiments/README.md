# CAPO Experiments

Reproducible experiments for the CAPO paper.

## Structure

```
experiments/
├── analysis/           # Run collection + paper artifact generation
├── docs/               # Design docs and tutorial
├── recipe/capo/        # Main entrypoint + Hydra configs
└── scripts/            # Data prep + utility scripts
```

## Quick Start

```bash
# 1. Prepare CountDown dataset
python -m capo.experiments.scripts.data.prepare_countdown_dataset \
    --out_dir data/countdown

# 2. Run experiment
python -m capo.experiments.recipe.capo.main_capo \
    algorithm.adv_estimator=capo_eb

# 3. Collect results
python -m capo.experiments.analysis.collect_runs \
    --runs_dir outputs --out artifacts/collected

# 4. Generate paper figures/tables
python -m capo.experiments.analysis.make_paper_artifacts \
    --collected artifacts/collected --out artifacts
```

## Experiment Scripts

| Script | Description |
|--------|-------------|
| `E1_main_comparison.sh` | CAPO vs baselines |
| `E2_dynamics.sh` | Learning curves |
| `E3_stability_sweep.sh` | Sensitivity sweep |
| `E4_length_deciles.sh` | Length-stratified eval |

## Configuration

Base config: `recipe/capo/config/capo_trainer.yaml`

Key options:
```yaml
algorithm:
  adv_estimator: capo_eb  # capo | capo_eb_lite | capo_eb
```

See [docs/TUTORIAL.md](docs/TUTORIAL.md) for detailed instructions.
