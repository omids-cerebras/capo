# CAPO paper recipe

This recipe is the **authoritative interface** for running the CAPO paper experiments.

It follows a simple structure:

- one Hydra config: `config/capo_trainer.yaml`
- one thin entrypoint: `main_capo.py`
- thin launch scripts in `scripts/` (paper-minimum suite)
- deterministic outputs under `outputs/` (Hydra default)

## How CAPO is selected

Set:

- `algorithm.adv_estimator: capo | capo_eb_lite | capo_eb`

These advantage estimators are registered in the vendored VERL:
- `experiments/verl/trainer/ppo/core_algos.py`

CAPO hyperparameters live under the `capo:` config block.

## Paper artifacts: script → figure/table mapping

This mapping is the contract between the repo and the LaTeX paper.
If you rename a figure/table in LaTeX, update this mapping (and the artifact builder) accordingly.

| Script | What it runs | Paper artifacts produced (via `analysis/make_paper_artifacts.py`) |
|---|---|---|
| `scripts/E1_main_comparison.sh` | CAPO variants vs baselines at a fixed budget (multi-seed) | `tab_main_accuracy.tex` |
| `scripts/E2_dynamics.sh` | learning curves / dynamics logging | `fig_dynamics.pdf` |
| `scripts/E3_stability_sweep.sh` | stability/sensitivity sweep over CAPO knobs | `fig_stability.pdf`, `tab_stability_efficiency.tex` |
| `scripts/E4_length_deciles.sh` | length-stratified evaluation (deciles) | `fig_length_deciles.pdf` |

### Artifact build steps

1) Collect runs:

```bash
python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected
```

2) Generate all figures/tables:

```bash
python experiments/capo_paper/analysis/make_paper_artifacts.py --collected artifacts/collected --out artifacts
```

The artifact builder is intentionally forgiving: if a run or metric is missing it emits
compile-safe placeholders (dashes / empty plots) rather than crashing.

## Configuration

- Base config: `config/capo_trainer.yaml`
- CAPO block:

```yaml
capo:
  epsilon: 1e-8
  eb_lite:
    max_iters: 20
    tol: 1e-4
  eb_full:
    max_iters: 50
    tol: 1e-4
    k_band: 8
```

Override any field from the CLI, e.g.:

```bash
cd src/capo/experiments
python recipe/capo/main_capo.py \
  algorithm.adv_estimator=capo_eb \
  capo.eb_full.k_band=16
```
