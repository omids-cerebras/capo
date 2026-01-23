# CAPO paper experiments

This directory contains a **self-contained experiment harness** for the CAPO paper.

Design goals:
- **Single entrypoint + simple scripts** (copy/paste runnable).
- **Reproducible run directories** (Hydra outputs) with **local metrics logging** (`metrics.jsonl`).
- **One command** to convert runs into **paper figures/tables** (PDF + LaTeX).

Directory overview:

```
experiments/capo_paper/
  verl/                         # vendored VERL runtime (lightly patched for local metrics)
  recipe/capo/                  # CAPO entrypoint + config + launch scripts
  analysis/                     # collectors + paper artifact builder
  scripts/a10/                  # A10 smoke tests + environment diagnostics
  docs/                         # design + tutorial
  third_party/                  # upstream licenses/notices (do not edit/remove)
```

Quickstart
----------

1) Install CAPO (editable) and the analysis extras:

```bash
pip install -e ".[analysis]"
```

2) Run an A10 sanity check (fast, fails loudly if something is wrong):

```bash
bash experiments/capo_paper/scripts/a10/a10_diagnose_env.sh
bash experiments/capo_paper/scripts/a10/a10_smoke_tiny.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

3) Run paper experiments (see mapping in `recipe/capo/README.md`):

```bash
bash experiments/capo_paper/recipe/capo/scripts/E1_main_comparison.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
bash experiments/capo_paper/recipe/capo/scripts/E2_dynamics.sh        --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
bash experiments/capo_paper/recipe/capo/scripts/E3_stability_sweep.sh --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
bash experiments/capo_paper/recipe/capo/scripts/E4_length_deciles.sh  --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE>
```

4) Build the paper artifacts (all figures/tables) from the runs:

```bash
python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected
python experiments/capo_paper/analysis/make_paper_artifacts.py --collected artifacts/collected --out artifacts/paper
```

Documentation
-------------
- Design/architecture: `experiments/capo_paper/docs/DESIGN.md`
- Hands-on tutorial:   `experiments/capo_paper/docs/TUTORIAL.md`
- Script → figure/table mapping (authoritative): `experiments/capo_paper/recipe/capo/README.md`
