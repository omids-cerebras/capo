#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# One-command runner for the CAPO paper experiments.
#
# This script intentionally mirrors the "single entrypoint + simple bash" style
# popularized by DeltaL-style repositories.
#
# What it does (in order):
#   1) prepare the CountDown dataset (parquet)
#   2) run a 1-GPU smoke test (fast correctness check)
#   3) run E1 (short) main comparison
#   4) run E3 (short) CAPO k-band sweep
#   5) run E4 (long) length-decile evaluation
#   6) collect runs + build paper artifacts (figures/tables)
#   7) optionally copy artifacts into the LaTeX report directory
#
# Notes:
#   - The defaults are designed to be realistic on 8×A10 in a few days.
#   - You can override steps, seeds, and output locations via CLI flags.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

usage() {
  cat <<USAGE
Usage:
  bash experiments/capo_paper/scripts/run_all.sh \
    --model <HF_MODEL_OR_PATH> \
    --workdir <WORKDIR> \
    [--paper_dir <PATH_TO_LATEX_REPORT>]

Required:
  --model      HuggingFace model name or local path (e.g., Qwen/Qwen2.5-1.5B-Instruct)
  --workdir    Where to put outputs and prepared data (will be created)

Optional:
  --paper_dir  If set and contains artifacts/paper/, copy built figures/tables there
  --steps_e1   Total training steps for E1 (default: 600)
  --steps_e3   Total training steps for E3 (default: 400)
  --steps_e4   Total training steps for E4 (default: 400)
  --seeds      Space-separated seeds for E1 (default: "0 1")
  --max_train  Truncate train split for quick iterations (default: 0 = full)
  --max_test   Truncate test split for quick iterations (default: 0 = full)

Examples:
  bash experiments/capo_paper/scripts/run_all.sh \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --workdir /mnt/efs/capo_runs \
    --paper_dir /mnt/efs/capo_paper_latex
USAGE
}

MODEL=""
WORKDIR=""
PAPER_DIR=""

STEPS_E1=600
STEPS_E3=400
STEPS_E4=400
SEEDS_STR="0 1"
MAX_TRAIN=0
MAX_TEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --paper_dir) PAPER_DIR="$2"; shift 2 ;;
    --steps_e1) STEPS_E1="$2"; shift 2 ;;
    --steps_e3) STEPS_E3="$2"; shift 2 ;;
    --steps_e4) STEPS_E4="$2"; shift 2 ;;
    --seeds) SEEDS_STR="$2"; shift 2 ;;
    --max_train) MAX_TRAIN="$2"; shift 2 ;;
    --max_test) MAX_TEST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$WORKDIR" ]]; then
  echo "ERROR: --model and --workdir are required" >&2
  usage
  exit 1
fi

WORKDIR="$(cd "$(dirname "$WORKDIR")" && pwd)/$(basename "$WORKDIR")"
mkdir -p "$WORKDIR"

DATA_DIR="$WORKDIR/data/countdown"
OUT_ROOT="$WORKDIR/outputs"
COLLECTED="$WORKDIR/collected"
ARTIFACTS="$WORKDIR/artifacts"

echo "=== [1/7] Prepare CountDown dataset ==="
python "$REPO_ROOT/experiments/capo_paper/scripts/data/prepare_countdown_dataset.py" \
  --out_dir "$DATA_DIR" \
  --seed 0 \
  --max_train "$MAX_TRAIN" \
  --max_test "$MAX_TEST"

echo "=== [2/7] Smoke test (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 \
  bash "$REPO_ROOT/experiments/capo_paper/scripts/a10/a10_smoke_tiny.sh" \
    --model "$MODEL" \
    --train "$DATA_DIR/train.parquet" \
    --val "$DATA_DIR/test.parquet" \
    --adv capo_eb \
    --steps 30 \
    --seed 0

echo "=== [3/7] E1 (short) main comparison ==="
bash "$REPO_ROOT/experiments/capo_paper/recipe/capo/scripts/E1_main_comparison.sh" \
  --model "$MODEL" \
  --data_dir "$DATA_DIR" \
  --output_root "$OUT_ROOT" \
  --steps "$STEPS_E1" \
  --max_resp_len 2048 \
  --seeds "$SEEDS_STR"

echo "=== [4/7] E3 (short) CAPO k-band sweep ==="
bash "$REPO_ROOT/experiments/capo_paper/recipe/capo/scripts/E3_stability_sweep.sh" \
  --model "$MODEL" \
  --data_dir "$DATA_DIR" \
  --output_root "$OUT_ROOT" \
  --steps "$STEPS_E3" \
  --max_resp_len 2048 \
  --seeds "0"

echo "=== [5/7] E4 (long) length-decile evaluation ==="
bash "$REPO_ROOT/experiments/capo_paper/recipe/capo/scripts/E4_length_deciles.sh" \
  --model "$MODEL" \
  --data_dir "$DATA_DIR" \
  --output_root "$OUT_ROOT" \
  --steps "$STEPS_E4" \
  --max_resp_len 8192 \
  --seeds "0"

echo "=== [6/7] Collect runs + build artifacts ==="
python "$REPO_ROOT/experiments/capo_paper/analysis/collect_runs.py" \
  --outputs_root "$OUT_ROOT" \
  --out "$COLLECTED"

python "$REPO_ROOT/experiments/capo_paper/analysis/make_paper_artifacts.py" \
  --collected "$COLLECTED" \
  --out "$ARTIFACTS"

echo "Artifacts written to: $ARTIFACTS/paper"

echo "=== [7/7] Copy artifacts into LaTeX report (optional) ==="
if [[ -n "$PAPER_DIR" ]]; then
  if [[ -d "$PAPER_DIR/artifacts/paper" ]]; then
    cp -v "$ARTIFACTS/paper/"* "$PAPER_DIR/artifacts/paper/"
    echo "Copied artifacts into: $PAPER_DIR/artifacts/paper"
  else
    echo "WARNING: paper_dir does not contain artifacts/paper/: $PAPER_DIR" >&2
  fi
else
  echo "(skipping) --paper_dir not provided"
fi

echo
echo "Done."
echo "Outputs root:  $OUT_ROOT"
echo "Collected:     $COLLECTED"
echo "Artifacts:     $ARTIFACTS"
