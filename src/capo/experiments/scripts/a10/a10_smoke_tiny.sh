#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# A10 smoke test: tiny training run to catch plumbing issues early.
#
# Intended for a single A10 (1 GPU). Uses very small budgets to:
# - validate dataloader + tokenization
# - validate rollout + reward plumbing
# - validate CAPO advantage estimator execution (no NaNs/shape issues)
# - validate local metrics logging (metrics.jsonl)
#
# Expected runtime: 5–20 minutes (depends on model/data).
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Ensure vendored VERL is used.
export PYTHONPATH="$REPO_ROOT/experiments/capo_paper:$REPO_ROOT:${PYTHONPATH:-}"

usage() {
  cat <<USAGE
Usage:
  bash experiments/capo_paper/scripts/a10/a10_smoke_tiny.sh \
    --model <MODEL_PATH> --train <TRAIN_FILE> --val <VAL_FILE> [--adv <capo|capo_eb_lite|capo_eb>]

Required:
  --model PATH   HuggingFace model checkpoint path (local or remote, per VERL)
  --train FILE   Training file(s) supported by the runner (e.g., parquet/jsonl)
  --val FILE     Validation file(s)

Optional:
  --adv NAME     Advantage estimator (default: capo_eb)
  --steps N      Total training steps (default: 50)
  --seed S       Seed (default: 0)
USAGE
}

MODEL=""
TRAIN=""
VAL=""
ADV="capo_eb"
STEPS="50"
SEED="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --train) TRAIN="$2"; shift 2 ;;
    --val)   VAL="$2"; shift 2 ;;
    --adv)   ADV="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed)  SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TRAIN" || -z "$VAL" ]]; then
  echo "ERROR: --model, --train, and --val are required" >&2
  usage
  exit 1
fi

ENTRYPOINT="$REPO_ROOT/experiments/capo_paper/recipe/capo/main_capo.py"
if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "ERROR: entrypoint not found: $ENTRYPOINT" >&2
  exit 1
fi

# Put metrics under a deterministic experiment name.
EXP_NAME="smoke_a10_${ADV}"
NOW="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$REPO_ROOT/outputs/${EXP_NAME}/${NOW}"

mkdir -p "$OUTDIR"

python "$ENTRYPOINT" \
  hydra.run.dir="$OUTDIR" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.local_metrics_path="$OUTDIR/metrics.jsonl" \
  trainer.validation_data_dir="$OUTDIR/val_generations" \
  trainer.val_before_train=True \
  trainer.test_freq=25 \
  trainer.save_freq=-1 \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.train_batch_size=16 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.val_kwargs.n=2 \
  algorithm.adv_estimator="$ADV" \
  seed="$SEED"

echo "Smoke run complete."
echo "Run dir: $OUTDIR"
echo "Metrics: $OUTDIR/metrics.jsonl"
