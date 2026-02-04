#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Common helpers for CAPO paper experiment scripts.
#
# Goals:
#   - Keep each experiment script short and readable (DeltaL-style).
#   - Provide a single, consistent CLI across scripts.
#   - Ensure the vendored VERL runtime is used (via PYTHONPATH).
#   - Make output directories deterministic and paper-artifact friendly.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Ensure vendored VERL is imported instead of any site-packages `verl`.
export PYTHONPATH="$REPO_ROOT/experiments/capo_paper:$REPO_ROOT:${PYTHONPATH:-}"

common_usage() {
  cat <<USAGE
Usage (shared across all E* scripts):
  bash <SCRIPT>.sh --model <HF_MODEL> --data_dir <COUNTDOWN_DIR> [options]

Required:
  --model PATH         HuggingFace model name or local path (e.g. Qwen/Qwen2.5-1.5B-Instruct)
  --data_dir DIR       Directory containing train/val parquet files (train.parquet, test.parquet)

Common options:
  --train FILE         Override train parquet path (default: <data_dir>/train.parquet)
  --val FILE           Override val parquet path   (default: <data_dir>/test.parquet)
  --output_root DIR    Output root for Hydra run dirs (default: <repo>/outputs)
  --nnodes N           Number of nodes (default: 1)
  --gpus_per_node N    GPUs per node (default: 8)
  --steps N            Total training steps (default: 500)
  --train_bsz N        Global train batch size (default: 128)
  --rollout_n N        Number of responses per prompt during training (default: 8)
  --val_n N            Number of responses per prompt during validation (default: 8)
  --max_prompt_len N   Max prompt length (default: 256)
  --max_resp_len N     Max response length (default: 2048)
  --test_freq N        Validation frequency in steps (default: 50)
  --save_freq N        Checkpoint frequency in steps (default: -1, disabled)
  --seeds "0 1"        Space-separated list of seeds (default: "0 1")

Notes:
  - Scripts are written for A10x8 (e.g. g5.48xlarge). Adjust --gpus_per_node if needed.
  - Set WANDB_API_KEY (and optionally WANDB_PROJECT) to enable Weights & Biases logging.
USAGE
}

# Defaults (sane for 8×A10; override per-script for long-context runs).
MODEL=""
DATA_DIR=""
TRAIN_FILE=""
VAL_FILE=""
OUTPUT_ROOT=""
NNODES="1"
GPUS_PER_NODE="8"
STEPS="500"
TRAIN_BSZ="128"
ROLLOUT_N="8"
VAL_N="8"
MAX_PROMPT_LEN="256"
MAX_RESP_LEN="2048"
TEST_FREQ="50"
SAVE_FREQ="-1"
SEEDS=(0 1)

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --data_dir) DATA_DIR="$2"; shift 2 ;;
      --train) TRAIN_FILE="$2"; shift 2 ;;
      --val) VAL_FILE="$2"; shift 2 ;;
      --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
      --nnodes) NNODES="$2"; shift 2 ;;
      --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
      --steps) STEPS="$2"; shift 2 ;;
      --train_bsz) TRAIN_BSZ="$2"; shift 2 ;;
      --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
      --val_n) VAL_N="$2"; shift 2 ;;
      --max_prompt_len) MAX_PROMPT_LEN="$2"; shift 2 ;;
      --max_resp_len) MAX_RESP_LEN="$2"; shift 2 ;;
      --test_freq) TEST_FREQ="$2"; shift 2 ;;
      --save_freq) SAVE_FREQ="$2"; shift 2 ;;
      --seeds)
        # shellcheck disable=SC2206
        SEEDS=($2)
        shift 2
        ;;
      -h|--help) common_usage; exit 0 ;;
      *) echo "Unknown arg: $1" >&2; common_usage; exit 1 ;;
    esac
  done

  if [[ -z "$MODEL" || -z "$DATA_DIR" ]]; then
    echo "ERROR: --model and --data_dir are required." >&2
    common_usage
    exit 1
  fi

  if [[ -z "$TRAIN_FILE" ]]; then
    TRAIN_FILE="$DATA_DIR/train.parquet"
  fi
  if [[ -z "$VAL_FILE" ]]; then
    VAL_FILE="$DATA_DIR/test.parquet"
  fi
  if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="$REPO_ROOT/outputs"
  fi
}

run_one() {
  local exp_name="$1"; shift

  local now
  now="$(date +%Y%m%d_%H%M%S)"

  local outdir="$OUTPUT_ROOT/$exp_name/$now"
  mkdir -p "$outdir"

  local entrypoint="$REPO_ROOT/experiments/capo_paper/recipe/capo/main_capo.py"
  if [[ ! -f "$entrypoint" ]]; then
    echo "ERROR: entrypoint not found: $entrypoint" >&2
    exit 1
  fi

  # Common overrides across all experiments.
  # Keep these overrides minimal and explicit; algorithm-specific knobs belong in the caller.
  python "$entrypoint" \
    hydra.run.dir="$outdir" \
    seed="${SEED:-0}" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.total_training_steps="$STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.project_name="${WANDB_PROJECT:-capo}" \
    trainer.experiment_name="$exp_name" \
    trainer.local_metrics_path="$outdir/metrics.jsonl" \
    trainer.validation_data_dir="$outdir/val_generations" \
    trainer.val_before_train=True \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BSZ" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESP_LEN" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    "$@"

  echo
  echo "Run complete: $exp_name"
  echo "  outdir:   $outdir"
  echo "  metrics:  $outdir/metrics.jsonl"
  echo "  val dump: $outdir/val_generations"
}
