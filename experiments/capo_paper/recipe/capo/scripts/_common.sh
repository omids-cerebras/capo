#!/usr/bin/env bash
set -euo pipefail

# Shared utilities for CAPO upstream-style scripts.

usage_common() {
  cat <<USAGE
Required flags:
  --model PATH         HuggingFace checkpoint path (local or remote, per VERL)
  --train FILE         Training parquet (or list supported by VERL)
  --val FILE           Validation parquet (or list supported by VERL)

Optional flags:
  --exp_name NAME      Experiment name (default: derived from script)
  --nnodes N           Number of nodes (default: 1)
  --gpus_per_node N    GPUs per node (default: 1)
  --max_steps N        Max train steps (default: 200)
  --max_prompt_len N   (default: 512)
  --max_resp_len N     (default: 512)
  --train_bsz N        (default: 64)
  --adv_est NAME       capo | capo_eb_lite | capo_eb (default: capo_eb)

Environment:
  OUTPUT_DIR           If set, overrides Hydra output dir (default: ./outputs)
USAGE
}

parse_common() {
  # defaults
  EXP_NAME=""
  MODEL=""
  TRAIN=""
  VAL=""
  NNODES=1
  GPUS_PER_NODE=1
  MAX_STEPS=200
  MAX_PROMPT_LEN=512
  MAX_RESP_LEN=512
  TRAIN_BSZ=64
  ADV_EST="capo_eb"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2;;
      --train) TRAIN="$2"; shift 2;;
      --val) VAL="$2"; shift 2;;
      --exp_name) EXP_NAME="$2"; shift 2;;
      --nnodes) NNODES="$2"; shift 2;;
      --gpus_per_node) GPUS_PER_NODE="$2"; shift 2;;
      --max_steps) MAX_STEPS="$2"; shift 2;;
      --max_prompt_len) MAX_PROMPT_LEN="$2"; shift 2;;
      --max_resp_len) MAX_RESP_LEN="$2"; shift 2;;
      --train_bsz) TRAIN_BSZ="$2"; shift 2;;
      --adv_est) ADV_EST="$2"; shift 2;;
      -h|--help) usage_common; exit 0;;
      *) echo "Unknown flag: $1"; usage_common; exit 1;;
    esac
  done

  if [[ -z "$MODEL" || -z "$TRAIN" || -z "$VAL" ]]; then
    echo "Missing required flags."; usage_common; exit 1
  fi

  if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$(basename "$0" .sh)"
  fi

  OUTPUT_DIR_ROOT="${OUTPUT_DIR:-outputs}"
}

hydra_run() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local recipe_dir
  recipe_dir="$(cd "$script_dir/.." && pwd)"

  # Ensure vendored VERL is used.
  export PYTHONPATH="$recipe_dir/../..:${PYTHONPATH:-}"

  python "$recipe_dir/main_capo.py" \
    hydra.run.dir="$OUTPUT_DIR_ROOT/$EXP_NAME/${now:%Y%m%d_%H%M%S}" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.total_training_steps="$MAX_STEPS" \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESP_LEN" \
    data.train_batch_size="$TRAIN_BSZ" \
    actor_rollout_ref.model.path="$MODEL" \
    algorithm.adv_estimator="$ADV_EST" \
    "$@"
}
