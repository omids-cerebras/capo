#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"

# Main comparison: baselines vs CAPO variants.
# Produces: tab_main_accuracy, fig_main

parse_common_args "$@"

RUNS=(
  "baseline_grpo:algorithm.adv_estimator=grpo"
  "capo:algorithm.adv_estimator=capo"
  "capo_eb_lite:algorithm.adv_estimator=capo_eb_lite"
  "capo_eb:algorithm.adv_estimator=capo_eb"
)

for item in "${RUNS[@]}"; do
  name="${item%%:*}"
  override="${item#*:}"
  exp_name="E1_${name}"

  echo "[E1] launching $exp_name ($override)"
  (cd "$ROOT_DIR" && \
    python experiments/capo_paper/recipe/capo/main_capo.py \
      +exp_name="$exp_name" \
      actor_rollout_ref.model.path="$MODEL" \
      data.train_files="$TRAIN_FILE" \
      data.val_files="$VAL_FILE" \
      data.max_prompt_length="$MAX_PROMPT_LEN" \
      data.max_response_length="$MAX_RESP_LEN" \
      data.train_batch_size="$TRAIN_BSZ" \
      trainer.total_training_steps="$STEPS" \
      trainer.nnodes="$NNODES" \
      trainer.n_gpus_per_node="$GPUS_PER_NODE" \
      $override \
      ${EXTRA_OVERRIDES[@]:-}
  )
done
