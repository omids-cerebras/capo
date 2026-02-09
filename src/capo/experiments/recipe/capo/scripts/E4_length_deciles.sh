#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# E4: Length-stratified evaluation
#
# Runs long-context training/evaluation (typically max_response_length=8192)
# for a small subset of methods, then relies on the validation-generation
# dumps to compute accuracy stratified by response-length deciles.
#
# The default subset is:
#   - GRPO++ (alpha=1.0) as the strongest length-only baseline
#   - LV-CAPO as the covariance-aware method
#
# The artifact builder consumes the per-run `val_generations/*.jsonl`.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

parse_common_args "$@"

# Long-context defaults (the caller may still override via --max_resp_len / --test_freq).
MAX_RESP_LEN="8192"
TEST_FREQ="50"
SEEDS=("${SEEDS[0]}")

METHODS=(
  "grpopp_alpha_1p00"
  "capo_eb"
)

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export SEED="$seed"
    exp="E4_${method}_L${MAX_RESP_LEN}_seed${seed}"

    case "$method" in
      grpopp_alpha_1p00)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=False \
          algorithm.use_grpopp=True \
          algorithm.grpopp_config.alpha=1.0 \
          actor_rollout_ref.actor.loss_agg_mode=token-mean
        ;;

      capo_eb)
        run_one "$exp" \
          algorithm.adv_estimator=capo_eb \
          actor_rollout_ref.actor.loss_agg_mode=token-mean
        ;;

      *)
        echo "Unknown method: $method" >&2
        exit 1
        ;;
    esac
  done
done

echo
echo "E4 complete. The paper artifact builder will use these runs to produce fig_length_deciles.pdf."
