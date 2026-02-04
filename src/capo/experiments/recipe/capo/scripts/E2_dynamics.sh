#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# E2: Training dynamics
#
# Runs a small subset of methods with a more frequent validation cadence...
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

parse_common_args "$@"

# For dynamics we want more frequent evaluation for smoother curves.
TEST_FREQ="20"

# E2 is primarily about qualitative curve shape; one seed is typically sufficient.
SEEDS=("${SEEDS[0]}")

METHODS=(
  "grpopp_alpha_1p00"
  "capo_eb"
)

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export SEED="$seed"
    exp="E2_${method}_L${MAX_RESP_LEN}_seed${seed}"

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
echo "E2 complete. The paper artifact builder will use these runs to produce fig_dynamics.pdf."
