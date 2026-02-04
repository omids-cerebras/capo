#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# E3: Stability / sensitivity sweep
#
# Purpose:
#   - stress-test CAPO's empirical Bayes estimators
#   - evaluate sensitivity to the covariance window size (k-band)
#   - provide data for stability/efficiency summaries
#
# Design:
#   - sweep a small grid of k_band values for CAPO-EB
#   - keep all other knobs identical
#
# Output:
#   - runs are aggregated by the artifact builder into fig_stability.pdf and
#     tab_stability_efficiency.tex
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

parse_common_args "$@"

# E3 focuses on stability metrics; validation cadence should be reasonably fine.
TEST_FREQ="25"
SEEDS=("${SEEDS[0]}")

K_BANDS=(4 8 16)

for k in "${K_BANDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export SEED="$seed"
    exp="E3_capo_eb_k${k}_L${MAX_RESP_LEN}_seed${seed}"
    run_one "$exp" \
      algorithm.adv_estimator=capo_eb \
      capo.eb_full.k_band="$k" \
      actor_rollout_ref.actor.loss_agg_mode=token-mean
  done
done

echo
echo "E3 complete. The paper artifact builder will use these runs to produce fig_stability.pdf and tab_stability_efficiency.tex."
