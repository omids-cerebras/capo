#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

# E2: Alpha sweep for length-based normalization (CountDown, short-context).
# Generates inputs for: fig_dynamics overlays + stability diagnostics.

parse_common "$@"

ALPHAS=(0.5 0.75 1.0)

for a in "${ALPHAS[@]}"; do
  name="deltaL_a${a}"
  echo "[E2] alpha=$a (seed=$SEED, L=$MAX_RESP_LEN)"
  hydra_run +method.name="$name"     algorithm.adv_estimator=grpo     algorithm.use_grpopp=true     algorithm.grpopp_config.alpha="$a"     actor_rollout_ref.actor.use_grpopp=true     actor_rollout_ref.actor.grpopp_config.alpha="$a"
done
