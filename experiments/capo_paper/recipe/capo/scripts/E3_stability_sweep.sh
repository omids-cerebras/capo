#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"

# Stability sweep over a few EB hyperparameters.
# Produces: fig_stability, tab_stability_efficiency

parse_common_args "$@"

EXPNAME_BASE=${EXPNAME_BASE:-"capo_E3_stability"}

SWEEPS=(
  "k64:capo.eb_full.k_band=64" 
  "k128:capo.eb_full.k_band=128" 
  "ema0.9:capo.eb_full.ema_beta=0.9 capo.eb_full.ema_xi=0.9" 
)

for s in "${SWEEPS[@]}"; do
  TAG=${s%%:*}
  OVERRIDES=${s#*:}
  echo "[E3] running $TAG ($OVERRIDES)"
  run_one \
    "${EXPNAME_BASE}_${TAG}" \
    $OVERRIDES
  echo
done
