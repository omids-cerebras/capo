#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"

# Training dynamics: run a single config with extra logging.
# Produces: fig_dynamics

parse_common_args "$@"

EXPNAME_BASE=${EXPNAME_BASE:-"capo_E2_dynamics"}
ADV=${ADV:-"capo_eb"}
EXTRA_OVERRIDES=(
  "trainer.project_name=capo"
  "algorithm.adv_estimator=$ADV"
  "trainer.total_steps=${TOTAL_STEPS:-1000}"
)

run_hydra_experiment "$ROOT_DIR/experiments/capo_paper/recipe/capo/main_capo.py" \
  "$EXPNAME_BASE" \
  "$MODEL" \
  "$TRAIN" \
  "$VAL" \
  "${EXTRA_OVERRIDES[@]}"
