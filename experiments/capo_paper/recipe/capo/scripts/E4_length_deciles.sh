#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"

# Length-stratified evaluation.
# Produces: fig_length_deciles

parse_common_args "$@"

EXPNAME_BASE=${EXPNAME_BASE:-"capo_E4_length_deciles"}
OVERRIDES=(
  "algorithm.adv_estimator=capo_eb"
  "trainer.project_name=capo"
)

launch_run "$ROOT_DIR/experiments/capo_paper/recipe/capo/main_capo.py" \
  "$EXPNAME_BASE" \
  "${OVERRIDES[@]}"
