#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# E1: Main comparison (CountDown, Qwen2.5-1.5B-Instruct)
#
# Runs a minimal but comprehensive comparison suite where the ONLY degrees of
# freedom are (a) aggregation / normalization choices and (b) CAPO statistics.
#
# This script is intentionally "DeltaL-style": simple bash loops with explicit
# Hydra overrides.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

parse_common_args "$@"

# NOTE: For long-context runs (e.g. 8192), override --max_resp_len and consider
# lowering --train_bsz or --rollout_n if you hit OOM.

METHODS=(
  "dapo_norm"
  "grpo_norm"
  "dr_grpo_norm"
  "grpopp_alpha_0p75"
  "grpopp_alpha_1p00"
  "capo_eb_lite"
  "capo_eb"
)

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export SEED="$seed"
    exp="E1_${method}_L${MAX_RESP_LEN}_seed${seed}"

    case "$method" in
      dapo_norm)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=False \
          algorithm.use_grpopp=False \
          actor_rollout_ref.actor.loss_agg_mode=token-mean
        ;;

      grpo_norm)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=False \
          algorithm.use_grpopp=False \
          actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
        ;;

      dr_grpo_norm)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=True \
          algorithm.use_grpopp=False \
          actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
        ;;

      grpopp_alpha_0p75)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=False \
          algorithm.use_grpopp=True \
          algorithm.grpopp_config.alpha=0.75 \
          actor_rollout_ref.actor.loss_agg_mode=token-mean
        ;;

      grpopp_alpha_1p00)
        run_one "$exp" \
          algorithm.adv_estimator=grpo \
          algorithm.use_dr_grpo=False \
          algorithm.use_grpopp=True \
          algorithm.grpopp_config.alpha=1.0 \
          actor_rollout_ref.actor.loss_agg_mode=token-mean
        ;;

      capo_eb_lite)
        run_one "$exp" \
          algorithm.adv_estimator=capo_eb_lite \
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
echo "E1 complete. Next steps:"
echo "  - Run artifact collection/build:" 
echo "      python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected"
echo "      python experiments/capo_paper/analysis/make_paper_artifacts.py --collected artifacts/collected --out artifacts/paper"