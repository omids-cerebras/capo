#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CAPO container entrypoint
#
# Behaviour:
#   1. Print GPU topology + NCCL diagnostics (helpful for debugging comms).
#   2. If the first argument is a known sub-command, dispatch to it.
#   3. Otherwise exec whatever the user passed (default: bash).
#
# Sub-commands:
#   train   – launch CAPO training (delegates to main_capo.py)
#   prepare – prepare the CountDown dataset
#   test    – run the test suite
#   diag    – environment diagnostics (no training)
# =============================================================================

echo "=== CAPO container ==="
echo "  Python : $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA   : $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "  GPUs   : $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""

# ---- sub-command dispatch ---------------------------------------------------
case "${1:-}" in

  train)
    shift
    echo ">>> Launching CAPO training..."

    # Derive GPU count from NVIDIA_VISIBLE_DEVICES or torch
    N_GPUS="${N_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
    NNODES="${NNODES:-1}"

    ENTRYPOINT="/workspace/src/capo/experiments/recipe/capo/main_capo.py"

    # Default Hydra overrides for multi-GPU; user args come last and win.
    exec python3 "$ENTRYPOINT" \
      trainer.nnodes="$NNODES" \
      trainer.n_gpus_per_node="$N_GPUS" \
      "$@"
    ;;

  prepare)
    shift
    echo ">>> Preparing CountDown dataset..."
    exec python3 -m capo.experiments.scripts.data.prepare_countdown_dataset "$@"
    ;;

  test)
    shift
    echo ">>> Running test suite..."
    exec pytest "${@:--ra -q}"
    ;;

  diag)
    shift
    echo ">>> Running environment diagnostics..."
    python3 - <<'PY'
import sys, os

print(f"Python     : {sys.version}")
try:
    import torch
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA avail : {torch.cuda.is_available()}")
    print(f"GPU count  : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / (1024**3)
        print(f"  GPU {i}: {props.name}  ({mem_gb:.1f} GB)")
    print(f"NCCL ver   : {'.'.join(map(str, torch.cuda.nccl.version()))}")
except Exception as e:
    print(f"  torch error: {e}")

try:
    import ray; print(f"Ray        : {ray.__version__}")
except ImportError:
    print("Ray        : NOT INSTALLED")

try:
    import vllm; print(f"vLLM       : {vllm.__version__}")
except ImportError:
    print("vLLM       : NOT INSTALLED")

try:
    import flash_attn; print(f"FlashAttn  : {flash_attn.__version__}")
except ImportError:
    print("FlashAttn  : NOT INSTALLED")

try:
    from capo.verl_integration.adv_estimators import (
        compute_capo_advantage,
        compute_capo_eb_lite_advantage,
        compute_capo_eb_full_advantage,
    )
    print("CAPO adv   : all 3 estimators importable ✓")
except Exception as e:
    print(f"CAPO adv   : IMPORT ERROR – {e}")

print("\nEnvironment OK ✓")
PY
    ;;

  *)
    # Pass through: let user run arbitrary commands (default CMD is bash)
    exec "$@"
    ;;

esac
