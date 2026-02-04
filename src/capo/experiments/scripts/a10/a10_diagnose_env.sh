#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# A10 diagnostics: environment + import sanity checks (no training).
#
# What this checks:
# - Python / Torch versions
# - CUDA visibility
# - Ray import
# - Vendored VERL import
# - CAPO advantage estimators registered in VERL registry
#
# Expected runtime: < 1 minute.
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

python - <<PY
import sys
from pathlib import Path

repo_root = Path(r"$REPO_ROOT")
paper_root = repo_root / "experiments" / "capo_paper"

print("python:", sys.version.replace("\n", " "))

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device_count:", torch.cuda.device_count())
        print("cuda_device_name_0:", torch.cuda.get_device_name(0))
except Exception as e:
    raise RuntimeError(f"Torch import/diagnostics failed: {e}")

try:
    import ray  # noqa: F401
    print("ray_import_ok: True")
except Exception as e:
    raise RuntimeError(f"Ray import failed: {e}")

# Ensure vendored VERL is importable with highest precedence.
# The sys.path entry must be the *parent* of the 'verl' package directory.
if str(paper_root) not in sys.path:
    sys.path.insert(0, str(paper_root))

try:
    import verl  # noqa: F401
    from verl.trainer.ppo import core_algos
    print("verl_import_ok: True")
except Exception as e:
    raise RuntimeError(f"Vendored VERL import failed: {e}")

# Ensure CAPO is importable and that the advantage estimators are registered.
try:
    import capo  # noqa: F401
    print("capo_import_ok: True")
except Exception as e:
    raise RuntimeError(f"CAPO import failed: {e}")

expected = ["capo", "capo_eb_lite", "capo_eb"]
for name in expected:
    fn = core_algos.get_adv_estimator_fn(name)
    if fn is None:
        raise RuntimeError(f"CAPO adv estimator not registered: {name}")
    print("adv_est_registered:", name, "->", getattr(fn, "__name__", str(fn)))

print("OK")
PY
