#!/usr/bin/env bash
set -euo pipefail
# Build all paper artifacts (tables/figures) from completed runs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_DIR="${1:-$REPO_ROOT/outputs}"
OUT_DIR="${2:-$REPO_ROOT/paper/artifacts/paper}"

python "$REPO_ROOT/experiments/capo_paper/analysis/make_paper_artifacts.py" \
  --runs_dir "$RUNS_DIR" \
  --out_dir "$OUT_DIR"

echo "Artifacts written to: $OUT_DIR"
