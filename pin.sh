#!/usr/bin/env bash
#
# pin.sh
#
# Compile requirements.in into a fully pinned requirements file using uv.
#
# Unlike the previous version, this script no longer assumes that uv is
# already installed. If uv is not found on PATH, we:
#
#   1. Create a temporary virtual environment.
#   2. Install uv into that environment with pip.
#   3. Run `python -m uv pip compile ...` inside that environment.
#   4. Remove the temporary environment.
#
# Usage:
#   ./pin.sh                 # compiles requirements.in → pinned-requirements.txt
#   ./pin.sh custom.in       # compiles custom.in → pinned-requirements.txt
#
# You can override the output filename via the OUTPUT_FILE env var:
#   OUTPUT_FILE=requirements.lock.txt ./pin.sh
#

set -euo pipefail

INPUT_FILE="${1:-requirements.in}"
OUTPUT_FILE="${OUTPUT_FILE:-pinned-requirements.txt}"

if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file '$INPUT_FILE' not found." >&2
  exit 1
fi

# Choose a Python interpreter.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  # Fallback to `python` if `python3` is not available.
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Neither 'python3' nor 'python' was found on PATH." >&2
  exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

# Fast path: uv is already installed globally.
if command -v uv >/dev/null 2>&1; then
  echo "Found 'uv' on PATH; using it directly."
  echo "Compiling $INPUT_FILE → $OUTPUT_FILE ..."
  uv pip compile "$INPUT_FILE" --output-file "$OUTPUT_FILE"
  echo "Done."
  exit 0
fi

echo "No 'uv' found on PATH; bootstrapping uv in a temporary virtualenv."

# Create a temporary directory for the bootstrap venv.
# We use mktemp if available; otherwise we fall back to a fixed name.
if command -v mktemp >/dev/null 2>&1; then
  TMP_ENV="$(mktemp -d .uv-bootstrap.XXXXXX)"
else
  TMP_ENV=".uv-bootstrap-env"
  if [ -d "$TMP_ENV" ]; then
    echo "Removing existing temporary env at $TMP_ENV"
    rm -rf "$TMP_ENV"
  fi
fi

cleanup() {
  if [ -n "${TMP_ENV:-}" ] && [ -d "$TMP_ENV" ]; then
    echo "Cleaning up temporary env at $TMP_ENV"
    rm -rf "$TMP_ENV"
  fi
}
trap cleanup EXIT

echo "Creating temporary virtualenv at: $TMP_ENV"
"$PYTHON_BIN" -m venv "$TMP_ENV"

# Figure out the Python executable inside the venv (POSIX vs Windows).
VENV_PY="$TMP_ENV/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY="$TMP_ENV/Scripts/python.exe"
fi

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: Could not find Python inside temporary env '$TMP_ENV'." >&2
  exit 1
fi

echo "Installing uv into the temporary env..."
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PY" -m pip install uv >/dev/null

echo "Compiling $INPUT_FILE → $OUTPUT_FILE using 'python -m uv pip compile'..."
"$VENV_PY" -m uv pip compile "$INPUT_FILE" --output-file "$OUTPUT_FILE"

echo "Pinned requirements written to: $OUTPUT_FILE"
echo "Temporary env will now be removed."
# cleanup() via trap will remove TMP_ENV
