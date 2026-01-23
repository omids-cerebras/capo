#!/usr/bin/env bash
#
# create_env.sh
#
# Create a virtual environment, install pinned dependencies from
# pinned-requirements.txt, and then install CAPO in editable mode.
#
# Behavior:
#   If PYTHON_BIN is set, use exactly that interpreter.
#     If it doesn't exist or is < 3.10, fail.
#   Otherwise, try python3, then python, and require >= 3.10.
#   Create or reuse VENV_DIR.
#   Install uv inside that env.
#   Use uv to install pinned deps and CAPO (-e .) INTO THAT VENV,
#   even if a Conda env is active.
#
# Usage:
#   ./create_env.sh            # uses .venv and pinned-requirements.txt
#   ./create_env.sh myenv      # different venv directory
#
#   PYTHON_BIN=python3.11 ./create_env.sh
#     -> forces use of python3.11 to create the env.
#

set -euo pipefail

VENV_DIR="${1:-.venv}"
PINNED_FILE="pinned-requirements.txt"

if [ ! -f "$PINNED_FILE" ]; then
  echo "ERROR: '$PINNED_FILE' not found. Run ./pin.sh first." >&2
  exit 1
fi

###############################################################################
# Step 1: Choose the Python interpreter used to create the venv
###############################################################################

if [ "${PYTHON_BIN-}" != "" ]; then
  # User explicitly set PYTHON_BIN: respect it.
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN='$PYTHON_BIN' was specified, but that command is not on PATH." >&2
    exit 1
  fi
  CHOSEN_PY="$PYTHON_BIN"
  echo "Using explicitly specified Python interpreter: $CHOSEN_PY"
else
  # Auto-detect: prefer python3, then python.
  if command -v python3 >/dev/null 2>&1; then
    CHOSEN_PY="python3"
  elif command -v python >/dev/null 2>&1; then
    CHOSEN_PY="python"
  else
    echo "ERROR: Could not find 'python3' or 'python' on PATH." >&2
    echo "       Please install Python 3.10+ and/or set PYTHON_BIN explicitly." >&2
    exit 1
  fi
  echo "Using auto-detected Python interpreter: $CHOSEN_PY"
fi

###############################################################################
# Step 2: Create or reuse the venv
###############################################################################

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at: $VENV_DIR"
  "$CHOSEN_PY" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at: $VENV_DIR (reusing)"
fi

ENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$ENV_PY" ]; then
  echo "ERROR: Could not find Python inside '$VENV_DIR' (expected '$VENV_DIR/bin/python')." >&2
  exit 1
fi

###############################################################################
# Step 3: Check venv Python version against requires-python >= 3.10
###############################################################################

PY_MAJOR=$("$ENV_PY" -c "import sys; print(sys.version_info[0])")
PY_MINOR=$("$ENV_PY" -c "import sys; print(sys.version_info[1])")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "ERROR: The virtualenv Python is ${PY_MAJOR}.${PY_MINOR}, but CAPO requires >= 3.10." >&2
  echo
  if [ "${PYTHON_BIN-}" != "" ]; then
    echo "       You explicitly requested PYTHON_BIN='$PYTHON_BIN', which is too old." >&2
  fi
  echo "       Please install a newer Python (e.g. 3.10 or 3.11) and re-run, e.g.:" >&2
  echo "         PYTHON_BIN=python3.11 ./create_env.sh" >&2
  echo
  echo "       Or adjust 'requires-python' in pyproject.toml if you truly want" >&2
  echo "       to support Python ${PY_MAJOR}.${PY_MINOR} (not recommended)." >&2
  exit 1
fi

echo "Venv Python version is ${PY_MAJOR}.${PY_MINOR} (OK: >= 3.10)."

###############################################################################
# Step 4: Install uv inside the venv
###############################################################################

echo
echo "Upgrading pip inside the environment..."
"$ENV_PY" -m pip install --upgrade pip

echo
echo "Installing uv inside the environment..."
"$ENV_PY" -m pip install uv

UV_BIN="$VENV_DIR/bin/uv"
if [ ! -x "$UV_BIN" ]; then
  echo "ERROR: uv executable not found in '$VENV_DIR/bin' after installation." >&2
  exit 1
fi

###############################################################################
# Step 5: Install pinned deps and CAPO INTO THIS VENV using uv
#
# We explicitly:
#   set VIRTUAL_ENV="$VENV_DIR"
#   and pass --python "$ENV_PY"
# so uv targets this venv even if a Conda env is active.
###############################################################################

echo
echo "Installing pinned dependencies from '$PINNED_FILE' into $VENV_DIR using uv..."
VIRTUAL_ENV="$VENV_DIR" "$UV_BIN" pip install --python "$ENV_PY" -r "$PINNED_FILE"

echo
echo "Installing CAPO in editable mode (pip-style -e .) into $VENV_DIR using uv..."
VIRTUAL_ENV="$VENV_DIR" "$UV_BIN" pip install --python "$ENV_PY" -e .

echo
echo "Environment setup complete."
echo
echo "To activate the environment, run:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Then you can run tests with:"
echo "  pytest"
