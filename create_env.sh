#!/bin/bash
# Create and configure the CAPO conda environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="capo"

echo "=== CAPO Environment Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n "${ENV_NAME}" -y
fi

# Create environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f "${SCRIPT_DIR}/environment.yml"

# Activate and install package in editable mode
echo "Installing CAPO package in editable mode..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

# Install tensordict (needs PyPI, not PyTorch index)
pip install tensordict

pip install -e "${SCRIPT_DIR}"

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

echo ""
echo "=== Environment created successfully! ==="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation:"
echo "  python -c 'import capo; print(f\"CAPO v{capo.__version__}\")'"
echo ""
