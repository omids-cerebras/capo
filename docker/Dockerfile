FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY . /workspace

# Install a CUDA-enabled PyTorch wheel.
# NOTE: adjust 'cu121' if you build this image with a different CUDA runtime.
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install pinned deps (torch is intentionally excluded from the lock file).
RUN pip install -r pinned-requirements.txt

# Install CAPO in editable mode.
RUN pip install -e .

# Ensure vendored VERL is importable.
ENV PYTHONPATH=/workspace/experiments/capo_paper:/workspace:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1

CMD ["bash"]
