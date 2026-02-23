# =============================================================================
# CAPO: Multi-GPU training image (8×A10 / A100 / similar Ampere/Ada GPUs)
#
# Provides:
#   - PyTorch 2.6 + CUDA 12.4  (matches VERL / vLLM matrix)
#   - vLLM 0.8.5       (fast rollout engine)
#   - FlashAttention-2  (fused attention kernels)
#   - Ray               (distributed orchestration)
#   - VERL + CAPO       (training loop + EB advantage estimators)
#
# Build:
#   docker build -t capo:latest -f docker/Dockerfile .
#
# Run (see docker-compose.yml for the turnkey 8×A10 setup):
#   docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 \
#     -v $PWD:/workspace -w /workspace capo:latest bash
# =============================================================================

# ---------- base image with CUDA 12.4 + cuDNN 9 -----------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# System packages (build tools, git for pip installs, OpenCV deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        git curl wget \
        build-essential ninja-build \
        libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# pip / setuptools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------- PyTorch + inference / attention kernels --------------------------
# PyTorch (cu124 wheels)
RUN pip install --no-cache-dir \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Flash-Attention 2  (pre-built wheel → no 20-min compile)
RUN wget -qO /tmp/flash_attn.whl \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
    pip install --no-cache-dir /tmp/flash_attn.whl && rm /tmp/flash_attn.whl

# vLLM (fast rollout backend) + FlashInfer kernel
RUN pip install --no-cache-dir "vllm==0.8.5.post1" && \
    wget -qO /tmp/flashinfer.whl \
      https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir /tmp/flashinfer.whl && rm /tmp/flashinfer.whl

# ---------- Python ML / RL stack ---------------------------------------------
# Ray, HF ecosystem, VERL runtime deps
RUN pip install --no-cache-dir \
    "ray[default]>=2.10" \
    "transformers[hf_xet]>=4.51.0" \
    accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    hydra-core omegaconf \
    codetiming cloudpickle dill pybind11 \
    wandb liger-kernel \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    pylatexenc qwen-vl-utils \
    "tensordict==0.6.2" torchdata \
    pytest

# ---------- CAPO + vendored VERL ---------------------------------------------
WORKDIR /workspace
COPY . /workspace

# Install CAPO as an editable package
RUN pip install --no-cache-dir -e ".[dev]"

# ---------- runtime environment ----------------------------------------------
# NCCL / comms tuning for multi-GPU (A10 uses NVLink-less PCIe topology)
ENV NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_DEBUG=WARN

# General
ENV PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=true \
    VLLM_LOGGING_LEVEL=WARN

# Ensure vendored VERL is on the path (CAPO patches live there)
ENV PYTHONPATH="/workspace/src/capo/experiments:/workspace/src:${PYTHONPATH:-}"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
