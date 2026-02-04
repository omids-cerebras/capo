FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY . /workspace

# Install PyTorch with CUDA support
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install CAPO and dependencies
RUN pip install -e . verl==0.3.0.post1

ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1

CMD ["bash"]
