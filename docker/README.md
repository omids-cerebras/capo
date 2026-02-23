# Docker: Multi-GPU CAPO Training

Run CAPO rollout → advantage estimation → training on 8×A10 GPUs (or any
multi-GPU Ampere/Ada node) with a single `docker compose` command.

## What's in the image

| Component         | Version    | Why                                       |
|-------------------|------------|-------------------------------------------|
| CUDA              | 12.4       | Matches vLLM / FlashAttention wheel ABI   |
| PyTorch           | 2.6        | FSDP improvements, cu124 wheels           |
| vLLM              | 0.8.5      | Fast rollout engine for generation         |
| FlashAttention-2  | 2.7.4      | Fused attention kernels                    |
| Ray               | ≥ 2.10     | Distributed orchestration (RayPPOTrainer) |
| VERL (vendored)   | —          | PPO/GRPO trainer with CAPO patches         |
| CAPO              | editable   | EB advantage estimators + reward function  |

## Quick start

```bash
cd /path/to/capo   # repo root

# 1. Build
docker compose -f docker/docker-compose.yml build

# 2. Sanity-check the environment
docker compose -f docker/docker-compose.yml run --rm capo diag

# 3. Prepare CountDown dataset
docker compose -f docker/docker-compose.yml run --rm capo prepare \
    --out_dir /data/countdown

# 4. Train (8×A10, EB-CAPO on Qwen-1.5B)
docker compose -f docker/docker-compose.yml run --rm capo train \
    data.train_files=/data/countdown/train.parquet \
    data.val_files=/data/countdown/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    algorithm.adv_estimator=capo_eb \
    trainer.total_training_steps=500 \
    data.train_batch_size=128 \
    data.max_response_length=2048
```

## Entrypoint sub-commands

| Command   | Description                                                 |
|-----------|-------------------------------------------------------------|
| `train`   | Launch CAPO training (auto-detects GPU count)               |
| `prepare` | Prepare CountDown dataset to `/data/countdown`              |
| `test`    | Run pytest suite                                            |
| `diag`    | Print GPU topology, library versions, CAPO import check     |
| *(other)* | Passed through to exec (e.g. `bash`, `python …`)           |

## Configuration

All Hydra overrides are passed as trailing arguments to `train`. Key knobs:

```bash
docker compose -f docker/docker-compose.yml run --rm capo train \
    algorithm.adv_estimator=capo_eb \       # capo | capo_eb_lite | capo_eb
    trainer.n_gpus_per_node=8 \             # auto-detected if omitted
    trainer.nnodes=1 \
    data.train_batch_size=128 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.name=vllm \   # vllm (fast) or hf (simple)
    actor_rollout_ref.rollout.n=8 \         # responses per prompt
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
```

## Experiment scripts (full paper reproduction)

The experiment scripts from `src/capo/experiments/recipe/capo/scripts/` work
inside the container. Example:

```bash
docker compose -f docker/docker-compose.yml run --rm capo bash -c \
    "bash src/capo/experiments/recipe/capo/scripts/E1_main_comparison.sh \
         --model Qwen/Qwen2.5-1.5B-Instruct \
         --data_dir /data/countdown \
         --gpus_per_node 8 \
         --steps 500"
```

## GPU count

The compose file requests `count: all` GPUs. To use fewer:

```yaml
# In docker-compose.yml, change:
count: all
# to:
count: 4
```

Or set `NVIDIA_VISIBLE_DEVICES`:
```bash
NVIDIA_VISIBLE_DEVICES=0,1,2,3 docker compose -f docker/docker-compose.yml run --rm capo train ...
```

## Volumes

| Volume          | Mount point                  | Purpose                |
|-----------------|------------------------------|------------------------|
| `capo-data`     | `/data`                      | Datasets, checkpoints  |
| `capo-hf-cache` | `/root/.cache/huggingface`   | Model cache            |
| bind mount      | `/workspace`                 | Source code (live edit) |

## WandB / HuggingFace tokens

Set in your shell or a `.env` file next to `docker-compose.yml`:

```bash
export WANDB_API_KEY=your_key
export WANDB_MODE=online        # default: offline
export HF_TOKEN=your_hf_token   # for gated models like Llama
```

## Without docker compose

```bash
# Build
docker build -t capo:latest -f docker/Dockerfile .

# Run
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    -v $PWD:/workspace \
    -v capo-data:/data \
    -w /workspace \
    -e PYTHONPATH=/workspace/src/capo/experiments:/workspace/src \
    capo:latest train \
        data.train_files=/data/countdown/train.parquet \
        data.val_files=/data/countdown/test.parquet \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
        algorithm.adv_estimator=capo_eb
```

## Notes

- **A10 (24 GB)**: Qwen-1.5B fits comfortably with FSDP FULL_SHARD.
  For 7B models, reduce `data.train_batch_size` or `rollout.n`.
- **NCCL tuning**: The defaults (`IB_DISABLE=1`, `P2P_DISABLE=0`) work for
  PCIe-connected A10s (e.g., `g5.48xlarge`). For NVLink machines (A100/H100),
  you can remove `NCCL_IB_DISABLE`.
- Set `WANDB_MODE=offline` if your cluster has no internet access.
- Outputs land in `outputs/` by default (Hydra working directory).
