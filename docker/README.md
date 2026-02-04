# Docker quickstart (GPU)

This repository vendors VERL under `experiments/capo_paper/verl`. The Docker
image configures `PYTHONPATH` so that the vendored copy is imported (no `pip
install verl` is required).

## Build

From the repository root:

```bash
docker build -t capo:latest -f docker/Dockerfile .
```

## Run

```bash
docker run --rm -it --gpus all \
  -v $PWD:/workspace \
  -w /workspace \
  capo:latest
```

Inside the container you can then run:

```bash
python experiments/capo_paper/scripts/data/prepare_countdown_dataset.py --out_dir data/countdown
bash experiments/capo_paper/scripts/run_all.sh --model Qwen/Qwen2.5-1.5B-Instruct --data_dir data/countdown
```

## Notes

- If you build against a different CUDA runtime, adjust the `--index-url` used to
  install PyTorch in the Dockerfile.
- If you do not want WandB network access, set `WANDB_MODE=offline` in your
  environment.