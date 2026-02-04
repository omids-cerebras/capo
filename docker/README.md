# Docker

Build and run CAPO in a container.

## Build

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

Inside the container:

```bash
# Prepare data
python -m capo.experiments.scripts.data.prepare_countdown_dataset --out_dir data/countdown

# Run training
python -m capo.experiments.recipe.capo.main_capo \
    algorithm.adv_estimator=capo_eb
```

## Notes

- Adjust CUDA version in Dockerfile if needed
- Set `WANDB_MODE=offline` if no network access
  environment.