# Vendored VERL

This directory contains a vendored copy of [VERL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs).

## Why Vendor?

CAPO requires modifications to VERL's core advantage estimation logic. Specifically:

1. **Custom advantage estimators**: CAPO registers `capo`, `capo_eb_lite`, and `capo_eb` advantage estimators in `verl/trainer/ppo/core_algos.py`.

2. **API stability**: Vendoring pins the exact VERL version used for the paper experiments, ensuring reproducibility regardless of upstream changes.

3. **Hydra config inheritance**: The experiment configs (`capo_trainer.yaml`) inherit from VERL's `ppo_trainer.yaml`. Vendoring makes config discovery trivial.

## VERL Version

This vendored copy is based on VERL v0.3.0 with the following CAPO-specific patches:

- `verl/trainer/ppo/core_algos.py`: Added CAPO advantage estimator registration
- Integration with `capo.verl_integration.adv_estimators`

## Upstream Repository

- **GitHub**: https://github.com/volcengine/verl
- **Paper**: [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- **License**: Apache 2.0

## Usage

The vendored VERL is automatically used when running experiments via the scripts in `recipe/capo/scripts/`. The `PYTHONPATH` is set to prioritize this vendored copy over any installed `verl` package.

```bash
# This uses vendored verl automatically
bash scripts/E1_main_comparison.sh --model <MODEL> --data_dir <DATA>
```

## Updating

If you need to update the vendored VERL:

1. Copy the new VERL source to this directory
2. Re-apply the CAPO patches to `verl/trainer/ppo/core_algos.py`
3. Test that all experiments still work
