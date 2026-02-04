# CAPO: Covariance-Aware Policy Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**CAPO** is a Bayesian gradient aggregation method for Reinforcement Learning with Verifiable Rewards (RLVR). It provides precision-optimal trajectory weighting that accounts for length heterogeneity and within-trajectory token dependence.

## Key Ideas

Standard policy gradient methods treat all trajectories equally, but longer trajectories have different variance characteristics. CAPO addresses this by:

1. **Modeling variance** as $v_i = \sigma^2 L_i^\beta s(L_i; \xi)$ where:
   - $\beta$ is the length exponent (how variance scales with length)
   - $s(L; \xi)$ captures within-trajectory dependence structure

2. **Estimating parameters** via Empirical Bayes from batch statistics

3. **Reweighting trajectories** with precision-optimal weights $w_i \propto L_i^{-\beta} / s(L_i; \xi)$

This recovers the $\Delta L$ normalization as a special case ($\beta=1$, no dependence) while enabling adaptive, data-driven weighting.

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate capo

# Install CAPO
pip install -e .
```

For CPU-only or different CUDA versions, edit `environment.yml` accordingly.

## Quick Start

```python
import capo.verl_integration  # Register with VERL

# In your VERL config:
# algorithm:
#   adv_estimator: capo_eb
```

## Repository Structure

```
src/
└── capo/
    ├── __init__.py
    ├── eb_core.py              # Core EB algorithms
    ├── verl_integration/       # VERL integration
    ├── experiments/            # Paper reproduction code
    └── tests/                  # Test suite

notebooks/tutorials/            # Interactive tutorials
docs/                           # MkDocs documentation
```

## Tutorials

Interactive notebooks to get started:

| Notebook | Description |
|----------|-------------|
| [01_capo_basics](notebooks/tutorials/01_capo_basics.ipynb) | Core concepts and weighting |
| [02_comparing_estimators](notebooks/tutorials/02_comparing_estimators.ipynb) | Compare all three estimators |
| [03_empirical_bayes](notebooks/tutorials/03_empirical_bayes.ipynb) | EB estimation deep dive |

## Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `capo` | Standard CAPO with fixed weights | Baseline |
| `capo_eb_lite` | EB estimation of β only | Fast, no dependence modeling |
| `capo_eb` | Full EB with (β, ρ, η) | Best quality, handles dependence |

## Running Experiments

See [src/capo/experiments/docs/TUTORIAL.md](src/capo/experiments/docs/TUTORIAL.md) for reproduction instructions.

```bash
# Prepare data
python -m capo.experiments.scripts.data.prepare_countdown_dataset --out_dir data/countdown

# Run training
python -m capo.experiments.recipe.capo.main_capo
```

## Documentation

```bash
# Build docs
pip install -e ".[docs]"
mkdocs serve
```

## Testing

```bash
pytest
```

## Citation

```bibtex
@article{solari2024capo,
  title={Covariance-Aware Policy Optimization for RLVR},
  author={Solari, Omid Shams},
  year={2024}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
