# CAPO: Covariance-Aware Policy Optimization

**CAPO** is a Bayesian gradient aggregation method for Reinforcement Learning with Verifiable Rewards (RLVR).

## Overview

Standard policy gradient methods treat all trajectories equally, but this is suboptimal when variance depends on trajectory length. CAPO addresses this by:

1. **Modeling variance** as $v_i = \sigma^2 L_i^\beta s(L_i; \xi)$
2. **Estimating parameters** via Empirical Bayes from batch statistics
3. **Reweighting trajectories** with precision-optimal weights

## Installation

```bash
conda env create -f environment.yml
conda activate capo
pip install -e .
```

## Quick Start

```python
import capo.verl_integration  # Register with VERL

# In your VERL config:
# algorithm:
#   adv_estimator: capo_eb
```

## Documentation

- [Setup & Notation](setup.md) — Mathematical foundations
- [CAPO Method](method.md) — Core methodology
- [Algorithms](algorithms.md) — EB-lite, ACF-moment, k-banded weights
- [Evaluations](evaluations.md) — Experimental results
- [API Reference](api.md) — Code documentation
- [Experiments](experiments.md) — Reproducing paper results
- [Technical Report](report/capo.pdf) — Full paper

## Citation

```bibtex
@article{solari2024capo,
  title={Covariance-Aware Policy Optimization for RLVR},
  author={Solari, Omid Shams},
  year={2024}
}
```
