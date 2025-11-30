# CAPO: Covariance-Aware Policy Optimization

This repository contains a light-weight plugin package that implements:

- **CAPO-style credit assignment** for VERL-style RL (RLHF / RLVR).
  - Outcome + process-based reward composition (C, P table).
  - Optional token-level rewards via a custom `RewardManager`.
- An **Empirical Bayes advantage estimator** that can be plugged into
  VERL's PPO / GRPO trainer.

The goal is to keep this repository **small and focused** on the method,
while depending on VERL as an upstream library.

> **Note:** This is a *skeleton* implementation with clear extension points.
> You are expected to fill in the actual GenPRM client (LLM calls), exact
> CAPO hyperparameters, and any paper-specific details.

---

## Installation

You need:

- Python ≥ 3.10
- PyTorch ≥ 2.1
- VERL (installed from GitHub)

### 1. Install VERL

```bash
pip install "git+https://github.com/volcengine/verl.git"
