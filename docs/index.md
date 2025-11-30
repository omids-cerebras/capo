# CAPO: Credit Assignment Policy Optimization

This repository implements **CAPO** and **EB–CAPO** as drop-in
extensions to the [VERL](https://github.com/volcengine/verl) RL
framework.

The core problem is: given a batch of trajectories with **heterogeneous
lengths** and potentially **correlated token-level noise**, how should
we:

1. Aggregate noisy per-trajectory signals into a low-variance estimate
   of a common quantity (e.g. a gradient or reward mean), and
2. Use those same weights to construct per-token **advantages** for
   policy optimization?

CAPO answers this with a family of length- and dependence-aware
weights, estimated via an **Empirical Bayes** procedure that learns
the exponent on length and a dependence correction from the data.

## Structure of the docs

- [Setup](setup.md) – problem formulation, notation, and MVU weights.
- [Method](method.md) – variance model, EB objective, dependence
  family \(s(L;\xi)\).
- [Algorithms](algorithms.md) – EB–CAPO workflow, EB-lite, and
  ACF-moment, and how they map to code.
- [Evaluations](evaluations.md) – planned experiments and metrics.

For the implementation details, see:

- `capo/verl_integration/adv_estimators.py` – three advantage
  estimators:
  - `capo` – plain CAPO (z-normalized token rewards),
  - `capo_eb_lite` – EB–CAPO-lite (length-only EB),
  - `capo_eb` – full EB–CAPO (length + dependence).
- `capo/verl_integration/reward_fn.py` – CAPO reward composition
  (outcome + process).
- `capo/verl_integration/reward_manager.py` – VERL `RewardManager`
  producing token-level CAPO rewards.
