"""
CAPO: Covariance-Aware Policy Optimization with Empirical Bayes Weighting.

This package provides precision-optimal trajectory weighting for RLHF training,
designed to handle the heterogeneous lengths and correlated noise that arise
in reasoning tasks (math, coding, etc.).

Overview
--------
CAPO addresses a fundamental challenge in policy gradient methods: how to
aggregate noisy trajectory-level signals when trajectories have different
lengths and potentially correlated token-level noise. The key insight is
that treating all trajectories equally (as in standard GRPO) is suboptimal
when variance depends on length.

Key Components
--------------
**Empirical Bayes Core** (`capo.eb_core`):
    - Length exponent β estimation (L-CAPO algorithm)
    - Dependence shape ξ = (ρ, η) estimation (ACF-moment algorithm)
    - k-banded covariance weights for precision-optimal aggregation
    - Joint EB updates for online parameter learning

**VERL Integration** (`capo.verl_integration`):
    - `CAPORewardManager`: Token-level CAPO reward composition
    - Advantage estimators: `capo`, `capo_eb_lite`, `capo_eb`
    - GenPRM client interface for process reward models

Quick Start
-----------
To use CAPO with VERL, import the integration module to trigger registration:

    >>> import capo.verl_integration  # noqa: F401

This registers CAPO components with VERL's internal registries. Then configure
your VERL trainer YAML to use CAPO:

    algorithm:
      adv_estimator: capo_eb  # or capo_eb_lite, capo

Mathematical Background
-----------------------
Given G trajectories with lengths L_i and scalarized returns g_i, CAPO models:

    Var(g_i) = σ² L_i^β s(L_i; ξ)

where β is the length exponent and s(L; ξ) captures dependence structure.
The Empirical Bayes procedure estimates (β, ξ) from the batch and uses
precision-optimal weights w_i ∝ L_i^{-β} / s(L_i; ξ) for aggregation.

See Also
--------
- README.md for installation and usage instructions
- docs/ for detailed mathematical derivations
- capo/tests/ for example usage patterns

Version
-------
"""

__version__ = "0.1.0"
__author__ = "Omid Shams Solari"
__email__ = "omid.solari@cerebras.net"

__all__ = ["eb_core", "verl_integration", "__version__"]
