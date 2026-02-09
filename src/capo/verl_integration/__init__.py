"""
VERL integration for CAPO.

This module registers CAPO components with VERL's plugin system:

- ``CAPORewardManager`` — registered as ``reward_manager: "capo"``
- ``compute_capo_empirical_bayes_advantage`` — registered as ``adv_estimator: "capo_eb"``

Usage::

    import capo.verl_integration  # registers CAPO with VERL

    # Then in your VERL config:
    # reward_model:
    #   reward_manager: capo
    # algorithm:
    #   adv_estimator: capo_eb

Submodules
----------
reward_fn
    CAPO reward function and GenPRM client interface.
reward_manager
    Token-level reward manager for VERL's PPO/GRPO trainer.
adv_estimators
    Empirical Bayes advantage estimators (L-CAPO and full EB).
"""

from .adv_estimators import compute_capo_empirical_bayes_advantage
from .reward_fn import CAPOConfig, capo_reward_fn
from .reward_manager import CAPORewardManager

__all__ = [
    "CAPOConfig",
    "capo_reward_fn",
    "CAPORewardManager",
    "compute_capo_empirical_bayes_advantage",
]
