"""
VERL integration for the CAPO method.

Importing this module will:

- Register `CAPORewardManager` as `reward_manager: "capo"` in VERL.
- Register `compute_capo_empirical_bayes_advantage` as
  `adv_estimator: "capo_eb"` in VERL.

You typically do not need to import the individual symbols directly.
Instead, just ensure this side-effect import happens once:

    import capo.verl_integration  # noqa: F401

The actual implementations live in:

- `reward_fn.py`      – CAPO reward function + GenPRM client.
- `reward_manager.py` – VERL-compatible token-level CAPORewardManager.
- `adv_estimators.py` – Empirical Bayes advantage estimator.

Note: The reward_manager and adv_estimators modules require VERL to be installed.
The reward_fn module can be used independently for testing purposes.
"""

# Always import reward_fn (no VERL dependency)
from .reward_fn import CAPOConfig, capo_reward_fn  # noqa: F401

# Conditionally import VERL-dependent modules
try:
    from .reward_manager import CAPORewardManager  # noqa: F401
    from .adv_estimators import compute_capo_empirical_bayes_advantage  # noqa: F401

    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False
    CAPORewardManager = None  # type: ignore
    compute_capo_empirical_bayes_advantage = None  # type: ignore

__all__ = [
    "CAPOConfig",
    "capo_reward_fn",
    "CAPORewardManager",
    "compute_capo_empirical_bayes_advantage",
]
