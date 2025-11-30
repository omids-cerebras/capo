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
"""

# Import modules for side-effects (registration)
from .reward_fn import CAPOConfig, capo_reward_fn  # noqa: F401
from .reward_manager import CAPORewardManager      # noqa: F401
from .adv_estimators import (                      # noqa: F401
    compute_capo_empirical_bayes_advantage,
)
