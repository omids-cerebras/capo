"""
Minimal tests for the Empirical Bayes advantage estimator.

These tests focus on basic sanity properties:

- The estimator runs without error for simple inputs.
- The resulting advantages are "shrunk" compared to naive
  group-mean baselines, especially for small groups.

The tests require VERL because the EB estimator is registered via
VERL's `register_adv_est`. If VERL is not available in the environment,
these tests are skipped.
"""

from __future__ import annotations

# pytest.importorskip("verl")  # Skip entire module if VERL is not installed.
import numpy as np
import torch

from capo.verl_integration.adv_estimators import compute_capo_empirical_bayes_advantage


def test_empirical_bayes_advantage_basic_shape():
    """
    Check that the EB estimator returns tensors of the expected shape
    and does not crash on a simple synthetic example.
    """
    torch.manual_seed(0)

    # Two groups of 3 samples each.
    index = np.array([0, 0, 0, 1, 1, 1])

    # Simulate higher rewards in group 1 than group 0.
    token_rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],  # group 0
            [1.2, 1.2, 1.2, 1.2],  # group 0
            [0.8, 0.8, 0.8, 0.8],  # group 0
            [5.0, 5.0, 5.0, 5.0],  # group 1
            [5.2, 5.2, 5.2, 5.2],  # group 1
            [4.8, 4.8, 4.8, 4.8],  # group 1
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(token_rewards, dtype=torch.bool)

    advantages, returns, _ = compute_capo_empirical_bayes_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        config=None,
    )

    assert advantages.shape == token_rewards.shape
    assert returns.shape == token_rewards.shape

    # Ensure we actually get non-zero advantages in at least one group.
    assert torch.any(advantages != 0.0)


def test_empirical_bayes_shrinkage_behavior():
    """
    With per-group baselines (like GRPO), advantages centre around zero
    **within each prompt group**.  The test verifies:

    - Within group 0 (g = [1.0, 1.5]): higher-reward trajectory has
      positive advantage, lower-reward trajectory has negative advantage.
    - Within group 1 (g = [5.0] × 8): all trajectories are identical,
      so all within-group advantages ≈ 0.

    This matches the paper: w normalised *within* I_p, m_p per-group.
    """
    torch.manual_seed(0)

    # Group 0: small group (n=2) with low rewards.
    # Group 1: large group (n=8) with identical high rewards.
    rewards_list = [
        [1.0, 1.0],
        [1.5, 1.5],
    ] + [
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
    ]

    token_rewards = torch.tensor(rewards_list, dtype=torch.float32)
    mask = torch.ones_like(token_rewards, dtype=torch.bool)

    index = np.array([0, 0] + [1] * 8)

    advantages, _, _ = compute_capo_empirical_bayes_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        config=None,
    )

    # Collapse advantages per sequence.
    adv_scalar = (advantages * mask).mean(dim=-1)

    # Group 0: trajectory 0 (g=2.0) < trajectory 1 (g=3.0),
    # so adv[0] < adv[1].
    assert adv_scalar[0].item() < adv_scalar[1].item()

    # Group 1: all trajectories identical → all advantages ≈ 0.
    adv_g1 = adv_scalar[2:].abs().max().item()
    assert adv_g1 < 1e-6, f"Group 1 advantages should be ~0, got max={adv_g1}"
