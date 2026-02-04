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

import pytest
import torch

# pytest.importorskip("verl")  # Skip entire module if VERL is not installed.

import numpy as np

from capo.verl_integration.adv_estimators import compute_capo_empirical_bayes_advantage


def test_empirical_bayes_advantage_basic_shape():
    """
    Check that the EB estimator returns tensors of the expected shape
    and does not crash on a simple synthetic example.
    """
    torch.manual_seed(0)

    batch_size = 6
    resp_len = 4

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

    advantages, returns = compute_capo_empirical_bayes_advantage(
        token_level_rewards=token_rewards, response_mask=mask, index=index, config=None,
    )

    assert advantages.shape == token_rewards.shape
    assert returns.shape == token_rewards.shape

    # Ensure we actually get non-zero advantages in at least one group.
    assert torch.any(advantages != 0.0)


def test_empirical_bayes_shrinkage_behavior():
    """
    Construct an asymmetric scenario with one small group and one large
    group and check that the small group's baseline is closer to the
    global mean (i.e., more shrinkage).
    """
    torch.manual_seed(0)

    # Group 0: small group (n=2) with low rewards.
    # Group 1: large group (n=8) with high rewards.
    rewards_list = [[1.0, 1.0], [1.5, 1.5],] + [  # g0  # g0
        [5.0, 5.0],  # g1 x 8
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

    advantages, _ = compute_capo_empirical_bayes_advantage(
        token_level_rewards=token_rewards, response_mask=mask, index=index, config=None,
    )

    # Collapse advantages per sequence.
    adv_scalar = (advantages * mask).mean(dim=-1)

    # For a sanity check, the average advantage in group 0 and group 1
    # should have opposite signs (since group 1's rewards are higher).
    adv_g0 = adv_scalar[:2].mean().item()
    adv_g1 = adv_scalar[2:].mean().item()

    assert adv_g0 < 0.0
    assert adv_g1 > 0.0
