# tests/test_adv_capo_base.py
import types

import pytest
import torch

pytest.importorskip("verl")  # ensure VERL is installed before importing adv_estimators

from capo.verl_integration.adv_estimators import compute_capo_advantage


def _make_config(norm: bool = True):
    cfg = types.SimpleNamespace()
    cfg.norm_adv_by_std_in_grpo = norm
    return cfg


def test_capo_advantage_shape_and_masking():
    B, T = 3, 4
    rewards = torch.arange(B * T, dtype=torch.float32).view(B, T)
    # Mark last token of last sequence as padding.
    mask = torch.ones_like(rewards, dtype=torch.long)
    mask[-1, -1] = 0

    adv, returns = compute_capo_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=None,
        config=_make_config(norm=True),
    )

    assert adv.shape == rewards.shape
    assert returns.shape == rewards.shape

    # Padding positions must have zero advantage and zero return.
    assert torch.all(adv[mask == 0] == 0)
    assert torch.all(returns[mask == 0] == 0)


def test_capo_advantage_zero_mean_unit_std_on_valid_tokens():
    B, T = 4, 5
    rewards = torch.randn(B, T)
    mask = torch.ones_like(rewards, dtype=torch.long)

    adv, _ = compute_capo_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=None,
        config=_make_config(norm=True),
    )

    valid = mask > 0
    mean_adv = adv[valid].mean().item()
    std_adv = adv[valid].std().item()

    # With enough tokens this should be close to 0 mean and unit std.
    assert abs(mean_adv) < 1e-5
    assert abs(std_adv - 1.0) < 1e-5
