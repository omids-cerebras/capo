# tests/test_eb_lite_and_full.py
import types

import pytest
import torch

pytest.importorskip("verl")

from capo.verl_integration.adv_estimators import (
    eb_lite_fit_beta_and_weights,
    compute_capo_eb_lite_advantage,
    compute_capo_eb_full_advantage,
)


def _make_config(norm: bool = True):
    cfg = types.SimpleNamespace()
    cfg.norm_adv_by_std_in_grpo = norm
    return cfg


def test_eb_lite_fit_beta_and_weights_basic_properties():
    # Simple synthetic (g_i, L_i)
    g = torch.tensor([0.8, 1.0, 1.1], dtype=torch.float32)
    L = torch.tensor([16.0, 64.0, 256.0], dtype=torch.float32)

    beta_hat, w, m = eb_lite_fit_beta_and_weights(g=g, L=L, eps=1e-8, max_iters=5, tol=1e-5)

    assert isinstance(beta_hat, float)
    assert torch.isfinite(torch.tensor(beta_hat))

    assert w.shape == L.shape
    assert torch.all(w >= 0)
    assert abs(w.sum().item() - 1.0) < 1e-6

    # Aggregator m should be in [min g_i, max g_i].
    assert g.min().item() <= m.item() <= g.max().item()


def test_eb_lite_weights_invariant_to_global_length_scaling():
    g = torch.tensor([0.8, 1.0, 1.1], dtype=torch.float32)
    L = torch.tensor([16.0, 64.0, 256.0], dtype=torch.float32)
    L_scaled = 2.0 * L

    beta_hat_1, w_1, _ = eb_lite_fit_beta_and_weights(g=g, L=L, eps=1e-8, max_iters=5, tol=1e-5)
    beta_hat_2, w_2, _ = eb_lite_fit_beta_and_weights(
        g=g, L=L_scaled, eps=1e-8, max_iters=5, tol=1e-5
    )

    # Global rescaling L -> cL should not change weights in the
    # x_i ∝ L_i^{-β} family; EB-lite should respect that approximately.
    assert torch.allclose(w_1, w_2, atol=1e-4)


def test_compute_capo_eb_lite_advantage_constant_within_trajectory():
    B, T = 4, 5
    rewards = torch.randn(B, T)
    mask = torch.ones_like(rewards, dtype=torch.long)

    adv, returns = compute_capo_eb_lite_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=None,
        config=_make_config(norm=False),  # EB-lite already normalizes across trajectories
    )

    assert adv.shape == rewards.shape
    assert returns.shape == rewards.shape

    # EB-lite produces scalar A_i per trajectory, broadcast over tokens.
    for i in range(B):
        valid = mask[i] > 0
        assert torch.allclose(
            adv[i, valid], adv[i, valid][0].expand_as(adv[i, valid]), atol=1e-6
        )


def test_compute_capo_eb_full_advantage_runs_and_has_reasonable_shape():
    B, T = 6, 10
    rewards = torch.randn(B, T)
    mask = torch.ones_like(rewards, dtype=torch.long)

    # Use same rewards as "increments" for a cheap ACF-moment run.
    adv, returns = compute_capo_eb_full_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=None,
        config=_make_config(norm=True),
        increments=rewards,
        increments_mask=mask,
        beta_steps=1,
        xi_steps=1,
    )

    assert adv.shape == rewards.shape
    assert returns.shape == rewards.shape

    valid = mask > 0
    assert torch.all(torch.isfinite(adv[valid]))
    assert torch.all(torch.isfinite(returns[valid]))
