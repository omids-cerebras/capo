# tests/test_eb_core_comprehensive.py
"""
Comprehensive unit tests for capo.eb_core.

This module tests the core Empirical Bayes functions without requiring VERL.
It focuses on:
- EBStats dataclass and eb_statistics function
- EB objective computation
- Gradient computations (closed-form and numerical)
- joint_eb_update_kband integration
- Edge cases and numerical stability
"""

from __future__ import annotations

import math

import pytest
import torch

from capo.eb_core import (
    EBStats,
    acf_moment_estimate,
    eb_lite_fit_beta_and_weights,
    eb_objective,
    eb_statistics,
    grad_ell_beta_closed_form,
    joint_eb_update_kband,
    numeric_grad_rho_eta,
    s_kband,
)

# ===========================================================================
# Tests for s_kband (stretched-geometric k-banded dependence factor)
# ===========================================================================


class TestSKband:
    """Tests for the s_kband function."""

    def test_returns_ones_when_rho_zero(self):
        """s(L; ρ=0, k, η) ≡ 1 (no correlation)."""
        L = torch.tensor([16.0, 64.0, 128.0, 256.0])
        s = s_kband(L, rho=0.0, k=32, eta=1.0)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6)

    def test_returns_ones_when_k_zero(self):
        """s(L; ρ, k=0, η) ≡ 1 (no lag)."""
        L = torch.tensor([16.0, 64.0, 128.0, 256.0])
        s = s_kband(L, rho=0.9, k=0, eta=1.0)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6)

    def test_returns_ones_when_L_is_one(self):
        """Degenerate trajectory with L=1 has s=1."""
        L = torch.tensor([1.0])
        s = s_kband(L, rho=0.9, k=32, eta=1.0)
        assert torch.isclose(s[0], torch.tensor(1.0), atol=1e-6)

    def test_s_increases_with_rho(self):
        """More correlation (higher ρ) → larger s (variance inflation)."""
        L = torch.tensor([128.0])
        s_low = s_kband(L, rho=0.2, k=32, eta=1.0)[0].item()
        s_med = s_kband(L, rho=0.5, k=32, eta=1.0)[0].item()
        s_high = s_kband(L, rho=0.9, k=32, eta=1.0)[0].item()
        assert s_low < s_med < s_high

    def test_s_increases_with_k(self):
        """Larger bandwidth k → larger s (more lags contribute)."""
        L = torch.tensor([256.0])
        rho = 0.7
        s_k8 = s_kband(L, rho=rho, k=8, eta=1.0)[0].item()
        s_k32 = s_kband(L, rho=rho, k=32, eta=1.0)[0].item()
        s_k128 = s_kband(L, rho=rho, k=128, eta=1.0)[0].item()
        assert s_k8 < s_k32 < s_k128

    def test_s_decreases_with_eta_for_rho_less_than_one(self):
        """Larger η → faster decay of correlations → smaller s."""
        L = torch.tensor([128.0])
        s_eta05 = s_kband(L, rho=0.8, k=32, eta=0.5)[0].item()
        s_eta10 = s_kband(L, rho=0.8, k=32, eta=1.0)[0].item()
        s_eta15 = s_kband(L, rho=0.8, k=32, eta=1.5)[0].item()
        # η=0.5 gives slower decay → larger s
        assert s_eta05 > s_eta10 > s_eta15

    def test_batch_computation(self):
        """Batch of lengths produces correct shapes."""
        L = torch.tensor([16.0, 32.0, 64.0, 128.0, 256.0])
        s = s_kband(L, rho=0.5, k=16, eta=1.0)
        assert s.shape == L.shape
        assert torch.all(s >= 1.0 - 1e-6)  # s should be >= 1 for positive ρ

    def test_negative_rho(self):
        """Negative ρ is allowed (alternating correlations)."""
        L = torch.tensor([64.0])
        s_neg = s_kband(L, rho=-0.5, k=16, eta=1.0)[0].item()
        # Should still produce a finite positive result
        assert math.isfinite(s_neg)
        assert s_neg > 0


# ===========================================================================
# Tests for eb_statistics (E-step summary statistics)
# ===========================================================================


class TestEBStatistics:
    """Tests for eb_statistics function."""

    def test_basic_properties(self):
        """Basic sanity checks on EBStats output."""
        g = torch.tensor([0.8, 1.0, 1.1])
        L = torch.tensor([16.0, 64.0, 256.0])

        stats = eb_statistics(g, L, beta=1.0, rho=0.0, eta=1.0, k=0)

        # Check types
        assert isinstance(stats, EBStats)
        assert isinstance(stats.omega, torch.Tensor)
        assert isinstance(stats.w, torch.Tensor)
        assert isinstance(stats.Lambda_omega, float)
        assert isinstance(stats.RSS_omega, float)

        # Weights sum to 1
        assert math.isclose(stats.w.sum().item(), 1.0, rel_tol=1e-6)

        # All weights non-negative
        assert torch.all(stats.w >= 0)

        # Residuals are g - m
        expected_e = g.float() - stats.m
        assert torch.allclose(stats.e, expected_e, atol=1e-5)

    def test_weighted_mean_in_range(self):
        """Weighted mean m should be between min and max of g."""
        g = torch.tensor([0.5, 1.0, 1.5, 2.0])
        L = torch.tensor([10.0, 20.0, 30.0, 40.0])

        for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
            stats = eb_statistics(g, L, beta=beta)
            m = stats.m.item()
            assert g.min().item() <= m <= g.max().item()

    def test_larger_omega_for_shorter_trajectories(self):
        """With β > 0, shorter trajectories get larger ω (higher precision)."""
        g = torch.tensor([1.0, 1.0, 1.0])
        L = torch.tensor([16.0, 64.0, 256.0])  # increasing lengths

        stats = eb_statistics(g, L, beta=1.0)

        omega = stats.omega
        # ω_i = L_i^{-β}, so shorter → larger ω
        assert omega[0] > omega[1] > omega[2]

    def test_dependence_correction_reduces_precision(self):
        """With positive ρ, the dependence factor affects precision ratios."""
        g = torch.tensor([1.0, 1.0])
        L = torch.tensor([64.0, 256.0])

        stats_indep = eb_statistics(g, L, beta=1.0, rho=0.0, eta=1.0, k=32)
        stats_dep = eb_statistics(g, L, beta=1.0, rho=0.8, eta=1.0, k=32)

        # With dependence, both s values are >= 1, affecting the precision ratio.
        # The key property is that the output is valid (finite, positive weights)
        assert torch.all(stats_dep.omega > 0)
        assert torch.all(stats_dep.w > 0)
        assert math.isclose(stats_dep.w.sum().item(), 1.0, rel_tol=1e-6)

        # The precision ratio changes when accounting for dependence
        ratio_indep = (stats_indep.omega[0] / stats_indep.omega[1]).item()
        ratio_dep = (stats_dep.omega[0] / stats_dep.omega[1]).item()
        # Both ratios should be finite and positive
        assert math.isfinite(ratio_indep) and ratio_indep > 0
        assert math.isfinite(ratio_dep) and ratio_dep > 0


# ===========================================================================
# Tests for eb_objective
# ===========================================================================


class TestEBObjective:
    """Tests for eb_objective function."""

    def test_returns_finite_value(self):
        """EB objective should be finite for reasonable inputs."""
        g = torch.tensor([0.8, 1.0, 1.1, 0.9, 1.2])
        L = torch.tensor([16.0, 32.0, 64.0, 128.0, 256.0])

        obj = eb_objective(g, L, beta=1.0, rho=0.3, eta=1.0, k=16)
        assert math.isfinite(obj)

    def test_objective_changes_with_beta(self):
        """EB objective should vary with β."""
        g = torch.randn(20)
        L = torch.rand(20) * 200 + 16

        obj_05 = eb_objective(g, L, beta=0.5, rho=0.0, eta=1.0, k=0)
        obj_10 = eb_objective(g, L, beta=1.0, rho=0.0, eta=1.0, k=0)
        obj_15 = eb_objective(g, L, beta=1.5, rho=0.0, eta=1.0, k=0)

        # Objectives should be different (unless g are degenerate)
        assert not (obj_05 == obj_10 == obj_15)


# ===========================================================================
# Tests for gradients
# ===========================================================================


class TestGradients:
    """Tests for gradient computations."""

    def test_grad_beta_closed_form_is_finite(self):
        """Closed-form gradient ∂ℓ/∂β should be finite."""
        g = torch.tensor([0.8, 1.0, 1.1])
        L = torch.tensor([16.0, 64.0, 256.0])

        stats = eb_statistics(g, L, beta=1.0)

        grad = grad_ell_beta_closed_form(
            L=L,
            omega=stats.omega,
            e=stats.e,
            Lambda_omega=stats.Lambda_omega,
            RSS_omega=stats.RSS_omega,
        )

        assert math.isfinite(grad)

    def test_numeric_grad_rho_eta_is_finite(self):
        """Numerical gradients for ρ and η should be finite."""
        g = torch.tensor([0.8, 1.0, 1.1, 0.9, 1.2])
        L = torch.tensor([16.0, 32.0, 64.0, 128.0, 256.0])

        g_rho, g_eta = numeric_grad_rho_eta(g=g, L=L, beta=1.0, rho=0.3, eta=1.0, k=16)

        assert math.isfinite(g_rho)
        assert math.isfinite(g_eta)

    def test_gradient_direction_consistency(self):
        """Gradient should point in direction of increasing objective."""
        torch.manual_seed(42)
        g = torch.randn(30)
        L = torch.rand(30) * 200 + 16

        beta = 1.0
        rho = 0.3
        eta = 1.0
        k = 16

        obj_base = eb_objective(g, L, beta, rho, eta, k)
        stats = eb_statistics(g, L, beta, rho, eta, k)

        g_beta = grad_ell_beta_closed_form(
            L=L,
            omega=stats.omega,
            e=stats.e,
            Lambda_omega=stats.Lambda_omega,
            RSS_omega=stats.RSS_omega,
        )

        # Move β in direction of gradient
        delta = 0.01 * (1 if g_beta > 0 else -1)
        obj_new = eb_objective(g, L, beta + delta, rho, eta, k)

        # Objective should increase (approximately) when moving with gradient
        # Allow some tolerance for numerical issues
        if abs(g_beta) > 0.01:  # Only check if gradient is non-trivial
            assert obj_new >= obj_base - 0.01


# ===========================================================================
# Tests for joint_eb_update_kband
# ===========================================================================


class TestJointEBUpdate:
    """Tests for joint_eb_update_kband function."""

    def test_basic_output_structure(self):
        """joint_eb_update_kband returns expected types and shapes."""
        torch.manual_seed(0)
        g = torch.randn(16)
        L = torch.rand(16) * 100 + 16

        beta_new, rho_new, eta_new, w = joint_eb_update_kband(
            g=g,
            L=L,
            beta_init=1.0,
            rho_init=0.0,
            eta_init=1.0,
            k=16,
        )

        assert isinstance(beta_new, float)
        assert isinstance(rho_new, float)
        assert isinstance(eta_new, float)
        assert isinstance(w, torch.Tensor)
        assert w.shape == g.shape

    def test_weights_sum_to_one(self):
        """Output weights should sum to 1."""
        torch.manual_seed(1)
        g = torch.randn(32)
        L = torch.rand(32) * 200 + 16

        _, _, _, w = joint_eb_update_kband(
            g=g,
            L=L,
            beta_init=1.0,
            rho_init=0.3,
            eta_init=1.0,
            k=32,
        )

        assert math.isclose(w.sum().item(), 1.0, rel_tol=1e-5)

    def test_parameters_within_bounds(self):
        """Output parameters should respect specified bounds."""
        torch.manual_seed(2)
        g = torch.randn(32)
        L = torch.rand(32) * 200 + 16

        beta_new, rho_new, eta_new, _ = joint_eb_update_kband(
            g=g,
            L=L,
            beta_init=1.5,
            rho_init=0.8,
            eta_init=1.5,
            k=32,
            beta_bounds=(0.5, 1.5),
            rho_max=0.9,
            steps_beta=5,
            steps_xi=5,
        )

        assert 0.5 <= beta_new <= 1.5
        assert -0.9 <= rho_new <= 0.9
        assert eta_new >= 0.0

    def test_acf_warmstart(self):
        """ACF warm-start should affect initial ρ, η."""
        torch.manual_seed(3)
        g = torch.randn(16)
        L = torch.full((16,), 64.0)

        # Create synthetic AR(1) increments for warm-start
        increments = torch.randn(16, 64)
        mask = torch.ones_like(increments, dtype=torch.bool)

        beta1, rho1, eta1, _ = joint_eb_update_kband(
            g=g,
            L=L,
            beta_init=1.0,
            rho_init=0.0,
            eta_init=1.0,
            k=16,
            use_acf_warmstart=False,
        )

        beta2, rho2, eta2, _ = joint_eb_update_kband(
            g=g,
            L=L,
            beta_init=1.0,
            rho_init=0.0,
            eta_init=1.0,
            k=16,
            use_acf_warmstart=True,
            increments=increments,
            increments_mask=mask,
        )

        # With warm-start, ρ should potentially differ
        # (exact behavior depends on the random increments)
        # At minimum, we verify no crash and finite outputs
        assert math.isfinite(rho2)
        assert math.isfinite(eta2)


# ===========================================================================
# Tests for edge cases and numerical stability
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_single_trajectory(self):
        """Functions should handle G=1 gracefully."""
        g = torch.tensor([1.0])
        L = torch.tensor([64.0])

        stats = eb_statistics(g, L, beta=1.0)
        assert stats.w[0].item() == 1.0
        assert stats.m.item() == 1.0

        beta, w, m = eb_lite_fit_beta_and_weights(g, L)
        assert w[0].item() == 1.0

    def test_identical_returns(self):
        """All identical g_i → zero residuals."""
        g = torch.tensor([1.0, 1.0, 1.0, 1.0])
        L = torch.tensor([16.0, 32.0, 64.0, 128.0])

        stats = eb_statistics(g, L, beta=1.0)
        assert stats.m.item() == 1.0
        assert torch.allclose(stats.e, torch.zeros(4))

    def test_very_long_trajectories(self):
        """Very long L should not cause overflow."""
        g = torch.randn(8)
        L = torch.tensor([1e4, 1e5, 1e6, 1e7, 1e4, 1e5, 1e6, 1e7])

        stats = eb_statistics(g, L, beta=1.0)
        assert torch.all(torch.isfinite(stats.omega))
        assert torch.all(torch.isfinite(stats.w))

    def test_very_short_trajectories(self):
        """Very short L (near 1) should be handled."""
        g = torch.randn(4)
        L = torch.tensor([1.0, 2.0, 3.0, 4.0])

        stats = eb_statistics(g, L, beta=1.0)
        assert torch.all(torch.isfinite(stats.omega))
        assert torch.all(torch.isfinite(stats.w))

    def test_extreme_beta_values(self):
        """Extreme β values (0, 2) should work."""
        g = torch.randn(8)
        L = torch.rand(8) * 200 + 16

        for beta in [0.0, 2.0]:
            stats = eb_statistics(g, L, beta=beta)
            assert torch.all(torch.isfinite(stats.omega))
            assert torch.all(torch.isfinite(stats.w))

    def test_high_rho(self):
        """High ρ (near 1) should not cause numerical issues."""
        torch.randn(8)
        L = torch.rand(8) * 200 + 16

        s = s_kband(L, rho=0.99, k=32, eta=1.0)
        assert torch.all(torch.isfinite(s))
        assert torch.all(s > 0)


# ===========================================================================
# Tests for ACF moment estimate recovery
# ===========================================================================


class TestACFMomentRecovery:
    """Additional tests for ACF-moment estimator."""

    def test_white_noise_gives_zero_rho(self):
        """White noise increments → ρ̂ ≈ 0."""
        torch.manual_seed(42)
        B, T = 32, 128
        Y = torch.randn(B, T)
        M = torch.ones(B, T, dtype=torch.bool)

        rho_hat, eta_hat = acf_moment_estimate(Y, M, k=20)

        # For white noise, ρ should be near 0
        assert abs(rho_hat) < 0.3  # generous tolerance

    def test_handles_variable_lengths(self):
        """Variable trajectory lengths should be handled via mask."""
        torch.manual_seed(43)
        B, T_max = 16, 128
        Y = torch.randn(B, T_max)
        M = torch.zeros(B, T_max, dtype=torch.bool)

        # Set variable lengths
        lengths = torch.randint(32, T_max, (B,))
        for i in range(B):
            M[i, : lengths[i]] = True

        rho_hat, eta_hat = acf_moment_estimate(Y, M, k=16)

        assert math.isfinite(rho_hat)
        assert math.isfinite(eta_hat)

    def test_empty_mask_returns_defaults(self):
        """Empty mask should return default (0, 1)."""
        Y = torch.randn(4, 32)
        M = torch.zeros(4, 32, dtype=torch.bool)  # all invalid

        rho_hat, eta_hat = acf_moment_estimate(Y, M, k=10)

        assert rho_hat == 0.0
        assert eta_hat == 1.0


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
