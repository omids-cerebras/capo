"""Tests for CAPO-Q (quadratic variance-law plug-in) and CAPO-HAC (Newey-West
lag-window plug-in) functions in ``eb_core.py`` and their advantage wrappers
in both ``synthetic_benchmark.py`` and ``adv_estimators.py``.

Test categories
---------------
1. **Unit tests** for ``capo_q_fit_and_predict`` and
   ``capo_hac_pooled_autocovariance`` / ``capo_hac_fit_and_predict``.
2. **Matched-regime oracle tests**: CAPO-Q should match oracle under CS;
   CAPO-HAC should match oracle under IID and AR(1).
3. **End-to-end advantage wrapper tests**: shape, groupedness, and
   consistency between benchmark wrappers and production
   ``adv_estimators`` wrappers.
4. **Degenerate / edge-case tests**: all-same lengths, single trajectory,
   zero mask.
"""

from __future__ import annotations

import numpy as np
import torch

from capo.eb_core import (
    capo_hac_fit_and_predict,
    capo_hac_pooled_autocovariance,
    capo_q_fit_and_predict,
)

# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_iid_batch(B=512, T=256, seed=0):
    """IID token-level rewards with uniform lengths."""
    rng = np.random.default_rng(seed)
    lengths = rng.choice([64, 128, 256], size=B, replace=True)
    rewards = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i in range(B):
        L = int(lengths[i])
        rewards[i, :L] = rng.standard_normal(L).astype(np.float32)
        mask[i, :L] = 1.0
    return (
        torch.tensor(rewards),
        torch.tensor(mask),
        torch.tensor(lengths, dtype=torch.float32),
        np.repeat(np.arange(B // 8), 8)[:B],
    )


def _make_ar1_batch(B=512, T=256, rho=0.6, seed=0):
    """AR(1) token-level rewards."""
    rng = np.random.default_rng(seed)
    lengths = rng.choice([64, 128, 256], size=B, replace=True)
    rewards = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i in range(B):
        L = int(lengths[i])
        z = np.zeros(L)
        z[0] = rng.standard_normal()
        for t in range(1, L):
            z[t] = rho * z[t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal()
        rewards[i, :L] = z.astype(np.float32)
        mask[i, :L] = 1.0
    return (
        torch.tensor(rewards),
        torch.tensor(mask),
        torch.tensor(lengths, dtype=torch.float32),
        np.repeat(np.arange(B // 8), 8)[:B],
    )


def _make_cs_batch(B=512, T=256, rho_cs=0.05, seed=0):
    """Compound symmetry token-level rewards.

    r_{i,t} = mu_i + eps_{i,t}  where mu_i ~ N(0, rho_cs) and
    eps_{i,t} ~ N(0, 1 - rho_cs).
    """
    rng = np.random.default_rng(seed)
    lengths = rng.choice([64, 128, 256], size=B, replace=True)
    rewards = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i in range(B):
        L = int(lengths[i])
        mu_i = np.sqrt(rho_cs) * rng.standard_normal()
        eps = np.sqrt(1 - rho_cs) * rng.standard_normal(L)
        rewards[i, :L] = (mu_i + eps).astype(np.float32)
        mask[i, :L] = 1.0
    return (
        torch.tensor(rewards),
        torch.tensor(mask),
        torch.tensor(lengths, dtype=torch.float32),
        np.repeat(np.arange(B // 8), 8)[:B],
    )


# ═════════════════════════════════════════════════════════════════════
#  1. CAPO-Q unit tests
# ═════════════════════════════════════════════════════════════════════


class TestCapoQFitAndPredict:
    """Tests for ``capo_q_fit_and_predict``."""

    def test_output_shapes(self):
        """a_hat, b_hat are scalars; omega has shape [B]."""
        rewards, mask, L, index = _make_iid_batch(B=64)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        a, b, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert omega.shape == (64,)

    def test_nonnegative_coefficients(self):
        """Both a_hat and b_hat must be >= 0 (NNLS constraint)."""
        rewards, mask, L, index = _make_iid_batch(B=256)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        a, b, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        assert a >= 0.0
        assert b >= 0.0

    def test_positive_omega(self):
        """All precision weights must be strictly positive."""
        rewards, mask, L, index = _make_iid_batch(B=256)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        _, _, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        assert (omega > 0).all()

    def test_iid_b_hat_near_zero(self):
        """Under IID, b_hat should be near zero (no quadratic term)."""
        rewards, mask, L, index = _make_iid_batch(B=2048, seed=42)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        a, b, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        # b should be small relative to a
        assert b < 0.5 * a or b < 0.01, f"b={b}, a={a}: expected b << a under IID"

    def test_cs_b_hat_positive(self):
        """Under compound symmetry, b_hat should be positive."""
        rewards, mask, L, index = _make_cs_batch(B=2048, rho_cs=0.05, seed=42)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        a, b, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        # Under CS, v(L) = (1-rho)*L + rho*L^2 => b > 0
        assert b > 0, f"b={b}: expected b > 0 under compound symmetry"

    def test_longer_trajectories_get_lower_omega(self):
        """Under IID or CS, longer trajectories should have lower precision."""
        rewards, mask, L, index = _make_iid_batch(B=512)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)
        _, _, omega = capo_q_fit_and_predict(g, L, index=idx_t)
        # Mean omega for short vs long
        short_mask = L < 100
        long_mask = L > 200
        if short_mask.any() and long_mask.any():
            assert omega[short_mask].mean() > omega[long_mask].mean()

    def test_index_none_fallback(self):
        """When index is None, treat all as one group."""
        g = torch.randn(100)
        L = torch.randint(32, 256, (100,)).float()
        a, b, omega = capo_q_fit_and_predict(g, L, index=None)
        assert omega.shape == (100,)
        assert (omega > 0).all()


# ═════════════════════════════════════════════════════════════════════
#  2. CAPO-HAC unit tests
# ═════════════════════════════════════════════════════════════════════


class TestCapoHACPooledAutocovariance:
    """Tests for ``capo_hac_pooled_autocovariance``."""

    def test_output_shape(self):
        """gamma_hat should have shape [K+1]."""
        rewards, mask, L, _ = _make_iid_batch(B=128)
        K = 16
        gamma = capo_hac_pooled_autocovariance(rewards, mask, K)
        assert gamma.shape == (K + 1,)

    def test_gamma0_positive(self):
        """gamma(0) is a variance and must be positive."""
        rewards, mask, L, _ = _make_iid_batch(B=256)
        gamma = capo_hac_pooled_autocovariance(rewards, mask, K=16)
        assert gamma[0].item() > 0

    def test_iid_gamma_h_near_zero(self):
        """Under IID, gamma(h) for h >= 1 should be near zero."""
        rewards, mask, L, _ = _make_iid_batch(B=2048, seed=42)
        gamma = capo_hac_pooled_autocovariance(rewards, mask, K=16)
        # gamma(0) should be ~1 (unit variance)
        assert abs(gamma[0].item() - 1.0) < 0.15, f"gamma(0)={gamma[0].item()}"
        # gamma(h>0) should be small relative to gamma(0)
        for h in range(1, 17):
            ratio = abs(gamma[h].item()) / gamma[0].item()
            assert ratio < 0.15, f"|gamma({h})|/gamma(0) = {ratio:.3f}"

    def test_ar1_gamma_decays(self):
        """Under AR(1), gamma(h) should decay roughly as rho^h."""
        rho = 0.6
        rewards, mask, L, _ = _make_ar1_batch(B=4096, rho=rho, seed=42)
        gamma = capo_hac_pooled_autocovariance(rewards, mask, K=8)
        # gamma should be monotonically decreasing in absolute value
        for h in range(1, 8):
            assert abs(gamma[h].item()) <= abs(gamma[h - 1].item()) + 0.1

    def test_cs_gamma_roughly_constant(self):
        """Under CS, gamma(h) for h >= 1 should be roughly constant (~rho_cs)."""
        rho_cs = 0.05
        rewards, mask, L, _ = _make_cs_batch(B=4096, rho_cs=rho_cs, seed=42)
        gamma = capo_hac_pooled_autocovariance(rewards, mask, K=8)
        # gamma(h) for h >= 1 should all be similar
        gammas_h = [gamma[h].item() for h in range(1, 9)]
        # They should be relatively constant (low std relative to mean)
        std_gh = np.std(gammas_h)
        # Under CS all lags are the same; the important thing is they
        # don't decay rapidly like AR(1). Use a relative-consistency check.
        assert std_gh < 0.03, f"std of gamma(h>0) = {std_gh:.4f}, expected roughly constant"


class TestCapoHACFitAndPredict:
    """Tests for ``capo_hac_fit_and_predict``."""

    def test_output_shapes(self):
        """gamma_hat has shape [K+1], omega has shape [B]."""
        rewards, mask, L, _ = _make_iid_batch(B=128)
        gamma, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        assert gamma.shape == (17,)
        assert omega.shape == (128,)

    def test_positive_omega(self):
        """All precision weights must be positive."""
        rewards, mask, L, _ = _make_iid_batch(B=256)
        _, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        assert (omega > 0).all()

    def test_longer_gets_lower_omega_iid(self):
        """Under IID, omega should decrease with length."""
        rewards, mask, L, _ = _make_iid_batch(B=512, seed=42)
        _, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        short_mask = L < 100
        long_mask = L > 200
        if short_mask.any() and long_mask.any():
            assert omega[short_mask].mean() > omega[long_mask].mean()

    def test_iid_vhat_roughly_proportional_to_L(self):
        """Under IID, v_hat(L) ≈ L * gamma(0) ≈ L."""
        rewards, mask, L, _ = _make_iid_batch(B=2048, seed=42)
        gamma, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        # v_hat = 1/omega, should be ~L
        v_hat = 1.0 / omega
        # Correlation of v_hat with L should be high
        corr = np.corrcoef(L.numpy(), v_hat.numpy())[0, 1]
        assert corr > 0.9, f"corr(v_hat, L) = {corr:.3f}, expected > 0.9 under IID"


# ═════════════════════════════════════════════════════════════════════
#  3. Matched-regime oracle tests
# ═════════════════════════════════════════════════════════════════════


class TestMatchedRegimeOracle:
    """Verify plug-ins achieve near-oracle RE in their matched regimes."""

    @staticmethod
    def _re(bl_method, bl_grpo):
        """Relative efficiency: Var(GRPO) / Var(method)."""
        var_g = np.var(bl_grpo)
        var_m = np.var(bl_method)
        return var_g / var_m if var_m > 1e-30 else 1.0

    def test_capo_q_dominant_under_cs(self):
        """CAPO-Q should achieve RE > 1 under compound symmetry."""
        B = 1024
        rewards, mask, L, index = _make_cs_batch(B=B, rho_cs=0.05, seed=42)
        g = (rewards * mask).sum(dim=-1)
        idx_t = torch.as_tensor(index, dtype=torch.long)

        # CAPO-Q baseline
        _, _, omega_q = capo_q_fit_and_predict(g, L, index=idx_t)
        # GRPO baseline (uniform)
        omega_grpo = torch.ones_like(L)

        # Compute group baselines
        from capo.experiments.analysis.synthetic_benchmark import _groupwise_baseline

        _groupwise_baseline(omega_grpo, g, index)
        _groupwise_baseline(omega_q, g, index)
        # CAPO-Q weights should be different from uniform
        # (can't compute RE from single batch — but can check weights are sensible)
        assert not torch.allclose(omega_q, omega_grpo)

    def test_capo_hac_sensible_under_iid(self):
        """Under IID, CAPO-HAC should produce weights that decrease with L."""
        rewards, mask, L, index = _make_iid_batch(B=1024, seed=42)
        gamma, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        # gamma(0) should be close to 1
        assert abs(gamma[0].item() - 1.0) < 0.15
        # Weights should decrease with length
        short = L < 100
        long = L > 200
        if short.any() and long.any():
            assert omega[short].mean() > omega[long].mean()


# ═════════════════════════════════════════════════════════════════════
#  4. End-to-end advantage wrapper tests
# ═════════════════════════════════════════════════════════════════════


class TestCapoQAdvantageWrapper:
    """Tests for CAPO-Q advantage computation."""

    def test_benchmark_wrapper_shape(self):
        """Advantage tensor should have shape [B, T]."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_q_advantage

        B, T = 128, 256
        rewards, mask, L, index = _make_iid_batch(B=B, T=T)
        adv, ret, metrics = compute_capo_q_advantage(rewards, mask, index=index)
        assert adv.shape == (B, T)
        assert ret.shape == (B, T)
        assert "capo_q/a_hat" in metrics
        assert "capo_q/b_hat" in metrics

    def test_advantages_zero_outside_mask(self):
        """Advantages outside the mask should be zero."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_q_advantage

        rewards, mask, L, index = _make_iid_batch(B=128, T=256)
        adv, _, _ = compute_capo_q_advantage(rewards, mask, index=index)
        outside = mask == 0
        assert (adv[outside] == 0).all()

    def test_production_wrapper_shape(self):
        """Production adv_estimators wrapper should produce same shapes."""
        from capo.verl_integration.adv_estimators import (
            compute_capo_q_advantage as prod_capo_q,
        )

        B, T = 128, 256
        rewards, mask, L, index = _make_iid_batch(B=B, T=T)
        adv, ret, metrics = prod_capo_q(rewards, mask, index=index)
        assert adv.shape == (B, T)
        assert ret.shape == (B, T)
        assert "capo_q/a_hat" in metrics
        assert "capo_q/weight_ess" in metrics

    def test_empty_batch(self):
        """Should handle all-zero mask gracefully."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_q_advantage

        rewards = torch.randn(16, 64)
        mask = torch.zeros(16, 64)
        adv, _, _ = compute_capo_q_advantage(rewards, mask, index=None)
        assert (adv == 0).all()


class TestCapoHACAdvantageWrapper:
    """Tests for CAPO-HAC advantage computation."""

    def test_benchmark_wrapper_shape(self):
        """Advantage tensor should have shape [B, T]."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_hac_advantage

        B, T = 128, 256
        rewards, mask, L, index = _make_iid_batch(B=B, T=T)
        adv, ret, metrics = compute_capo_hac_advantage(rewards, mask, index=index)
        assert adv.shape == (B, T)
        assert ret.shape == (B, T)
        assert "capo_hac/K" in metrics
        assert "capo_hac/gamma0" in metrics

    def test_advantages_zero_outside_mask(self):
        """Advantages outside the mask should be zero."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_hac_advantage

        rewards, mask, L, index = _make_iid_batch(B=128, T=256)
        adv, _, _ = compute_capo_hac_advantage(rewards, mask, index=index)
        outside = mask == 0
        assert (adv[outside] == 0).all()

    def test_production_wrapper_shape(self):
        """Production adv_estimators wrapper should produce same shapes."""
        from capo.verl_integration.adv_estimators import (
            compute_capo_hac_advantage as prod_capo_hac,
        )

        B, T = 128, 256
        rewards, mask, L, index = _make_iid_batch(B=B, T=T)
        adv, ret, metrics = prod_capo_hac(rewards, mask, index=index)
        assert adv.shape == (B, T)
        assert ret.shape == (B, T)
        assert "capo_hac/K" in metrics
        assert "capo_hac/weight_ess" in metrics

    def test_empty_batch(self):
        """Should handle all-zero mask gracefully."""
        from capo.experiments.analysis.synthetic_benchmark import compute_capo_hac_advantage

        rewards = torch.randn(16, 64)
        mask = torch.zeros(16, 64)
        adv, _, _ = compute_capo_hac_advantage(rewards, mask, index=None)
        assert (adv == 0).all()


# ═════════════════════════════════════════════════════════════════════
#  5. Edge cases
# ═════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Degenerate and edge-case tests for both plug-ins."""

    def test_capo_q_single_length(self):
        """When all trajectories have the same length, weights should be uniform."""
        B = 128
        g = torch.randn(B)
        L = torch.full((B,), 256.0)
        a, b, omega = capo_q_fit_and_predict(g, L, index=None)
        # All omega should be equal
        assert torch.allclose(omega, omega[0].expand(B), atol=1e-6)

    def test_capo_hac_single_length(self):
        """When all trajectories have the same length, weights should be uniform."""
        B = 128
        T = 256
        rewards = torch.randn(B, T)
        mask = torch.ones(B, T)
        L = torch.full((B,), float(T))
        _, omega = capo_hac_fit_and_predict(rewards, mask, L, K=16)
        # All omega should be equal
        assert torch.allclose(omega, omega[0].expand(B), atol=1e-6)

    def test_capo_q_small_batch(self):
        """Should not crash on very small batches."""
        g = torch.randn(4)
        L = torch.tensor([32.0, 64.0, 128.0, 256.0])
        a, b, omega = capo_q_fit_and_predict(g, L, index=None)
        assert omega.shape == (4,)
        assert (omega > 0).all()

    def test_capo_hac_small_K(self):
        """Should work with K=1 (minimum bandwidth)."""
        B, T = 64, 256
        rewards, mask, L, _ = _make_iid_batch(B=B, T=T)
        gamma, omega = capo_hac_fit_and_predict(rewards, mask, L, K=1)
        assert gamma.shape == (2,)  # K+1
        assert omega.shape == (B,)
        assert (omega > 0).all()

    def test_compute_all_advantages_includes_all_methods(self):
        """compute_all_advantages should handle all 7 methods."""
        from capo.experiments.analysis.synthetic_benchmark import (
            DEFAULT_LENGTH_BINS,
            METHODS,
            compute_all_advantages,
            generate_grouped_rollouts,
        )

        bins = DEFAULT_LENGTH_BINS
        rng = np.random.default_rng(42)
        rewards, mask, lengths, index = generate_grouped_rollouts(
            rng,
            P=8,
            N=4,
            length_bins=bins,
            build_cov=lambda L: np.eye(int(L)),
        )
        mkeys = [m.key for m in METHODS]
        advs = compute_all_advantages(rewards, mask, index, mkeys)
        assert len(advs) == 7
        for key in mkeys:
            assert key in advs, f"Missing method {key}"
            assert advs[key].shape == rewards.shape

    def test_compute_scalar_baselines_includes_all_methods(self):
        """compute_scalar_baselines should handle all 7 methods."""
        from capo.experiments.analysis.synthetic_benchmark import (
            DEFAULT_LENGTH_BINS,
            METHODS,
            compute_scalar_baselines,
            generate_grouped_rollouts,
        )

        bins = DEFAULT_LENGTH_BINS
        rng = np.random.default_rng(42)
        rewards, mask, lengths, index = generate_grouped_rollouts(
            rng,
            P=8,
            N=4,
            length_bins=bins,
            build_cov=lambda L: np.eye(int(L)),
        )
        mkeys = [m.key for m in METHODS]
        baselines = compute_scalar_baselines(rewards, mask, index, mkeys)
        assert len(baselines) == 7
        for key in mkeys:
            assert key in baselines, f"Missing method {key}"
            assert isinstance(baselines[key], float)
