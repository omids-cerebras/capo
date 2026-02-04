# tests/test_acf_moment.py
"""
Tests for the ACF-moment estimator (Algorithm~\\ref{alg:acf-moment}).

We generate synthetic AR(1) increments with known ρ_star and η_star=1
and check that `acf_moment_estimate` recovers a reasonable ρ̂.
"""

import math

import torch

from capo.eb_core import acf_moment_estimate


def _simulate_ar1_increments(
    rho_star: float,
    num_traj: int = 64,
    length: int = 256,
    sigma_eps: float = 1.0,
    seed: int = 0,
):
    rng = torch.Generator().manual_seed(seed)
    Y = torch.zeros(num_traj, length, dtype=torch.float32)
    M = torch.zeros(num_traj, length, dtype=torch.bool)

    for i in range(num_traj):
        # Simple AR(1): Y_{τ} = ρ Y_{τ-1} + ε_τ, ε_τ ~ N(0, σ^2)
        eps = torch.normal(mean=0.0, std=sigma_eps, size=(length,), generator=rng,)
        y = torch.zeros(length, dtype=torch.float32)
        for t in range(1, length):
            y[t] = rho_star * y[t - 1] + eps[t]
        Y[i] = y
        M[i, :length] = True

    return Y, M


def test_acf_moment_estimate_ar1():
    rho_star = 0.7
    Y, M = _simulate_ar1_increments(
        rho_star=rho_star, num_traj=64, length=256, seed=123
    )

    rho_hat, eta_hat = acf_moment_estimate(increments=Y, mask=M, k=20)

    # ρ̂ should be in the right ballpark; η̂ should be ≈ 1 for AR(1).
    assert math.isfinite(rho_hat)
    assert 0.0 <= rho_hat <= 0.99
    assert abs(rho_hat - rho_star) < 0.3

    assert math.isfinite(eta_hat)
    assert eta_hat >= 0.0
