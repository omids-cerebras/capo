# tests/test_eb_lite.py
"""
Tests for EB-lite (Algorithm~\\ref{alg:eb-lite}).

We generate synthetic data with a known length exponent β_star and
check that `eb_lite_fit_beta_and_weights` recovers a β̂ close to β_star.
"""

import math

import torch

from capo.eb_core import eb_lite_fit_beta_and_weights


def _simulate_length_variance_data(
    beta_star: float,
    num_traj: int = 64,
    sigma: float = 1.0,
    seed: int = 0,
):
    g_list = []
    L_list = []

    rng = torch.Generator().manual_seed(seed)

    # Sample lengths from a small discrete set to match realistic regimes.
    possible_L = torch.tensor([16.0, 32.0, 64.0, 128.0, 256.0])
    for _ in range(num_traj):
        L = possible_L[torch.randint(0, len(possible_L), (1,), generator=rng)]
        # Var(g_i | L_i) = σ^2 L_i^{β_star}
        std = sigma * (L ** (0.5 * beta_star))
        noise = torch.normal(0.0, std.item(), generator=rng)
        g_list.append(noise)
        L_list.append(L)

    g = torch.stack(g_list)
    L = torch.stack(L_list)
    return g, L


def test_eb_lite_recovers_beta():
    beta_star = 0.7
    g, L = _simulate_length_variance_data(beta_star=beta_star, num_traj=256, seed=42)

    beta_hat, w, m = eb_lite_fit_beta_and_weights(g=g, L=L, max_iters=20, tol=1e-4)

    # EB-lite is noisy, but β̂ should be in a reasonable neighborhood of β_star.
    assert math.isfinite(beta_hat)
    assert abs(beta_hat - beta_star) < 0.3
    # Weights sum to 1
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
