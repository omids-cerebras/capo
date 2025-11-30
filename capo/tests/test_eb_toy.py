# tests/test_eb_toy.py
"""
Numerical check of the toy example in the EB--CAPO write-up (§EB examples).

We consider G=3 trajectories with lengths

    L_1 = 16, L_2 = 64, L_3 = 256,

and scalarized returns

    g_1 = 0.8, g_2 = 1.0, g_3 = 1.1.

For a length-only variance model v_i ∝ L_i^{β}, precisions are
ω_i(β) = L_i^{-β}. This test reproduces the normalized weights

    w_i(β) = ω_i / Σ_j ω_j

and aggregated estimate

    m(β) = Σ_i w_i(β) g_i

for β=1 (ΔL-style) and β=0.5 (EB-favored when longer rollouts are
less noisy), matching the examples in the text.
"""

import math

import torch


def _weights_and_mean(L, g, beta: float):
    L = torch.tensor(L, dtype=torch.float64)
    g = torch.tensor(g, dtype=torch.float64)
    omega = L.pow(-beta)
    Lam = omega.sum()
    w = omega / Lam
    m = (w * g).sum()
    return w, m


def test_toy_example_beta_1_and_beta_half():
    L = [16.0, 64.0, 256.0]
    g = [0.8, 1.0, 1.1]

    w1, m1 = _weights_and_mean(L, g, beta=1.0)
    w2, m2 = _weights_and_mean(L, g, beta=0.5)

    # Expected from exact computation (no rounding):
    expected_w1 = torch.tensor(
        [0.7619047619047619, 0.19047619047619047, 0.047619047619047616],
        dtype=torch.float64,
    )
    expected_m1 = 0.8523809523809525

    expected_w2 = torch.tensor(
        [0.5714285714285714, 0.2857142857142857, 0.14285714285714285],
        dtype=torch.float64,
    )
    expected_m2 = 0.9

    assert torch.allclose(w1, expected_w1, atol=1e-10)
    assert abs(m1.item() - expected_m1) < 1e-10

    assert torch.allclose(w2, expected_w2, atol=1e-10)
    assert abs(m2.item() - expected_m2) < 1e-10
