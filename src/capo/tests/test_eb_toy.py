# tests/test_eb_toy.py
"""
Tests for EB weighting on the 3-trajectory toy example from the paper.

We verify that, for the length-only case s(L; ξ) ≡ 1, the normalized
weights w_i ∝ L_i^{-β} and the aggregated mean m(β) reproduce the
values given in Section 4:

- Case β = 1 ("ΔL" rule),
- Case β = 0.5 (EB-favored when longer rollouts are less noisy).
"""

import math

import torch

from capo.eb_core import kband_weights


def _compute_m_from_weights(w: torch.Tensor, g: torch.Tensor) -> float:
    """Helper: m(β) = Σ_i w_i g_i."""
    return float((w * g).sum().item())


def test_toy_example_beta_1():
    # L_1=16, L_2=64, L_3=256, g=[0.8, 1.0, 1.1]
    L = torch.tensor([16.0, 64.0, 256.0])
    g = torch.tensor([0.8, 1.0, 1.1])

    # β = 1, s(L) ≡ 1 ⇒ k=0 or ρ=0 gives s_i=1
    s, w = kband_weights(L=L, beta=1.0, rho=0.0, eta=1.0, k=0)

    # Weights should sum to 1
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)

    m = _compute_m_from_weights(w, g)
    # w_i ∝ L_i^{-1} → w ≈ [0.762, 0.190, 0.048], m ≈ 0.852
    assert math.isclose(m, 0.852, rel_tol=5e-3, abs_tol=5e-3)


def test_toy_example_beta_point_five():
    # L_1=16, L_2=64, L_3=256, g=[0.8, 1.0, 1.1]
    L = torch.tensor([16.0, 64.0, 256.0])
    g = torch.tensor([0.8, 1.0, 1.1])

    # β = 0.5, s(L) ≡ 1
    s, w = kband_weights(L=L, beta=0.5, rho=0.0, eta=1.0, k=0)

    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)

    m = _compute_m_from_weights(w, g)
    # w_i ∝ L_i^{-0.5} → w ≈ [0.571, 0.286, 0.143], m ≈ 0.900
    assert math.isclose(m, 0.900, rel_tol=5e-3, abs_tol=5e-3)
