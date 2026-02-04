# tests/test_s_kband_and_grad.py
import math

import torch

from capo.eb_core import (
    eb_statistics,
    grad_ell_beta_closed_form,
    numeric_grad_rho_eta,
    s_kband,
)


def test_s_kband_reduces_to_one_when_rho_zero_or_k_zero():
    L = torch.tensor([16.0, 64.0, 256.0])

    s_rho0 = s_kband(L, rho=0.0, k=32, eta=1.0)
    s_k0 = s_kband(L, rho=0.8, k=0, eta=1.0)

    assert torch.allclose(s_rho0, torch.ones_like(L))
    assert torch.allclose(s_k0, torch.ones_like(L))


def test_s_kband_monotone_in_rho_for_fixed_L():
    L = torch.tensor([128.0])
    s_low = s_kband(L, rho=0.3, k=32, eta=1.0)[0].item()
    s_high = s_kband(L, rho=0.9, k=32, eta=1.0)[0].item()
    assert s_high > s_low  # more dependence → larger s


def test_eb_stats_and_gradients_are_finite():
    L = torch.tensor([16.0, 64.0, 256.0], dtype=torch.float32)
    g = torch.tensor([0.8, 1.0, 1.1], dtype=torch.float32)

    beta = 1.0
    rho = 0.5
    k = 16
    eta = 1.0

    stats = eb_statistics(g, L, beta=beta, rho=rho, k=k, eta=eta)

    assert stats.omega.shape == L.shape
    assert stats.w.shape == L.shape
    assert torch.all(stats.w >= 0)
    assert abs(stats.w.sum().item() - 1.0) < 1e-6

    # Baseline and residuals behave as expected.
    assert min(g).item() <= stats.m.item() <= max(g).item()
    assert torch.allclose(stats.e, g - stats.m)

    grad_beta = grad_ell_beta_closed_form(
        L=L,
        omega=stats.omega,
        e=stats.e,
        Lambda_omega=stats.Lambda_omega,
        RSS_omega=stats.RSS_omega,
        dlog_pi_beta=0.0,
    )
    assert isinstance(grad_beta, float)
    assert math.isfinite(grad_beta)

    grad_rho, grad_eta = numeric_grad_rho_eta(
        L=L,
        g=g,
        beta=beta,
        rho=rho,
        k=k,
        eta=eta,
        dlog_pi_rho=0.0,
        dlog_pi_eta=0.0,
    )
    assert math.isfinite(grad_rho)
    assert math.isfinite(grad_eta)
