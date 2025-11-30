# tests/test_acf_moment.py
import pytest
import torch

pytest.importorskip("verl")

from capo.verl_integration.adv_estimators import acf_moment_fit


def test_acf_moment_fit_recovers_positive_rho():
    torch.manual_seed(0)
    B, T = 32, 64
    rho_true = 0.8
    sigma_eps = 0.1

    Y = torch.zeros(B, T, dtype=torch.float32)
    mask = torch.ones_like(Y, dtype=torch.bool)

    for i in range(B):
        eps = sigma_eps * torch.randn(T)
        for t in range(1, T):
            Y[i, t] = rho_true * Y[i, t - 1] + eps[t]

    rho_hat, eta_hat = acf_moment_fit(
        Y=Y,
        mask=mask,
        k=5,
        rho_max=0.99,
        eta_max=2.0,
        n_rho=21,
        n_eta=11,
    )

    # At least detect positive serial correlation.
    assert 0.0 <= rho_hat <= 1.0
    assert rho_hat > 0.3
    assert 0.0 <= eta_hat <= 2.0
