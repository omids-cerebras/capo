# capo/eb_core.py
"""
Core Empirical Bayes (EB) utilities for CAPO.

This module implements the algorithms described in the "Algorithms:
CAPO with Empirical Bayes length and dependence estimation" section of
the paper:

- Algorithm EB-lite (Algorithm~\\ref{alg:eb-lite}):
  A single-parameter length-only EB fit ignoring dependence
  (s(L; ξ) ≡ 1).

- Algorithm ACF-moment (Algorithm~\\ref{alg:acf-moment}):
  A moment-based estimator for the dependence shape ξ = (ρ, η, ...),
  based on pooled autocorrelation of token-level increments Y_{i,τ}.

- k-banded covariance weights (Algorithm~\\ref{alg:kband-weights}):
  Dependence-corrected precision weights under the stretched-geometric
  k-band family s(L; ρ, k, η).

- Joint EB update for (β, ρ, η) (Algorithm~\\ref{alg:joint-eb-kband}):
  A gradient-ascent step on the EB objective ℓ(β, ρ, η), followed by
  projection and optional EMA smoothing.

These routines are "policy-agnostic": they act only on scalarized
trajectory returns g_i, lengths L_i, and token-level increments Y_{i,τ}.
The VERL integration code (advantage estimators) calls into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class EBStats:
    """
    Sufficient statistics of the EB model for a single batch.

    Notation follows Section 4 / Algorithms:

    - g_i: scalarized trajectory returns (shape [G])
    - L_i: trajectory lengths (shape [G])
    - ω_i(β, ξ): precisions up to global scale
    - w_i(β, ξ): normalized weights
    - m(β, ξ): precision-weighted mean
    - e_i: residuals g_i - m
    - Λ_ω: sum of precisions
    - RSS_ω: weighted residual sum of squares

    Attributes
    ----------
    omega : Tensor, shape [G]
        Per-trajectory "precision weights" ω_i(β, ξ), unnormalized.
    w : Tensor, shape [G]
        Normalized precision weights w_i = ω_i / Λ_ω.
    m : Tensor, scalar
        Weighted mean m = Σ_i w_i g_i.
    e : Tensor, shape [G]
        Residuals e_i = g_i - m.
    Lambda_omega : float
        Sum of precisions Λ_ω = Σ_i ω_i.
    RSS_omega : float
        Weighted residual sum of squares RSS_ω = Σ_i ω_i e_i^2.
    """

    omega: Tensor
    w: Tensor
    m: Tensor
    e: Tensor
    Lambda_omega: float
    RSS_omega: float


def s_kband(L: Tensor, rho: float, k: int, eta: float, eps: float = 1e-8,) -> Tensor:
    """
    Stretched-geometric k-banded dependence factor s(L; ρ, k, η).

    This implements the shape function of Definition~\\ref{def:kband}:

        s(L; ρ, k, η)
          = 1 + (2 / L) * Σ_{h=1}^{m(L)} (L - h) ρ^{h^η},

    where m(L) = min{k, L - 1}.

    Parameters
    ----------
    L : Tensor of shape [G]
        Trajectory lengths L_i.
    rho : float
        Correlation parameter. If |ρ| is very small or k <= 0, we
        fall back to s(L) ≡ 1.
    k : int
        Maximum lag for the k-banded family.
    eta : float
        Stretch exponent; η = 1 recovers geometric decay, η ≠ 1 gives
        stretched-geometric decay.
    eps : float
        Numerical floor for stability.

    Returns
    -------
    s : Tensor of shape [G]
        Dependence factors s_i = s(L_i; ρ, k, η).
    """
    L = torch.as_tensor(L, dtype=torch.double)
    G = L.numel()
    s_vals = torch.ones(G, dtype=torch.double)

    if k <= 0 or abs(rho) < eps:
        # Independence / no effective correlation: s ≡ 1.
        return s_vals.float()

    rho = float(rho)
    eta = float(eta)

    for idx in range(G):
        Li = float(L[idx].item())
        if Li <= 1:
            # Degenerate trajectory: no internal correlation.
            s_vals[idx] = 1.0
            continue

        m_i = min(k, int(Li) - 1)
        if m_i <= 0:
            s_vals[idx] = 1.0
            continue

        h = torch.arange(1, m_i + 1, dtype=torch.double)
        # Use h^η in the exponent, as in the stretched-geometric model.
        powers = rho ** (h ** eta)
        summand = (Li - h) * powers
        s_vals[idx] = 1.0 + (2.0 / Li) * summand.sum()

    # Numerical floor to avoid division by zero downstream.
    return s_vals.clamp_min(eps).float()


def eb_statistics(
    g: Tensor,
    L: Tensor,
    beta: float,
    rho: float = 0.0,
    eta: float = 0.0,
    k: int = 0,
    eps: float = 1e-8,
) -> EBStats:
    """
    Compute EB statistics (ω_i, w_i, m, e_i, Λ_ω, RSS_ω) for a batch.

    This function implements the E-step summaries described in
    Algorithm~\\ref{alg:eb-capo} and Algorithm~\\ref{alg:joint-eb-kband}:

    - ω_i(β, ξ) = [L_i^β s(L_i; ξ)]^{-1}
    - w_i = ω_i / Λ_ω
    - m(β, ξ) = Σ_i w_i g_i
    - e_i = g_i - m
    - Λ_ω = Σ_i ω_i
    - RSS_ω = Σ_i ω_i e_i^2

    Parameters
    ----------
    g : Tensor, shape [G]
        Scalarized returns g_i (e.g., mean token-level CAPO reward per
        trajectory).
    L : Tensor, shape [G]
        Trajectory lengths L_i.
    beta : float
        Current length exponent β.
    rho : float, optional
        Dependence parameter ρ.
    eta : float, optional
        Dependence parameter η.
    k : int, optional
        k-band width for s(L; ρ, k, η).
    eps : float
        Numerical floor to stabilize divisions and logs.

    Returns
    -------
    EBStats
        Dataclass containing ω_i, w_i, m, e_i, Λ_ω, RSS_ω.
    """
    g = torch.as_tensor(g, dtype=torch.double)
    L = torch.as_tensor(L, dtype=torch.double).clamp_min(1.0)

    # Dependence factor s_i = s(L_i; ρ, k, η).
    s_vals = s_kband(L, rho=rho, k=k, eta=eta, eps=eps).double()

    # ω_i(β, ξ) = [L_i^β s_i]^{-1}
    omega = (L ** (-beta)) / s_vals
    omega = omega.clamp_min(eps)

    Lambda_omega = float(omega.sum().item())
    if Lambda_omega <= eps:
        # Degenerate case: fall back to uniform weights.
        w = torch.full_like(omega, 1.0 / omega.numel())
    else:
        w = omega / Lambda_omega

    m = (w * g).sum()
    e = g - m
    RSS_omega = float((omega * e * e).sum().item())

    return EBStats(
        omega=omega.float(),
        w=w.float(),
        m=m.float(),
        e=e.float(),
        Lambda_omega=Lambda_omega,
        RSS_omega=RSS_omega,
    )


def eb_objective(
    g: Tensor,
    L: Tensor,
    beta: float,
    rho: float,
    eta: float,
    k: int,
    eps: float = 1e-8,
) -> float:
    """
    Empirical Bayes objective ℓ(β, ξ) up to additive constants.

    We use the canonical form (cf. Section~\\ref{sec:eb-theory}):

        ℓ(β, ξ)
        = 0.5 Σ_i log ω_i(β, ξ)
          - 0.5 log Λ_ω(β, ξ)
          - 0.5 (G - 1) log RSS_ω(β, ξ)
        + (log priors),

    where ω_i, Λ_ω, RSS_ω are the statistics produced by `eb_statistics`.
    Here we omit explicit log-prior terms; the caller may add
    ∂/∂θ log π(θ) separately as "prior gradients".

    This function is primarily used for *numerical* gradients w.r.t.
    ρ and η in the joint EB update.

    Parameters
    ----------
    g, L, beta, rho, eta, k, eps
        As in `eb_statistics`.

    Returns
    -------
    float
        EB objective value ℓ(β, ξ) up to an additive constant.
    """
    stats = eb_statistics(g, L, beta=beta, rho=rho, eta=eta, k=k, eps=eps)
    G = int(g.numel())

    omega = stats.omega.double().clamp_min(eps)
    Lambda = stats.Lambda_omega + eps
    RSS = stats.RSS_omega + eps

    term1 = 0.5 * torch.log(omega).sum()
    term2 = -0.5 * torch.log(torch.tensor(Lambda))
    term3 = -0.5 * (G - 1) * torch.log(torch.tensor(RSS))

    return float((term1 + term2 + term3).item())


def grad_ell_beta_closed_form(
    L: Tensor,
    omega: Tensor,
    e: Tensor,
    Lambda_omega: float,
    RSS_omega: float,
    dlog_pi_beta: float = 0.0,
    eps: float = 1e-8,
) -> float:
    """
    Closed-form EB gradient ∂ℓ/∂β from Corollary~\\ref{cor:grad-beta}.

    The expression used here matches the update line in
    Algorithm~\\ref{alg:eb-capo}:

        g_β = 0.5 [
                  -Σ_i log L_i
                  + (Σ_i ω_i log L_i) / Λ_ω
                ]
              + 0.5 (G - 1) / RSS_ω * Σ_i ω_i e_i^2 log L_i
              + ∂_β log π_β(β),

    where:
      - Λ_ω = Σ_i ω_i,
      - RSS_ω = Σ_i ω_i e_i^2,
      - G is the batch size,
      - π_β is an optional prior on β.

    Parameters
    ----------
    L : Tensor, shape [G]
        Trajectory lengths L_i.
    omega : Tensor, shape [G]
        Precisions ω_i (from `eb_statistics`).
    e : Tensor, shape [G]
        Residuals e_i = g_i - m.
    Lambda_omega : float
        Sum of precisions Λ_ω.
    RSS_omega : float
        Weighted RSS.
    dlog_pi_beta : float, optional
        Prior gradient ∂_β log π_β(β).
    eps : float
        Numerical floor.

    Returns
    -------
    float
        Gradient g_β = ∂ℓ/∂β evaluated at the given statistics.
    """
    L = torch.as_tensor(L, dtype=torch.double).clamp_min(1.0)
    omega = torch.as_tensor(omega, dtype=torch.double)
    e = torch.as_tensor(e, dtype=torch.double)

    logL = torch.log(L)
    sum_logL = logL.sum()
    sum_omega_logL = (omega * logL).sum()

    G = int(L.numel())
    Lambda = Lambda_omega + eps
    RSS = RSS_omega + eps

    term1 = 0.5 * (-sum_logL + sum_omega_logL / Lambda)

    sum_omega_e2_logL = (omega * (e ** 2) * logL).sum()
    term2 = 0.5 * (G - 1) * sum_omega_e2_logL / RSS

    g_beta = term1 + term2 + dlog_pi_beta
    return float(g_beta.item())


def numeric_grad_rho_eta(
    g: Tensor,
    L: Tensor,
    beta: float,
    rho: float,
    eta: float,
    k: int,
    h_rho: float = 1e-3,
    h_eta: float = 1e-3,
    dlog_pi_rho: float = 0.0,
    dlog_pi_eta: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    Numerical gradients (∂ℓ/∂ρ, ∂ℓ/∂η) via central finite differences.

    This is a practical implementation of the "M-step on ξ" described in
    Algorithm~\\ref{alg:joint-eb-kband} when the exact expressions from
    Proposition~\\ref{prop:grad} and Lemma~\\ref{lem:kband-deriv} are
    inconvenient to maintain.

    We approximate:

        ∂ℓ/∂ρ ≈ [ℓ(β, ρ + h_ρ, η) - ℓ(β, ρ - h_ρ, η)] / (2 h_ρ),
        ∂ℓ/∂η ≈ [ℓ(β, ρ, η + h_η) - ℓ(β, ρ, η - h_η)] / (2 h_η),

    and then add the prior gradients ∂_ρ log π_ρ(ρ),
    ∂_η log π_η(η).

    Parameters
    ----------
    g, L, beta, rho, eta, k, eps :
        As in `eb_objective`.
    h_rho, h_eta : float
        Step sizes for finite differences.
    dlog_pi_rho, dlog_pi_eta : float
        Prior gradients on ρ and η.

    Returns
    -------
    g_rho, g_eta : float
        Numerical approximations to ∂ℓ/∂ρ and ∂ℓ/∂η.
    """
    # ρ-gradient (central difference, if possible)
    ell_plus = eb_objective(g, L, beta, rho + h_rho, eta, k, eps=eps)
    ell_minus = eb_objective(g, L, beta, rho - h_rho, eta, k, eps=eps)
    g_rho = (ell_plus - ell_minus) / (2.0 * h_rho) + dlog_pi_rho

    # η-gradient: ensure η - h_eta >= 0 (project to forward difference if needed).
    if eta - h_eta <= 0.0:
        ell_base = eb_objective(g, L, beta, rho, eta, k, eps=eps)
        ell_plus_eta = eb_objective(g, L, beta, rho, eta + h_eta, k, eps=eps)
        g_eta = (ell_plus_eta - ell_base) / h_eta + dlog_pi_eta
    else:
        ell_plus_eta = eb_objective(g, L, beta, rho, eta + h_eta, k, eps=eps)
        ell_minus_eta = eb_objective(g, L, beta, rho, eta - h_eta, k, eps=eps)
        g_eta = (ell_plus_eta - ell_minus_eta) / (2.0 * h_eta) + dlog_pi_eta

    return float(g_rho), float(g_eta)


def acf_moment_estimate(
    increments: Tensor,
    mask: Tensor,
    k: int,
    rho_grid: Optional[Tensor] = None,
    eta_grid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    ACF-moment estimator for dependence shape ξ = (ρ, η).

    Implements Algorithm~\\ref{alg:acf-moment}:

    1. For each trajectory i:
       - Center Y_{i,τ} by its within-trajectory mean.
       - Compute empirical autocovariances γ̂_i(h) up to lag k.

    2. Pool across trajectories using weights (L_i - h) to obtain
       pooled autocovariances γ̄(h).

    3. Form pooled autocorrelations r̄(h) = γ̄(h) / γ̄(0).

    4. Fit (ρ, η) by least squares against the stretched-geometric
       correlation model ρ^{h^η} on h = 1,...,k.

    Parameters
    ----------
    increments : Tensor, shape [B, T_max]
        Token-level increments Y_{i,τ} for each trajectory i, padded
        on the right (unused positions arbitrary).
    mask : Tensor[bool], shape [B, T_max]
        Boolean mask indicating which positions in `increments` are
        valid (True = valid token).
    k : int
        Maximum lag for which we compute autocovariances.
    rho_grid, eta_grid : Tensor, optional
        1D grid of candidate ρ and η values. If None, we use:
        - ρ ∈ linspace(0, 0.99, 21)
        - η ∈ linspace(0, 2.0, 11)
    eps : float
        Numerical floor.

    Returns
    -------
    rho_hat, eta_hat : float
        Moment-based estimates of ρ and η. If the data is essentially
        uncorrelated (γ̄(0) ≤ 0), we return (0.0, 1.0).
    """
    Y = torch.as_tensor(increments, dtype=torch.double)
    M = torch.as_tensor(mask, dtype=torch.bool)
    B, T_max = Y.shape

    k = max(0, min(int(k), T_max - 1))
    if k == 0:
        # No lag information -> effectively independent.
        return 0.0, 1.0

    # Pooled autocovariances γ̄(h) and counts of contributing pairs.
    gamma = torch.zeros(k + 1, dtype=torch.double)
    counts = torch.zeros(k + 1, dtype=torch.double)

    for i in range(B):
        valid_i = M[i]
        L_i = int(valid_i.sum().item())
        if L_i <= 1:
            continue

        y_i = Y[i, :L_i]
        y_i = y_i - y_i.mean()

        for h in range(0, k + 1):
            if L_i - h <= 1:
                continue
            # Pairs (τ, τ + h) for τ = 1,...,L_i - h.
            y0 = y_i[: L_i - h]
            y1 = y_i[h:]
            gamma_i_h = (y0 * y1).mean()
            gamma[h] += (L_i - h) * gamma_i_h
            counts[h] += L_i - h

    # Normalize by total number of pairs at each lag.
    valid_counts = counts > 0
    gamma[valid_counts] = gamma[valid_counts] / counts[valid_counts]

    gamma0 = gamma[0]
    if gamma0 <= eps:
        # Essentially white noise; no reliable correlation structure.
        return 0.0, 1.0

    # Pooled autocorrelation r̄(h) for h >= 1.
    r = gamma[1:] / gamma0
    valid_lags = valid_counts[1:]
    r = r[valid_lags]
    if r.numel() == 0:
        return 0.0, 1.0

    # Lags that we actually have data for.
    h_vals = torch.arange(1, k + 1, dtype=torch.double)[valid_lags]

    # Default grids for (ρ, η).
    if rho_grid is None:
        rho_grid = torch.linspace(0.0, 0.99, steps=21, dtype=torch.double)
    if eta_grid is None:
        eta_grid = torch.linspace(0.0, 2.0, steps=11, dtype=torch.double)

    best_loss = float("inf")
    best_rho, best_eta = 0.0, 1.0

    # Brute-force grid search: small (21 x 11) grid is cheap.
    for rho in rho_grid:
        for eta in eta_grid:
            # Predicted correlation ρ^{h^η} for observed lags h.
            preds = rho ** (h_vals ** eta)
            loss = torch.mean((r - preds) ** 2).item()
            if loss < best_loss:
                best_loss = loss
                best_rho = float(rho.item())
                best_eta = float(eta.item())

    return best_rho, best_eta


def eb_lite_fit_beta_and_weights(
    g: Tensor, L: Tensor, eps: float = 1e-8, max_iters: int = 20, tol: float = 1e-4,
) -> Tuple[float, Tensor, Tensor]:
    """
    EB-lite (Algorithm~\\ref{alg:eb-lite}): length-only EB estimator.

    The EB-lite algorithm ignores dependence (s(L; ξ) ≡ 1) and fits only
    the length exponent β in the variance model

        Var(g_i | L_i) ≈ σ^2 L_i^β,

    via the approximation:

        log((g_i - m)^2) ≈ c - β log L_i,

    when g_i is centered at m ≈ μ.

    Steps
    -----
    1. Initialize m^{(0)} as the mean of {g_i}.
    2. Iterate:
       - e_i = g_i - m^{(k)},
       - z_i = log(e_i^2 + ε),
       - OLS regression z_i = c - β log L_i + ε_i yields slope b,
         set β^{(k+1)} = -b,
       - weights ω_i ∝ L_i^{-β^{(k+1)}}, normalized to w_i,
       - update m^{(k+1)} = Σ_i w_i g_i.
    3. Stop when |m^{(k+1)} - m^{(k)}| < tol or after max_iters.

    Parameters
    ----------
    g : Tensor, shape [G]
        Scalarized returns g_i.
    L : Tensor, shape [G]
        Trajectory lengths L_i.
    eps : float
        Numerical floor used in log(e_i^2 + eps).
    max_iters : int
        Maximum number of EB-lite iterations.
    tol : float
        Convergence tolerance for |m^{(k+1)} - m^{(k)}|.

    Returns
    -------
    beta_hat : float
        Estimated length exponent β̂.
    w : Tensor, shape [G]
        Normalized weights w_i ∝ L_i^{-β̂}.
    m : Tensor, scalar
        Precision-weighted mean m(β̂) = Σ_i w_i g_i.
    """
    g = torch.as_tensor(g, dtype=torch.double)
    L = torch.as_tensor(L, dtype=torch.double).clamp_min(1.0)

    # Initial aggregator m^(0): simple mean of g_i
    m = g.mean()
    beta_hat = 1.0  # neutral ΔL exponent

    logL = torch.log(L)

    for _ in range(max_iters):
        e = g - m
        z = torch.log(e * e + eps)

        x = logL
        x_mean = x.mean()
        z_mean = z.mean()
        cov_xz = ((x - x_mean) * (z - z_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum().clamp_min(eps)

        slope = cov_xz / var_x
        beta_new = float(-slope.item())

        # Provisional weights: ω_i ∝ L_i^{-β_new}
        omega = L.pow(-beta_new)
        omega_sum = omega.sum()
        if omega_sum <= eps:
            w = torch.full_like(omega, 1.0 / omega.numel())
        else:
            w = omega / omega_sum

        m_new = (w * g).sum()

        if torch.abs(m_new - m) < tol:
            m = m_new
            beta_hat = beta_new
            break

        m = m_new
        beta_hat = beta_new

    # Final weights at β̂.
    omega = L.pow(-beta_hat)
    omega_sum = omega.sum()
    if omega_sum <= eps:
        w = torch.full_like(omega, 1.0 / omega.numel())
    else:
        w = omega / omega_sum

    m = (w * g).sum()
    return float(beta_hat), w.float(), m.float()


def kband_weights(
    L: Tensor, beta: float, rho: float, eta: float, k: int, eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    k-banded covariance weights (Algorithm~\\ref{alg:kband-weights}).

    Given a batch of lengths L_i and a current EB parameter triple
    (β, ρ, η), compute:

    - s_i = s(L_i; ρ, k, η)
          = 1 + (2 / L_i) Σ_{h=1}^{m_i} (L_i - h) ρ^{h^η},
      where m_i = min{k, L_i - 1},

    - ω_i = [L_i^β s_i]^{-1},
    - w_i = ω_i / Σ_j ω_j.

    Parameters
    ----------
    L : Tensor, shape [G]
        Trajectory lengths L_i.
    beta, rho, eta, k :
        Current EB parameters and k-band width.
    eps : float
        Numerical floor.

    Returns
    -------
    s : Tensor, shape [G]
        Shape factors s_i.
    w : Tensor, shape [G]
        Normalized weights w_i.
    """
    L = torch.as_tensor(L, dtype=torch.double).clamp_min(1.0)
    s = s_kband(L, rho=rho, k=k, eta=eta, eps=eps).double()
    omega = L.pow(-beta) / s
    omega = omega.clamp_min(eps)

    omega_sum = omega.sum()
    if omega_sum <= eps:
        w = torch.full_like(omega, 1.0 / omega.numel())
    else:
        w = omega / omega_sum

    return s.float(), w.float()


def joint_eb_update_kband(
    g: Tensor,
    L: Tensor,
    beta_init: float,
    rho_init: float,
    eta_init: float,
    k: int,
    lr_beta: float = 0.1,
    lr_rho: float = 0.1,
    lr_eta: float = 0.1,
    steps_beta: int = 1,
    steps_xi: int = 1,
    ema_beta: float = 1.0,
    ema_xi: float = 1.0,
    beta_bounds: Tuple[float, float] = (0.0, 2.0),
    rho_max: float = 0.99,
    dlog_pi_beta: float = 0.0,
    dlog_pi_rho: float = 0.0,
    dlog_pi_eta: float = 0.0,
    use_acf_warmstart: bool = False,
    increments: Optional[Tensor] = None,
    increments_mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[float, float, float, Tensor]:
    """
    Joint EB update for (β, ρ, η) under the k-banded dependence model.

    This function is a practical implementation of
    Algorithm~\\ref{alg:joint-eb-kband}:

    1. Optional ACF-moment warm-start for (ρ, η) using
       Algorithm~\\ref{alg:acf-moment}.

    2. E-step: compute EB statistics under current (β, ρ, η):
       - ω_i, w_i, m_t, e_i, Λ_ω, RSS_ω.

    3. M-step:
       - g_β = ∂ℓ/∂β via the closed-form expression from
         Corollary~\\ref{cor:grad-beta},
       - (g_ρ, g_η) via numerical gradients of ℓ(β, ρ, η) (Section 4.3),
         plus prior gradients dlog_pi_ρ, dlog_pi_η.

    4. Projected gradient ascent on (β, ρ, η), followed by EMA smoothing:

       β_t = (1 - γ_β) β_{t-1} + γ_β clip(β_{t-1} + η_β g_β, [β_min, β_max]),
       ρ_t = (1 - γ_ξ) ρ_{t-1} + γ_ξ clip(ρ_{t-1} + η_ρ g_ρ, [-ρ_max, ρ_max]),
       η_t = (1 - γ_ξ) η_{t-1} + γ_ξ max(η_{t-1} + η_η g_η, 0).

    5. Recompute k-banded weights w_i^{(t)} at (β_t, ρ_t, η_t) for use
       in the CAPO update.

    Parameters
    ----------
    g, L : Tensor, shape [G]
        Scalarized returns and lengths.
    beta_init, rho_init, eta_init : float
        Initial parameters (β_{t-1}, ρ_{t-1}, η_{t-1}).
    k : int
        k-band width for s(L; ρ, k, η).
    lr_beta, lr_rho, lr_eta : float
        Learning rates for β, ρ, η.
    steps_beta, steps_xi : int
        Number of gradient ascent steps for β and ξ = (ρ, η).
    ema_beta, ema_xi : float
        EMA smoothing coefficients γ_β, γ_ξ ∈ (0, 1].
    beta_bounds : (float, float)
        Bounds [β_min, β_max].
    rho_max : float
        Maximum |ρ| allowed after projection.
    dlog_pi_beta, dlog_pi_rho, dlog_pi_eta : float
        Prior gradients ∂ log π / ∂θ for θ ∈ {β, ρ, η}.
    use_acf_warmstart : bool
        Whether to warm-start (ρ, η) from ACF-moment estimates.
    increments, increments_mask : Tensor or None
        Token-level increments and mask for ACF-moment warm-start.
    eps : float
        Numerical floor.

    Returns
    -------
    beta_t, rho_t, eta_t : float
        Updated EB parameters after this batch.
    w : Tensor, shape [G]
        Final dependence-corrected weights at (β_t, ρ_t, η_t).
    """
    g = torch.as_tensor(g, dtype=torch.float32)
    L = torch.as_tensor(L, dtype=torch.float32).clamp_min(1.0)

    beta = float(beta_init)
    rho = float(rho_init)
    eta = float(eta_init)

    beta_min, beta_max = beta_bounds

    # Optional warm-start for (ρ, η) from token-level autocorrelations.
    if use_acf_warmstart and increments is not None and increments_mask is not None:
        rho_hat, eta_hat = acf_moment_estimate(
            increments=increments, mask=increments_mask, k=k
        )
        # Project into admissible region and blend with old parameters.
        rho = max(-rho_max, min(rho_hat, rho_max))
        eta = max(0.0, eta_hat)

    # Gradient-ascent steps for β.
    for _ in range(max(0, steps_beta)):
        stats = eb_statistics(g, L, beta=beta, rho=rho, eta=eta, k=k, eps=eps)
        g_beta = grad_ell_beta_closed_form(
            L=L,
            omega=stats.omega,
            e=stats.e,
            Lambda_omega=stats.Lambda_omega,
            RSS_omega=stats.RSS_omega,
            dlog_pi_beta=dlog_pi_beta,
            eps=eps,
        )
        beta_proposed = beta + lr_beta * g_beta
        beta_proposed = max(beta_min, min(beta_proposed, beta_max))
        beta = (1.0 - ema_beta) * beta + ema_beta * beta_proposed

    # Gradient-ascent steps for ξ = (ρ, η).
    for _ in range(max(0, steps_xi)):
        g_rho, g_eta = numeric_grad_rho_eta(
            g=g,
            L=L,
            beta=beta,
            rho=rho,
            eta=eta,
            k=k,
            dlog_pi_rho=dlog_pi_rho,
            dlog_pi_eta=dlog_pi_eta,
            eps=eps,
        )
        rho_proposed = rho + lr_rho * g_rho
        rho_proposed = max(-rho_max, min(rho_proposed, rho_max))

        eta_proposed = eta + lr_eta * g_eta
        eta_proposed = max(0.0, eta_proposed)

        rho = (1.0 - ema_xi) * rho + ema_xi * rho_proposed
        eta = (1.0 - ema_xi) * eta + ema_xi * eta_proposed

    # Final weights at (β_t, ρ_t, η_t).
    _, w = kband_weights(L=L, beta=beta, rho=rho, eta=eta, k=k, eps=eps)

    return beta, rho, eta, w
