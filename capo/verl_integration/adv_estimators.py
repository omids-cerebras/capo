"""
Advantage estimators for CAPO and EB–CAPO in VERL.

This module implements the three CAPO-side algorithms from the
Algorithms section (§4) of the write-up:

1. Plain CAPO advantage ("capo")
   --------------------------------
   - Token-wise GRPO-style z-normalization over CAPO rewards r_{i,t}.
   - No length or dependence modelling; this is the non-EB baseline.

2. EB–CAPO-lite ("capo_eb_lite")
   --------------------------------
   - Implements Algorithm~EB-lite (no dependence) from §4.
   - Length-only model (no s(L; ξ)): ω_i(β) = L_i^{-β}, β ∈ [0, 2].
   - β is estimated by robust regression of log(e_i^2) on log L_i:

         z_i = log(e_i^2 + ε) = c - β log L_i + ε_i,

     as in Algorithm~\ref{alg:eb-lite}, and weights are set to
     w_i ∝ L_i^{-β̂}, normalized.

3. Full EB–CAPO with k-banded dependence ("capo_eb")
   --------------------------------
   - Variance model / precision (Notation from §3, §4):

         v_i(θ) = σ^2 L_i^{β} s(L_i; ξ),
         ω_i(β, ξ) = 1 / v_i(θ) ∝ L_i^{-β} s(L_i; ξ)^{-1},

     with stretched-geometric k-band dependence shape:

         s(L; ρ, k, η) = 1 + (2 / L) Σ_{h=1}^{m} (L - h) ρ^{h^{η}},
         m = min{k, L - 1},        (Definition~\ref{def:kband})

   - EB objective (Theorem~\ref{thm:EB-obj}, eq.~(EB-obj)):

         ℓ(β, ξ)
           = log π_β(β) + log π_ξ(ξ)
             + 1/2 Σ_i log ω_i(β, ξ)
             - 1/2 log Λ_ω(β, ξ)
             - (G - 1)/2 log RSS_ω(β, ξ),

       where Λ_ω = Σ_i ω_i, RSS_ω = Σ_i ω_i (g_i - m)^2, and
       m(β, ξ) = Σ_i w_i g_i with w_i = ω_i / Λ_ω.

   - Gradients:
       * General ∂ℓ/∂φ from Prop.~\ref{prop:grad}, eq.~(EB-grad-general).
       * Length-direction ∂ℓ/∂β from Cor.~\ref{cor:grad-beta}:

             ∂ℓ/∂β
               = 1/2 [ - Σ_i log L_i
                        + (Σ_i ω_i log L_i) / Λ_ω ]
                 + (G - 1)/(2 RSS_ω) Σ_i ω_i e_i^2 log L_i
                 + ∂ log π_β(β) / ∂β.

       * Dependence directions ∂ℓ/∂ρ, ∂ℓ/∂η using eq.~(EB-grad-general)
         plus Lemma~\ref{lem:kband-deriv} for ∂s/∂ρ, ∂s/∂η.

   - ACF-moment estimator for ξ (Algorithm "ACF-moment"):
       * Computes empirical auto-covariances γ̂_i(h) and aggregated
         r̄(h), then fits (ρ, η) by least squares:

             (ρ̂, η̂) = argmin_{ρ,η} Σ_{h=1}^k (r̄(h) - ρ^{h^{η}})^2.

     We approximate the last step with a coarse grid search.

All EB-related notation (ω_i, Λ_ω, RSS_ω, s(L; ρ, k, η), β, ξ)
matches the paper so it is easy to cross-reference.
"""

from __future__ import annotations

from typing import Any, Tuple

import math
import torch

from verl.trainer.ppo.core_algos import register_adv_est


# ============================================================================
# 1. Plain CAPO advantage: token-wise z-normalization (no EB)
# ============================================================================


@register_adv_est("capo")
def compute_capo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config: Any = None,
    epsilon: float = 1e-8,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plain CAPO advantage estimator (no EB).

    Implements the GRPO-style token-wise z-normalization over CAPO
    rewards r_{i,t}:

        A_{i,t} = (r_{i,t} - μ_r) / (σ_r + ε),

    where μ_r, σ_r are computed over all valid response tokens
    (response_mask == 1). This corresponds to the "baseline CAPO"
    in the EB–CAPO write-up.

    Parameters
    ----------
    token_level_rewards:
        (B, T) tensor of CAPO token rewards r_{i,t}.

    response_mask:
        (B, T) tensor, 1 for valid response tokens.

    index:
        Group indices from VERL (ignored here).

    config:
        AlgoConfig. If it has `norm_adv_by_std_in_grpo == False`, we
        only center and do not divide by std.

    epsilon:
        Numerical guard when dividing by σ_r.

    Returns
    -------
    advantages:
        (B, T) CAPO token advantages.

    returns:
        (B, T) returns; here equal to the masked rewards.
    """
    device = token_level_rewards.device
    valid = response_mask > 0

    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards, device=device)
        returns = token_level_rewards.clone()
        return advantages, returns

    valid_rewards = token_level_rewards[valid]
    mean_r = valid_rewards.mean()
    centered = token_level_rewards - mean_r

    if config is None or getattr(config, "norm_adv_by_std_in_grpo", True):
        std_r = valid_rewards.std().clamp_min(epsilon)
        advantages = centered / std_r
    else:
        advantages = centered

    advantages = advantages * valid.float()
    returns = token_level_rewards * valid
    return advantages, returns


# ============================================================================
# Helper: collapse token-level rewards to (g_i, L_i)
# ============================================================================


def _lengths_and_scalar_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute trajectory lengths L_i and scalarized returns g_i.

    As in §2–3 of the write-up, we define:
      - L_i: number of valid response tokens in trajectory i,
      - g_i: mean of r_{i,t} over valid response tokens,

    which is the scalar quantity plugged into the EB model.

    Returns
    -------
    lengths:
        (B,) float tensor with L_i ≥ 1.

    returns_scalar:
        (B,) float tensor g_i.

    valid:
        (B, T) boolean mask of valid response tokens.
    """
    valid = response_mask > 0
    lengths = valid.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
    returns_scalar = (token_level_rewards * valid).sum(dim=-1) / lengths
    return lengths, returns_scalar, valid


# ============================================================================
# 2. EB–CAPO-lite: Algorithm "EB-lite (no dependence)" (ω_i = L_i^{-β})
# ============================================================================


def eb_lite_fit_beta_and_weights(
    g: torch.Tensor,
    L: torch.Tensor,
    eps: float = 1e-8,
    max_iters: int = 10,
    tol: float = 1e-4,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Algorithm EB-lite (no dependence) from §4 (Alg.~\\ref{alg:eb-lite}).

    Input:
      - Trajectory pairs {(g_i, L_i)}_{i=1}^G,
      - Small regularization ε > 0.

    Output:
      - β̂ and weights w_i ∝ L_i^{-β̂}.

    Paper pseudocode (simplified):

      1. Initialize m^{(0)} ← (1/G) Σ_i g_i   (or median{g_i}).
      2. Repeat until converged:
           e_i ← g_i - m^{(k)}
           z_i ← log(e_i^2 + ε)
           Fit z_i = c - β log L_i + ε_i by least squares,
             ⇒ β̂^{(k+1)} ← - (slope on log L_i)
           Set w_i ∝ L_i^{-β̂^{(k+1)}}, normalize Σ_i w_i = 1
           Update m^{(k+1)} ← Σ_i w_i g_i
      3. Return β̂, w.

    We implement a small number of outer iterations with a simple
    convergence check on m^{(k)}.
    """
    device = g.device
    g = g.to(dtype=torch.float64)
    L = L.to(dtype=torch.float64)

    if g.numel() != L.numel():
        raise ValueError("g and L must have the same length.")

    # Step 1: m^{(0)} = mean(g_i).
    m = g.mean()
    beta_hat = 1.0

    logL = torch.log(L.clamp_min(1.0))

    for _ in range(max_iters):
        # e_i = g_i - m^{(k)}
        e = g - m
        # z_i = log(e_i^2 + ε)
        z = torch.log(e * e + eps)

        # Fit z_i = c - β log L_i via OLS: z ~ a + b * x, x = log L_i, so β = -b.
        x = logL
        x_mean = x.mean()
        z_mean = z.mean()
        cov_xz = ((x - x_mean) * (z - z_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum().clamp_min(eps)
        slope = cov_xz / var_x  # slope on log L_i
        beta_new = float(-slope.item())  # β̂^{(k+1)} = - slope

        # w_i ∝ L_i^{-β_new}, normalize.
        omega = L.pow(-beta_new)
        omega_sum = omega.sum()
        if omega_sum <= 0:
            w = torch.full_like(L, 1.0 / float(L.numel()))
        else:
            w = omega / omega_sum

        # m^{(k+1)} = Σ_i w_i g_i.
        m_new = (w * g).sum()

        # Convergence test on m.
        if torch.abs(m_new - m) < tol:
            m = m_new
            beta_hat = beta_new
            break

        m = m_new
        beta_hat = beta_new

    # Final weights for β̂.
    omega = L.pow(beta_hat * -1.0)  # L_i^{-β̂}
    omega_sum = omega.sum()
    if omega_sum <= 0:
        w = torch.full_like(L, 1.0 / float(L.numel()))
    else:
        w = omega / omega_sum

    return float(beta_hat), w.to(dtype=torch.float32).to(device), m.to(dtype=torch.float32).to(device)


@register_adv_est("capo_eb_lite")
def compute_capo_eb_lite_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config: Any = None,
    epsilon: float = 1e-8,
    max_iters: int = 10,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EB–CAPO-lite advantage estimator (no dependence).

    This wraps Algorithm EB-lite and converts its batch-wise EB estimate
    into token-level advantages:

      1. Collapse to (g_i, L_i) using _lengths_and_scalar_returns.
      2. Run EB-lite to obtain:
           - β̂   (length exponent),
           - w_i (trajectory weights, w_i ∝ L_i^{-β̂}),
           - m(β̂) (aggregated estimate, eq. (μ-agg)).
      3. Define per-trajectory scalar advantages

           A_i = w_i (g_i - m(β̂)),

         which encode both centering and EB weight.
      4. Broadcast A_i to tokens in trajectory i.
      5. Optionally apply a GRPO-style std normalization over valid
         tokens if config.norm_adv_by_std_in_grpo is True.
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(
        token_level_rewards, response_mask
    )

    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards)
        returns = token_level_rewards.clone()
        return advantages, returns

    beta_hat, w, m = eb_lite_fit_beta_and_weights(
        g=returns_scalar,
        L=lengths,
        eps=epsilon,
        max_iters=max_iters,
        tol=float(kwargs.get("tol", 1e-4)),
    )

    # Per-trajectory scalar advantages A_i = w_i (g_i - m).
    adv_scalar = w * (returns_scalar - m)  # (B,)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        valid_adv = advantages[valid]
        std = valid_adv.std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid
    return advantages, returns


# ============================================================================
# 3. Full EB–CAPO: β and k-banded dependence ξ = (ρ, k, η)
#    s(L; ρ, k, η) from Definition~\ref{def:kband}, eq.~(s-family).
#    Gradients from Prop.~\ref{prop:grad} + Lem.~\ref{lem:kband-deriv}.
# ============================================================================


def s_kband(
    L: torch.Tensor,
    rho: float,
    k: int,
    eta: float,
) -> torch.Tensor:
    """
    Stretched-geometric k-banded dependence shape (Definition~\\ref{def:kband}).

    Direct implementation of eq.~(s-family):

        s(L; ρ, k, η)
          = 1 + (2 / L) Σ_{h=1}^{m} (L - h) ρ^{h^{η}},
        m = min{k, L - 1}.

    Parameters
    ----------
    L:
        1D tensor of lengths L_i.

    rho:
        Correlation parameter (|ρ| < 1). In practice we usually
        restrict ρ ∈ [0, ρ_max].

    k:
        Band-width (maximum lag).

    eta:
        Stretch exponent (η ≥ 0).

    Returns
    -------
    s_vals:
        1D tensor of s(L_i; ρ, k, η).
    """
    L = L.to(dtype=torch.float64)
    device = L.device
    s_vals = torch.ones_like(L, dtype=torch.float64, device=device)

    rho = float(rho)
    eta = float(eta)
    k = int(k)

    if abs(rho) < 1e-12 or k <= 0:
        return s_vals.to(dtype=torch.float32)

    for idx in range(L.shape[0]):
        Li = float(L[idx].item())
        if Li <= 1:
            s_vals[idx] = 1.0
            continue
        m_i = min(k, int(Li) - 1)
        if m_i <= 0:
            s_vals[idx] = 1.0
            continue

        h = torch.arange(1, m_i + 1, dtype=torch.float64, device=device)
        if abs(eta) < 1e-8:
            # η ≈ 0 ⇒ h^η ≈ 1 ⇒ ρ^{h^η} ≈ ρ.
            powers = torch.full_like(h, rho)
        else:
            powers = rho ** (h ** eta)
        summand = (Li - h) * powers
        s_vals[idx] = 1.0 + (2.0 / Li) * summand.sum()

    return s_vals.to(dtype=torch.float32)


def eb_stats(
    g: torch.Tensor,
    L: torch.Tensor,
    beta: float,
    rho: float,
    k: int,
    eta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    EB statistics for ℓ(β, ξ) (eq.~(EB-obj)):

      ω_i(β, ξ) = L_i^{-β} / s(L_i; ξ),
      Λ_ω       = Σ_i ω_i,
      w_i       = ω_i / Λ_ω,
      m(β, ξ)   = Σ_i w_i g_i,
      e_i       = g_i - m(β, ξ),
      RSS_ω     = Σ_i ω_i e_i^2.

    Returns ω_i, w_i, m, e_i, Λ_ω, RSS_ω.
    """
    device = g.device
    g = g.to(dtype=torch.double)
    L = L.to(dtype=torch.double)

    s_vals = s_kband(L, rho=rho, k=k, eta=eta).to(dtype=torch.double)
    s_vals = s_vals.clamp_min(1e-8)

    omega = L.pow(-beta) / s_vals
    Lambda_omega = omega.sum()
    if Lambda_omega <= 0:
        # Fallback: uniform weights if ω degenerates.
        w = torch.full_like(omega, 1.0 / float(omega.numel()))
        m = (w * g).sum()
        e = g - m
        RSS_omega = (w * e * e).sum()
    else:
        w = omega / Lambda_omega
        m = (w * g).sum()
        e = g - m
        RSS_omega = (omega * e * e).sum()

    return (
        omega.to(dtype=torch.float32),
        w.to(dtype=torch.float32),
        m.to(dtype=torch.float32),
        e.to(dtype=torch.float32),
        float(Lambda_omega.item()),
        float(RSS_omega.item()),
    )


def grad_ell_beta(
    L: torch.Tensor,
    omega: torch.Tensor,
    e: torch.Tensor,
    Lambda_omega: float,
    RSS_omega: float,
    dlog_pi_beta: float = 0.0,
) -> float:
    """
    ∂ℓ/∂β from Corollary~\\ref{cor:grad-beta}, eq.~(EB-grad-beta):

        ∂ℓ/∂β
          = 1/2 [ - Σ_i log L_i
                  + ( Σ_i ω_i log L_i ) / Λ_ω ]
            + (G - 1) / (2 RSS_ω) Σ_i ω_i e_i^2 log L_i
            + ∂ log π_β(β) / ∂β.

    Here we allow a general prior derivative dlog_pi_beta. For the
    simplest empirical Bayes implementation, set this to 0 (flat prior).
    """
    L = L.to(dtype=torch.double)
    omega = omega.to(dtype=torch.double)
    e = e.to(dtype=torch.double)

    G = L.numel()
    logL = torch.log(L.clamp_min(1.0))

    term1 = -logL.sum()
    term2 = (omega * logL).sum() / max(Lambda_omega, 1e-8)
    term3 = (omega * (e * e) * logL).sum()

    grad = 0.5 * (term1 + term2) + 0.5 * (G - 1) * term3 / max(RSS_omega, 1e-8)
    grad = grad + dlog_pi_beta
    return float(grad.item())


def grad_ell_rho_eta(
    L: torch.Tensor,
    g: torch.Tensor,
    beta: float,
    rho: float,
    k: int,
    eta: float,
    dlog_pi_rho: float = 0.0,
    dlog_pi_eta: float = 0.0,
) -> Tuple[float, float]:
    """
    ∂ℓ/∂ρ and ∂ℓ/∂η from Prop.~\\ref{prop:grad}, eq.~(EB-grad-general)
    combined with Lemma~\\ref{lem:kband-deriv}.

    For φ ∈ {ρ, η}, eq.~(EB-grad-general) says:

        ∂ℓ/∂φ
          = 1/2 [
                Σ_i ∂ log ω_i / ∂φ
                - (1/Λ_ω) Σ_i ∂ ω_i / ∂φ
              ]
            - (G - 1) / (2 RSS_ω)
                Σ_i ω_i e_i^2 ∂ log ω_i / ∂φ
            + ∂ log π(φ) / ∂φ.

    Lemma~\\ref{lem:kband-deriv} provides:

        ∂ log ω_i / ∂ρ = - (1 / s_i) ∂ s_i / ∂ρ,
        ∂ log ω_i / ∂η = - (1 / s_i) ∂ s_i / ∂η,

    with explicit formulas for ∂ s_i / ∂ρ and ∂ s_i / ∂η under the
    stretched-geometric k-band model. We implement those literally.
    """
    device = L.device
    Ld = L.to(dtype=torch.double)
    gd = g.to(dtype=torch.double)

    omega, w, m, e, Lambda_omega, RSS_omega = eb_stats(
        gd, Ld, beta=beta, rho=rho, k=k, eta=eta
    )
    omega = omega.to(dtype=torch.double)
    e = e.to(dtype=torch.double)
    G = Ld.numel()

    s_vals = s_kband(Ld, rho=rho, k=k, eta=eta).to(dtype=torch.double)
    s_vals = s_vals.clamp_min(1e-8)

    ds_drho_list = []
    ds_deta_list = []

    for idx in range(Ld.shape[0]):
        Li = float(Ld[idx].item())
        if Li <= 1 or k <= 0:
            ds_drho_list.append(0.0)
            ds_deta_list.append(0.0)
            continue

        m_i = min(k, int(Li) - 1)
        if m_i <= 0:
            ds_drho_list.append(0.0)
            ds_deta_list.append(0.0)
            continue

        h = torch.arange(1, m_i + 1, dtype=torch.double, device=device)
        if abs(eta) < 1e-8:
            # η ≈ 0 ⇒ h^η ≈ 1, ρ^{h^η} = ρ ⇒ ∂/∂ρ ρ^{h^η} = 1.
            ds_drho = (2.0 / Li) * (Li - h).sum()
            ds_deta = 0.0
        else:
            h_eta = h ** eta
            # ∂ s_i / ∂ρ = (2 / L_i) Σ (L_i - h) h^{η} ρ^{h^{η}-1}
            ds_drho = (2.0 / Li) * ((Li - h) * h_eta * (rho ** (h_eta - 1.0))).sum()
            # ∂ s_i / ∂η = (2 / L_i) Σ (L_i - h) ρ^{h^{η}} (log ρ) h^{η} log h
            if rho <= 0:
                ds_deta = 0.0
            else:
                log_rho = math.log(rho)
                ds_deta = (
                    (2.0 / Li)
                    * ((Li - h) * (rho ** h_eta) * log_rho * h_eta * torch.log(h))
                ).sum()

        ds_drho_list.append(float(ds_drho if isinstance(ds_drho, torch.Tensor) else ds_drho))
        ds_deta_list.append(float(ds_deta if isinstance(ds_deta, torch.Tensor) else ds_deta))

    ds_drho = torch.tensor(ds_drho_list, dtype=torch.double, device=device)
    ds_deta = torch.tensor(ds_deta_list, dtype=torch.double, device=device)

    # ∂ log ω_i / ∂ρ = - (1 / s_i) ∂ s_i / ∂ρ
    dlogw_drho = -ds_drho / s_vals
    domega_drho = omega * dlogw_drho

    # ∂ log ω_i / ∂η = - (1 / s_i) ∂ s_i / ∂η
    dlogw_deta = -ds_deta / s_vals
    domega_deta = omega * dlogw_deta

    def grad_phi(dlogw: torch.Tensor, domega: torch.Tensor, dlog_pi_phi: float) -> float:
        term1 = dlogw.sum()
        term2 = domega.sum() / max(Lambda_omega, 1e-8)
        term3 = (omega * (e * e) * dlogw).sum()
        grad = 0.5 * (term1 - term2) - 0.5 * (G - 1) * term3 / max(RSS_omega, 1e-8)
        grad = grad + dlog_pi_phi
        return float(grad.item())

    grad_rho = grad_phi(dlogw_drho, domega_drho, dlog_pi_rho)
    grad_eta = grad_phi(dlogw_deta, domega_deta, dlog_pi_eta)
    return grad_rho, grad_eta


def acf_moment_fit(
    Y: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    rho_max: float = 0.99,
    eta_max: float = 2.0,
    n_rho: int = 21,
    n_eta: int = 11,
) -> Tuple[float, float]:
    """
    Algorithm "ACF-moment" from §4: moment-based fit for ξ.

    Given token-level increments Y_{i,τ} and maximum lag k:

      1. For each trajectory i, compute empirical auto-covariances
         γ̂_i(h) for h = 0..k:

            γ̂_i(h)
              = (1 / (L_i - h)) Σ_{τ=1}^{L_i-h}
                  (Y_{i,τ} - Ȳ_i)(Y_{i,τ+h} - Ȳ_i).

      2. Aggregate:

            γ̄(h) = [ Σ_i (L_i - h) γ̂_i(h) ] / [ Σ_i (L_i - h) ].

      3. Normalize:

            r̄(h) = γ̄(h) / γ̄(0), h ≥ 1.

      4. Fit the stretched-geometric model r(h) ≈ ρ^{h^{η}} by
         least squares:

            (ρ̂, η̂) = argmin Σ_{h=1}^k (r̄(h) - ρ^{h^{η}})^2.

    We approximate step 4 via a coarse grid search over
    ρ ∈ [0, ρ_max], η ∈ [0, η_max].
    """
    device = Y.device
    Y = Y.to(dtype=torch.double)
    mask = mask.to(dtype=torch.bool)

    B, T = Y.shape
    L = mask.sum(dim=-1)

    max_h = k
    num = torch.zeros(max_h + 1, dtype=torch.double, device=device)
    den = torch.zeros(max_h + 1, dtype=torch.double, device=device)

    for i in range(B):
        Li = int(L[i].item())
        if Li <= 1:
            continue
        Yi = Y[i, :Li]
        Mi = mask[i, :Li]
        Yi = Yi[Mi]
        Li_eff = Yi.shape[0]
        if Li_eff <= 1:
            continue

        mu_i = Yi.mean()
        Yi_centered = Yi - mu_i

        for h in range(0, max_h + 1):
            if Li_eff <= h:
                continue
            length_h = Li_eff - h
            y1 = Yi_centered[:length_h]
            y2 = Yi_centered[h : h + length_h]
            gamma_i_h = (y1 * y2).sum() / float(length_h)
            num[h] += float(length_h) * gamma_i_h
            den[h] += float(length_h)

    gamma_bar = torch.zeros(max_h + 1, dtype=torch.double, device=device)
    nonzero = den > 0
    gamma_bar[nonzero] = num[nonzero] / den[nonzero]

    if gamma_bar[0] == 0 or not nonzero.any():
        return 0.0, 1.0

    r_bar = torch.zeros(max_h + 1, dtype=torch.double, device=device)
    for h in range(1, max_h + 1):
        if gamma_bar[h] == 0:
            r_bar[h] = 0.0
        else:
            r_bar[h] = gamma_bar[h] / gamma_bar[0]

    rho_grid = torch.linspace(0.0, rho_max, steps=n_rho, dtype=torch.double, device=device)
    eta_grid = torch.linspace(0.0, eta_max, steps=n_eta, dtype=torch.double, device=device)

    best_rho = 0.0
    best_eta = 1.0
    best_sse = float("inf")

    hs = torch.arange(1, max_h + 1, dtype=torch.double, device=device)

    for rho in rho_grid:
        for eta in eta_grid:
            if eta.item() == 0.0:
                h_eta = torch.ones_like(hs)
            else:
                h_eta = hs ** eta
            r_model = rho ** h_eta
            diffs = r_bar[1:] - r_model
            sse = float((diffs * diffs).sum().item())
            if sse < best_sse:
                best_sse = sse
                best_rho = float(rho.item())
                best_eta = float(eta.item())

    return best_rho, best_eta


@register_adv_est("capo_eb")
def compute_capo_eb_full_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config: Any = None,
    epsilon: float = 1e-8,
    beta_init: float = 1.0,
    rho_init: float = 0.0,
    eta_init: float = 1.0,
    k_band: int = 64,
    beta_steps: int = 3,
    xi_steps: int = 3,
    beta_lr: float = 0.1,
    rho_lr: float = 0.1,
    eta_lr: float = 0.1,
    rho_max: float = 0.99,
    eta_max: float = 3.0,
    use_acf_moment: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full EB–CAPO advantage estimator with k-banded dependence.

    This is a practical wrapper around the EB–CAPO workflow described
    in §4 and the theory in §3:

      - s(L; ρ, k, η): Definition~\\ref{def:kband}, eq.~(s-family).
      - ω_i(β, ξ) = L_i^{-β} / s(L_i; ξ).
      - ℓ(β, ξ): Theorem~\\ref{thm:EB-obj}, eq.~(EB-obj).
      - ∂ℓ/∂β: Corollary~\\ref{cor:grad-beta}, eq.~(EB-grad-beta).
      - ∂ℓ/∂ρ, ∂ℓ/∂η: Prop.~\\ref{prop:grad}, eq.~(EB-grad-general)
        + Lemma~\\ref{lem:kband-deriv}.
      - Optional ACF-based initialization of ξ via Algorithm "ACF-moment".

    We perform a small number of gradient-ascent steps in β and ξ,
    then use w_i(β̂, ξ̂) to define per-trajectory advantages:

        A_i = w_i(β̂, ξ̂) (g_i - m(β̂, ξ̂)),

    broadcast to all tokens of trajectory i, with an optional final
    std-normalization across valid tokens (GRPO-style).
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(
        token_level_rewards, response_mask
    )

    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards)
        returns = token_level_rewards.clone()
        return advantages, returns

    beta = float(beta_init)
    rho = float(rho_init)
    eta = float(eta_init)

    # Optional ACF-moment initialization of ξ = (ρ, η).
    if use_acf_moment and "increments" in kwargs and "increments_mask" in kwargs:
        Y = kwargs["increments"]
        Ymask = kwargs["increments_mask"]
        rho_hat, eta_hat = acf_moment_fit(Y=Y, mask=Ymask, k=k_band)
        alpha = float(kwargs.get("acf_init_smoothing", 1.0))
        rho = (1.0 - alpha) * rho + alpha * rho_hat
        eta = (1.0 - alpha) * eta + alpha * eta_hat

    dlog_pi_beta = float(kwargs.get("dlog_pi_beta", 0.0))
    dlog_pi_rho = float(kwargs.get("dlog_pi_rho", 0.0))
    dlog_pi_eta = float(kwargs.get("dlog_pi_eta", 0.0))

    # Gradient-ascent on β (length exponent).
    for _ in range(beta_steps):
        omega, w, m, e, Lambda_omega, RSS_omega = eb_stats(
            returns_scalar, lengths, beta=beta, rho=rho, k=k_band, eta=eta
        )
        g_beta = grad_ell_beta(
            L=lengths,
            omega=omega,
            e=e,
            Lambda_omega=Lambda_omega,
            RSS_omega=RSS_omega,
            dlog_pi_beta=dlog_pi_beta,
        )
        beta = beta + beta_lr * g_beta
        # Consistent with β ∈ [0, 2] in the length-only model.
        beta = min(max(beta, 0.0), 2.0)

    # Gradient-ascent on ξ = (ρ, η) (dependence shape).
    for _ in range(xi_steps):
        g_rho, g_eta = grad_ell_rho_eta(
            L=lengths,
            g=returns_scalar,
            beta=beta,
            rho=rho,
            k=k_band,
            eta=eta,
            dlog_pi_rho=dlog_pi_rho,
            dlog_pi_eta=dlog_pi_eta,
        )
        rho = rho + rho_lr * g_rho
        eta = eta + eta_lr * g_eta
        rho = max(0.0, min(rho, rho_max))
        eta = max(0.0, min(eta, eta_max))

    # Final EB stats with (β̂, ξ̂).
    omega, w, m, e, Lambda_omega, RSS_omega = eb_stats(
        returns_scalar, lengths, beta=beta, rho=rho, k=k_band, eta=eta
    )

    # Per-trajectory scalar advantages A_i = w_i (g_i - m).
    adv_scalar = w * (returns_scalar - m)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        valid_adv = advantages[valid]
        std = valid_adv.std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid
    return advantages, returns
