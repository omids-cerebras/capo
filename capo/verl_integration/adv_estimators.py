# capo/verl_integration/adv_estimators.py
"""
Advantage estimators for CAPO in VERL.

This module implements three CAPO-side advantage estimators:

- "capo": plain CAPO advantage, GRPO-style token-wise z-normalization
  of CAPO token-level rewards.

- "capo_eb_lite": EB–CAPO-lite (Algorithm~\\ref{alg:eb-lite}), which
  fits only a length exponent β (no dependence) using the EB-lite
  regression and reweights trajectories as w_i ∝ L_i^{-β̂}.

- "capo_eb": full EB–CAPO (Algorithm~\\ref{alg:eb-capo} and
  Algorithm~\\ref{alg:joint-eb-kband}), which treats both length and
  dependence under the k-banded stretched-geometric model and performs
  a joint EB update on (β, ρ, η) via `joint_eb_update_kband`.

All estimators take CAPO token-level rewards r_{i,t} as input and
produce token-level advantages A_{i,t} that VERL's PPO/GRPO trainer
consumes.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from capo.eb_core import (
    eb_lite_fit_beta_and_weights,
    joint_eb_update_kband,
)

Tensor = torch.Tensor


def _cfg_get(config, path, default):
    """Return config[path] for nested OmegaConf/dict/objects; else default."""
    if config is None:
        return default
    parts = path.split(".") if path else []
    cur = config
    for k in parts:
        if cur is None:
            return default
        # dict-like
        if isinstance(cur, dict):
            cur = cur.get(k, None)
            continue
        # OmegaConf / Namespace-like (attribute access)
        if hasattr(cur, k):
            cur = getattr(cur, k)
            continue
        # OmegaConf may allow key access
        try:
            cur = cur[k]
            continue
        except Exception:
            return default
    return default if cur is None else cur


def _lengths_and_scalar_returns(
    token_level_rewards: Tensor, response_mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute (L_i, g_i) from token-level rewards.

    Parameters
    ----------
    token_level_rewards : Tensor, shape [B, T]
        CAPO token-level rewards r_{i,t}.
    response_mask : Tensor[bool] or Tensor[int], shape [B, T]
        Mask indicating valid response tokens (1/True = valid).

    Returns
    -------
    lengths : Tensor, shape [B]
        Response lengths L_i = Σ_t 1_{mask_{i,t}}.
    returns_scalar : Tensor, shape [B]
        Scalarized returns g_i = \sum_{t \le L_i} r_{i,t}.
    valid : Tensor[bool], shape [B, T]
        Boolean mask of valid positions (same as response_mask > 0).
    """
    valid = response_mask > 0
    lengths = valid.sum(dim=-1).clamp_min(1).float()
    # Scalar return per trajectory.
    #
    # IMPORTANT: use a sum (not a mean). For outcome-only tasks like CountDown,
    # the reward is concentrated at the terminal token; a mean would
    # artificially shrink returns for longer responses.
    returns_scalar = (token_level_rewards * valid).sum(dim=-1)
    return lengths, returns_scalar, valid


def compute_capo_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: Any,
    config: Any = None,
    epsilon: float = 1e-8,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Plain CAPO advantage: GRPO-style token-wise z-normalization.

    This estimator corresponds to the "plain CAPO advantage" case
    in your description: we treat CAPO token rewards as the signal
    and perform the same kind of global z-norm that GRPO applies
    to its rewards.

    A_{i,t} = (r_{i,t} - μ_r) / σ_r,

    where μ_r and σ_r are the mean and std of r_{i,t} over all valid
    tokens in the batch.

    Parameters
    ----------
    token_level_rewards : Tensor, shape [B, T]
        CAPO token-level rewards r_{i,t}.
    response_mask : Tensor, shape [B, T]
        Mask for valid response tokens.
    index, config, epsilon, **kwargs :
        Additional arguments required by VERL's interface; `config`
        may carry flags such as `norm_adv_by_std_in_grpo`.

    Returns
    -------
    advantages : Tensor, shape [B, T]
        Token-level advantages.
    returns : Tensor, shape [B, T]
        Returns, here equal to token_level_rewards masked by response.
    """
    valid = response_mask > 0
    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards)
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


def compute_capo_eb_lite_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: Any,
    config: Any = None,
    epsilon: float = 1e-8,
    max_iters: int = 20,
    tol: float = 1e-4,
    **kwargs,
) -> Tuple[Tensor, Tensor, dict]:
    """
    EB–CAPO-lite advantage (Algorithm~\\ref{alg:eb-lite}).

    This estimator ignores dependence (s(L; ξ) ≡ 1) and fits only
    a length exponent β via the EB-lite regression. It then
    reweights trajectories as:

        w_i ∝ L_i^{-β̂},

    and constructs scalar advantages:

        A_i = w_i (g_i - m(β̂)),

    which are broadcast across all tokens in trajectory i.

    Parameters
    ----------
    token_level_rewards : Tensor, shape [B, T]
        CAPO token-level rewards r_{i,t}.
    response_mask : Tensor, shape [B, T]
        Mask for valid response tokens.
    index, config :
        Required by VERL; `config.norm_adv_by_std_in_grpo` controls
        a final std-normalization of the advantages.
    epsilon : float
        Numerical floor (used only if we decide to normalize again).
    max_iters : int
        Max iterations for EB-lite.
    tol : float
        Convergence tolerance for EB-lite.

    Returns
    -------
    advantages : Tensor, shape [B, T]
        Token-level advantages from EB–CAPO-lite.
    returns : Tensor, shape [B, T]
        Returns (here equal to CAPO token-level rewards masked).
    adv_metrics : dict
        Lightweight diagnostics for logging (JSON-serializable floats).
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(
        token_level_rewards, response_mask
    )

    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards)
        returns = token_level_rewards.clone()
        return advantages, returns, {}

    beta_hat, w, m = eb_lite_fit_beta_and_weights(
        g=returns_scalar, L=lengths, eps=epsilon, max_iters=max_iters, tol=tol,
    )

    # Per-trajectory scalar advantages A_i = w_i (g_i - m̂).
    adv_scalar = w * (returns_scalar - m)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    # Optional global std-normalization (GRPO-style).
    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        valid_adv = advantages[valid]
        std = valid_adv.std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid
    # Advantage estimator diagnostics (helpful for debugging / stability plots).
    with torch.no_grad():
        w_mean = w.mean()
        w_std = w.std(unbiased=False)
        w_cv = (w_std / (w_mean.abs() + epsilon)).item()
        w_centered = w - w_mean
        w_var = w_centered.pow(2).mean().clamp_min(epsilon)
        w_kurt = (w_centered.pow(4).mean() / (w_var * w_var)).item()
        ess = (w.sum().pow(2) / w.pow(2).sum().clamp_min(epsilon)).item()

        adv_metrics = {
            "capo/beta": float(beta_hat),
            "capo/weight_cv": float(w_cv),
            "capo/weight_kurtosis": float(w_kurt),
            "capo/weight_ess": float(ess),
        }

    return advantages, returns, adv_metrics


def compute_capo_eb_full_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: Any,
    config: Any = None,
    epsilon: float = 1e-8,
    beta_init: float = 1.0,
    rho_init: float = 0.0,
    eta_init: float = 1.0,
    k_band: int = 64,
    beta_steps: int = 1,
    xi_steps: int = 1,
    beta_lr: float = 0.1,
    rho_lr: float = 0.1,
    eta_lr: float = 0.1,
    rho_max: float = 0.99,
    beta_min: float = 0.0,
    beta_max: float = 2.0,
    ema_beta: float = 1.0,
    ema_xi: float = 1.0,
    use_acf_moment: bool = True,
    increments: Tensor | None = None,
    increments_mask: Tensor | None = None,
    dlog_pi_beta: float = 0.0,
    dlog_pi_rho: float = 0.0,
    dlog_pi_eta: float = 0.0,
    **kwargs,
) -> Tuple[Tensor, Tensor, dict]:
    """
    Full EB–CAPO advantage (Algorithms~\\ref{alg:eb-capo} and
    ~\\ref{alg:joint-eb-kband}).

    This estimator implements the full EB machinery:

    - A length exponent β_t,
    - A k-banded stretched-geometric dependence shape ξ_t = (ρ_t, η_t),
    - Per-batch EB updates of (β, ρ, η) via `joint_eb_update_kband`,
    - Trajectory weights w_i^{(t)} ∝ [L_i^{β_t} s(L_i; ρ_t, k, η_t)]^{-1},
    - Scalar advantages A_i = w_i^{(t)} (g_i - m_t).

    In the VERL integration, each call to this function corresponds
    to a single "outer step" t, using the current batch of trajectories
    in place of the full dataset ℬ_t.

    Parameters
    ----------
    token_level_rewards : Tensor, shape [B, T]
        CAPO token-level rewards r_{i,t}.
    response_mask : Tensor, shape [B, T]
        Mask for valid response tokens.
    index, config :
        Required by VERL; `config.norm_adv_by_std_in_grpo` controls
        a final std-normalization of the advantages.
    epsilon : float
        Numerical floor.
    beta_init, rho_init, eta_init : float
        Initial EB parameters (β_{t-1}, ρ_{t-1}, η_{t-1}) for this call.
        In a streaming setup, these can be carried in the Hydra config
        and updated over time.
    k_band : int
        Maximum lag k for the k-banded family s(L; ρ, k, η).
    beta_steps, xi_steps : int
        Number of gradient-ascent steps in β and ξ, respectively.
    beta_lr, rho_lr, eta_lr : float
        Learning rates for β, ρ, and η.
    rho_max : float
        Projection bound on |ρ|.
    beta_min, beta_max : float
        Bounding box [β_min, β_max] for β.
    ema_beta, ema_xi : float
        EMA smoothing coefficients γ_β and γ_ξ.
    use_acf_moment : bool
        Whether to use the ACF-moment estimator as a warm-start for
        (ρ, η).
    increments, increments_mask : Tensor, optional
        Token-level increments and mask for ACF-moment warm-start.
    dlog_pi_beta, dlog_pi_rho, dlog_pi_eta : float
        Prior gradients ∂ log π / ∂θ for θ ∈ {β, ρ, η}.

    Returns
    -------
    advantages : Tensor, shape [B, T]
        Token-level EB–CAPO advantages.
    returns : Tensor, shape [B, T]
        Returns (here equal to CAPO token-level rewards masked).
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(
        token_level_rewards, response_mask
    )

    if not torch.any(valid):
        advantages = torch.zeros_like(token_level_rewards)
        returns = token_level_rewards.clone()
        return advantages, returns, {}

    # Joint EB update for this batch.
    beta_t, rho_t, eta_t, w = joint_eb_update_kband(
        g=returns_scalar,
        L=lengths,
        beta_init=beta_init,
        rho_init=rho_init,
        eta_init=eta_init,
        k=k_band,
        lr_beta=beta_lr,
        lr_rho=rho_lr,
        lr_eta=eta_lr,
        steps_beta=beta_steps,
        steps_xi=xi_steps,
        ema_beta=ema_beta,
        ema_xi=ema_xi,
        beta_bounds=(beta_min, beta_max),
        rho_max=rho_max,
        dlog_pi_beta=dlog_pi_beta,
        dlog_pi_rho=dlog_pi_rho,
        dlog_pi_eta=dlog_pi_eta,
        use_acf_warmstart=use_acf_moment,
        increments=increments,
        increments_mask=increments_mask,
        eps=epsilon,
    )

    # Scalar advantages A_i = w_i (g_i - m_t), with m_t = Σ_i w_i g_i.
    m_t = (w * returns_scalar).sum()
    adv_scalar = w * (returns_scalar - m_t)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    # Optional global std-normalization (GRPO-style).
    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        valid_adv = advantages[valid]
        std = valid_adv.std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid

    # Diagnostics for logging.
    w = w.detach()
    w_mean = w.mean()
    w_std = w.std(unbiased=False)
    w_cv = w_std / (w_mean.abs() + epsilon)
    w_centered = w - w_mean
    w_var = w_centered.pow(2).mean().clamp_min(epsilon)
    w_kurt = w_centered.pow(4).mean() / (w_var.pow(2) + epsilon)
    w_ess = (w.sum().pow(2) / (w.pow(2).sum() + epsilon)).clamp_min(0.0)

    adv_metrics = {
        "capo/beta": float(beta_t),
        "capo/rho": float(rho_t),
        "capo/eta": float(eta_t),
        "capo/k_band": float(k_band),
        "capo/weight_cv": float(w_cv),
        "capo/weight_kurtosis": float(w_kurt),
        "capo/weight_ess": float(w_ess),
    }

    return advantages, returns, adv_metrics


# Alias for backward compatibility
compute_capo_empirical_bayes_advantage = compute_capo_eb_full_advantage

# Re-export functions from eb_core that tests expect to find here
from capo.eb_core import (  # noqa: F401, E402
    eb_lite_fit_beta_and_weights,
    s_kband,
    eb_statistics as eb_stats,
    grad_ell_beta_closed_form as grad_ell_beta,
    numeric_grad_rho_eta as grad_ell_rho_eta,
)
