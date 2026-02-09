r"""Synthetic benchmark — ten experiments showcasing CAPO.

This module generates **synthetic token-level rollouts** that mimic
LLM-generated trajectories with configurable within-trajectory
covariance structures, then feeds them through the **production**
CAPO / GRPO / ΔL advantage estimators to verify correctness and
compare methods on identical data.

Central thesis: **modelling within-trajectory token covariance**
($v(L) = L^\beta\, s(L;\xi)$) is strictly better than ignoring it
(GRPO) or using a fixed power-law correction ($\Delta L$).

Synthetic Rollout Generation
-----------------------------
Instead of generating scalar $(g_i, L_i)$ pairs, we generate full
token-level reward tensors of shape ``[B, T]`` with a response mask,
exactly mirroring what verl produces.  For each trajectory *i* of
length $L_i$, the token-level rewards $r_{i,1}, \ldots, r_{i,L_i}$
are drawn from a multivariate normal whose covariance encodes:

- **IID**: $\Sigma = I$
- **AR(1)**: $\Sigma_{st} = \rho^{|s-t|}$
- **Compound Symmetry**: $\Sigma_{st} = \rho \, \mathbb{1}[s \neq t]
  + \mathbb{1}[s=t]$
- **k-band stretched-geometric**: $\Sigma_{st} = \rho^{|s-t|^\eta}
  \, \mathbb{1}[|s-t| \le k]$

The sum $g_i = \sum_t r_{i,t}$ then has $\text{Var}(g_i) = v(L_i)$
matching the analytical variance laws used in the paper.

Ten Experiments
---------------
1.  Variance Law Landscape
2.  Weight Comparison (CAPO ≈ oracle)
3.  Relative Efficiency Heatmap (headline)
4.  MSE vs Correlation Strength  (monotone gain)
5.  Effective Sample Size
6.  Length-Bias Diagnostic
7.  Advantage Concentration
8.  L-CAPO Parameter Recovery
9.  Mixed-Dependence Stress Test
10. Scaling with Group Size N

All methods operate **within prompt groups** ($P$ groups of $N$
rollouts), matching the on-policy RLVR workflow.

Dependencies: numpy, matplotlib, seaborn, torch.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Production CAPO core — all EB algorithms live in eb_core (no verl dep).
# The advantage estimators in verl_integration/adv_estimators.py are thin
# wrappers that call these routines; we mirror that logic below to avoid
# pulling in the verl training stack.
from capo.eb_core import (
    eb_lite_fit_beta_and_weights,
    joint_eb_update_kband,
    s_kband,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Publication-quality defaults (seaborn + tuned rcParams)
# ═══════════════════════════════════════════════════════════════════════════

sns.set_theme(
    style="whitegrid",
    context="paper",
    font="serif",
    rc={
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
    },
)

# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

Length = int


@dataclass(frozen=True)
class Regime:
    """A covariance regime with a name and a covariance builder."""

    name: str
    build_cov: Callable[[int], np.ndarray]  # L -> (L, L) covariance
    v_of_L: Callable[[int], float]  # L -> scalar variance v(L)


@dataclass(frozen=True)
class Method:
    display: str
    key: str


# ═══════════════════════════════════════════════════════════════════════════
#  Covariance builders — produce (L, L) covariance matrices
# ═══════════════════════════════════════════════════════════════════════════


def cov_iid(L: int) -> np.ndarray:
    """Identity covariance: independent tokens."""
    return np.eye(L)


def cov_ar1(L: int, rho: float) -> np.ndarray:
    """AR(1) covariance: Sigma_{st} = rho^|s-t|."""
    idx = np.arange(L)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def cov_compound_symmetry(L: int, rho: float) -> np.ndarray:
    """Compound symmetry: Sigma_{st} = rho 1[s!=t] + 1[s=t]."""
    return rho * np.ones((L, L)) + (1.0 - rho) * np.eye(L)


def cov_kband(L: int, rho: float, k: int, eta: float) -> np.ndarray:
    """k-banded stretched-geometric covariance (vectorised)."""
    idx = np.arange(L)
    diff = np.abs(idx[:, None] - idx[None, :])  # (L, L)
    band = diff <= k
    C = np.where(band, rho ** (diff**eta), 0.0)
    np.fill_diagonal(C, 1.0)
    return C


# ═══════════════════════════════════════════════════════════════════════════
#  Analytical variance laws  v(L) = 1^T Sigma 1
# ═══════════════════════════════════════════════════════════════════════════


def v_iid(L: int) -> float:
    return float(L)


def v_ar1(L: int, rho: float) -> float:
    L = int(L)
    h = np.arange(1, L)
    s = float(np.sum((L - h) * rho**h))
    return float(L + 2.0 * s)


def v_kband(L: int, rho: float, k: int, eta: float) -> float:
    L = int(L)
    m = min(int(k), L - 1)
    h = np.arange(1, m + 1)
    s = float(np.sum((L - h) * rho ** (h**eta)))
    return float(L + 2.0 * s)


def v_compound_symmetry(L: int, rho_cs: float) -> float:
    L = int(L)
    return float(L + rho_cs * L * (L - 1))


def v_power_law(L: int, beta: float) -> float:
    return float(int(L) ** beta)


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic rollout generation — token-level rewards with covariance
# ═══════════════════════════════════════════════════════════════════════════


def _cholesky_cache(
    length_bins: np.ndarray,
    build_cov: Callable[[int], np.ndarray],
) -> dict[int, np.ndarray]:
    """Pre-compute Cholesky factors for each length bin."""
    cache: dict[int, np.ndarray] = {}
    for L in length_bins:
        L = int(L)
        C = build_cov(L)
        cache[L] = np.linalg.cholesky(C)
    return cache


def generate_rollouts(
    rng: np.random.Generator,
    B: int,
    length_bins: np.ndarray,
    build_cov: Callable[[int], np.ndarray],
    T_max: int | None = None,
    length_probs: np.ndarray | None = None,
    chol_cache: dict[int, np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate synthetic token-level rollouts with covariance structure.

    For each trajectory i:
      - Sample L_i uniformly from length_bins
      - Draw z ~ N(0, I_L) and set r = chol(Sigma_L) @ z
      - Pack into [B, T_max] with zero-padding

    Returns
    -------
    token_level_rewards : torch.Tensor, shape [B, T_max]
    response_mask : torch.Tensor, shape [B, T_max]
    lengths : np.ndarray, shape [B], dtype int
    """
    if T_max is None:
        T_max = int(length_bins.max())

    if chol_cache is None:
        chol_cache = _cholesky_cache(length_bins, build_cov)

    lengths = rng.choice(length_bins, size=B, replace=True, p=length_probs)
    rewards = np.zeros((B, T_max), dtype=np.float32)
    mask = np.zeros((B, T_max), dtype=np.float32)

    # Batch generation per length bin (one matmul per unique length)
    for L_val in np.unique(lengths):
        L_i = int(L_val)
        idxs = np.where(lengths == L_val)[0]
        n = len(idxs)
        chol = chol_cache[L_i]  # (L_i, L_i)
        Z = rng.standard_normal((n, L_i))  # (n, L_i)
        R = Z @ chol.T  # (n, L_i)
        rewards[idxs, :L_i] = R.astype(np.float32)
        mask[idxs, :L_i] = 1.0

    return (
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        lengths.astype(int),
    )


def generate_grouped_rollouts(
    rng: np.random.Generator,
    P: int,
    N: int,
    length_bins: np.ndarray,
    build_cov: Callable[[int], np.ndarray],
    T_max: int | None = None,
    chol_cache: dict[int, np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Generate P prompt groups of N rollouts each.

    Returns
    -------
    token_level_rewards : Tensor [P*N, T_max]
    response_mask : Tensor [P*N, T_max]
    lengths : np.ndarray [P*N]
    index : np.ndarray [P*N]  — prompt group IDs
    """
    B = P * N
    rewards, mask, lengths = generate_rollouts(
        rng,
        B,
        length_bins,
        build_cov,
        T_max,
        chol_cache=chol_cache,
    )
    index = np.repeat(np.arange(P), N)
    return rewards, mask, lengths, index


# ═══════════════════════════════════════════════════════════════════════════
#  Advantage estimators — wrappers around the real production code
# ═══════════════════════════════════════════════════════════════════════════


def _make_config(norm: bool = False):
    """Minimal config object for advantage estimators."""

    class _C:
        norm_adv_by_std_in_grpo = norm

    return _C()


# ---- Production-mirroring CAPO wrappers (no verl dependency) -----------
#
# These functions replicate the logic of
# ``capo.verl_integration.adv_estimators.compute_capo_eb_lite_advantage``
# and ``compute_capo_eb_full_advantage`` **verbatim** but without
# importing from ``verl_integration`` (whose ``__init__.py`` pulls in
# the verl training stack).  All the real work is done by ``eb_core``
# functions which are imported at module level above.


def _lengths_and_scalar_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same helper as ``adv_estimators._lengths_and_scalar_returns``."""
    valid = response_mask > 0
    lengths = valid.sum(dim=-1).clamp_min(1).float()
    returns_scalar = (token_level_rewards * valid).sum(dim=-1)
    return lengths, returns_scalar, valid


def _groupwise_advantages(
    omega: torch.Tensor,
    g: torch.Tensor,
    index,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group weight normalisation and baseline (mirrors production helper)."""
    _ = omega.shape[0]
    w = torch.empty_like(omega)
    adv_scalar = torch.empty_like(omega)

    if index is None:
        omega_sum = omega.sum().clamp_min(eps)
        w[:] = omega / omega_sum
        m = (w * g).sum()
        adv_scalar[:] = w * (g - m)
        return w, adv_scalar

    idx_t = torch.as_tensor(index, dtype=torch.long, device=omega.device)
    unique_groups = idx_t.unique()

    for gid in unique_groups:
        mask = idx_t == gid
        om_g = omega[mask]
        g_g = g[mask]
        om_sum = om_g.sum().clamp_min(eps)
        w_g = om_g / om_sum
        m_g = (w_g * g_g).sum()
        w[mask] = w_g
        adv_scalar[mask] = w_g * (g_g - m_g)

    return w, adv_scalar


def compute_capo_eb_lite_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    config=None,
    epsilon: float = 1e-8,
    max_iters: int = 20,
    tol: float = 1e-4,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """L-CAPO advantage — mirrors production ``adv_estimators``."""
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(token_level_rewards, response_mask)

    if not torch.any(valid):
        z = torch.zeros_like(token_level_rewards)
        return z, z.clone(), {}

    beta_hat, _w_global, _m_global = eb_lite_fit_beta_and_weights(
        g=returns_scalar,
        L=lengths,
        eps=epsilon,
        max_iters=max_iters,
        tol=tol,
    )

    omega = lengths.double().pow(-beta_hat).float()
    w, adv_scalar = _groupwise_advantages(omega, returns_scalar, index, eps=epsilon)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        std = advantages[valid].std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid
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
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    config=None,
    epsilon: float = 1e-8,
    beta_init: float = 1.0,
    rho_init: float = 0.0,
    eta_init: float = 1.0,
    k_band: int = 64,
    beta_steps: int = 10,
    xi_steps: int = 1,
    beta_lr: float = 0.001,
    rho_lr: float = 0.01,
    eta_lr: float = 0.01,
    rho_max: float = 0.99,
    beta_min: float = 0.0,
    beta_max: float = 2.0,
    ema_beta: float = 1.0,
    ema_xi: float = 1.0,
    use_acf_moment: bool = True,
    increments=None,
    increments_mask=None,
    dlog_pi_beta: float = 0.0,
    dlog_pi_rho: float = 0.0,
    dlog_pi_eta: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Full LV-CAPO advantage — mirrors production ``adv_estimators``.

    Note: defaults for beta_steps/xi_steps/beta_lr are tuned for the
    *single-batch* regime used in this benchmark.  In production (the
    streaming setting) the caller carries (beta, rho, eta) across
    batches and 1 step per batch suffices.
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(token_level_rewards, response_mask)

    if not torch.any(valid):
        z = torch.zeros_like(token_level_rewards)
        return z, z.clone(), {}

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

    s_vals = s_kband(lengths, rho_t, k_band, eta_t)
    omega = (lengths.double().pow(beta_t) * s_vals.double()).reciprocal().float().clamp_min(epsilon)
    w, adv_scalar = _groupwise_advantages(omega, returns_scalar, index, eps=epsilon)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()

    if config is not None and getattr(config, "norm_adv_by_std_in_grpo", False):
        std = advantages[valid].std().clamp_min(epsilon)
        advantages = advantages / std

    returns = token_level_rewards * valid
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


def compute_grpo_advantage_grouped(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""GRPO advantage using the unified weighted-baseline form.

    GRPO uses uniform precision weights $\omega_i = 1$, so the
    per-group baseline is the unweighted mean $m_p = \bar g_p$
    and the advantage is $A_i = w_i (g_i - m_p)$ with $w_i = 1/N_p$.

    This is structurally identical to CAPO with $\beta = 0$, making
    the comparison apples-to-apples.
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(token_level_rewards, response_mask)
    omega = torch.ones_like(lengths)  # uniform weights
    w, adv_scalar = _groupwise_advantages(omega, returns_scalar, index, eps=epsilon)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()
    returns = token_level_rewards * valid
    return advantages, returns


def compute_deltaL_advantage_grouped(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""$\Delta L$ advantage using the unified weighted-baseline form.

    Uses precision weights $\omega_i = L_i^{-\alpha}$, normalised
    within each prompt group — structurally identical to CAPO with a
    fixed exponent $\beta = \alpha$ and $s \equiv 1$.

    This replaces the original two-stage "GRPO z-norm then multiply
    by length factor" recipe with the same $A_i = w_i(g_i - m_p)$
    formula used by CAPO, isolating the **weighting rule** as the only
    variable across all methods.
    """
    lengths, returns_scalar, valid = _lengths_and_scalar_returns(token_level_rewards, response_mask)
    omega = lengths.pow(-alpha)  # fixed-exponent inverse-length weights
    w, adv_scalar = _groupwise_advantages(omega, returns_scalar, index, eps=epsilon)
    advantages = adv_scalar.unsqueeze(-1) * valid.float()
    returns = token_level_rewards * valid
    return advantages, returns


def compute_all_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    method_keys: list[str],
    eb_full_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Compute advantages for all requested methods on the same batch.

    Returns dict mapping method key -> advantages Tensor [B, T].
    """
    results = {}

    if "grpo" in method_keys:
        adv, _ = compute_grpo_advantage_grouped(token_level_rewards, response_mask, index)
        results["grpo"] = adv

    if "deltal_0.5" in method_keys:
        adv, _ = compute_deltaL_advantage_grouped(
            token_level_rewards, response_mask, index, alpha=0.5
        )
        results["deltal_0.5"] = adv

    if "deltal_1.0" in method_keys:
        adv, _ = compute_deltaL_advantage_grouped(
            token_level_rewards, response_mask, index, alpha=1.0
        )
        results["deltal_1.0"] = adv

    if "l_capo" in method_keys:
        adv, _, _ = compute_capo_eb_lite_advantage(
            token_level_rewards, response_mask, index=index, config=_make_config(norm=False)
        )
        results["l_capo"] = adv

    if "lv_capo" in method_keys:
        kw = eb_full_kwargs or {}
        adv, _, _ = compute_capo_eb_full_advantage(
            token_level_rewards, response_mask, index=index, config=_make_config(norm=False), **kw
        )
        results["lv_capo"] = adv

    return results


def _groupwise_baseline(
    omega: torch.Tensor,
    g: torch.Tensor,
    index: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Per-group weighted-mean baseline, averaged across groups.

    Uses the same per-group normalisation as ``_groupwise_advantages``:
    within each group, $w_j = \omega_j / \sum_{k \in I_p} \omega_k$,
    then $m_p = \sum_{j \in I_p} w_j g_j$.  Returns the simple
    average of per-group baselines.
    """
    idx_t = torch.as_tensor(index, dtype=torch.long, device=omega.device)
    m_groups = []
    for gid in idx_t.unique():
        mask = idx_t == gid
        om_g = omega[mask]
        g_g = g[mask]
        w_g = om_g / om_g.sum().clamp_min(eps)
        m_groups.append((w_g * g_g).sum().item())
    return float(np.mean(m_groups))


def compute_scalar_baselines(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    method_keys: list[str],
) -> dict[str, float]:
    """Compute the scalar weighted-mean baseline for each method.

    Every method uses the same per-group normalisation structure;
    only the precision weights $\omega_i$ differ.  This makes the
    comparison apples-to-apples.
    """
    lengths = response_mask.sum(dim=-1).clamp_min(1).float()
    returns_scalar = (token_level_rewards * response_mask).sum(dim=-1)
    L = lengths
    g = returns_scalar

    out: dict[str, float] = {}

    if "grpo" in method_keys:
        omega = torch.ones_like(L)
        out["grpo"] = _groupwise_baseline(omega, g, index)

    for alpha_val, key in [(0.5, "deltal_0.5"), (1.0, "deltal_1.0")]:
        if key in method_keys:
            omega = L.pow(-alpha_val)
            out[key] = _groupwise_baseline(omega, g, index)

    if "l_capo" in method_keys:
        beta_hat, _w, _m = eb_lite_fit_beta_and_weights(g, L)
        omega = L.double().pow(-beta_hat).float()
        out["l_capo"] = _groupwise_baseline(omega, g, index)

    if "lv_capo" in method_keys:
        beta_t, rho_t, eta_t, _w = joint_eb_update_kband(
            g,
            L,
            beta_init=1.0,
            rho_init=0.0,
            eta_init=1.0,
            k=64,
            steps_beta=10,
            steps_xi=1,
            lr_beta=0.001,
            lr_rho=0.01,
            lr_eta=0.01,
        )
        s_vals = s_kband(L, rho_t, 64, eta_t)
        omega = (L.double().pow(beta_t) * s_vals.double()).reciprocal().float().clamp_min(1e-8)
        out["lv_capo"] = _groupwise_baseline(omega, g, index)

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Styling — seaborn-compatible palette
# ═══════════════════════════════════════════════════════════════════════════

_PAL = {
    "grpo": "#d62728",
    "deltal_0.5": "#ff7f0e",
    "deltal_1.0": "#bcbd22",
    "l_capo": "#1f77b4",
    "lv_capo": "#2ca02c",
}
_MK = {"grpo": "o", "deltal_0.5": "v", "deltal_1.0": "^", "l_capo": "D", "lv_capo": "s"}
_LS = {"grpo": "--", "deltal_0.5": ":", "deltal_1.0": ":", "l_capo": "-", "lv_capo": "-"}
_ZO = {"grpo": 2, "deltal_0.5": 2, "deltal_1.0": 2, "l_capo": 5, "lv_capo": 6}
_LW = {"grpo": 1.6, "deltal_0.5": 1.4, "deltal_1.0": 1.4, "l_capo": 2.3, "lv_capo": 2.6}


def _sty(key: str) -> dict:
    return dict(
        color=_PAL.get(key, "#888"),
        marker=_MK.get(key, "o"),
        linestyle=_LS.get(key, "-"),
        linewidth=_LW.get(key, 1.8),
        zorder=_ZO.get(key, 3),
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=6,
    )


def _save(fig, stem: Path) -> None:
    """Save a figure as both PDF and high-quality PNG."""
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Standard catalogues  —  STRONG covariance, LLM-scale lengths
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_LENGTH_BINS = np.array([128, 256, 512, 1024, 2048])

REGIMES: list[Regime] = [
    Regime("IID", lambda L: cov_iid(int(L)), v_iid),
    Regime(r"AR(1) $\rho$=0.5", lambda L: cov_ar1(int(L), 0.5), lambda L: v_ar1(int(L), 0.5)),
    Regime(r"AR(1) $\rho$=0.8", lambda L: cov_ar1(int(L), 0.8), lambda L: v_ar1(int(L), 0.8)),
    Regime(r"AR(1) $\rho$=0.95", lambda L: cov_ar1(int(L), 0.95), lambda L: v_ar1(int(L), 0.95)),
    Regime(
        r"CS $\rho$=0.02",
        lambda L: cov_compound_symmetry(int(L), 0.02),
        lambda L: v_compound_symmetry(int(L), 0.02),
    ),
    Regime(
        r"CS $\rho$=0.05",
        lambda L: cov_compound_symmetry(int(L), 0.05),
        lambda L: v_compound_symmetry(int(L), 0.05),
    ),
    Regime(
        r"$k$-band (0.8,64,1.3)",
        lambda L: cov_kband(int(L), 0.8, 64, 1.3),
        lambda L: v_kband(int(L), 0.8, 64, 1.3),
    ),
]

METHODS: list[Method] = [
    Method("GRPO", "grpo"),
    Method(r"$\Delta L$ ($\alpha$=0.5)", "deltal_0.5"),
    Method(r"$\Delta L$ ($\alpha$=1.0)", "deltal_1.0"),
    Method("L-CAPO", "l_capo"),
    Method("LV-CAPO", "lv_capo"),
]

METHODS4: list[Method] = [
    Method("GRPO", "grpo"),
    Method(r"$\Delta L$ ($\alpha$=1.0)", "deltal_1.0"),
    Method("L-CAPO", "l_capo"),
    Method("LV-CAPO", "lv_capo"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════════


def _relative_efficiency(baselines_dict: dict, ref_key: str = "grpo"):
    var_ref = float(np.var(baselines_dict[ref_key]))
    out = {}
    for k, vals in baselines_dict.items():
        var_k = float(np.var(vals))
        out[k] = var_ref / var_k if var_k > 1e-30 else 1.0
    return out


def _latex_table(data, row_labels, col_labels, bold_best="max", digits=2):
    col_spec = "l" + "c" * len(col_labels)
    header = ["Method"] + col_labels
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    best = {}
    for col in col_labels:
        vals = {r: data.get(col, {}).get(r, float("nan")) for r in row_labels}
        ok = {k: v for k, v in vals.items() if not math.isnan(v)}
        if ok:
            best[col] = (max if bold_best == "max" else min)(ok, key=ok.get)
    for row in row_labels:
        cells = [row]
        for col in col_labels:
            v = data.get(col, {}).get(row, float("nan"))
            c = f"{v:.{digits}f}" if not math.isnan(v) else "---"
            if best.get(col) == row and not math.isnan(v):
                c = f"\\textbf{{{c}}}"
            cells.append(c)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ######################################################################## #
#  EXPERIMENT 1 — Variance Law Landscape                                   #
# ######################################################################## #


def experiment_01_variance_landscape(out: Path) -> None:
    """How token dependence makes v(L) explode super-linearly."""
    L = np.linspace(8, 2048, 500)

    laws = [
        ("IID ($v=L$)", lambda x: v_iid(int(x)), "#aaaaaa", "-", 1.8),
        ("AR(1) $\\rho$=0.5", lambda x: v_ar1(int(x), 0.5), _PAL["deltal_0.5"], "--", 1.8),
        ("AR(1) $\\rho$=0.8", lambda x: v_ar1(int(x), 0.8), _PAL["deltal_1.0"], "--", 2.0),
        ("AR(1) $\\rho$=0.95", lambda x: v_ar1(int(x), 0.95), _PAL["grpo"], "-", 2.2),
        ("CS $\\rho$=0.02", lambda x: v_compound_symmetry(int(x), 0.02), "#9467bd", "-.", 1.8),
        ("CS $\\rho$=0.05", lambda x: v_compound_symmetry(int(x), 0.05), "#e377c2", "-.", 2.0),
        (
            "$k$-band (0.8,64,1.3)",
            lambda x: v_kband(int(x), 0.8, 64, 1.3),
            _PAL["l_capo"],
            "-",
            2.0,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) v(L)
    ax = axes[0]
    for nm, vf, c, ls, lw in laws:
        ax.semilogy(L, [vf(x) for x in L], color=c, ls=ls, lw=lw, label=nm)
    ax.set_xlabel("Trajectory length $L$")
    ax.set_ylabel("$v(L)$  [log scale]")
    ax.set_title("(a) Induced variance $v(L)$", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # (b) v(L)/L
    ax = axes[1]
    for nm, vf, c, ls, lw in laws:
        vs = np.array([vf(x) for x in L])
        ax.plot(L, vs / L, color=c, ls=ls, lw=lw, label=nm)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("Trajectory length $L$")
    ax.set_yscale("log")
    ax.set_ylabel("$v(L)/L$  (amplification)")
    ax.set_title("(b) Per-token variance amplification", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # (c) omega(L) = 1/v(L) normalised
    ax = axes[2]
    for nm, vf, c, ls, lw in laws:
        vs = np.array([vf(x) for x in L])
        om = 1.0 / vs
        om /= om[0]
        ax.plot(L, om, color=c, ls=ls, lw=lw, label=nm)
    ax.set_xlabel("Trajectory length $L$")
    ax.set_yscale("log")
    ax.set_ylabel("$\\omega(L)/\\omega(L_{\\min})$")
    ax.set_title("(c) CAPO precision weight", fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

    fig.suptitle(
        "Experiment 1 — Variance Law Landscape: Token Dependence "
        "Causes Super-Linear Variance Growth",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_01_variance_landscape")
    print("  [1/10] Variance landscape done")


# ######################################################################## #
#  EXPERIMENT 2 — Weight Comparison                                        #
# ######################################################################## #


def experiment_02_weight_comparison(out: Path) -> None:
    """CAPO's weights approximate oracle; GRPO and DeltaL don't."""
    bins = DEFAULT_LENGTH_BINS
    panels = [
        ("AR(1) $\\rho$=0.8", lambda L: v_ar1(int(L), 0.8)),
        ("CS $\\rho$=0.05", lambda L: v_compound_symmetry(int(L), 0.05)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for pi, (pname, vfn) in enumerate(panels):
        ax = axes[pi]
        v = np.array([vfn(L) for L in bins])

        # Oracle weights from CAPO theory
        w_oracle = 1.0 / v
        w_oracle /= w_oracle.sum()
        # GRPO: uniform
        w_grpo = np.ones(len(bins)) / len(bins)
        # DeltaL(alpha=1): w proportional to 1/L
        w_dl = bins.astype(float) ** (-1.0)
        w_dl /= w_dl.sum()
        # L-CAPO: use production eb_lite_fit to find beta from oracle v(L)
        L_t = torch.tensor(bins, dtype=torch.float32)
        g_t = torch.tensor(v, dtype=torch.float32).sqrt()  # dummy g with right scale
        beta_hat, w_t, _ = eb_lite_fit_beta_and_weights(g_t, L_t)
        w_lite = L_t.pow(-beta_hat).numpy()
        w_lite /= w_lite.sum()

        x = np.arange(len(bins))
        w = 0.18
        ax.bar(
            x - 1.5 * w,
            w_oracle,
            w,
            label="Oracle ($1/v(L)$)",
            color=_PAL["lv_capo"],
            edgecolor="white",
            lw=0.8,
            zorder=4,
        )
        ax.bar(
            x - 0.5 * w,
            w_lite,
            w,
            label=f"L-CAPO ($\\beta$={beta_hat:.2f})",
            color=_PAL["l_capo"],
            edgecolor="white",
            lw=0.8,
            zorder=3,
        )
        ax.bar(
            x + 0.5 * w,
            w_dl,
            w,
            label="$\\Delta L$ ($\\alpha$=1)",
            color=_PAL["deltal_1.0"],
            edgecolor="white",
            lw=0.8,
            zorder=2,
        )
        ax.bar(
            x + 1.5 * w,
            w_grpo,
            w,
            label="GRPO (uniform)",
            color=_PAL["grpo"],
            edgecolor="white",
            lw=0.8,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"$L$={L}" for L in bins])
        ax.set_ylabel("Normalised weight $w(L)$")
        ax.set_title(f"({chr(97+pi)}) {pname}", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0)

    fig.suptitle(
        "Experiment 2 — Weight Comparison: CAPO Approximates " "the Oracle",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_02_weight_comparison")
    print("  [2/10] Weight comparison done")


# ######################################################################## #
#  EXPERIMENT 3 — Relative Efficiency Heatmap                              #
# ######################################################################## #


def experiment_03_relative_efficiency(out, seed, P, N, bins, n_mc=200):
    """Headline result: CAPO dominates every covariance regime."""
    regimes = REGIMES
    methods = METHODS
    mkeys = [m.key for m in methods]
    T_max = int(bins.max())

    eff: dict[str, dict[str, float]] = {}

    for reg in regimes:
        chol_cache = _cholesky_cache(bins, reg.build_cov)
        bl: dict[str, list] = {k: [] for k in mkeys}

        for r in range(n_mc):
            rng = np.random.default_rng(seed * 100_000 + r)
            rewards, mask, lengths, index = generate_grouped_rollouts(
                rng, P, N, bins, reg.build_cov, T_max, chol_cache
            )
            b = compute_scalar_baselines(rewards, mask, index, mkeys)
            for k in mkeys:
                bl[k].append(b[k])

        var_u = float(np.var(bl["grpo"]))
        eff[reg.name] = {}
        for m in methods:
            var_m = float(np.var(bl[m.key]))
            eff[reg.name][m.display] = var_u / var_m if var_m > 1e-30 else 1.0

    mat = np.array([[eff[reg.name][m.display] for reg in regimes] for m in methods])

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = LinearSegmentedColormap.from_list(
        "eff", ["#ffffff", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
    )
    vmax = max(float(mat.max()) * 1.05, 2.0)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.8, vmax=vmax)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.name for r in regimes], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.display for m in methods], fontsize=10)

    for ri in range(len(regimes)):
        best = int(np.argmax(mat[:, ri]))
        for mi in range(len(methods)):
            v = mat[mi, ri]
            tc = "white" if v > 0.6 * vmax else "black"
            txt = f"{v:.1f}x" if v >= 10 else f"{v:.2f}x"
            ax.text(
                ri,
                mi,
                txt,
                ha="center",
                va="center",
                fontsize=11 if mi == best else 9,
                color=tc,
                fontweight="bold" if mi == best else "normal",
            )

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Relative Efficiency  Var(GRPO)/Var(method)", fontsize=10)
    ax.set_title(
        "Experiment 3 — Relative Efficiency: CAPO Dominates " "Every Regime",
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    _save(fig, out / "fig_03_relative_efficiency")

    tab: dict[str, dict[str, float]] = {}
    for reg in regimes:
        tab[reg.name] = {m.display: eff[reg.name][m.display] for m in methods}
    tex = _latex_table(
        tab, [m.display for m in methods], [r.name for r in regimes], bold_best="max"
    )
    (out / "tab_03_relative_efficiency.tex").write_text(tex, encoding="utf-8")
    print("  [3/10] Relative efficiency done")
    return eff


# ######################################################################## #
#  EXPERIMENT 4 — MSE vs Correlation Strength                              #
# ######################################################################## #


def experiment_04_mse_vs_correlation(out, seed, P, N, bins, n_mc=200):
    """Show CAPO's gain from modelling covariance, deconfounded from length.

    Panel (a): RE vs GRPO — all methods improve with rho.
    Panel (b): RE vs DeltaL(alpha=1) — isolates covariance-modelling gain.
    """
    methods = METHODS4
    rhos = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    T_max = int(bins.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel_idx, ref_key in enumerate(["grpo", "deltal_1.0"]):
        ax = axes[panel_idx]
        res: dict[str, list] = {m.key: [] for m in methods}

        for rho in rhos:
            build_cov = (
                (lambda L, _r=rho: cov_ar1(int(L), _r)) if rho > 0 else (lambda L: cov_iid(int(L)))
            )
            chol_cache = _cholesky_cache(bins, build_cov)

            mkeys = [m.key for m in methods]
            bl: dict[str, list] = {k: [] for k in mkeys}
            for r in range(n_mc):
                rng = np.random.default_rng(seed * 100_000 + r)
                rewards, mask, lengths, index = generate_grouped_rollouts(
                    rng, P, N, bins, build_cov, T_max, chol_cache
                )
                b = compute_scalar_baselines(rewards, mask, index, mkeys)
                for k in mkeys:
                    bl[k].append(b[k])
            re = _relative_efficiency(bl, ref_key=ref_key)
            for m in methods:
                res[m.key].append(re[m.key])

        for m in methods:
            ax.plot(rhos, res[m.key], label=m.display, **_sty(m.key))
        ax.axhline(1.0, color="grey", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel(r"$\rho$  (AR(1) within-trajectory correlation)")

        if panel_idx == 0:
            ax.set_ylabel("Relative Efficiency vs GRPO")
            ax.set_title("(a) RE vs GRPO (all gains)", fontweight="bold")
        else:
            ax.set_ylabel(r"Relative Efficiency vs $\Delta L$ ($\alpha$=1)")
            ax.set_title(r"(b) RE vs $\Delta L$ (covariance-only gain)", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Experiment 4 — Deconfounding Length from Covariance: "
        "CAPO's Gain is from Modelling Dependence",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_04_mse_vs_correlation")
    print("  [4/10] MSE vs correlation done")


# ######################################################################## #
#  EXPERIMENT 5 — Effective Sample Size                                    #
# ######################################################################## #


def experiment_05_effective_sample_size(out, seed, P, N, bins, n_mc=200):
    regimes = [r for r in REGIMES if "IID" not in r.name]
    methods = METHODS4
    T_max = int(bins.max())

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(regimes))
    bw = 0.18
    m_ess: dict[str, list] = {m.key: [] for m in methods}

    for _ri, reg in enumerate(regimes):
        chol_cache = _cholesky_cache(bins, reg.build_cov)
        mkeys = [m.key for m in methods]
        bl: dict[str, list] = {k: [] for k in mkeys}

        for r in range(n_mc):
            rng = np.random.default_rng(seed * 100_000 + r)
            rewards, mask, lengths, index = generate_grouped_rollouts(
                rng, P, N, bins, reg.build_cov, T_max, chol_cache
            )
            b = compute_scalar_baselines(rewards, mask, index, mkeys)
            for k in mkeys:
                bl[k].append(b[k])

        var_grpo = float(np.var(bl["grpo"]))
        for m in methods:
            var_m = float(np.var(bl[m.key]))
            m_ess[m.key].append(N * (var_grpo / var_m) if var_m > 1e-30 else N)

    for mi, m in enumerate(methods):
        bars = ax.bar(
            x + mi * bw,
            m_ess[m.key],
            bw,
            label=m.display,
            color=_PAL.get(m.key, "#888"),
            edgecolor="white",
            lw=0.8,
        )
        for bar, val in zip(bars, m_ess[m.key], strict=False):
            if val > N * 1.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if "capo" in m.key else "normal",
                )

    ax.axhline(N, color="grey", lw=1.2, ls="--", label=f"Nominal $N$={N}")
    ax.set_xticks(x + bw * (len(methods) - 1) / 2)
    ax.set_xticklabels([r.name for r in regimes], fontsize=9)
    ax.set_ylabel("Effective Sample Size (ESS)")
    ax.set_title(
        "Experiment 5 — Effective Sample Size: CAPO Extracts " "More from the Same Batch",
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0)
    fig.tight_layout()
    _save(fig, out / "fig_05_effective_sample_size")
    print("  [5/10] Effective sample size done")


# ######################################################################## #
#  EXPERIMENT 6 — Length-Bias Diagnostic                                   #
# ######################################################################## #


def experiment_06_length_bias(out, seed, P, N, bins, n_mc=300):
    """GRPO over-weights long noisy trajectories; CAPO eliminates this."""

    def build_cov(L, _r=0.8):
        return cov_ar1(int(L), _r)

    methods = METHODS4
    T_max = int(bins.max())
    chol_cache = _cholesky_cache(bins, build_cov)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) E[A^2 | L]
    ax = axes[0]
    msa_acc: dict[str, dict[int, list]] = {m.key: {int(L): [] for L in bins} for m in methods}

    for rep in range(n_mc):
        rng = np.random.default_rng(seed * 100_000 + rep)
        rewards, mask, lengths, index = generate_grouped_rollouts(
            rng, P, N, bins, build_cov, T_max, chol_cache
        )

        advs = compute_all_advantages(rewards, mask, index, [m.key for m in methods])
        for m in methods:
            # Scalar advantage per trajectory: sum of token advantages
            A_scalar = (advs[m.key] * mask).sum(dim=-1)
            for j in range(len(lengths)):
                Li = int(lengths[j])
                msa_acc[m.key][Li].append(float(A_scalar[j].item() ** 2))

    x = np.arange(len(bins))
    bw = 0.18
    for mi, m in enumerate(methods):
        vals = [float(np.mean(msa_acc[m.key][int(L)])) for L in bins]
        ax.bar(
            x + mi * bw,
            vals,
            bw,
            label=m.display,
            color=_PAL.get(m.key, "#888"),
            edgecolor="white",
            lw=0.8,
        )
    ax.set_xticks(x + bw * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"$L$={L}" for L in bins])
    ax.set_ylabel("$\\mathbb{E}[A^2 \\mid L]$  (gradient energy)")
    ax.set_title("(a) Gradient energy by length", fontweight="bold")
    ax.legend(fontsize=8)

    # (b) Long/short ratio
    ax = axes[1]
    L_min, L_max = int(bins.min()), int(bins.max())
    ratio_acc: dict[str, list] = {m.key: [] for m in methods}

    for rep in range(n_mc):
        rng = np.random.default_rng(seed * 100_000 + rep)
        rewards, mask, lengths, index = generate_grouped_rollouts(
            rng, P, N, bins, build_cov, T_max, chol_cache
        )

        advs = compute_all_advantages(rewards, mask, index, [m.key for m in methods])
        for m in methods:
            A_scalar = (advs[m.key] * mask).sum(dim=-1)
            A2 = A_scalar**2
            ms = float(A2[lengths == L_min].mean().item()) if (lengths == L_min).any() else 1e-30
            ml = float(A2[lengths == L_max].mean().item()) if (lengths == L_max).any() else 1e-30
            ratio_acc[m.key].append(ml / (ms + 1e-30))

    bp = ax.boxplot(
        [ratio_acc[m.key] for m in methods],
        positions=range(len(methods)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "lw": 1.5},
    )
    for patch, m in zip(bp["boxes"], methods, strict=False):
        patch.set_facecolor(_PAL.get(m.key, "#888"))
        patch.set_alpha(0.7)
    ax.axhline(1.0, color="grey", lw=1.0, ls="--", label="Unbiased (=1)")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.display for m in methods], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(f"$\\mathbb{{E}}[A^2|L={L_max}]$ / " f"$\\mathbb{{E}}[A^2|L={L_min}]$")
    ax.set_title("(b) Long/short gradient energy ratio", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Experiment 6 — Length-Bias Diagnostic: GRPO Over-Weights " "Noisy Long Trajectories",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_06_length_bias")
    print("  [6/10] Length-bias diagnostic done")


# ######################################################################## #
#  EXPERIMENT 7 — Advantage Concentration                                  #
# ######################################################################## #


def experiment_07_advantage_concentration(out, seed, P, N, bins):
    panels = [
        Regime(r"AR(1) $\rho$=0.8", lambda L: cov_ar1(int(L), 0.8), lambda L: v_ar1(int(L), 0.8)),
        Regime(
            r"CS $\rho$=0.05",
            lambda L: cov_compound_symmetry(int(L), 0.05),
            lambda L: v_compound_symmetry(int(L), 0.05),
        ),
    ]
    methods = METHODS4
    T_max = int(bins.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for pi, reg in enumerate(panels):
        chol_cache = _cholesky_cache(bins, reg.build_cov)
        rng = np.random.default_rng(seed)
        rewards, mask, lengths, index = generate_grouped_rollouts(
            rng, P, N, bins, reg.build_cov, T_max, chol_cache
        )

        advs = compute_all_advantages(rewards, mask, index, [m.key for m in methods])
        ax = axes[pi]
        for m in methods:
            # Scalar advantage per trajectory
            A_scalar = (advs[m.key] * mask).sum(dim=-1).numpy()
            An = (A_scalar - A_scalar.mean()) / (A_scalar.std() + 1e-12)
            sns.kdeplot(
                An,
                ax=ax,
                label=m.display,
                color=_PAL.get(m.key, "#888"),
                linewidth=_LW.get(m.key, 1.5),
                linestyle=_LS.get(m.key, "-"),
                zorder=_ZO.get(m.key, 3),
                clip=(-6, 6),
            )
        ax.set_xlabel("Normalised advantage $\\tilde{A}$")
        ax.set_ylabel("Density")
        ax.set_title(f"({chr(97+pi)}) {reg.name}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-5, 5)

    fig.suptitle(
        "Experiment 7 — Advantage Concentration: CAPO Produces "
        "Tighter, More Informative Advantages",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_07_advantage_concentration")
    print("  [7/10] Advantage concentration done")


# ######################################################################## #
#  EXPERIMENT 8 — L-CAPO Parameter Recovery                               #
# ######################################################################## #


def experiment_08_eb_recovery(out, seed, P=512, N=8, n_mc=100):
    """L-CAPO (production code) recovers the true beta under power-law v(L)."""
    bins = DEFAULT_LENGTH_BINS
    betas = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
    T_max = int(bins.max())
    bhats: dict[float, list] = {b: [] for b in betas}

    # Collect baselines for RE computation
    grpo_bl: dict[float, list] = {b: [] for b in betas}
    oracle_bl: dict[float, list] = {b: [] for b in betas}
    fitted_bl: dict[float, list] = {b: [] for b in betas}

    for bs in betas:
        # Power-law covariance: diagonal with Var(g_i) = L^beta
        # For a pure power-law test, use IID tokens with
        # per-token variance scaled so that Var(sum) = L^beta,
        # i.e., per-token variance = L^{beta-1}
        def _build_cov_pl(L_int, _b=bs):
            L_int = int(L_int)
            sigma2 = float(L_int) ** (_b - 1.0)
            return sigma2 * np.eye(L_int)

        chol_cache = _cholesky_cache(bins, _build_cov_pl)

        for rep in range(n_mc):
            rng = np.random.default_rng(seed * 100_000 + rep)
            rewards, mask, lengths, index = generate_grouped_rollouts(
                rng, P, N, bins, _build_cov_pl, T_max, chol_cache
            )

            # Scalar returns and lengths
            g = (rewards * mask).sum(dim=-1)
            L = mask.sum(dim=-1).clamp_min(1).float()

            # Use production L-CAPO
            beta_hat, w_fit, m_fit = eb_lite_fit_beta_and_weights(g, L)
            bhats[bs].append(beta_hat)

            # Oracle weights
            v_arr = L.pow(bs)
            w_oracle = 1.0 / v_arr
            w_oracle = w_oracle / w_oracle.sum()
            m_oracle = float((w_oracle * g).sum().item())

            # Collect baselines for RE
            grpo_bl[bs].append(float(g.mean().item()))
            oracle_bl[bs].append(m_oracle)
            fitted_bl[bs].append(float(m_fit.item()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bp = ax.boxplot(
        [bhats[b] for b in betas],
        positions=range(len(betas)),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "lw": 2},
    )
    for p in bp["boxes"]:
        p.set_facecolor(_PAL["l_capo"])
        p.set_alpha(0.55)
    ax.plot(
        range(len(betas)),
        betas,
        "s--",
        color=_PAL["grpo"],
        lw=2,
        markersize=7,
        label="True $\\beta^*$",
        zorder=5,
    )
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_xlabel("True $\\beta^*$")
    ax.set_ylabel("Recovered $\\hat{\\beta}$")
    ax.set_title("(a) L-CAPO parameter recovery", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    re_o = [
        float(np.var(grpo_bl[b])) / float(np.var(oracle_bl[b]))
        if np.var(oracle_bl[b]) > 1e-30
        else 1.0
        for b in betas
    ]
    re_f = [
        float(np.var(grpo_bl[b])) / float(np.var(fitted_bl[b]))
        if np.var(fitted_bl[b]) > 1e-30
        else 1.0
        for b in betas
    ]
    ax.plot(
        betas,
        re_o,
        "s-",
        color=_PAL["lv_capo"],
        lw=2.5,
        markersize=7,
        label="LV-CAPO (oracle $\\beta^*$)",
        zorder=5,
    )
    ax.plot(
        betas,
        re_f,
        "D-",
        color=_PAL["l_capo"],
        lw=2.5,
        markersize=7,
        label="L-CAPO (fitted $\\hat{\\beta}$)",
        zorder=5,
    )
    ax.axhline(1.0, color=_PAL["grpo"], lw=1.5, ls="--", label="GRPO")
    ax.set_xlabel("True $\\beta^*$")
    ax.set_ylabel("Relative Efficiency vs GRPO")
    ax.set_title("(b) Fitted efficiency matches oracle", fontweight="bold")
    ax.legend(fontsize=8)

    rmse = float(np.sqrt(np.mean([(bh - b) ** 2 for b in betas for bh in bhats[b]])))
    fig.suptitle(
        f"Experiment 8 — L-CAPO Parameter Recovery " f"(RMSE = {rmse:.3f})",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_08_eb_recovery")
    print("  [8/10] EB recovery done")
    return {"recovery_rmse": rmse}


# ######################################################################## #
#  EXPERIMENT 9 — Mixed-Dependence Stress Test                             #
# ######################################################################## #


def experiment_09_mixed_dependence(out, seed, P, N, bins, n_mc=200):
    """Test robustness when different lengths have different covariance."""
    methods = METHODS4
    T_max = int(bins.max())

    def _mx1(b):
        """Short=IID, Long=AR(0.8)."""

        def build(L):
            L = int(L)
            return cov_iid(L) if L <= 256 else cov_ar1(L, 0.8)

        return build, "Short=IID\nLong=AR(0.8)"

    def _mx2(b):
        """Gradual rho: 0.1 to 0.9."""
        rhos = np.linspace(0.1, 0.9, len(b))
        rho_map = {int(L): float(r) for L, r in zip(b, rhos, strict=False)}

        def build(L):
            L = int(L)
            return cov_ar1(L, rho_map[L])

        return build, "Gradual\n$\\rho$: 0.1 to 0.9"

    def _mx3(b):
        """Short=CS(0.03), Long=k-band."""

        def build(L):
            L = int(L)
            return cov_compound_symmetry(L, 0.03) if L <= 256 else cov_kband(L, 0.8, 64, 1.3)

        return build, "Short=CS(0.03)\nLong=$k$-band"

    def _mx4(b):
        """Uniform CS(0.05)."""

        def build(L):
            return cov_compound_symmetry(int(L), 0.05)

        return build, "Uniform\nCS(0.05)"

    def _mx5(b):
        """Adversarial: rho proportional to log L."""

        def build(L):
            L = int(L)
            rho = min(0.08, 0.01 * np.log2(L))
            return cov_compound_symmetry(L, rho)

        return build, "Adversarial\n$\\rho \\propto \\log L$"

    builders = [_mx1, _mx2, _mx3, _mx4, _mx5]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    names = []
    m_res: dict[str, list] = {m.key: [] for m in methods}

    for build_fn in builders:
        build_cov, sn = build_fn(bins)
        names.append(sn)
        chol_cache = _cholesky_cache(bins, build_cov)
        mkeys = [m.key for m in methods]
        bl: dict[str, list] = {k: [] for k in mkeys}

        for r in range(n_mc):
            rng = np.random.default_rng(seed * 100_000 + r)
            rewards, mask, lengths, index = generate_grouped_rollouts(
                rng, P, N, bins, build_cov, T_max, chol_cache
            )
            b = compute_scalar_baselines(rewards, mask, index, mkeys)
            for k in mkeys:
                bl[k].append(b[k])

        re = _relative_efficiency(bl)
        for m in methods:
            m_res[m.key].append(re[m.key])

    x = np.arange(len(names))
    bw = 0.18
    for mi, m in enumerate(methods):
        bars = ax.bar(
            x + mi * bw,
            m_res[m.key],
            bw,
            label=m.display,
            color=_PAL.get(m.key, "#888"),
            edgecolor="white",
            lw=0.8,
        )
        for bar, val in zip(bars, m_res[m.key], strict=False):
            if val > 1.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if "capo" in m.key else "normal",
                )
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xticks(x + bw * (len(methods) - 1) / 2)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Relative Efficiency vs GRPO")
    ax.set_title(
        "Experiment 9 — Mixed-Dependence Stress Test: CAPO " "Handles Heterogeneous Covariance",
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0)
    fig.tight_layout()
    _save(fig, out / "fig_09_mixed_dependence")
    print("  [9/10] Mixed-dependence done")


# ######################################################################## #
#  EXPERIMENT 10 — Scaling with Group Size N                               #
# ######################################################################## #


def experiment_10_scaling_with_N(out, seed, P, bins, n_mc=200):
    """CAPO's advantage persists and grows with group size N."""
    Nvals = np.array([4, 8, 16, 32, 64])
    methods = METHODS4
    T_max = int(bins.max())
    panels = [
        Regime(r"AR(1) $\rho$=0.8", lambda L: cov_ar1(int(L), 0.8), lambda L: v_ar1(int(L), 0.8)),
        Regime(
            r"$k$-band (0.8,64,1.3)",
            lambda L: cov_kband(int(L), 0.8, 64, 1.3),
            lambda L: v_kband(int(L), 0.8, 64, 1.3),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for pi, reg in enumerate(panels):
        ax = axes[pi]
        chol_cache = _cholesky_cache(bins, reg.build_cov)
        res: dict[str, list] = {m.key: [] for m in methods}

        for Nv in Nvals:
            mkeys = [m.key for m in methods]
            P_use = min(P, 500)
            bl: dict[str, list] = {k: [] for k in mkeys}

            for r in range(n_mc):
                rng = np.random.default_rng(seed * 100_000 + r)
                rewards, mask, lengths, index = generate_grouped_rollouts(
                    rng, P_use, int(Nv), bins, reg.build_cov, T_max, chol_cache
                )
                b = compute_scalar_baselines(rewards, mask, index, mkeys)
                for k in mkeys:
                    bl[k].append(b[k])

            re = _relative_efficiency(bl)
            for m in methods:
                res[m.key].append(re[m.key])

        for m in methods:
            ax.plot(Nvals, res[m.key], label=m.display, **_sty(m.key))
        ax.axhline(1.0, color="grey", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Group size $N$ (rollouts per prompt)")
        ax.set_ylabel("Relative Efficiency vs GRPO")
        ax.set_title(f"({chr(97+pi)}) {reg.name}", fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(Nvals)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(bottom=0.8)

    fig.suptitle(
        "Experiment 10 — Scaling with Group Size $N$: " "CAPO's Advantage Persists and Grows",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "fig_10_scaling_with_N")
    print("  [10/10] Scaling with N done")


# ######################################################################## #
#  Master entry points                                                     #
# ######################################################################## #


def build_all_experiments(out_dir, *, seed=0, P=4000, N=8, fast=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bins = DEFAULT_LENGTH_BINS

    if fast:
        n_mc = 40
        P0 = min(P, 300)
        n_mc_bias = 50
        n_mc_eb = 50
        n_mc_mixed = 40
        n_mc_scale = 40
    else:
        n_mc = 300
        P0 = P
        n_mc_bias = 400
        n_mc_eb = 150
        n_mc_mixed = 300
        n_mc_scale = 250

    print("=" * 60)
    print(f"Generating 10 experiments  (fast={fast}, P={P0}, N={N})")
    print("=" * 60)

    experiment_01_variance_landscape(out)
    experiment_02_weight_comparison(out)
    experiment_03_relative_efficiency(out, seed, P0, N, bins, n_mc)
    experiment_04_mse_vs_correlation(out, seed, P0, N, bins, n_mc)
    experiment_05_effective_sample_size(out, seed, P0, N, bins, n_mc)
    experiment_06_length_bias(out, seed, P0, N, bins, n_mc_bias)
    experiment_07_advantage_concentration(out, seed, P0, N, bins)
    experiment_08_eb_recovery(out, seed, P=P0, N=N, n_mc=n_mc_eb)
    experiment_09_mixed_dependence(out, seed, P0, N, bins, n_mc_mixed)
    experiment_10_scaling_with_N(out, seed, P0, bins, n_mc_scale)

    print("=" * 60)
    print(f"All 10 experiments -> {out}/")
    print("=" * 60)


def build_paper_artifacts(out_dir, *, seed=0, P=4000, N=8, **kw):
    build_all_experiments(out_dir, seed=seed, P=P, N=N, **kw)


def main():
    ap = argparse.ArgumentParser(description="Synthetic benchmark (CAPO)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--P", type=int, default=4000)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    build_all_experiments(a.out, seed=a.seed, P=a.P, N=a.N, fast=a.fast)


if __name__ == "__main__":
    main()
