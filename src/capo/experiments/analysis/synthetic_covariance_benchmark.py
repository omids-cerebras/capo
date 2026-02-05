"""Synthetic covariance benchmark: CAPO vs. GRPO and \Delta L.

This module implements a base-model-free simulator that generates scalar
rollout statistics (g_i, L_i) under configurable within-trajectory covariance
regimes. It is used to isolate how different normalization / aggregation rules
transform the same batch into per-trajectory scalar advantages A_i.

The paper uses this benchmark to highlight length amplification and signal
allocation effects that arise when within-trajectory dependence induces
structured heteroscedasticity.

Outputs
-------
The primary entrypoint, :func:`build_synthetic_covariance_artifacts`, writes
compile-ready artifacts into a user-specified output directory:

  - fig_synthetic_covariance.pdf
  - fig_synthetic_signal_share.pdf
  - tab_synthetic_covariance.tex

These filenames match the LaTeX includes in docs/report.

Reproducibility
---------------
The benchmark is stochastic but fully seedable. Default parameters match the
paper's artifacts.

Dependencies
------------
Only numpy and matplotlib are required.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Use a non-interactive backend for headless environments.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Length = int


@dataclass(frozen=True)
class Regime:
    """A synthetic covariance regime specified via an induced variance law v(L)."""

    name: str
    v_of_L: Callable[[Length], float]


@dataclass(frozen=True)
class Method:
    """A scalar-advantage construction."""

    display: str
    key: str


# -----------------------------------------------------------------------------
# Induced variance laws
# -----------------------------------------------------------------------------


def _v_iid(L: Length) -> float:
    """IID token noise: Var(sum_{t=1}^L Y_t) = L for Var(Y_t)=1."""

    return float(int(L))


def _v_ar1(L: Length, rho: float) -> float:
    """AR(1)/geometric dependence with unit marginal variance.

    If Corr(Y_t, Y_{t-h}) = rho^h, then
        Var(sum_{t=1}^L Y_t) = L + 2 * sum_{h=1}^{L-1} (L-h) rho^h.
    """

    L = int(L)
    s = 0.0
    for h in range(1, L):
        s += (L - h) * (rho**h)
    return float(L + 2.0 * s)


def _v_kband(L: Length, rho: float, k: int, eta: float) -> float:
    """k-banded stretched-geometric dependence.

    Autocovariance is gamma(h) = rho^{h^{eta}} for 1<=h<=k and gamma(h)=0 for h>k.
    """

    L = int(L)
    m = min(int(k), L - 1)
    s = 0.0
    for h in range(1, m + 1):
        s += (L - h) * (rho ** (h**eta))
    return float(L + 2.0 * s)


def _v_compound_symmetry(L: Length, rho_cs: float) -> float:
    """Compound symmetry: constant off-diagonal correlation rho_cs."""

    L = int(L)
    return float(L + rho_cs * L * (L - 1))


# -----------------------------------------------------------------------------
# Simulation + advantage constructions
# -----------------------------------------------------------------------------


def _simulate_batch(
    rng: np.random.Generator,
    P: int,
    N: int,
    length_bins: np.ndarray,
    v_map: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a prompt-grouped batch of (lengths, returns).

    Returns
    -------
    lengths:
        Integer array of shape (P, N).
    g:
        Float array of shape (P, N) with g_{p,i} ~ Normal(0, v(L_{p,i})).
    """

    lengths = rng.choice(length_bins, size=(int(P), int(N)), replace=True)
    var = np.vectorize(lambda x: v_map[int(x)])(lengths).astype(float)
    g = rng.normal(loc=0.0, scale=np.sqrt(var), size=(int(P), int(N)))
    return lengths, g


def _grpo_advantages(g: np.ndarray, eps: float) -> np.ndarray:
    """Group-centered and standardized advantages (GRPO)."""

    mean = g.mean(axis=1, keepdims=True)
    centered = g - mean
    std = np.sqrt(np.mean(centered**2, axis=1, keepdims=True)) + float(eps)
    return centered / std


def _deltaL_advantages(
    A_grpo: np.ndarray, lengths: np.ndarray, alpha: float, eps: float
) -> np.ndarray:
    """GRPO++ / Delta-L scaling: A = r_i(alpha) * A_grpo with mean(r_i)=1 in each group."""

    meanL = lengths.mean(axis=1, keepdims=True)
    f = (lengths / meanL) ** (-float(alpha))
    f_norm = f / (np.mean(f, axis=1, keepdims=True) + float(eps))
    return f_norm * A_grpo


def _capo_advantages(g_flat: np.ndarray, w_flat: np.ndarray) -> np.ndarray:
    """CAPO-style centered, weighted scalar advantage: A_i = w_i (g_i - m)."""

    w = w_flat.astype(float)
    w = w / np.sum(w)
    m = float(np.sum(w * g_flat))
    return w * (g_flat - m)


def _msa_by_length(
    A_flat: np.ndarray, L_flat: np.ndarray, length_bins: np.ndarray
) -> dict[int, float]:
    out: dict[int, float] = {}
    for L in length_bins:
        L_int = int(L)
        mask = L_flat == L_int
        out[L_int] = float(np.mean(A_flat[mask] ** 2))
    return out


def _share_by_length(
    A_flat: np.ndarray, L_flat: np.ndarray, length_bins: np.ndarray
) -> dict[int, float]:
    out: dict[int, float] = {}
    denom = float(np.sum(A_flat**2))
    for L in length_bins:
        L_int = int(L)
        mask = L_flat == L_int
        out[L_int] = float(np.sum(A_flat[mask] ** 2) / denom)
    return out


# -----------------------------------------------------------------------------
# Artifact building
# -----------------------------------------------------------------------------


def _plot_msa_ratio(
    msa_ratio: Mapping[str, Mapping[str, Mapping[int, float]]],
    regimes: Sequence[Regime],
    methods: Sequence[Method],
    length_bins: np.ndarray,
    outpath: Path,
) -> None:
    nreg = len(regimes)
    ncols = 3
    nrows = int(math.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 6 * nrows),
        sharey=True,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, reg in enumerate(regimes):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        for m in methods:
            y = [msa_ratio[reg.name][m.display][int(L)] for L in length_bins]
            ax.plot(length_bins, y, marker="o", label=m.display)

        ax.set_title(reg.name, fontsize=14)
        ax.set_xlabel("Length $L$")
        if c == 0:
            ax.set_ylabel(r"$\mathbb{E}[A^2\mid L]$ normalized by $L=L_{\min}$")
        ax.set_xticks(length_bins)
        ax.grid(True, alpha=0.3)

    # Delete empty axes if any.
    for j in range(nreg, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    # Global legend (use handles from first axis).
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_signal_share(
    share: Mapping[str, Mapping[str, Mapping[int, float]]],
    regimes: Sequence[Regime],
    methods: Sequence[Method],
    length_bins: np.ndarray,
    outpath: Path,
) -> None:
    nreg = len(regimes)
    ncols = 3
    nrows = int(math.ceil(nreg / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 6 * nrows),
        sharey=True,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    # Use default color cycle for stacks.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stack_colors = colors[: len(length_bins)]

    for idx, reg in enumerate(regimes):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        x = np.arange(len(methods))
        bottoms = np.zeros(len(methods))

        for j, L in enumerate(length_bins):
            vals = [share[reg.name][m.display][int(L)] for m in methods]
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=f"L={int(L)}" if idx == 0 else None,
                color=stack_colors[j],
            )
            bottoms += np.array(vals)

        ax.set_title(reg.name, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                m.display.replace(" (oracle $\\beta=1$)", "").replace(" (oracle)", "")
                for m in methods
            ],
            rotation=30,
            ha="right",
        )
        if c == 0:
            ax.set_ylabel("Share of total $A^2$")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(nreg, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(length_bins),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _latex_amp_table(
    amp: Mapping[str, Mapping[str, float]],
    regimes: Sequence[Regime],
    methods: Sequence[Method],
    digits: int = 2,
) -> str:
    col_spec = "l" + "c" * len(regimes)
    header = ["Method"] + [r.name for r in regimes]

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for m in methods:
        row = [m.display]
        for r in regimes:
            row.append(f"{amp[r.name][m.display]:.{digits}f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def build_synthetic_covariance_artifacts(
    out_dir: Path,
    *,
    seed: int = 0,
    P: int = 8000,
    N: int = 8,
    eps: float = 1e-8,
    length_bins: Sequence[int] = (32, 64, 128, 256),
) -> None:
    """Generate synthetic-covariance artifacts used in the paper.

    Parameters
    ----------
    out_dir:
        Output directory to which the figure/table artifacts are written.
    seed:
        RNG seed.
    P:
        Number of prompt groups.
    N:
        Number of rollouts per group.
    eps:
        Stabilizer used in GRPO standardization and \Delta L normalization.
    length_bins:
        Discrete length bins used by the simulator.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L_bins = np.asarray(list(length_bins), dtype=int)
    L_min = int(np.min(L_bins))
    L_max = int(np.max(L_bins))

    # Regimes (match Appendix E).
    regimes: list[Regime] = [
        Regime("IID", lambda L: _v_iid(L)),
        Regime("AR(1) ($\\rho=0.7$)", lambda L: _v_ar1(L, 0.7)),
        Regime("AR(1) ($\\rho=0.9$)", lambda L: _v_ar1(L, 0.9)),
        Regime(
            "$k$-band ($\\rho=0.7,\\,k=16,\\,\\eta=1.5$)",
            lambda L: _v_kband(L, 0.7, 16, 1.5),
        ),
        Regime("CS ($\\rho=0.01$)", lambda L: _v_compound_symmetry(L, 0.01)),
        Regime("CS ($\\rho=0.05$)", lambda L: _v_compound_symmetry(L, 0.05)),
    ]

    # Methods (match the LaTeX captions).
    methods: list[Method] = [
        Method("GRPO (z-score)", "grpo"),
        Method("$\\Delta L$ ($\\alpha=0.75$)", "deltal_0.75"),
        Method("$\\Delta L$ ($\\alpha=1.0$)", "deltal_1.0"),
        Method("CAPO-EB-lite (oracle $\\beta=1$)", "capo_lite"),
        Method("CAPO-EB (oracle)", "capo_full"),
    ]

    rng = np.random.default_rng(int(seed))

    # Outputs keyed by (regime_name, method_display).
    msa_ratio: dict[str, dict[str, dict[int, float]]] = {}
    amp: dict[str, dict[str, float]] = {}
    share: dict[str, dict[str, dict[int, float]]] = {}

    for reg in regimes:
        # Precompute v(L) for the discrete bins.
        v_map = {int(L): float(reg.v_of_L(int(L))) for L in L_bins}

        lengths, g = _simulate_batch(rng, P=P, N=N, length_bins=L_bins, v_map=v_map)

        # GRPO / Delta-L operate within groups.
        A_grpo = _grpo_advantages(g, eps=eps)
        A_d075 = _deltaL_advantages(A_grpo, lengths, alpha=0.75, eps=eps)
        A_d1 = _deltaL_advantages(A_grpo, lengths, alpha=1.0, eps=eps)

        # CAPO weights operate across full batch.
        L_flat = lengths.reshape(-1)
        g_flat = g.reshape(-1)

        w_lite = 1.0 / (L_flat.astype(float) ** 1.0)
        A_lite = _capo_advantages(g_flat, w_lite)

        w_full = 1.0 / np.vectorize(lambda x, v=v_map: v[int(x)])(L_flat).astype(float)
        A_full = _capo_advantages(g_flat, w_full)

        A_by_key = {
            "grpo": A_grpo.reshape(-1),
            "deltal_0.75": A_d075.reshape(-1),
            "deltal_1.0": A_d1.reshape(-1),
            "capo_lite": A_lite,
            "capo_full": A_full,
        }

        msa_ratio[reg.name] = {}
        amp[reg.name] = {}
        share[reg.name] = {}

        for m in methods:
            A = A_by_key[m.key]
            msa = _msa_by_length(A, L_flat, L_bins)
            ratios = {L: msa[L] / msa[L_min] for L in msa}
            msa_ratio[reg.name][m.display] = ratios
            amp[reg.name][m.display] = ratios[L_max]
            share[reg.name][m.display] = _share_by_length(A, L_flat, L_bins)

    # Write LaTeX table.
    (out_dir / "tab_synthetic_covariance.tex").write_text(
        _latex_amp_table(amp, regimes, methods, digits=2),
        encoding="utf-8",
    )

    # Write figures.
    _plot_msa_ratio(
        msa_ratio,
        regimes,
        methods,
        L_bins,
        outpath=out_dir / "fig_synthetic_covariance.pdf",
    )
    _plot_signal_share(
        share,
        regimes,
        methods,
        L_bins,
        outpath=out_dir / "fig_synthetic_signal_share.pdf",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for paper artifacts (e.g., docs/report/artifacts/paper)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--P", type=int, default=8000, help="# prompt groups")
    ap.add_argument("--N", type=int, default=8, help="# rollouts per group")
    args = ap.parse_args()

    build_synthetic_covariance_artifacts(
        args.out,
        seed=args.seed,
        P=args.P,
        N=args.N,
    )


if __name__ == "__main__":
    main()
