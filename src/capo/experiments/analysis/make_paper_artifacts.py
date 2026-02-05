#!/usr/bin/env python
"""Build paper figures + LaTeX tables from VERL run directories.

This script is intentionally self-contained and "forgiving": if a particular
metric or run is missing, it produces compile-safe placeholders (dashes / empty
plots) rather than crashing. This makes it suitable for incremental experiment
completion while keeping the LaTeX report building clean.

Inputs
------
The script consumes `runs.json` produced by `collect_runs.py`. Each entry in
that file contains:
  - run_dir: path to a Hydra run directory (containing `metrics.jsonl`)
  - metrics: the final metrics dict (last JSONL row)
  - best_val_metric: optional best validation metric (computed by collect_runs)
  - config: the resolved Hydra config for that run

Outputs
-------
Writes the paper artifacts into `<out>/paper/`:
  - fig_dynamics.pdf
  - fig_stability.pdf
  - fig_length_deciles.pdf
  - tab_main_accuracy.tex        (tabular only; the paper provides the table env)
  - tab_stability_efficiency.tex (tabular only)

Metric conventions
------------------
For CountDown, VERL reports validation summaries under:
  val-core/<data_source>/<reward_key>/mean@K

With our dataset prep script, `data_source` is "countdown" and the reward
function provides `acc` (0/1). Therefore we default to:
  VAL_ACC_KEY = "val-core/countdown/acc/mean@8"

"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Defaults (edit here if you change validation n or dataset data_source)
# -----------------------------------------------------------------------------

VAL_N = 8
DATA_SOURCE = "countdown"
VAL_ACC_KEY = f"val-core/{DATA_SOURCE}/acc/mean@{VAL_N}"
VAL_PASS_KEY = f"val-core/{DATA_SOURCE}/acc/best@{VAL_N}/mean"  # best@K/mean acts like pass@K

# Stability table definitions
TTT_THRESHOLD = 0.50  # accuracy threshold for tokens-to-target


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if _safe_float(x) is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def _fmt_cell(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "\\textemdash"
    return f"{v:.{digits}f}"


def _latex_tabular(col_spec: str, header: list[str], rows: list[list[str]]) -> str:
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = statistics.mean([(x - mx) ** 2 for x in xs])
    vy = statistics.mean([(y - my) ** 2 for y in ys])
    if vx <= 0 or vy <= 0:
        return None
    cov = statistics.mean([(x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)])
    return cov / math.sqrt(vx * vy)


# -----------------------------------------------------------------------------
# Run parsing / method naming
# -----------------------------------------------------------------------------


@dataclass
class Run:
    run_dir: Path
    config: dict[str, Any]
    best_val_metric: float | None

    @property
    def max_resp_len(self) -> int | None:
        try:
            return int(self.config.get("data", {}).get("max_response_length"))
        except Exception:
            return None

    @property
    def seed(self) -> int | None:
        try:
            return int(self.config.get("seed"))
        except Exception:
            return None

    @property
    def adv_estimator(self) -> str:
        adv = self.config.get("algorithm", {}).get("adv_estimator")
        return str(adv) if adv is not None else "unknown"

    def method_key(self) -> str:
        """Internal method identifier used for grouping."""
        algo = self.config.get("algorithm", {})
        adv = str(algo.get("adv_estimator", "unknown"))

        # CAPO family
        if adv in {"capo", "capo_eb", "capo_eb_lite"}:
            return adv

        # GRPO family
        if adv == "grpo":
            use_grpopp = bool(algo.get("use_grpopp", False))
            use_dr_grpo = bool(algo.get("use_dr_grpo", False))
            if use_grpopp:
                alpha = algo.get("grpopp_config", {}).get("alpha", None)
                try:
                    alpha_f = float(alpha)
                except Exception:
                    alpha_f = None
                if alpha_f is None:
                    return "grpopp"
                return f"grpopp_alpha={alpha_f:.2f}"
            if use_dr_grpo:
                return "dr_grpo_norm"

            # Differentiate GRPO vs DAPO via loss aggregation mode.
            loss_agg = (
                self.config.get("actor_rollout_ref", {})
                .get("actor", {})
                .get("loss_agg_mode", "token-mean")
            )
            if str(loss_agg) == "seq-mean-token-mean":
                return "grpo_norm"
            return "dapo_norm"

        return adv

    def method_display(self) -> str:
        """Display name used in LaTeX tables / plots."""
        k = self.method_key()
        if k == "dapo_norm":
            return "DAPO Norm"
        if k == "grpo_norm":
            return "GRPO Norm"
        if k == "dr_grpo_norm":
            return "Dr.~GRPO Norm"
        if k.startswith("grpopp_alpha="):
            alpha = k.split("=", 1)[1]
            return f"$\\Delta L$ norm ($\\alpha={alpha}$)"
        if k == "capo":
            return "CAPO"
        if k == "capo_eb_lite":
            return "CAPO-EB-lite"
        if k == "capo_eb":
            return "CAPO-EB"
        return k


def _load_runs(collected_runs_json: Path) -> list[Run]:
    raw = _read_json(collected_runs_json)
    runs: list[Run] = []
    for r in raw:
        try:
            run_dir = Path(r["run_dir"])
        except Exception:
            continue
        cfg = r.get("config") or {}
        best = _safe_float(r.get("best_val_metric"))
        runs.append(Run(run_dir=run_dir, config=cfg, best_val_metric=best))
    return runs


def _val_series(metrics_rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    vals: list[float] = []
    for row in metrics_rows:
        if key not in row:
            continue
        v = _safe_float(row.get(key))
        if v is None:
            continue
        step = row.get("step")
        try:
            step_i = int(step)
        except Exception:
            continue
        steps.append(step_i)
        vals.append(v)
    return steps, vals


def _cumulative_tokens_to_step(
    metrics_rows: list[dict[str, Any]], step_target: int
) -> float | None:
    total = 0.0
    have_any = False
    for row in metrics_rows:
        try:
            step = int(row.get("step", 0))
        except Exception:
            continue
        if step > step_target:
            break
        v = _safe_float(row.get("perf/total_num_tokens"))
        if v is None:
            continue
        total += v
        have_any = True
    return total if have_any else None


def _stability_metrics(
    metrics_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    steps, acc = _val_series(metrics_rows, VAL_ACC_KEY)
    if len(steps) < 2:
        return {"monotonicity": None, "ndr": None, "ttt": None}

    # Monotonicity: Pearson correlation between step and accuracy.
    mono = _pearson_corr([float(s) for s in steps], acc)

    # Negative drift rate: fraction of evaluation intervals with a decrease.
    decreases = 0
    for a0, a1 in zip(acc[:-1], acc[1:], strict=False):
        if a1 < a0:
            decreases += 1
    ndr = decreases / max(1, (len(acc) - 1))

    # Tokens-to-target: cumulative tokens until accuracy first reaches threshold.
    ttt_tokens = None
    for s, a in zip(steps, acc, strict=False):
        if a >= TTT_THRESHOLD:
            ttt_tokens = _cumulative_tokens_to_step(metrics_rows, s)
            break

    return {"monotonicity": mono, "ndr": ndr, "ttt": ttt_tokens}


# -----------------------------------------------------------------------------
# Artifact generation
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", type=Path, required=True, help="Directory containing runs.json")
    ap.add_argument("--out", type=Path, required=True, help="Artifact root directory")
    ap.add_argument(
        "--skip_synthetic",
        action="store_true",
        help="Skip synthetic covariance benchmark artifacts.",
    )
    args = ap.parse_args()

    collected = args.collected
    out_root = args.out
    paper_out = out_root / "paper"
    paper_out.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(collected / "runs.json")

    # We only support CountDown in the paper harness (by construction).
    contexts = [2048, 8192]

    # Canonical method ordering (matches the LaTeX table template).
    method_keys = [
        "grpo_norm",
        "dapo_norm",
        "dr_grpo_norm",
        "grpopp_alpha=0.75",
        "grpopp_alpha=1.00",
        "capo",
        "capo_eb_lite",
        "capo_eb",
    ]

    # ---------------- Table: main accuracy ----------------
    # For each method/context, average best_val_metric across seeds.
    table_rows: list[list[str]] = []
    for mk in method_keys:
        row: list[str] = []
        # display name from a dummy run-like object
        display = None
        for r in runs:
            if r.method_key() == mk:
                display = r.method_display()
                break
        row.append(display or mk)

        for ctx in contexts:
            vals: list[float] = []
            for r in runs:
                if r.method_key() != mk:
                    continue
                if r.max_resp_len != ctx:
                    continue
                if r.best_val_metric is None:
                    continue
                vals.append(r.best_val_metric)
            mean_v, _std_v = _mean_std(vals)
            row.append(_fmt_cell(mean_v, digits=3))
        table_rows.append(row)

    tab_main = _latex_tabular(
        "lcc",
        [
            "Method",
            "CountDown (1.5B, 2048)",
            "CountDown (1.5B, 8192)",
        ],
        table_rows,
    )
    (paper_out / "tab_main_accuracy.tex").write_text(tab_main, encoding="utf-8")

    # ---------------- Dynamics figure ----------------
    # Plot validation acc vs step for a small subset of methods (best run per method).
    dyn_methods = ["grpo_norm", "grpopp_alpha=1.00", "capo_eb"]
    plt.figure()
    any_line = False
    for mk in dyn_methods:
        # Choose best run among available contexts=2048 for dynamics.
        candidates = [r for r in runs if r.method_key() == mk and r.max_resp_len == 2048]
        candidates = [r for r in candidates if (r.run_dir / "metrics.jsonl").exists()]
        if not candidates:
            continue
        best_run = max(candidates, key=lambda r: (r.best_val_metric or float("-inf")))
        rows = _read_jsonl(best_run.run_dir / "metrics.jsonl")
        steps, acc = _val_series(rows, VAL_ACC_KEY)
        if not steps:
            continue
        plt.plot(steps, acc, label=best_run.method_display())
        any_line = True

    plt.xlabel("Training step")
    plt.ylabel(f"Validation accuracy (Avg@{VAL_N})")
    if any_line:
        plt.legend()
    plt.tight_layout()
    plt.savefig(paper_out / "fig_dynamics.pdf")
    plt.close()

    # ---------------- Stability figure + table ----------------
    # Compute stability metrics per run, then average across seeds per method.
    stab_rows: list[list[str]] = []
    method_points: list[tuple[str, float, float]] = []  # (label, mono, ndr)

    # Precompute per-run stability metrics (2048 context).
    per_run_stab: dict[tuple[str, Path], dict[str, float | None]] = {}
    for r in runs:
        if r.max_resp_len != 2048:
            continue
        metrics_path = r.run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        rows = _read_jsonl(metrics_path)
        per_run_stab[(r.method_key(), r.run_dir)] = _stability_metrics(rows)

    # Aggregate per method.
    agg: dict[str, dict[str, float | None]] = {}
    for mk in method_keys:
        monos: list[float] = []
        ndrs: list[float] = []
        ttts: list[float] = []
        for (mk2, _rd), m in per_run_stab.items():
            if mk2 != mk:
                continue
            if m.get("monotonicity") is not None:
                monos.append(float(m["monotonicity"]))
            if m.get("ndr") is not None:
                ndrs.append(float(m["ndr"]))
            if m.get("ttt") is not None:
                ttts.append(float(m["ttt"]))

        mono_mean, _ = _mean_std(monos)
        ndr_mean, _ = _mean_std(ndrs)
        ttt_mean, _ = _mean_std(ttts)
        agg[mk] = {"monotonicity": mono_mean, "ndr": ndr_mean, "ttt": ttt_mean}

    # Determine column-wise bests for bolding.
    mono_best = None
    ndr_best = None
    ttt_best = None
    for mk in method_keys:
        m = agg.get(mk, {})
        if m.get("monotonicity") is not None:
            mono_best = (
                max(mono_best, m["monotonicity"]) if mono_best is not None else m["monotonicity"]
            )
        if m.get("ndr") is not None:
            ndr_best = min(ndr_best, m["ndr"]) if ndr_best is not None else m["ndr"]
        if m.get("ttt") is not None:
            ttt_best = min(ttt_best, m["ttt"]) if ttt_best is not None else m["ttt"]

    def _maybe_bold(v: float | None, best: float | None, higher_is_better: bool) -> str:
        if v is None:
            return "\\textemdash"
        s = _fmt_cell(v, digits=3)
        if best is None:
            return s
        # A small tolerance because of float formatting.
        tol = 1e-12
        if higher_is_better and v >= best - tol:
            return f"\\mathbf{{{s}}}"
        if (not higher_is_better) and v <= best + tol:
            return f"\\mathbf{{{s}}}"
        return s

    for mk in method_keys:
        label = None
        for r in runs:
            if r.method_key() == mk:
                label = r.method_display()
                break
        label = label or mk
        m = agg.get(mk, {})
        mono = m.get("monotonicity")
        ndr = m.get("ndr")
        ttt = m.get("ttt")
        stab_rows.append(
            [
                label,
                _maybe_bold(mono, mono_best, True),
                _maybe_bold(ndr, ndr_best, False),
                _maybe_bold(ttt, ttt_best, False),
            ]
        )

        if mono is not None and ndr is not None:
            method_points.append((label, float(mono), float(ndr)))

    tab_stab = _latex_tabular(
        "lccc",
        ["Setting", "Monotonicity$\\uparrow$", "NDR$\\downarrow$", "TTT$\\downarrow$"],
        stab_rows,
    )
    (paper_out / "tab_stability_efficiency.tex").write_text(tab_stab, encoding="utf-8")

    # Scatter plot for stability figure.
    plt.figure()
    for label, mono, ndr in method_points:
        plt.scatter([mono], [ndr])
        plt.annotate(
            label,
            (mono, ndr),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
        )
    plt.xlabel("Monotonicity (corr(step, acc))")
    plt.ylabel("Negative drift rate")
    plt.tight_layout()
    plt.savefig(paper_out / "fig_stability.pdf")
    plt.close()

    # ---------------- Length deciles figure ----------------
    # Use the latest validation generations from the best long-context run of
    # CAPO-EB and ΔL (alpha=1.00), if available.

    def _latest_val_jsonl(run_dir: Path) -> Path | None:
        val_dir = run_dir / "val_generations"
        if not val_dir.exists():
            return None
        candidates = sorted(val_dir.glob("*.jsonl"))
        if not candidates:
            return None

        # choose by numeric step if possible
        def _step(p: Path) -> int:
            """Extract the leading global step from filenames like '100_countdown.jsonl'."""
            stem = p.stem
            head = stem.split("_")[0]
            try:
                return int(head)
            except Exception:
                return -1

        return max(candidates, key=_step)

    # Tokenizer for length computation (optional; fall back to string length).
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        # Prefer model path from any run config (they should all match).
        model_path = None
        for r in runs:
            mp = r.config.get("actor_rollout_ref", {}).get("model", {}).get("path", None)
            if mp:
                model_path = mp
                break
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = None

    def _resp_len(x: str) -> int:
        if tokenizer is None:
            return len(x)
        try:
            return len(tokenizer(x, add_special_tokens=False).input_ids)
        except Exception:
            return len(x)

    def _deciles_acc(val_file: Path) -> tuple[list[float], list[float]] | None:
        rows = _read_jsonl(val_file)
        if not rows:
            return None
        lens: list[int] = []
        accs: list[float] = []
        for r in rows:
            out = r.get("output", "")
            a = _safe_float(r.get("acc"))
            if a is None:
                continue
            lens.append(_resp_len(str(out)))
            accs.append(a)
        if not lens:
            return None
        # Compute decile boundaries by sorting.
        idx = sorted(range(len(lens)), key=lambda i: lens[i])
        lens_sorted = [lens[i] for i in idx]
        acc_sorted = [accs[i] for i in idx]
        n = len(lens_sorted)
        xs = []
        ys = []
        for d in range(10):
            lo = int(d * n / 10)
            hi = int((d + 1) * n / 10)
            if hi <= lo:
                continue
            xs.append(d + 1)
            ys.append(statistics.mean(acc_sorted[lo:hi]))
        return xs, ys

    long_methods = ["grpopp_alpha=1.00", "capo_eb"]
    plt.figure()
    any_bars = False
    for mk in long_methods:
        candidates = [r for r in runs if r.method_key() == mk and r.max_resp_len == 8192]
        candidates = [r for r in candidates if (r.run_dir / "val_generations").exists()]
        if not candidates:
            continue
        best_run = max(candidates, key=lambda r: (r.best_val_metric or float("-inf")))
        val_jsonl = _latest_val_jsonl(best_run.run_dir)
        if val_jsonl is None:
            continue
        res = _deciles_acc(val_jsonl)
        if res is None:
            continue
        xs, ys = res
        plt.plot(xs, ys, marker="o", label=best_run.method_display())
        any_bars = True

    plt.xlabel("Response length decile (short $\\rightarrow$ long)")
    plt.ylabel(f"Validation accuracy (Avg@{VAL_N})")
    if any_bars:
        plt.legend()
    plt.tight_layout()
    plt.savefig(paper_out / "fig_length_deciles.pdf")
    plt.close()

    # Synthetic covariance benchmark (base-model-free, deterministic up to RNG seed).
    if not args.skip_synthetic:
        from capo.experiments.analysis.synthetic_covariance_benchmark import (
            build_synthetic_covariance_artifacts,
        )

        build_synthetic_covariance_artifacts(paper_out)

    print(f"Wrote paper artifacts -> {paper_out}")


if __name__ == "__main__":
    main()
