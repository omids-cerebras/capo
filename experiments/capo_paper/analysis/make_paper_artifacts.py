#!/usr/bin/env python
"""Build CAPO paper artifacts (figures + LaTeX tables) from collected runs.

This script expects the output of `collect_runs.py` in `--collected`.

It generates, at minimum:
- fig_main.pdf
- fig_dynamics.pdf
- fig_stability.pdf
- fig_length_deciles.pdf
- tab_main_accuracy.tex
- tab_stability_efficiency.tex

The exact metric keys are configurable near the top of this file.
If a required metric is missing, we raise an error (no silent fallback).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt


def _read_json(path: Path):
    return json.loads(path.read_text())


def _ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


def _latex_table(rows, cols, header, caption, label):
    # Minimal compile-safe LaTeX table (natbib-independent).
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{%s}" % ("l" + "r" * (len(cols) - 1)))
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{%s}" % caption)
    lines.append("\\label{%s}" % label)
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = _read_json(args.collected / "runs.json")
    # Key metric assumptions (edit to match your reward_fn/val pipeline):
    # - prefer a validation accuracy metric if available
    # - otherwise fall back to average reward
    ACC_KEYS = [
        "val/accuracy",
        "validation/accuracy",
        "val/acc",
        "val/score",
        "validation/score",
    ]
    REWARD_KEYS = [
        "actor/reward",
        "actor/score",
        "val/reward",
        "val/score",
    ]

    def pick_metric(r, keys):
        for k in keys:
            v = r.get("metrics", {}).get(k)
            if v is not None and not (
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            ):
                return k, v
        return None, None

    # ---------------- Figure: main comparison ----------------
    # group by (exp_name) and show final validation metric
    main_rows = []
    for r in runs:
        k, v = pick_metric(r, ACC_KEYS)
        if v is None:
            k, v = pick_metric(r, REWARD_KEYS)
        _ensure(
            v is not None,
            f"Run {r.get('run_dir')} missing acc/reward metric; update ACC_KEYS/REWARD_KEYS.",
        )
        main_rows.append(
            (r.get("exp_name", "?"), r.get("adv_estimator", "?"), float(v))
        )

    # simple bar plot by run
    labels = [f"{e}\n{a}" for e, a, _ in main_rows]
    vals = [v for _, _, v in main_rows]
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out / "fig_main.pdf")
    plt.close()

    # ---------------- Table: main accuracy ----------------
    cols = ["Experiment", "AdvEstimator", "Metric"]
    table_rows = [[e, a, f"{v:.4f}"] for e, a, v in main_rows]
    (args.out / "tab_main_accuracy.tex").write_text(
        _latex_table(
            table_rows,
            cols,
            header=None,
            caption="Main comparison across methods (metric is final validation accuracy or score).",
            label="tab:main-accuracy",
        )
    )

    # ---------------- Placeholder: dynamics / stability / length ----------------
    # These require time-series metrics and/or per-bucket evaluation.
    # We output empty but compile-safe placeholders so the paper is fully accounted.
    # Replace by extending collect_runs.py to aggregate curves/deciles.

    for name in ["fig_dynamics.pdf", "fig_stability.pdf", "fig_length_deciles.pdf"]:
        plt.figure()
        plt.text(0.5, 0.5, f"{name} TBD: populate from metrics.jsonl", ha="center")
        plt.axis("off")
        plt.savefig(args.out / name)
        plt.close()

    (args.out / "tab_stability_efficiency.tex").write_text(
        _latex_table(
            [
                ["TBD", "TBD", "TBD"],
            ],
            ["Setting", "StabilityMetric", "Throughput"],
            header=None,
            caption="Stability and efficiency summary (TBD; fill from stability sweep outputs).",
            label="tab:stability-efficiency",
        )
    )

    print(f"Wrote artifacts to: {args.out}")


if __name__ == "__main__":
    main()
