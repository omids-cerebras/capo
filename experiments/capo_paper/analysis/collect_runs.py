#!/usr/bin/env python
"""Collect upstream-style VERL runs into a normalized table.

This collector is intentionally conservative:
- It only relies on artifacts we control locally:
  - Hydra output dir: `outputs/<exp>/<time>/...`
  - Metrics log: `metrics.jsonl` (enabled via `trainer.local_metrics_path`)
  - Resolved Hydra config: `.hydra/config.yaml`

Outputs
-------
- <out>/runs.json (list of dicts)
- <out>/runs.csv

Example
-------
python experiments/capo_paper/analysis/collect_runs.py --runs_dir outputs --out artifacts/collected
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_hydra_cfg(cfg_path: Path) -> Dict[str, Any]:
    # No yaml dependency: parse as JSON when possible; else do minimal YAML parse.
    try:
        import yaml  # type: ignore

        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        # ultra-minimal fallback
        out: Dict[str, Any] = {}
        for line in cfg_path.read_text().splitlines():
            if ":" not in line:
                continue
            if line.strip().startswith("#"):
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        return out


@dataclass
class RunSummary:
    run_dir: Path
    exp_name: str
    timestamp: str
    adv_estimator: str
    last_step: int
    last_metrics: Dict[str, Any]
    best_val_step: Optional[int]
    best_val_metric: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_dir": str(self.run_dir),
            "exp_name": self.exp_name,
            "timestamp": self.timestamp,
            "adv_estimator": self.adv_estimator,
            "last_step": self.last_step,
            "best_val_step": self.best_val_step,
            "best_val_metric": self.best_val_metric,
        }
        # flatten a few common keys if present
        for key in [
            "val/score",
            "val/acc",
            "val/reward",
            "actor/kl",
            "actor/loss",
            "actor/entropy",
            "timing/total",
        ]:
            if key in self.last_metrics:
                d[key] = _safe_float(self.last_metrics[key])
        return d


def _extract_best_val(rows: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
    # Prefer val/acc, else val/score, else val/reward.
    candidates = ["val/acc", "val/score", "val/reward"]
    best_step: Optional[int] = None
    best_val: Optional[float] = None
    for r in rows:
        step = r.get("step")
        for k in candidates:
            v = _safe_float(r.get(k))
            if v is None:
                continue
            if best_val is None or v > best_val:
                best_val = v
                best_step = int(step) if step is not None else None
    return best_step, best_val


def summarize_run(run_dir: Path) -> Optional[RunSummary]:
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not hydra_cfg.exists() or not metrics_path.exists():
        return None

    cfg = _read_hydra_cfg(hydra_cfg)
    adv = None
    try:
        adv = cfg.get("algorithm", {}).get("adv_estimator", None)
    except Exception:
        adv = None
    adv_estimator = str(adv) if adv is not None else "unknown"

    rows = _read_jsonl(metrics_path)
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: int(r.get("step", 0)))
    last = rows_sorted[-1]
    last_step = int(last.get("step", 0))

    best_step, best_val = _extract_best_val(rows_sorted)

    # Hydra default output structure: outputs/<exp>/<timestamp>/
    parts = run_dir.parts
    exp_name = run_dir.parent.name
    ts = run_dir.name

    return RunSummary(
        run_dir=run_dir,
        exp_name=exp_name,
        timestamp=ts,
        adv_estimator=adv_estimator,
        last_step=last_step,
        last_metrics=last,
        best_val_step=best_step,
        best_val_metric=best_val,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="outputs", help="Hydra outputs directory")
    ap.add_argument("--out", type=str, default="artifacts/collected", help="Output directory")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [p for p in runs_dir.glob("*/**/") if (p / "metrics.jsonl").exists()]
    summaries: List[RunSummary] = []
    for rd in sorted(set(run_dirs)):
        s = summarize_run(rd)
        if s is not None:
            summaries.append(s)

    rows = [s.to_dict() for s in summaries]
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # CSV
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"Collected {len(rows)} runs -> {out_dir}")


if __name__ == "__main__":
    main()
