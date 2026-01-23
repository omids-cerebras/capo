#!/usr/bin/env python
"""Prepare a CountDown-only RLVR dataset in VERL parquet format.

This script produces deterministic CountDown datasets suitable for:
- 1xA10 smoke tests
- 8xA10 paper runs (scale n as desired)

Output schema follows VERL's RLHF dataset format:
  - data_source: "countdown"
  - prompt: list[{"role":"user","content":...}]  (already chat-formatted)
  - reward_model: {"style":"rule","ground_truth":{"target":int,"numbers":list[int]}}

The CountDown verifier in VERL expects the model to output an equation inside:
  <answer> ... </answer>
and the equation must use each provided number exactly once.

We generate instances that are guaranteed solvable by construction:
  target = sum(numbers) and a valid solution is simply "a+b+c+d".

Example
-------
python experiments/capo_paper/scripts/data/prepare_countdown_parquet.py \
  --out_dir experiments/capo_paper/data/countdown \
  --n_train 2048 --n_val 256 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


def make_example(
    rng: random.Random, n_numbers: int = 4, lo: int = 1, hi: int = 9
) -> Dict:
    numbers = [rng.randint(lo, hi) for _ in range(n_numbers)]
    target = sum(numbers)
    prompt = (
        "You are given a target integer and a multiset of integers. "
        "Write a single arithmetic expression using each number exactly once "
        "with operations +, -, *, / and parentheses, such that the expression "
        "evaluates to the target.\n\n"
        f"Target: {target}\n"
        f"Numbers: {numbers}\n\n"
        "Return your final expression in the following XML format on the last line:\n"
        "<answer>EXPRESSION</answer>\n"
        "Do not include any additional text after the </answer> tag."
    )
    return {
        "data_source": "countdown",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": int(target),
                "numbers": [int(x) for x in numbers],
            },
        },
        "uid": None,  # filled later
    }


def build_split(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict] = []
    for i in range(n):
        ex = make_example(rng=rng)
        ex["uid"] = f"countdown_{seed}_{i}"
        rows.append(ex)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_split(args.n_train, seed=args.seed)
    val_df = build_split(args.n_val, seed=args.seed + 1)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    meta = {
        "task": "countdown",
        "model_output_contract": "final line must contain <answer>...</answer>",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote:\\n  {train_path}\\n  {val_path}")


if __name__ == "__main__":
    main()
