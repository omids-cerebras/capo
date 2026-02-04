"""Prepare the CountDown dataset in a VERL-compatible parquet format.

The CAPO recipe in this repository uses VERL's `RLHFDataset`, which expects a
parquet dataset with a chat-style `prompt` column and a `reward_model` payload
containing task-specific ground truth.

This script writes two files:

  - `train.parquet`
  - `test.parquet`

Each row has the following schema:

  - `prompt`: list[dict] chat messages, e.g. [{"role": "user", "content": "..."}]
  - `data_source`: str ("countdown")
  - `reward_model`: {"ground_truth": {"target": int, "numbers": list[int]}}
  - `extra_info`: dict (optional; includes a stable `index`)

By default, the script downloads the canonical CountDown dataset used in the
DeltaL normalization repository ("Jiayi-Pan/Countdown-Tasks-3to4") and applies a
simple train/test split. If the dataset cannot be downloaded (e.g., offline
execution), the script falls back to a small synthetic CountDown-like dataset so
that smoke tests remain runnable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DEFAULT_HF_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def _build_user_prompt(target: int, numbers: List[int]) -> str:
    """Build the user-facing prompt text.

    We require the final answer to appear inside <answer>...</answer>, because
    the CountDown scorer used by VERL extracts solutions from that tag.
    """

    nums = ", ".join(map(str, numbers))
    return (
        "You are given a target and a multiset of numbers. "
        "Construct a single arithmetic expression using each number exactly once "
        "and using only +, -, *, /, and parentheses so that the expression evaluates "
        f"to the target.\n\nTarget: {target}\nNumbers: {nums}\n\n"
        "Format your response as:\n"
        "<think>...optional reasoning...</think>\n"
        "<answer>EXPRESSION</answer>\n\n"
        "The expression in <answer> must use each number exactly once."
    )


def _to_verl_row(target: int, numbers: List[int], index: int) -> Dict[str, Any]:
    """Convert a (target, numbers) instance to a VERL RLHF-parquet row."""

    prompt_msg = [{"role": "user", "content": _build_user_prompt(target, numbers)}]
    return {
        "prompt": prompt_msg,
        "data_source": "countdown",
        "reward_model": {
            "ground_truth": {"target": int(target), "numbers": list(map(int, numbers))}
        },
        "extra_info": {"index": int(index)},
    }


def _try_load_hf_dataset(
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and split the canonical CountDown dataset from HuggingFace.

    Returns:
        (train_rows, test_rows)
    """

    from datasets import load_dataset

    ds = load_dataset(DEFAULT_HF_DATASET, split="train")
    ds = ds.train_test_split(test_size=0.1, seed=seed)
    train = ds["train"]
    test = ds["test"]

    train_rows = []
    for i, ex in enumerate(train):
        train_rows.append(_to_verl_row(ex["target"], ex["nums"], index=i))

    test_rows = []
    for i, ex in enumerate(test):
        test_rows.append(_to_verl_row(ex["target"], ex["nums"], index=i))

    return train_rows, test_rows


def _synthetic_instances(n: int, seed: int) -> List[Tuple[int, List[int]]]:
    """Generate a small synthetic CountDown-like dataset.

    This is a fallback when the HuggingFace dataset cannot be downloaded.

    The generator builds instances by sampling 3-4 integers and constructing a
    random expression that evaluates to an integer target.
    """

    rng = random.Random(seed)

    def safe_div(a: int, b: int) -> int:
        if b == 0:
            return a
        if a % b == 0:
            return a // b
        return a

    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", safe_div),
    ]

    out: List[Tuple[int, List[int]]] = []
    for _ in range(n):
        k = rng.choice([3, 4])
        numbers = [rng.randint(1, 9) for _ in range(k)]
        val = numbers[0]
        for j in range(1, k):
            _, f = rng.choice(ops)
            val = f(val, numbers[j])
        target = int(val)
        out.append((target, numbers))
    return out


def _write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="If >0, truncate the training split to this many examples (useful for quick tests).",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=0,
        help="If >0, truncate the test split to this many examples (useful for quick tests).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"

    try:
        train_rows, test_rows = _try_load_hf_dataset(seed=args.seed)
        source = "hf"
        source_name = DEFAULT_HF_DATASET
    except Exception as e:  # noqa: BLE001
        # Fall back to a synthetic dataset so that smoke tests remain runnable offline.
        print(f"[WARN] Failed to load HF dataset '{DEFAULT_HF_DATASET}': {e}")
        print("[WARN] Falling back to a synthetic CountDown-like dataset.")
        source = "synthetic"
        source_name = "synthetic"

        syn_train = _synthetic_instances(n=2000, seed=args.seed)
        syn_test = _synthetic_instances(n=200, seed=args.seed + 1)

        train_rows = [
            _to_verl_row(t, nums, index=i) for i, (t, nums) in enumerate(syn_train)
        ]
        test_rows = [
            _to_verl_row(t, nums, index=i) for i, (t, nums) in enumerate(syn_test)
        ]

    if args.max_train and args.max_train > 0:
        train_rows = train_rows[: args.max_train]
    if args.max_test and args.max_test > 0:
        test_rows = test_rows[: args.max_test]

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    _write_parquet(train_rows, train_path)
    _write_parquet(test_rows, test_path)

    meta = {
        "source": source,
        "source_name": source_name,
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_file": str(train_path),
        "test_file": str(test_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Wrote:")
    print("  ", train_path)
    print("  ", test_path)
    print("  ", meta_path)


if __name__ == "__main__":
    main()
