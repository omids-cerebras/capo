"""
Unit tests for the CAPO reward function.

These tests focus on the (C, P) outcome + process reward table and
do *not* depend on VERL. To avoid calling a real LLM-as-GenPRM, we
inject a dummy GenPRM client that returns hard-coded step judgements.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pytest

from capo.verl_integration import reward_fn as rf


class DummyGenPRMClient(rf.GenPRMClient):
    """
    Simple deterministic GenPRMClient for testing.

    This client is initialized with a list of lists of booleans, where
    each inner list is the per-step correctness judgements for a call
    to `judge_steps`.

    Each call to `judge_steps` consumes one entry of this list.
    """

    def __init__(self, config: rf.CAPOConfig, step_sequences: Sequence[Sequence[bool]]):
        super().__init__(config=config)
        self._step_sequences = list(step_sequences)
        self._call_idx = 0

    def judge_steps(
        self,
        question: str,
        solution: str,
        ground_truth: str | None,
        steps: Sequence[str],
    ) -> Sequence[bool]:
        if self._call_idx >= len(self._step_sequences):
            raise RuntimeError("Not enough dummy step sequences for test.")
        seq = self._step_sequences[self._call_idx]
        self._call_idx += 1
        # We ignore `steps` length mismatches for simplicity; real
        # code should check.
        return list(seq)


def _run_capo_reward(
    is_correct: bool,
    step_sequence: Sequence[bool],
    correct_reward: float = 2.0,
    process_penalty: float = 1.0,
) -> Dict[str, Any]:
    """
    Helper to run the CAPO reward function with a dummy GenPRM.
    """
    cfg = rf.CAPOConfig(
        correct_reward=correct_reward,
        process_penalty=process_penalty,
        num_critiques=1,
        vote_mode="intersection",
    )

    client = DummyGenPRMClient(config=cfg, step_sequences=[step_sequence])

    # Monkeypatch the internal client creation helper so that the
    # reward function uses our dummy client.
    def _fake_get_client(
        config: rf.CAPOConfig, reward_kwargs: Dict[str, Any]
    ) -> rf.GenPRMClient:
        return client

    # Store original helper and restore afterwards.
    orig_get_client = rf._get_or_create_genprm_client
    rf._get_or_create_genprm_client = _fake_get_client  # type: ignore[attr-defined]

    try:
        result = rf.capo_reward_fn(
            data_source="dummy_source",
            solution_str="Step 1\nStep 2",
            ground_truth=None,
            extra_info={"is_correct": is_correct},
            correct_reward=correct_reward,
            process_penalty=process_penalty,
            num_critiques=1,
            vote_mode="intersection",
        )
    finally:
        # Restore original helper.
        rf._get_or_create_genprm_client = orig_get_client  # type: ignore[attr-defined]

    return result


def test_capo_reward_correct_and_all_steps_correct():
    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True, True],
        correct_reward=2.0,
        process_penalty=1.0,
    )
    assert result["score"] == pytest.approx(2.0)


def test_capo_reward_correct_and_some_steps_wrong():
    # One correct, one wrong -> process-bad case.
    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True, False],
        correct_reward=2.0,
        process_penalty=1.0,
    )
    # Expected: C - P = 1.0
    assert result["score"] == pytest.approx(1.0)


def test_capo_reward_incorrect_and_all_steps_correct():
    result = _run_capo_reward(
        is_correct=False,
        step_sequence=[True, True],
        correct_reward=2.0,
        process_penalty=1.0,
    )
    # Expected: 0.0
    assert result["score"] == pytest.approx(0.0)


def test_capo_reward_incorrect_and_some_steps_wrong():
    result = _run_capo_reward(
        is_correct=False,
        step_sequence=[True, False],
        correct_reward=2.0,
        process_penalty=1.0,
    )
    # Expected: -P = -1.0
    assert result["score"] == pytest.approx(-1.0)


def test_capo_reward_missing_is_correct_raises():
    cfg = rf.CAPOConfig()
    client = DummyGenPRMClient(config=cfg, step_sequences=[[True, True]])

    def _fake_get_client(
        config: rf.CAPOConfig, reward_kwargs: Dict[str, Any]
    ) -> rf.GenPRMClient:
        return client

    orig_get_client = rf._get_or_create_genprm_client
    rf._get_or_create_genprm_client = _fake_get_client  # type: ignore[attr-defined]

    try:
        with pytest.raises(ValueError):
            rf.capo_reward_fn(
                data_source="dummy",
                solution_str="Step 1\nStep 2",
                ground_truth=None,
                extra_info={},  # no "is_correct"
            )
    finally:
        rf._get_or_create_genprm_client = orig_get_client  # type: ignore[attr-defined]
