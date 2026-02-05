"""
Unit tests for the CAPO reward function.

These tests focus on the (C, P) outcome + process reward table and
do *not* depend on VERL. To avoid calling a real LLM-as-GenPRM, we
inject a dummy GenPRM client that returns hard-coded step judgements.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

# Create a module-level alias for compatibility with existing tests
import capo.verl_integration.reward_fn as rf

# Import directly from reward_fn to avoid pulling in VERL dependency
# through the verl_integration __init__.py
from capo.verl_integration.reward_fn import GenPRMClient


class DummyGenPRMClient(GenPRMClient):
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
) -> dict[str, Any]:
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
        config: rf.CAPOConfig, reward_kwargs: dict[str, Any]
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
        config: rf.CAPOConfig, reward_kwargs: dict[str, Any]
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


# ===========================================================================
# Additional tests for CAPO reward edge cases and configurations
# ===========================================================================


def test_capo_reward_custom_cp_values():
    """Test with non-default C and P values."""
    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True, True],
        correct_reward=5.0,
        process_penalty=2.5,
    )
    assert result["score"] == pytest.approx(5.0)

    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True, False],
        correct_reward=5.0,
        process_penalty=2.5,
    )
    # C - P = 5.0 - 2.5 = 2.5
    assert result["score"] == pytest.approx(2.5)


def test_capo_reward_returns_steps():
    """Check that the result contains step information."""
    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True, True],
    )
    assert "steps" in result
    assert "step_correctness" in result
    assert "wrong_step_indices" in result


def test_capo_reward_all_wrong_steps():
    """All steps wrong case."""
    result = _run_capo_reward(
        is_correct=False,
        step_sequence=[False, False, False],
    )
    # -P = -1.0
    assert result["score"] == pytest.approx(-1.0)
    assert result["wrong_step_indices"] == [0, 1, 2]


def test_capo_reward_single_step():
    """Single step solution."""
    result = _run_capo_reward(
        is_correct=True,
        step_sequence=[True],
    )
    assert result["score"] == pytest.approx(2.0)
    assert len(result["steps"]) >= 1  # At least one step


class TestCAPOConfig:
    """Tests for CAPOConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        cfg = rf.CAPOConfig()
        assert cfg.correct_reward == 2.0
        assert cfg.process_penalty == 1.0
        assert cfg.num_critiques == 4
        assert cfg.vote_mode == "intersection"
        assert cfg.genprm_temperature == 0.0

    def test_custom_values(self):
        """Custom config values are stored correctly."""
        cfg = rf.CAPOConfig(
            correct_reward=10.0,
            process_penalty=5.0,
            num_critiques=8,
            vote_mode="majority",
            genprm_model_name="test-model",
            genprm_temperature=0.5,
        )
        assert cfg.correct_reward == 10.0
        assert cfg.process_penalty == 5.0
        assert cfg.num_critiques == 8
        assert cfg.vote_mode == "majority"
        assert cfg.genprm_model_name == "test-model"
        assert cfg.genprm_temperature == 0.5


class TestGenPRMClient:
    """Tests for GenPRMClient base class."""

    def test_base_client_raises_not_implemented(self):
        """Base GenPRMClient.judge_steps raises NotImplementedError."""
        cfg = rf.CAPOConfig()
        client = rf.GenPRMClient(config=cfg)

        with pytest.raises(NotImplementedError):
            client.judge_steps(
                question="Q",
                solution="S",
                ground_truth="G",
                steps=["step1", "step2"],
            )


class TestVoteModes:
    """Tests for different vote modes in aggregation."""

    def test_intersection_requires_all_wrong(self):
        """Intersection mode: step is wrong only if ALL critiques mark it wrong."""
        cfg = rf.CAPOConfig(num_critiques=3, vote_mode="intersection")

        # Create client that returns different critiques:
        # Critique 1: [True, False]  (step 1 wrong)
        # Critique 2: [True, False]  (step 1 wrong)
        # Critique 3: [True, True]   (step 1 correct)
        # With intersection, step 1 should be considered CORRECT
        # because not all critiques marked it wrong.
        step_sequences = [
            [True, False],  # critique 1
            [True, False],  # critique 2
            [True, True],  # critique 3
        ]
        client = DummyGenPRMClient(config=cfg, step_sequences=step_sequences)

        def _fake_get_client(config, reward_kwargs):
            return client

        orig_get_client = rf._get_or_create_genprm_client
        rf._get_or_create_genprm_client = _fake_get_client

        try:
            result = rf.capo_reward_fn(
                data_source="test",
                solution_str="Step 1\nStep 2",
                ground_truth=None,
                extra_info={"is_correct": True},
                num_critiques=3,
                vote_mode="intersection",
            )
            # Since intersection mode requires all critiques to agree,
            # and critique 3 says step 1 is correct, step 1 is not in wrong_step_indices
            assert 1 not in result["wrong_step_indices"]
        finally:
            rf._get_or_create_genprm_client = orig_get_client


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
