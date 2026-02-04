# tests/test_utils.py
"""
Unit tests for capo.utils.

Tests for parsing and prompt utilities that don't depend on VERL.
"""

from __future__ import annotations

import pytest

from capo.utils.parsing import segment_solution_into_steps
from capo.utils.prompts import build_dummy_prompt


class TestSegmentSolution:
    """Tests for segment_solution_into_steps."""

    def test_basic_segmentation(self):
        """Basic multi-line solution is split into steps."""
        solution = """Step 1: Find the derivative.
Step 2: Set it to zero.
Step 3: Solve for x."""
        steps = segment_solution_into_steps(solution)
        assert len(steps) == 3
        assert steps[0] == "Step 1: Find the derivative."
        assert steps[1] == "Step 2: Set it to zero."
        assert steps[2] == "Step 3: Solve for x."

    def test_strips_whitespace(self):
        """Leading and trailing whitespace is stripped from each step."""
        solution = "  First step  \n  Second step  \n"
        steps = segment_solution_into_steps(solution)
        assert steps == ["First step", "Second step"]

    def test_filters_empty_lines(self):
        """Empty lines are filtered out."""
        solution = "Step 1\n\nStep 2\n\n\nStep 3"
        steps = segment_solution_into_steps(solution)
        assert len(steps) == 3

    def test_empty_string(self):
        """Empty string produces empty list."""
        steps = segment_solution_into_steps("")
        assert steps == []

    def test_whitespace_only_string(self):
        """Whitespace-only string produces empty list."""
        steps = segment_solution_into_steps("   \n\t\n   ")
        assert steps == []

    def test_single_line(self):
        """Single line without newlines."""
        solution = "The answer is 42."
        steps = segment_solution_into_steps(solution)
        assert steps == ["The answer is 42."]

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        solution = "ステップ1: 計算する\nステップ2: 確認する"
        steps = segment_solution_into_steps(solution)
        assert len(steps) == 2
        assert "計算" in steps[0]

    def test_crlf_line_endings(self):
        """Windows-style CRLF line endings work."""
        solution = "Step 1\r\nStep 2\r\nStep 3"
        steps = segment_solution_into_steps(solution)
        assert len(steps) == 3


class TestBuildDummyPrompt:
    """Tests for build_dummy_prompt."""

    def test_returns_list_of_messages(self):
        """Returns a list of chat messages."""
        messages = build_dummy_prompt(
            question="What is 2+2?",
            solution="2+2 = 4",
            ground_truth="4",
        )
        assert isinstance(messages, list)
        assert len(messages) >= 2

    def test_messages_have_role_and_content(self):
        """Each message has 'role' and 'content' keys."""
        messages = build_dummy_prompt(
            question="Test question",
            solution="Test solution",
            ground_truth=None,
        )
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_includes_question_and_solution(self):
        """Question and solution appear in the messages."""
        messages = build_dummy_prompt(
            question="What is the capital of France?",
            solution="The capital is Paris.",
            ground_truth=None,
        )
        all_content = " ".join(msg["content"] for msg in messages)
        assert "France" in all_content
        assert "Paris" in all_content

    def test_includes_ground_truth_when_provided(self):
        """Ground truth is included when provided."""
        messages = build_dummy_prompt(
            question="Q",
            solution="A",
            ground_truth="correct_answer_xyz",
        )
        all_content = " ".join(msg["content"] for msg in messages)
        assert "correct_answer_xyz" in all_content

    def test_works_without_ground_truth(self):
        """Works when ground_truth is None."""
        messages = build_dummy_prompt(
            question="Q",
            solution="A",
            ground_truth=None,
        )
        # Should have system and user messages at minimum
        assert len(messages) >= 2


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
