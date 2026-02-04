"""
Simple text parsing utilities for CAPO.

These helpers are intentionally minimal and should be adapted to your
dataset and method. Keeping them in a separate module makes it easier
to test and to evolve without touching the core VERL integration.
"""

from __future__ import annotations


def segment_solution_into_steps(solution_str: str) -> list[str]:
    """
    Segment a solution string into reasoning steps.

    Default behavior:
    -----------------
    - Split on newline characters.
    - Strip leading/trailing whitespace from each line.
    - Filter out empty lines.

    Parameters
    ----------
    solution_str:
        Full model output for a single sample.

    Returns
    -------
    List[str]
        List of "steps" in order they appear in the solution.

    Notes
    -----
    In practice, you may want a more sophisticated parser. Common
    strategies include:

    - Looking for markers like "Step 1:", "Step 2:", etc.
    - Splitting on LaTeX `\\` markers.
    - Leveraging special tokens inserted by the policy model itself.

    You can replace this function wholesale with whatever matches
    your paper's implementation best.
    """
    raw_lines = solution_str.splitlines()
    steps = [line.strip() for line in raw_lines if line.strip()]
    return steps
