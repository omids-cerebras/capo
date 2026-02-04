"""
Prompt templates for the GenPRM.

This module is a placeholder to keep GenPRM prompt construction
separate from the core reward logic. You can define functions here
that build the system/user messages for your LLM-as-GenPRM, e.g.:

    - `build_math_genprm_prompt(question, solution, ground_truth)`
    - `build_code_genprm_prompt(problem_statement, code, tests)`

At the moment, no concrete prompts are implemented; this is left to
you to fill based on your paper / use case.
"""

from __future__ import annotations


def build_dummy_prompt(
    question: str, solution: str, ground_truth: str | None
) -> list[dict[str, str]]:
    """
    Example placeholder for a chat-style prompt.

    Parameters
    ----------
    question:
        Original problem prompt.

    solution:
        Model's solution to be critiqued.

    ground_truth:
        Reference answer(s), if any.

    Returns
    -------
    List[Dict[str, str]]
        A list of chat messages compatible with many chat-based LLM
        APIs (role/content structure). This is only a stub and should
        be replaced with your actual prompt.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a rigorous mathematics teaching assistant. "
                "Given a student's solution, you will analyze the reasoning "
                "step by step and decide which steps are correct."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Student's solution:\n{solution}\n\n"
                "Please evaluate the solution step by step."
            ),
        },
    ]
    if ground_truth is not None:
        messages.append(
            {
                "role": "user",
                "content": f"For reference, the correct final answer is: {ground_truth}",
            }
        )
    return messages
