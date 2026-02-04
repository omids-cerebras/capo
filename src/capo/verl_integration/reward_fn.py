"""
CAPO reward function and GenPRM client.

This file is deliberately agnostic to VERL. It defines:

- `CAPOConfig`: a dataclass capturing CAPO hyperparameters.
- `GenPRMClient`: a stub you should implement to call your LLM-as-GenPRM.
- `capo_reward_fn`: the core CAPO reward composition function.

`capo_reward_fn` is intended to be plugged into VERL via
`reward_model.custom_reward_function` or called directly by
`CAPORewardManager` (see `reward_manager.py`).

The reward follows the standard CAPO outcome + process table:

    is_correct    all_steps_correct    reward
    ----------    -----------------    ------
       True              True           +C
       True              False          +C - P
       False             True            0
       False             False           -P

where C = `correct_reward`, P = `process_penalty`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class CAPOConfig:
    """
    Configuration for CAPO reward computation.

    Parameters
    ----------
    correct_reward:
        Reward C for an answer that is correct *and* whose reasoning
        steps are all judged correct by the GenPRM.

    process_penalty:
        Penalty P applied when there exists at least one flawed step.
        This is subtracted from the reward in the "process-bad" cases.

    num_critiques:
        Number of independent GenPRM critiques to draw for each sample.
        The critiques are combined via `vote_mode` to determine which
        steps are finally labeled as incorrect.

    vote_mode:
        Voting scheme for aggregating GenPRM critiques.

        Recommended options:
        - "intersection": a step is wrong only if *all* critiques mark
          it as wrong (high precision).
        - "majority": a step is wrong if a strict majority of critiques
          mark it as wrong (higher recall).

    genprm_model_name:
        Identifier for the GenPRM model (e.g., a model name that your
        client understands). This file does not enforce any particular
        backend; you are free to implement it with OpenAI, vLLM, etc.

    genprm_temperature:
        Temperature for sampling critiques from the GenPRM. For many
        uses, a near-deterministic setting such as 0.0 or 0.1 works
        well to reduce variance.

    """

    correct_reward: float = 2.0
    process_penalty: float = 1.0
    num_critiques: int = 4
    vote_mode: str = "intersection"
    genprm_model_name: str | None = None
    genprm_temperature: float = 0.0


class GenPRMClient:
    """
    Stub for a Generative Process Reward Model client.

    This class should encapsulate *all* details of how you query
    an LLM-as-GenPRM:

    - Prompt construction.
    - Sampling parameters.
    - Parsing the model's output into per-step judgements.

    The skeleton implementation here simply raises NotImplementedError
    so that you are forced to plug in your own logic.

    You can also create a subclass (e.g., `OpenAIChatGenPRMClient`)
    and inject it into `capo_reward_fn` via `reward_kwargs`, or wire
    the specific client in `CAPORewardManager`.
    """

    def __init__(self, config: CAPOConfig) -> None:
        self.config = config

    def judge_steps(
        self,
        question: str,
        solution: str,
        ground_truth: str | None,
        steps: Sequence[str],
    ) -> Sequence[bool]:
        """
        Run the GenPRM to judge each reasoning step.

        Parameters
        ----------
        question:
            Original problem prompt (the "Q" in math datasets).

        solution:
            Full model solution in natural language (the "A").

        ground_truth:
            Reference answer(s), if available. This may be used to
            help the GenPRM calibrate its critique.

        steps:
            A sequence of step strings, typically obtained by segmenting
            `solution` with `segment_solution_into_steps`.

        Returns
        -------
        Sequence[bool]
            `is_step_correct[j]` indicates whether step `j` is
            considered correct (True) or flawed (False).

        Notes
        -----
        - In *this* skeleton, this method raises NotImplementedError.
          You must implement it for your method to work.
        """
        raise NotImplementedError(
            "GenPRMClient.judge_steps must be implemented with your "
            "actual LLM-as-GenPRM backend."
        )


def _get_or_create_config(reward_kwargs: dict[str, Any]) -> CAPOConfig:
    """
    Build a CAPOConfig from generic reward_kwargs.

    This helper makes it convenient to pass scalar parameters (C, P,
    etc.) from a Hydra config without having to manually construct
    CAPOConfig elsewhere.
    """
    return CAPOConfig(
        correct_reward=float(reward_kwargs.get("correct_reward", 2.0)),
        process_penalty=float(reward_kwargs.get("process_penalty", 1.0)),
        num_critiques=int(reward_kwargs.get("num_critiques", 4)),
        vote_mode=str(reward_kwargs.get("vote_mode", "intersection")),
        genprm_model_name=reward_kwargs.get("genprm_model_name"),
        genprm_temperature=float(reward_kwargs.get("genprm_temperature", 0.0)),
    )


def _get_or_create_genprm_client(
    config: CAPOConfig,
    reward_kwargs: dict[str, Any],
) -> GenPRMClient:
    """
    Obtain a GenPRMClient for CAPO.

    The default behavior is to create a vanilla `GenPRMClient` with
    the given config (which raises NotImplementedError). If you want
    to use a concrete subclass, you can inject it via `reward_kwargs`:

        reward_kwargs:
          genprm_client_cls: path.to.YourClientClass

    or pass an already instantiated client:

        reward_kwargs:
          genprm_client: <YourClientInstance>
    """
    # If an instance is explicitly provided, use it.
    client = reward_kwargs.get("genprm_client")
    if client is not None:
        return client

    # If a class is provided, instantiate it with the config.
    client_cls = reward_kwargs.get("genprm_client_cls")
    if client_cls is not None:
        return client_cls(config=config)

    # Fallback: plain stub (will raise NotImplementedError upon use)
    return GenPRMClient(config=config)


def _segment_solution_into_steps(solution_str: str) -> list[str]:
    """
    Very simple solution segmentation into steps.

    Current behavior:
        - Split the solution on newline characters.
        - Strip leading/trailing whitespace from each line.
        - Filter out empty lines.

    This is intentionally naive. In your production code, consider a
    more robust parser (e.g., using markers like "Step 1:", LaTeX
    delimiters, or other dataset-specific heuristics).

    Keeping this function local to the reward function simplifies
    testing and reduces cross-module dependencies.
    """
    raw_lines = solution_str.splitlines()
    steps = [line.strip() for line in raw_lines if line.strip()]
    return steps


def _aggregate_step_judgements(
    critiques: Sequence[Sequence[bool]],
    vote_mode: str,
) -> tuple[list[bool], list[int]]:
    """
    Aggregate multiple GenPRM critiques into final per-step judgements.

    Parameters
    ----------
    critiques:
        A sequence over critiques, where each critique is a sequence
        of `bool` values of length `num_steps`. All critiques must
        have the same length.

    vote_mode:
        Voting scheme. Recognized:
        - "intersection": a step is marked incorrect only if *all*
          critiques mark it incorrect.
        - "majority": a step is marked incorrect if a strict majority
          of critiques mark it incorrect.

    Returns
    -------
    step_correctness:
        List[bool] of length `num_steps`. `True` means "step is correct".

    wrong_step_indices:
        List[int] of indices j where `step_correctness[j]` is False.
    """
    if not critiques:
        # No critiques at all; default to "all steps correct".
        return [], []

    num_critiques = len(critiques)
    num_steps = len(critiques[0])

    # Sanity check: ensure all critiques agree on number of steps.
    for idx, c in enumerate(critiques):
        if len(c) != num_steps:
            raise ValueError(
                f"Inconsistent number of steps in critique {idx}: "
                f"expected {num_steps}, got {len(c)}."
            )

    wrong_step_indices: list[int] = []
    step_correctness: list[bool] = []

    for j in range(num_steps):
        # Collect the decision for step j across all critiques.
        decisions_j = [not c[j] for c in critiques]  # True if "wrong"
        num_wrong = sum(decisions_j)

        if vote_mode == "intersection":
            is_wrong = num_wrong == num_critiques
        elif vote_mode == "majority":
            is_wrong = num_wrong > num_critiques // 2
        else:
            raise ValueError(f"Unsupported vote_mode: {vote_mode}")

        if is_wrong:
            wrong_step_indices.append(j)
            step_correctness.append(False)
        else:
            step_correctness.append(True)

    return step_correctness, wrong_step_indices


def capo_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    """
    Core CAPO reward function: outcome + process composition.

    This function is designed to be called in two typical ways:

    1. Directly from VERL as a `custom_reward_function`.
    2. Indirectly via `CAPORewardManager`, which calls this function
       inside its `__call__` implementation.

    Parameters
    ----------
    data_source:
        A string identifying the dataset or task. You can use this to
        switch between different correctness checkers, prompt templates,
        etc.

    solution_str:
        Full model output for the current sample (the "solution").

    ground_truth:
        Reference answer(s) if available. May be `None` for tasks where
        correctness cannot be determined automatically.

    extra_info:
        Optional dictionary with additional metadata. This function
        expects (but does not *require*) the key `"is_correct"`:

            extra_info.get("is_correct") -> bool or None

        If `is_correct` is missing, you must implement your own
        correctness checker or extend this function accordingly.

    reward_kwargs:
        Free-form keyword arguments that are used to construct a
        `CAPOConfig` and optionally a `GenPRMClient`. Typical keys:

        - "correct_reward": float   (C)
        - "process_penalty": float  (P)
        - "num_critiques": int
        - "vote_mode": str
        - "genprm_model_name": str

    Returns
    -------
    Dict[str, Any]
        A dictionary with at least:

        - "score": float
              The scalar CAPO reward for this sample.

        - "steps": List[str]
              The segmented reasoning steps (output of the internal
              step segmentation function).

        - "step_correctness": List[bool]
              Whether each step is judged correct.

        - "wrong_step_indices": List[int]
              Indices of steps judged incorrect.

        Downstream code (e.g., `CAPORewardManager`) may use these
        fields to generate token-level reward masks.

    Notes
    -----
    The reward table implemented is:

        is_correct & all_steps_correct -> +C
        is_correct & any_wrong_step    -> +C - P
        not is_correct & all_steps_correct -> 0
        not is_correct & any_wrong_step    -> -P

    This corresponds to Eq. (1) in the CAPO paper, if you interpret
    C and P as outcome and process weights respectively.
    """
    extra_info = extra_info or {}

    config = _get_or_create_config(reward_kwargs)
    genprm_client = _get_or_create_genprm_client(config, reward_kwargs)

    # 1. Determine correctness of the *final answer*.
    #    In most practical setups, this boolean is computed by a
    #    separate rule-based evaluator (e.g., numeric equality for
    #    MATH) and injected via extra_info["is_correct"].
    is_correct = extra_info.get("is_correct")
    if is_correct is None:
        # In a real implementation, you would route to a dataset-
        # or task-specific correctness checker here.
        raise ValueError(
            "extra_info['is_correct'] is required for capo_reward_fn. "
            "Please compute correctness upstream and pass it in."
        )

    # 2. Segment the solution into discrete reasoning steps.
    steps = _segment_solution_into_steps(solution_str)

    # 3. Run GenPRM several times (num_critiques) and collect per-step
    #    judgements. Each critique is a sequence of booleans of length
    #    len(steps), where False indicates a flawed step.
    critiques: list[Sequence[bool]] = []
    for _ in range(config.num_critiques):
        step_is_correct = genprm_client.judge_steps(
            question=data_source,
            solution=solution_str,
            ground_truth=ground_truth,
            steps=steps,
        )
        critiques.append(step_is_correct)

    step_correctness, wrong_step_indices = _aggregate_step_judgements(
        critiques=critiques,
        vote_mode=config.vote_mode,
    )

    # If there are no steps, treat as "no process information".
    if not steps:
        step_correctness = []
        wrong_step_indices = []

    any_wrong_step = bool(wrong_step_indices)

    # 4. Apply the CAPO reward table.
    C = config.correct_reward
    P = config.process_penalty

    if is_correct and not any_wrong_step:
        # Correct answer, clean reasoning.
        score = +C
    elif is_correct and any_wrong_step:
        # Correct answer, but flawed reasoning somewhere.
        score = +C - P
    elif (not is_correct) and not any_wrong_step:
        # Incorrect answer, clean reasoning (often rare).
        score = 0.0
    else:
        # Incorrect answer and flawed reasoning.
        score = -P

    return {
        "score": float(score),
        "steps": steps,
        "step_correctness": step_correctness,
        "wrong_step_indices": wrong_step_indices,
    }
