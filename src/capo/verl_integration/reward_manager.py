"""
VERL-compatible reward manager for CAPO.

This module implements a `CAPORewardManager` that:

1. Decodes prompts and responses from token IDs.
2. Calls `capo_reward_fn` to obtain:
   - a scalar CAPO reward ("score"),
   - segmented steps,
   - which steps are flawed.
3. Converts flawed-step information into a *token-level* reward tensor
   aligned with VERL's expectation for reward managers.

The result is a `(batch_size, response_length)` tensor of per-token
rewards, which can then be fed into a GRPO / PPO trainer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from capo.verl_integration.reward_fn import CAPOConfig, capo_reward_fn

# These imports rely on VERL. They will only succeed in an environment
# where VERL is installed. This is intentional: this module is not
# supposed to be imported in isolation without VERL.
try:
    from verl import DataProto
except ImportError as exc:  # pragma: no cover - only triggered w/o VERL
    raise ImportError(
        "CAPORewardManager requires VERL to be installed. "
        "Please install VERL first (see README)."
    ) from exc


def _build_wrong_step_token_mask(
    tokenizer: Any,
    response_ids: torch.Tensor,
    steps: Sequence[str],
    wrong_step_indices: Sequence[int],
) -> torch.Tensor:
    """
    Build a boolean token mask from incorrect step indices.

    Parameters
    ----------
    tokenizer:
        Tokenizer object used by VERL. It must implement a method
        `encode(text, add_special_tokens=False)` returning a list[int].

    response_ids:
        1D tensor of token IDs corresponding to the model's response
        (not including the prompt).

    steps:
        List of step strings as returned by `capo_reward_fn`.

    wrong_step_indices:
        Indices of steps that are considered incorrect.

    Returns
    -------
    torch.Tensor
        1D boolean tensor of shape `(len(response_ids),)` where
        `True` indicates that the token belongs to an erroneous step.

    Notes
    -----
    This implementation uses a simple alignment strategy:

    1. Encode each step independently with the tokenizer.
    2. Concatenate these token sequences in order.
    3. Check that they match the prefix of `response_ids`.
    4. Assign "wrong" to tokens belonging to steps in
       `wrong_step_indices`.

    This approach *assumes* the response is effectively the concatenation
    of the step strings (plus possibly trailing tokens). If your
    formatting is more complex, consider implementing a more robust
    alignment that uses character-level spans or special markers.

    If alignment fails, the function falls back to a mask of all False.
    """
    device = response_ids.device
    resp_len = int(response_ids.shape[0])

    mask = torch.zeros(resp_len, dtype=torch.bool, device=device)

    if not steps:
        # No steps at all: no penalties.
        return mask

    wrong_step_indices_set = set(int(i) for i in wrong_step_indices)

    # Encode each step and move through the response token sequence.
    offset = 0
    for step_idx, step_text in enumerate(steps):
        step_token_ids = tokenizer.encode(step_text, add_special_tokens=False)
        step_token_ids_tensor = torch.tensor(
            step_token_ids, dtype=response_ids.dtype, device=device
        )
        step_len = int(step_token_ids_tensor.shape[0])

        # If the step would overrun the response, alignment fails.
        if offset + step_len > resp_len:
            # Alignment failure: we choose not to raise, but to return
            # a zero mask and log a warning.
            # In a real implementation you might want to log this.
            return mask

        # Check that the response tokens match the step tokens at this offset.
        if not torch.equal(
            response_ids[offset : offset + step_len],
            step_token_ids_tensor,
        ):
            # Alignment failure; see comment above.
            return mask

        # If this step is incorrect, mark its tokens as True.
        if step_idx in wrong_step_indices_set:
            mask[offset : offset + step_len] = True

        offset += step_len

    # Any trailing tokens (offset:resp_len) are left as False.
    return mask


class CAPORewardManager:
    """
    VERL `RewardManager` for CAPO.

    Responsibilities
    ----------------
    - Implement the `__call__` protocol expected by VERL.
    - For each sample in the batch:
        * Decode prompt & response token IDs to strings.
        * Call `capo_reward_fn` to obtain:
          - scalar CAPO reward ("score"),
          - steps,
          - wrong step indices.
        * Convert wrong-step indices into a token-level mask over the
          response tokens.
        * Combine outcome and process components into per-token rewards.

    Token-Level Reward Shape
    ------------------------
    The returned tensor has the same shape as VERL's `responses` tensor,
    i.e.:

        reward_tensor: (batch_size, response_length)

    Only valid response tokens (as indicated by the attention mask) are
    filled; padding tokens remain zero.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **reward_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        tokenizer:
            Tokenizer used by the VERL actor. Must support `.decode`
            and `.encode` methods.

        num_examine:
            How many examples per `data_source` to print to stdout for
            debugging. You can set this to 0 to disable printing.

        compute_score:
            Optional override for the scoring function. If `None`
            (default), the `capo_reward_fn` from this package is used.
            You can set this to any callable with the same signature.

        reward_fn_key:
            Key under which the data source is stored in the
            `non_tensor_batch` of VERL's `DataProto`.

        reward_kwargs:
            Additional keyword arguments forwarded to `capo_reward_fn`
            (and also used to construct a `CAPOConfig` internally).
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Default scoring function is `capo_reward_fn` from this repo.
        self.compute_score = compute_score or capo_reward_fn

        # Configuration is stored for potential debugging.
        self.capo_config = CAPOConfig(
            correct_reward=float(reward_kwargs.get("correct_reward", 2.0)),
            process_penalty=float(reward_kwargs.get("process_penalty", 1.0)),
            num_critiques=int(reward_kwargs.get("num_critiques", 4)),
            vote_mode=str(reward_kwargs.get("vote_mode", "intersection")),
            genprm_model_name=reward_kwargs.get("genprm_model_name"),
            genprm_temperature=float(reward_kwargs.get("genprm_temperature", 0.0)),
        )
        self.reward_kwargs = reward_kwargs

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """
        Compute CAPO token-level rewards for a batch of data.

        Parameters
        ----------
        data:
            VERL `DataProto` object containing both tensors and
            non-tensor metadata for the current batch.

        return_dict:
            If True, return a dict with "reward_tensor" and
            "reward_extra_info" keys. If False (default), return only
            the reward tensor.

        Returns
        -------
        torch.Tensor or Dict[str, Any]
            Either a `(batch, response_length)` tensor of token-level
            rewards, or a dict wrapping that tensor along with
            additional logging info.
        """
        # If somebody pre-computed token-level rewards upstream, just
        # forward them. This is compatible with other setups that might
        # bypass the CAPO logic.
        if "token_level_rewards" in data.batch:
            reward_tensor = data.batch["token_level_rewards"]
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            return reward_tensor

        # Initialize reward tensor with zeros.
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(
            responses,
            dtype=torch.float32,
            device=responses.device,
        )

        reward_extra_info: dict[str, list[Any]] = defaultdict(list)
        already_print_data_sources: dict[str, int] = {}

        batch_size = len(data)

        for i in range(batch_size):
            # Access sample i's data.
            data_item = data[i]

            # Extract tensors.
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attention_mask = data_item.batch["attention_mask"]

            # Compute lengths from mask: [prompt | response].
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(attention_mask[:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            valid_response_length = int(attention_mask[prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # Decode prompt and response into strings.
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            solution_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            # Extract non-tensor metadata.
            non_tensor = data_item.non_tensor_batch
            ground_truth = non_tensor.get("reward_model", {}).get("ground_truth")
            data_source = non_tensor.get(self.reward_fn_key, "unknown_source")
            extra_info = dict(non_tensor.get("extra_info", {}))

            # We also propagate a few VERL-specific keys into extra_info
            # so they are visible to the reward function if needed.
            extra_info["__num_turns__"] = non_tensor.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = non_tensor.get("reward_scores", {})

            # 1. Compute CAPO score & step statistics.
            result = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.reward_kwargs,
            )

            if not isinstance(result, dict) or "score" not in result:
                raise TypeError(
                    "CAPORewardManager expected compute_score to return a dict "
                    "with at least a 'score' key."
                )

            score = float(result["score"])
            steps: list[str] = list(result.get("steps", []))
            wrong_step_indices: list[int] = list(result.get("wrong_step_indices", []))

            # 2. Convert flawed-step info into a token mask.
            wrong_token_mask = _build_wrong_step_token_mask(
                tokenizer=self.tokenizer,
                response_ids=valid_response_ids,
                steps=steps,
                wrong_step_indices=wrong_step_indices,
            )

            # 3. Combine outcome and process components into per-token reward.
            #    For simplicity, we treat the scalar CAPO reward `score`
            #    as the base per-token reward, and subtract `P` from
            #    tokens in erroneous steps.
            base = float(score)
            per_token_reward = torch.full(
                (valid_response_length,),
                fill_value=base,
                dtype=torch.float32,
                device=responses.device,
            )

            # Apply per-token penalty P to tokens marked as wrong.
            P = self.capo_config.process_penalty
            per_token_reward[wrong_token_mask] -= float(P)

            # Store in the batched reward tensor.
            reward_tensor[i, :valid_response_length] = per_token_reward

            # 4. Collect extra info for logging if requested.
            for key, value in result.items():
                reward_extra_info[key].append(value)

            # 5. Optional pretty-printing for manual inspection.
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n[CAPORewardManager] Example")
                print("[data_source]", data_source)
                print("[prompt]", prompt_str)
                print("[solution]", solution_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                print("[steps]", steps)
                print("[wrong_step_indices]", wrong_step_indices)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }

        return reward_tensor
