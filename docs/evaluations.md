# Evaluation Protocol

This section summarizes the experimental protocol hinted at in the
LaTeX evaluations section.

## Benchmarks and models

- **Benchmarks**:
  - Countdown-like synthetic tasks with controllable difficulty
    and length distribution.
  - Math-style reasoning datasets with long, structured solutions
    (e.g. competition-style problems with chain-of-thought).
- **Models**:
  - Qwen2-style models at two scales (e.g. 3B and 7B) trained with
    VERL-style RL on the above benchmarks.

The aim is to compare **CAPO** versus **ΔL-style baselines** on
stability, sample efficiency and final accuracy, under matched
token budgets and comparable hyperparameters (policy, teacher,
decoding temperature, PPO clipping, etc.).

## Main accuracy metrics

For each benchmark and model size we track:

- **Avg@k** – average correctness over the top‑k responses.
- **Pass@k** – probability that at least one of the top‑k responses
  is correct.

The LaTeX draft reports “best‑of‑run” results (e.g. best Avg@8 /
Pass@8 over a training run) and compares CAPO with ΔL under matched
total tokens consumed.

Qualitative takeaway:

- CAPO improves weighted Math metrics and Countdown metrics at both
  model sizes, with modest but consistent gains.
- CAPO generally dominates ΔL on the stability–accuracy frontier.

## Sample efficiency and tokens-to-target

To compare **sample efficiency**, we track **tokens‑to‑target (TTT)**:
the number of training tokens required to reach a fixed Pass@k
threshold.

- For each method, we fit or smooth the Pass@k curve over training
  steps.
- We then solve for the earliest step that reaches the target
  accuracy and convert it to tokens consumed.

On synthetic and Math-style benchmarks, EB–CAPO typically reduces
TTT relative to ΔL, reflecting the variance reduction predicted by
the theory.

## Stability and monotonicity

Stability is important for practical training:

- **Coefficient of variation (CV)** of accuracy across runs at a
  given step.
- **Monotonicity score** – correlation between step index and held‑out
  accuracy, measuring how often training backslides.

CAPO’s EB weights tend to:

- reduce the kurtosis of the effective weight distribution,
- lead to smoother learning curves with fewer catastrophic spikes,
- increase monotonicity scores relative to ΔL.

This matches the intuition that EB–CAPO avoids overreacting to
noisy, atypically long trajectories.

## Length robustness and equity

Because EB–CAPO explicitly models how variance contracts with length,
it is more robust under **sublinear dependence** (\(\alpha < 1\)):

- ΔL-style rules can over-penalize long trajectories even when they
  are less noisy on a per-token basis.
- EB–CAPO adjusts β and ξ to reflect the observed variance–length
  curve, leading to more equitable weighting across lengths.

In practice, this shows up as:

- better performance on batches mixing short and very long responses,
- reduced sensitivity to the long-tail of the length distribution,
- more balanced effective sample size across length buckets.
