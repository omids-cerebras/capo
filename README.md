# CAPO + VERL: Implementation & Training Guide

This document explains:

1. How VERL’s PPO/GRPO training loop works.  
2. How the **CAPO** method is implemented inside VERL.  
3. What each CAPO component does (reward + advantage).  
4. How to configure and **run a training experiment** with CAPO / EB–CAPO.  

Paste this README directly into your repo root (as `README.md`) or into `docs/howto_capo_verl.md`. The LaTeX math uses `$...$` and `$$...$$` so it renders correctly on GitHub / MkDocs.

---

## 0. High‑level: where CAPO plugs into VERL

A single PPO/GRPO step in VERL can be sketched as:

1. **Rollout**: actors sample trajectories from the current policy.  
2. **Reward**: a `RewardManager` converts trajectories into token‑level rewards.  
3. **Advantage estimation**: an advantage estimator converts rewards into token‑level advantages.  
4. **Policy update**: PPO/GRPO uses advantages and log‑prob ratios to update the policy.

CAPO integrates at steps **2** and **3**:

- `CAPORewardManager`: implements CAPO’s outcome + process reward and produces **token‑level CAPO rewards**.  
- `capo`, `capo_eb_lite`, `capo_eb`: three advantage estimators that compute **CAPO advantages** from those rewards, including Empirical Bayes weighting (EB–CAPO).

---

## 1. Repository structure & registration

The relevant CAPO code lives under:

```text
capo/
  __init__.py
  verl_integration/
    __init__.py
    reward_fn.py          # CAPO reward function + GenPRM interface
    reward_manager.py     # CAPORewardManager (token-level rewards for VERL)
    adv_estimators.py     # CAPO, EB–CAPO-lite, full EB–CAPO
```

To hook into VERL’s registries, `capo/verl_integration/__init__.py` imports the submodules:

```python
# capo/verl_integration/__init__.py
from . import reward_manager   # registers "capo" RewardManager
from . import adv_estimators   # registers "capo", "capo_eb_lite", "capo_eb"
```

Then, **before training starts**, you must import:

```python
import capo.verl_integration  # side effect: registers CAPORewardManager + adv estimators
```

After this, VERL can see:

- `reward_model.reward_manager: capo`  
- `algorithm.adv_estimator: capo`, `capo_eb_lite`, or `capo_eb`

in your Hydra config.

---

## 2. CAPO reward path in VERL

### 2.1 CAPO reward function (`reward_fn.py`)

File: `capo/verl_integration/reward_fn.py`

This file implements the **CAPO reward logic** at the example level.

#### CAPOConfig

A small config object collects CAPO hyperparameters, conceptually:

- `correct_reward` (denoted $C$ in the paper)  
- `process_penalty` (denoted $P$)  
- `num_critiques` (how many GenPRM calls / votes)  
- `vote_mode` (`"intersection"` or `"majority"`)  
- `genprm_model_name` (if you actually call an LLM as GenPRM)

#### GenPRMClient (process critic interface)

`GenPRMClient` is an abstraction for the **process critic**:

```python
class GenPRMClient:
    def judge_steps(self, question, solution, ground_truth, steps) -> list[bool]:
        ...
```

In a real setup, `judge_steps` runs an LLM or external verifier and returns a boolean per step (`True` = step is judged correct).

#### Step segmentation

CAPO needs the reasoning process split into steps:

```python
def _segment_solution_into_steps(solution_str: str) -> list[str]:
    # typically:
    # - split on newlines
    # - strip empty lines
    return steps
```

You can customize this depending on the dataset (e.g., explicit “Step 1:” markers).

#### Aggregating critiques

Given `num_critiques` sets of step judgments, we aggregate them:

- `intersection`: a step is **wrong** only if *all* critiques say wrong.  
- `majority`: a step is wrong if a strict majority say wrong.

This produces a final binary vector `step_correctness` and a list `wrong_step_indices`.

#### CAPO score table

Let:

- `is_correct` = final answer is correct (from `extra_info["is_correct"]`),  
- `any_wrong_step` = at least one step is judged wrong.

CAPO uses the table:

- Correct & all steps correct:  
  $$\text{score} = +C$$

- Correct & some steps wrong:  
  $$\text{score} = C - P$$

- Incorrect & all steps correct:  
  $$\text{score} = 0$$

- Incorrect & some steps wrong:  
  $$\text{score} = -P$$

So the core reward function looks like:

```python
def capo_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    config: CAPOConfig | None = None,
    genprm_client: GenPRMClient | None = None,
) -> dict:
    # 1) segment solution into steps
    # 2) get step correctness via GenPRMClient across num_critiques
    # 3) apply (C, P) table to get scalar "score"
    # 4) return score + per-step metadata
```

It returns a dict like:

```python
{
  "score": float,                 # CAPO scalar
  "steps": list[str],             # segmented steps
  "step_correctness": list[bool],
  "wrong_step_indices": list[int],
}
```

This function is **agnostic** to VERL; it just implements CAPO’s math.

---

### 2.2 CAPORewardManager (`reward_manager.py`)

File: `capo/verl_integration/reward_manager.py`

VERL expects a **RewardManager** that takes a batch and returns a tensor of token‑level rewards. You register one as:

```python
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("capo")
class CAPORewardManager(AbstractRewardManager):
    ...
```

In your config, you select it via:

```yaml
reward_model:
  reward_manager: capo
```

#### Constructor

Conceptually:

```python
def __init__(
    self,
    tokenizer,
    num_examine: int = 0,
    compute_score: callable | None = capo_reward_fn,
    reward_fn_key: str = "data_source",
    **reward_kwargs,
):
    # tokenizer: HF tokenizer used by the actor
    # num_examine: number of examples to pretty-print
    # compute_score: usually capo_reward_fn
    # reward_fn_key: key in non_tensor_batch holding data_source
    # reward_kwargs: forwarded to CAPOConfig / capo_reward_fn
```

#### `__call__(data: DataProto, return_dict=False)`

VERL passes a `DataProto` object with:

- `batch`: tensors  
  - `prompts`, `responses`, `attention_mask`  
- `non_tensor_batch`: Python objects  
  - `ground_truth`  
  - `data_source`  
  - `extra_info` (should contain `is_correct`)

`CAPORewardManager` does:

1. **Short-circuit** if `token_level_rewards` already exists in `data.batch` (to avoid double work).

2. For each item `i`:

   - Extract `prompt_ids`, `response_ids`, `response_mask` from the batch tensors.
   - Decode them to strings:
     ```python
     prompt_str   = tokenizer.decode(prompt_ids,   skip_special_tokens=True)
     solution_str = tokenizer.decode(response_ids, skip_special_tokens=True)
     ```

   - Read metadata from `non_tensor_batch`:
     - `ground_truth`,
     - `data_source`,
     - `extra_info["is_correct"]`.

   - Call CAPO:

     ```python
     result = compute_score(
         data_source=data_source,
         solution_str=solution_str,
         ground_truth=ground_truth,
         extra_info=extra_info,
         config=capo_config,
         genprm_client=genprm_client,
     )
     score = result["score"]
     steps = result["steps"]
     wrong_step_indices = result["wrong_step_indices"]
     ```

3. **Step → token alignment**:

   `_build_wrong_step_token_mask(tokenizer, response_ids, steps, wrong_step_indices)`:

   - Encode each `step` text into token IDs,  
   - Concatenate and align them to `response_ids`,  
   - Produce a boolean mask `wrong_step_token_mask` for response tokens belonging to **wrong** steps.

4. **Construct token‑level reward vector**:

   For each trajectory \(i\):

   - Let `score` be CAPO’s scalar reward,
   - Let `P` be the process penalty,
   - Initialize:

     ```python
     reward_vec = torch.full((response_length,), score, dtype=torch.float32)
     ```

   - For tokens where `wrong_step_token_mask[t]` is `True`, subtract `P`:

     ```python
     reward_vec[t] -= P
     ```

   - Assign:

     ```python
     reward_tensor[i, :response_length] = reward_vec
     ```

   - Set rewards for padding tokens to `0`.

5. **Return**:

   - If `return_dict=False` (default), return `reward_tensor`.  
   - If `return_dict=True`, return `{ "reward_tensor": reward_tensor, "reward_extra_info": ... }`.

VERL then writes this tensor into `data.batch["token_level_rewards"]`.

---

## 3. CAPO advantage estimators (`adv_estimators.py`)

File: `capo/verl_integration/adv_estimators.py`

Advantage estimators are registered via:

```python
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("capo")
def compute_capo_advantage(...): ...

@register_adv_est("capo_eb_lite")
def compute_capo_eb_lite_advantage(...): ...

@register_adv_est("capo_eb")
def compute_capo_eb_full_advantage(...): ...
```

In your config, you choose one via:

```yaml
algorithm:
  adv_estimator: capo         # or capo_eb_lite, capo_eb
```

The common signature is:

```python
def adv_estimator(
    token_level_rewards: torch.Tensor,  # (B, T)
    response_mask: torch.Tensor,       # (B, T)
    index,                             # group ids (not always used)
    config: Any = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns (advantages, returns)
```

### 3.1 Helper: $(g_i, L_i)$ from token‑level rewards

All EB variants use the scalarized per‑trajectory returns:

```python
def _lengths_and_scalar_returns(token_level_rewards, response_mask):
    valid = response_mask > 0
    lengths = valid.sum(dim=-1).clamp_min(1).float()              # L_i
    returns_scalar = (token_level_rewards * valid).sum(dim=-1) / lengths  # g_i
    return lengths, returns_scalar, valid
```

For each trajectory $i$:

- $L_i$ = number of valid response tokens,  
- $g_i$ = mean token reward per trajectory:
  $$
  g_i = \frac{1}{L_i} \sum_{t=1}^{L_i} r_{i,t}.
  $$

These $(g_i, L_i)$ pairs are the inputs to EB–CAPO.

---

### 3.2 Plain CAPO (`capo`)

This is the simplest CAPO estimator: **z‑normalize rewards token‑wise**, GRPO‑style.

Algorithm:

1. Compute `valid = (response_mask > 0)`.
2. Extract `valid_rewards = token_level_rewards[valid]`.
3. Compute:
   - mean $\mu_r = \text{mean}(r_{i,t})$ over valid tokens,
   - std $\sigma_r = \text{std}(r_{i,t})$.
4. Define advantages:
   $$
   A_{i,t} = \frac{r_{i,t} - \mu_r}{\sigma_r + \varepsilon}.
   $$
5. Zero out invalid positions and use `returns = token_level_rewards * response_mask`.

This is analogous to GRPO‑style advantage normalization, but using **CAPO shaped rewards**.

---

### 3.3 EB–CAPO‑lite (`capo_eb_lite`)

EB–CAPO‑lite uses a **length‑only** variance model.

#### 3.3.1 Variance model

For each trajectory $i$:

- $L_i$: length,
- $g_i$: scalarized return (mean CAPO reward),

assume:

```math
\mathrm{Var}(g_i \mid L_i) \propto L_i^{\beta}.
```

Define **precisions**:

```math
\omega_i(\beta) = L_i^{-\beta}.
```

MVU‑style weights are then $w_i(\beta) \propto \omega_i(\beta)$.

#### 3.3.2 EB‑lite log–log regression

`eb_lite_fit_beta_and_weights(g, L)` implements:

1. Initialize:

   ```math
   m^{(0)} = \frac{1}{G} \sum_{i=1}^G g_i.
   ```

2. Repeat:

   - Residuals $e_i^{(k)} = g_i - m^{(k)}$,
   - Define:

     ```math
     z_i^{(k)} = \log\!\big((e_i^{(k)})^2 + \varepsilon\big).
     ```

   - Regress:

     ```math
     z_i^{(k)} = c - \beta \log L_i + \epsilon_i
     ```

     by OLS. If the regression slope is $b$, set:

     ```math
     \hat{\beta}^{(k+1)} = -b.
     ```

   - Set weights:

     ```math
     w_i^{(k+1)} \propto L_i^{-\hat{\beta}^{(k+1)}},
     \quad \sum_i w_i^{(k+1)} = 1.
     ```

   - Update:

     ```math
     m^{(k+1)} = \sum_i w_i^{(k+1)} g_i.
     ```

   - Stop when $|m^{(k+1)} - m^{(k)}|$ is small or `max_iters` is reached.

3. Return $\hat{\beta}$, final `w`, and `m`.

#### 3.3.3 Advantages from EB‑lite

`compute_capo_eb_lite_advantage`:

1. Calls `_lengths_and_scalar_returns` → `(lengths, returns_scalar, valid)`.
2. Calls `eb_lite_fit_beta_and_weights(returns_scalar, lengths)`.
3. Computes per‑trajectory scalar advantages:

   ```math
   A_i = w_i(\hat{\beta}) \big(g_i - m(\hat{\beta})\big).
   ```

4. Broadcasts `A_i` across tokens in trajectory `i` (with `response_mask`).
5. Optionally normalizes advantages by std across valid tokens.

---

### 3.4 Full EB–CAPO (`capo_eb`)

Full EB–CAPO adds a **dependence factor** $s(L; \xi)$:

#### 3.4.1 Variance and precision

Variance model:

```math
\mathrm{Var}(g_i) = \sigma^2 L_i^{\beta} s(L_i; \xi),
```

with:

- length exponent $\beta$,
- dependence parameters $\xi = (\rho, k, \eta)$.

Precision (up to scale):

```math
\omega_i(\beta, \xi) = L_i^{-\beta} s(L_i; \xi)^{-1}.
```

#### 3.4.2 k‑banded dependence shape

`s_kband(L, rho, k, eta)` implements:

```math
s(L; \rho, k, \eta) =
  1 + \frac{2}{L}
      \sum_{h=1}^{m}
        (L - h)\,\rho^{h^{\eta}},
\qquad
m = \min\{k, L - 1\}.
```

#### 3.4.3 EB statistics

`eb_stats(g, L, beta, rho, k, eta)` computes:

- Precisions:

  ```math
  \omega_i(\beta,\xi) = \frac{L_i^{-\beta}}{s(L_i; \xi)}.
  ```

- Precision sum:

  ```math
  \Lambda_\omega(\beta,\xi) = \sum_i \omega_i(\beta,\xi).
  ```

- Normalized weights:

  ```math
  w_i(\beta,\xi) = \frac{\omega_i(\beta,\xi)}{\Lambda_\omega(\beta,\xi)}.
  ```

- Aggregated mean:

  ```math
  m(\beta,\xi) = \sum_i w_i(\beta,\xi)\,g_i.
  ```

- Residuals:

  ```math
  e_i(\beta,\xi) = g_i - m(\beta,\xi).
  ```

- Weighted RSS:

  ```math
  \mathrm{RSS}_\omega(\beta,\xi) =
    \sum_i \omega_i(\beta,\xi) \, e_i(\beta,\xi)^2.
  ```

#### 3.4.4 EB objective and gradients

The EB objective $\ell(\beta,\xi)$ (up to constants) can be written as:

```math
\ell(\beta,\xi)
 = \log \pi_\beta(\beta)
   + \log \pi_\xi(\xi)
   + \frac{1}{2}\sum_i \log \omega_i(\beta,\xi)
   - \frac{1}{2}\log \Lambda_\omega(\beta,\xi)
   - \frac{G-1}{2}\log \mathrm{RSS}_\omega(\beta,\xi)
   + \text{const}.
```

`grad_ell_beta(...)` implements the length‑direction derivative
$\partial \ell / \partial \beta$.
`grad_ell_rho_eta(...)` implements $\partial \ell / \partial \rho$ and
$\partial \ell / \partial \eta$, using:

- derivatives of $s(L; \xi)$ w.r.t. $\rho$ and $\eta$,
- the general EB gradient expression.

#### 3.4.5 ACF‑moment estimator

`acf_moment_fit(Y, mask, k)` is a moment‑based estimator for $(\rho,\eta)$:

1. For each trajectory, compute empirical autocovariances
   $\hat{\gamma}_i(h)$ up to lag $k$.
2. Aggregate to $\bar{\gamma}(h)$ and normalize to
   $\bar{r}(h) = \bar{\gamma}(h)/\bar{\gamma}(0)$.
3. Fit $\bar{r}(h) \approx \rho^{h^{\eta}}$ by grid search over $\rho$ and $\eta$.

The resulting $(\hat{\rho},\hat{\eta})$ is used as an initial guess for EB gradient ascent.

#### 3.4.6 Full EB–CAPO advantage flow

`compute_capo_eb_full_advantage`:

1. Compute `lengths`, `returns_scalar`, `valid` via `_lengths_and_scalar_returns(...)`.

2. Initialize:

   - $\beta \leftarrow \beta_{\text{init}}$,
   - $\rho \leftarrow \rho_{\text{init}}$,
   - $\eta \leftarrow \eta_{\text{init}}$,
   - `k_band`.

3. Optional ACF initialization:

   - If `increments` and `increments_mask` are passed and `use_acf_moment=True`,
   - Run `acf_moment_fit` to get $(\rho_{\text{acf}},\eta_{\text{acf}})$,
   - Blend into current $(\rho,\eta)$.

4. Gradient ascent on $\beta$:

   - For `beta_steps` iterations:
     - Call `eb_stats` → `omega, w, m, e, Lambda_omega, RSS_omega`.
     - Compute `g_beta = grad_ell_beta(...)`.
     - Update `beta ← beta + beta_lr * g_beta`, clamp to `[0, 2]`.

5. Gradient ascent on $\xi = (\rho,\eta)$:

   - For `xi_steps` iterations:
     - Compute `grad_rho, grad_eta = grad_ell_rho_eta(...)`.
     - Update:
       - `rho ← rho + rho_lr * grad_rho`,
       - `eta ← eta + eta_lr * grad_eta`,
     - Clamp `rho` to `[0, rho_max]`, `eta` to `[0, eta_max]`.

6. Final stats and advantages:

   - Recompute `omega, w, m, e` with final $(\beta,\rho,\eta)$.
   - Define per‑trajectory advantages:

     ```math
     A_i = w_i(\hat{\beta},\hat{\xi})
           \big(g_i - m(\hat{\beta},\hat{\xi})\big).
     ```

   - Broadcast `A_i` to token level using `response_mask`.
   - Optionally normalize by std over valid tokens.

---

## 4. Running a Training Experiment

### 4.1 Environment setup

From the repo root:

```bash
./pin.sh
./create_env.sh
source .venv/bin/activate

# Ensure VERL is installed if not already pinned
pip install "git+https://github.com/volcengine/verl.git"
```

Check imports:

```bash
python -c "import verl, capo; print('OK')"
```

### 4.2 Ensure CAPO is registered

Before constructing VERL’s trainer, import the integration module once so registration happens:

```python
import capo.verl_integration  # registers CAPORewardManager and adv_estimators
```

You can do this in your main script or in a package `__init__`.

### 4.3 Example experiment config (YAML sketch)

Save something like this as
`configs/experiments/math/qwen2_7b_capo_eb.yaml`:

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 1
  device: "cuda"
  project_name: "capo-math"
  experiment_name: "qwen2-7b-capo-eb"

data:
  train_files:
    - /path/to/train.parquet
  val_files:
    - /path/to/val.parquet
  # RLHFDataset options...
  # Each row should at least have "prompt" and whatever
  # fields you use in extra_info (e.g. ground_truth, is_correct).

actor_rollout_ref:
  model:
    name: Qwen/Qwen2.5-7B-Instruct  # or your model
  actor:
    strategy: fsdp
  rollout:
    strategy: fsdp
  ref:
    strategy: fsdp
  hybrid_engine: true

critic:
  enable: false              # GRPO-style without critic
  strategy: fsdp

reward_model:
  enable: false              # using function-based CAPO reward
  reward_manager: capo       # <-- CAPORewardManager
  reward_kwargs:
    correct_reward: 2.0
    process_penalty: 1.0
    num_critiques: 4
    vote_mode: intersection
    genprm_model_name: "qwen2.5-72b-instruct"

algorithm:
  name: grpo                 # or "ppo"
  adv_estimator: capo_eb     # one of: capo, capo_eb_lite, capo_eb
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: true

  adv_kwargs:
    beta_init: 1.0
    rho_init: 0.0
    eta_init: 1.0
    k_band: 64
    beta_steps: 3
    xi_steps: 3
    beta_lr: 0.1
    rho_lr: 0.1
    eta_lr: 0.1
    rho_max: 0.99
    eta_max: 3.0
    use_acf_moment: true      # if you pass increments & mask

logging:
  # configure wandb / tensorboard / CSV as needed
```

Notes:

- `reward_model.reward_manager: capo` selects `CAPORewardManager`.
- `algorithm.adv_estimator` selects which CAPO‑side algorithm you use:
  - `capo` – plain CAPO,
  - `capo_eb_lite` – EB–CAPO‑lite,
  - `capo_eb` – full EB–CAPO.

### 4.4 Example Python entrypoint

A minimal script to launch training:

```python
# run_capo_experiment.py
from omegaconf import OmegaConf

import capo.verl_integration  # ensure CAPO is registered

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.main_ppo import create_tokenizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.reward import RewardManager as VerlRewardManager


def main(config_path: str):
    config = OmegaConf.load(config_path)

    # 1. Tokenizer
    tokenizer = create_tokenizer(config.actor_rollout_ref.model)

    # 2. Worker roles and resource pool
    role_worker_mapping = {
        Role.ActorRolloutRef: ActorRolloutRefWorker,
        Role.Critic: CriticWorker,
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRolloutRef: global_pool_id,
        Role.Critic: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    # 3. RewardManager – CAPORewardManager is selected internally by VERL
    reward_fn = VerlRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = VerlRewardManager(tokenizer=tokenizer, num_examine=1)

    # 4. Trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/math/qwen2_7b_capo_eb.yaml"
    main(cfg_path)
```

Run:

```bash
source .venv/bin/activate
python run_capo_experiment.py configs/experiments/math/qwen2_7b_capo_eb.yaml
```

---

## 5. Sanity Checks and Debugging

Before launching a big training run:

1. **Unit tests**

   - Run all tests in `tests/` to verify:
     - EB‑lite weights match the 3‑trajectory toy example,
     - `s_kband`, `eb_stats`, `acf_moment_fit` behave as expected,
     - Advantage estimators return finite values.

2. **Dry run with a tiny config**

   - Set `trainer.max_steps = 1` and use a tiny dataset.
   - Check:
     - CAPORewardManager prints reasonable scores (if you set `num_examine > 0`),
     - `advantages` are non‑zero and finite,
     - no NaNs in losses.

3. **Monitor EB parameters**

   - Log `beta`, `rho`, `eta`, effective sample size (ESS), and EB weights over training.
   - Expect:
     - $\hat{\beta} < 1$ when longer rollouts are less noisy,
     - $\hat{\beta} > 1$ when very long rollouts are degraded.

4. **Compare variants**

   - Run the same experiment with:
     - `adv_estimator: capo`,
     - `adv_estimator: capo_eb_lite`,
     - `adv_estimator: capo_eb`.
   - Compare:
     - Pass@k / Avg@k curves,
     - training stability,
     - sample efficiency (tokens‑to‑target).

This gives you a complete end‑to‑end pipeline for using CAPO and EB–CAPO inside VERL on real RLHF / math‑style tasks.
