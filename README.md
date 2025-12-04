# CAPO: Outcome+Process RL with Empirical Bayes Weighting for VERL

CAPO is a method for **outcome + process** reinforcement learning from human feedback (RLHF), designed for reasoning tasks (math, coding, etc.). It has two key pieces:

1. A **CAPO reward** that combines:
   - final answer correctness (**C** term), and  
   - process/step correctness (**P** term), as judged by a process critic (GenPRM).

2. An **Empirical Bayes (EB)** module that estimates:
   - a **length exponent** $\beta$ (how variance scales with trajectory length), and  
   - a **dependence shape** $\xi = (\rho, \eta)$ (how token-level noise is correlated),  

   and uses them to produce **precision-optimal trajectory weights** for training.

This repository integrates CAPO into **VERL** (VolcEngine RL), in particular its PPO/GRPO trainer:

- CAPO plugs into VERL's **reward path** via a custom `RewardManager`.  
- EB–CAPO plugs into VERL's **advantage path** via custom advantage estimators.

The goal is to make CAPO easy to:

- reproducibly install and run on top of VERL,  
- inspect and modify algorithmically, and  
- test (both numerically and with synthetic experiments).

---

## Repository Layout

The main components relevant to CAPO and EB–CAPO are:

```text
capo/
  __init__.py
  eb_core.py                  # Core EB algorithms (β, ρ, η, k-banded weights, ACF moment)
  verl_integration/
    __init__.py               # Registers CAPO modules with VERL
    reward_fn.py              # CAPO reward (Outcome + Process) / GenPRM interface
    reward_manager.py         # CAPORewardManager -> token-level rewards for VERL
    adv_estimators.py         # "capo", "capo_eb_lite", "capo_eb" advantage estimators

docs/
  ...                         # MkDocs site; see docs/ for details

tests/
  test_eb_toy.py              # 3-trajectory toy example (β=1, β=0.5)
  test_eb_lite.py             # EB-lite recovers β on synthetic data
  test_acf_moment.py          # ACF-moment recovers ρ on AR(1) increments
  ...                         # (other tests if present)
```

The **VERL integration** is entirely contained under `capo/verl_integration/`. The **Empirical Bayes math** lives in `capo/eb_core.py`.

---

## Installation

### Requirements

- Python **3.10+** (3.11 recommended).
- A working C++ toolchain (if you compile PyTorch from source).
- Optionally: `uv` or `pip` for dependency management.
- VERL (installed from GitHub or via your internal distribution).

### Recommended: pinned environment via `pin.sh` / `create_env.sh`

If your repo includes `requirements.in`, `pin.sh`, and `create_env.sh`:

1. **Pin dependencies**

   ```bash
   ./pin.sh
   ```

   This produces a `pinned-requirements.txt` for your current platform.

2. **Create a virtual environment and install**

   ```bash
   ./create_env.sh .venv
   source .venv/bin/activate
   ```

   This will:

   - create (or reuse) `.venv/`,
   - install all pinned dependencies,
   - install the `capo` package in editable mode (`-e .`).

3. **Install VERL**

   If VERL is not already in pinned requirements, you can install from GitHub:

   ```bash
   pip install "git+https://github.com/volcengine/verl.git"
   ```

4. **Sanity check**

   ```bash
   python -c "import verl, capo; print('OK')"
   ```

---

## Quickstart with VERL

### 1. Import CAPO integration (registration)

Before constructing the VERL trainer, ensure the CAPO integration is imported:

```python
import capo.verl_integration
```

This has **side effects**:

- registers `CAPORewardManager` as `reward_model.reward_manager: "capo"`,  
- registers the advantage estimators `"capo"`, `"capo_eb_lite"`, `"capo_eb"`.

If you forget this import, VERL will not see CAPO's custom components.

### 2. Configure CAPO as VERL's reward manager

In your Hydra config (or equivalent), set:

```yaml
reward_model:
  enable: false            # we use a function-based reward, not a learned RM
  reward_manager: capo     # use CAPORewardManager
  reward_kwargs:
    correct_reward: 2.0    # C
    process_penalty: 1.0   # P
    num_critiques: 4
    vote_mode: intersection
    genprm_model_name: "qwen2.5-72b-instruct"
```

You must ensure your data loader populates (in `non_tensor_batch`):

- `ground_truth` (solution or labels),  
- `extra_info["is_correct"]` (boolean final correctness),  
- `data_source` (optional dataset key if you multiplex datasets).

### 3. Choose a CAPO advantage estimator

In your `algorithm` block:

```yaml
algorithm:
  name: grpo                    # or ppo; we focus on GRPO-style
  adv_estimator: capo_eb        # "capo", "capo_eb_lite", or "capo_eb"
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: true

  # EB hyperparameters for "capo_eb":
  adv_kwargs:
    beta_init: 1.0
    rho_init: 0.0
    eta_init: 1.0
    k_band: 64
    beta_steps: 1
    xi_steps: 1
    beta_lr: 0.1
    rho_lr: 0.1
    eta_lr: 0.1
    beta_min: 0.0
    beta_max: 2.0
    rho_max: 0.99
    ema_beta: 1.0
    ema_xi: 1.0
    use_acf_moment: true
```

You may tighten or relax the ranges and learning rates depending on the task.

### 4. Launch a training run

Example minimal Python entrypoint:

```python
from omegaconf import OmegaConf

import capo.verl_integration  # <-- important: register CAPO with VERL

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.main_ppo import create_tokenizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.reward import RewardManager as VerlRewardManager


def main(cfg_path: str):
    config = OmegaConf.load(cfg_path)
    tokenizer = create_tokenizer(config.actor_rollout_ref.model)

    # Allocate all roles to a single pool (simple single-node setup).
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

    reward_fn = VerlRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = VerlRewardManager(tokenizer=tokenizer, num_examine=1)

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
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/math/qwen2_7b_capo_eb.yaml"
    main(cfg)
```

Then:

```bash
python run_capo_experiment.py configs/experiments/math/qwen2_7b_capo_eb.yaml
```

---

## CAPO Reward: Outcome + Process

The CAPO reward function is implemented in:

- `capo/verl_integration/reward_fn.py`
- `capo/verl_integration/reward_manager.py`

### Conceptual behavior

Given a trajectory (question, solution, ground truth):

1. **Segment** the solution string into steps (usually per line).  
2. Use a **GenPRM process critic** (LLM or external model) to judge each step as correct/incorrect, possibly aggregating multiple critiques via `vote_mode ∈ {intersection, majority}`.  
3. Use the dataset’s answer checker to obtain `is_correct` (final answer correctness).  
4. Apply the CAPO table:

Let $C$ = `correct_reward`, $P$ = `process_penalty`.

- Final answer correct, all steps correct:  
  $\text{score} = +C$  

- Final answer correct, some steps wrong:  
  $\text{score} = C - P$  

- Final answer incorrect, all steps correct:  
  $\text{score} = 0$  

- Final answer incorrect, some steps wrong:  
  $\text{score} = -P$

5. Map wrong steps to tokens via the tokenizer and penalize tokens belonging to wrong steps (e.g., subtract $P$ on those tokens).

### Code structure

**`reward_fn.py`**:

- `CAPOConfig` — collects reward hyperparameters (`C`, `P`, `num_critiques`, `vote_mode`, `genprm_model_name`, etc.).
- `GenPRMClient` — abstract interface for a process critic; implement `judge_steps(...)`.
- `_segment_solution_into_steps` — splits solution string into a list of steps.
- `_aggregate_step_judgements` — combines multiple critiques per step.
- `capo_reward_fn` — implements the CAPO decision table and returns:

  ```python
  {
      "score": float,
      "steps": list[str],
      "step_correctness": list[bool],
      "wrong_step_indices": list[int],
  }
  ```

**`reward_manager.py`**:

- `CAPORewardManager` (registered as `"capo"`):

  - Takes a `DataProto` from VERL’s rollout worker.
  - Decodes prompts and responses to text using the tokenizer.
  - Calls `capo_reward_fn` per trajectory.
  - Aligns wrong steps to tokens and builds a `(B, T)` `reward_tensor`:

    - base value: `score` for each response token,  
    - minus `P` for tokens in wrong steps,  
    - 0 for prompt tokens and padding.

  - Returns `reward_tensor`, which VERL puts into `data.batch["token_level_rewards"]`.

---

## Empirical Bayes CAPO (EB–CAPO)

The EB layer lives in `capo/eb_core.py` and corresponds to your Algorithms section:

- **Section 4.1** — EB theory and objective $\ell(\beta, \xi)$  
- **Algorithm EB–CAPO** — integrated into the CAPO training loop  
- **Algorithm EB-lite** — length-only fit  
- **Algorithm ACF-moment** — moment-based dependence estimator  
- **Algorithm k-banded weights** — $k$-banded stretched-geometric dependence  
- **Algorithm joint EB update** — joint updates for $(\beta, \rho, \eta)$  

### 1. EB-lite (length-only)

Function:

```python
from capo.eb_core import eb_lite_fit_beta_and_weights
```

Implements Algorithm `EB-lite`:

- Model:  
  $\mathrm{Var}(g_i \mid L_i) \approx \sigma^2 L_i^{\beta}$,
  ignoring dependence ($s(L; \xi) \equiv 1$).
- Approximation:  
  $\log((g_i - m)^2) \approx c - \beta \log L_i$.
- Iterative procedure:

  1. Initialize $m^{(0)}$ as the mean of $\{g_i\}$.
  2. At each iteration:
     - $e_i = g_i - m^{(k)}$,
     - $z_i = \log(e_i^2 + \varepsilon)$,
     - regress $z_i$ on $\log L_i$ to get slope $b$, set $\beta^{(k+1)} = -b$,
     - set $w_i \propto L_i^{-\beta^{(k+1)}}$ and normalize,
     - $m^{(k+1)} = \sum_i w_i g_i$.
  3. Stop when $m$ stabilizes.

Returns:

- `beta_hat` — $\hat{\beta}$,  
- `w` — $w_i \propto L_i^{-\hat{\beta}}$,  
- `m` — $m(\hat{\beta})$.

Used by the **EB–CAPO-lite** advantage estimator.

### 2. ACF-moment estimator

Function:

```python
from capo.eb_core import acf_moment_estimate
```

Implements Algorithm `ACF-moment`:

- Input: token-level increments $Y_{i,\tau}$ and a boolean mask for valid tokens.  
- For each trajectory $i$:

  - center $Y_{i,\tau}$ by its mean $\bar{Y}_i$,  
  - compute empirical autocovariances $\hat{\gamma}_i(h)$ up to lag $k$.

- Pool across trajectories using weights $(L_i - h)$ to get $\bar{\gamma}(h)$.  
- Normalize to autocorrelations $\bar{r}(h) = \bar{\gamma}(h) / \bar{\gamma}(0)$.  
- Fit $(\rho, \eta)$ by minimizing:

  $$
  \sum_{h=1}^k \bigl(\bar{r}(h) - \rho^{h^{\eta}}\bigr)^2
  $$

  over a small grid of $\rho$ and $\eta$.

Used as a **warm-start** for $(\rho, \eta)$ in joint EB updates.

### 3. k-banded stretched-geometric weights

Functions:

```python
from capo.eb_core import s_kband, kband_weights
```

Implement Algorithm `k-banded covariance weights`:

- Shape function:

  $$
  s(L; \rho, k, \eta) = 1 + \frac{2}{L}
  \sum_{h=1}^{m(L)} (L - h)\, \rho^{h^{\eta}}, \quad
  m(L) = \min\{k, L - 1\}.
  $$

- Dependence-corrected weights:

  $$
  \omega_i(\beta, \xi) = [L_i^{\beta} s(L_i; \xi)]^{-1}, \quad
  w_i(\beta, \xi) = \frac{\omega_i}{\sum_j \omega_j}.
  $$

### 4. Joint EB update for $(\beta, \rho, \eta)$

Functions:

```python
from capo.eb_core import (
    eb_statistics,
    grad_ell_beta_closed_form,
    numeric_grad_rho_eta,
    joint_eb_update_kband,
)
```

- `eb_statistics` implements the E-step summary:

  - $\omega_i(\beta, \xi)$,  
  - $w_i(\beta, \xi)$,  
  - $m(\beta, \xi)$,  
  - $e_i = g_i - m(\beta, \xi)$,  
  - $\Lambda_{\omega} = \sum_i \omega_i$,  
  - $\mathrm{RSS}_{\omega} = \sum_i \omega_i e_i^2$.

- `grad_ell_beta_closed_form` implements the closed-form gradient $\partial \ell / \partial \beta$ from Corollary in the paper:

  $$
  g_{\beta} = \frac{1}{2}
  \left[
    -\sum_i \log L_i +
    \frac{\sum_i \omega_i \log L_i}{\Lambda_{\omega}}
  \right]
  +
  \frac{G - 1}{2} \frac{1}{\mathrm{RSS}_{\omega}}
  \sum_i \omega_i e_i^2 \log L_i
  + \partial_{\beta}\log \pi_{\beta}(\beta).
  $$

- `numeric_grad_rho_eta` approximates $\partial \ell / \partial \rho$ and $\partial \ell / \partial \eta$ using **central finite differences** (or forward difference near $\eta=0$).

- `joint_eb_update_kband` implements Algorithm `joint EB update`:

  - optional warm-start for $(\rho, \eta)$ via `acf_moment_estimate`,
  - gradient ascent on $\beta$ using `grad_ell_beta_closed_form`,
  - gradient ascent on $(\rho, \eta)$ using `numeric_grad_rho_eta`,
  - projection to constraints:
    - $\beta \in [\beta_{\min}, \beta_{\max}]$,
    - $|\rho| \le \rho_{\max} < 1$,
    - $\eta \ge 0$,
  - EMA smoothing for stability,
  - final call to `kband_weights` to get $w_i^{(t)}$ at $(\beta_t, \rho_t, \eta_t)$.

This is the **full EB–CAPO update** that the `"capo_eb"` advantage estimator uses.

---

## Advantage Estimators

Defined in `capo/verl_integration/adv_estimators.py` and registered with VERL under the names:

- `"capo"`
- `"capo_eb_lite"`
- `"capo_eb"`

### 1. `"capo"` — plain CAPO advantage

Formula:

- Let $r_{i,t}$ be CAPO token-level rewards over valid tokens.  
- Compute global mean and std over valid tokens across the batch:

  $$
  \mu_r = \mathrm{mean}(r_{i,t}), \quad
  \sigma_r = \mathrm{std}(r_{i,t}).
  $$

- Define advantages:

  $$
  A_{i,t} = \frac{r_{i,t} - \mu_r}{\sigma_r}.
  $$

This is *GRPO-style* z-normalization, used here on CAPO rewards instead of vanilla rewards.

### 2. `"capo_eb_lite"` — EB–CAPO-lite

Steps:

1. Collapse token-level rewards to `(g_i, L_i)` via `_lengths_and_scalar_returns`.  
2. Run `eb_lite_fit_beta_and_weights(g, L)` → `(beta_hat, w, m)`.  
3. Define per-trajectory scalar advantages:

   $$
   A_i = w_i(\hat{\beta}) \bigl(g_i - m(\hat{\beta})\bigr).
   $$

4. Broadcast to tokens:

   $$
   A_{i,t} = A_i \quad \text{for all valid tokens in trajectory } i.
   $$

5. Optionally apply final std-normalization over valid `A_{i,t}` if `norm_adv_by_std_in_grpo=True`.

### 3. `"capo_eb"` — full EB–CAPO

Steps:

1. Collapse token-level rewards to `(g_i, L_i)`.

2. Call `joint_eb_update_kband`:

   - optionally warm-start $(\rho, \eta)$ using `acf_moment_estimate` on token-level increments,  
   - run $N_{\beta}$ steps of gradient-ascent on $\beta$,  
   - run $N_{\xi}$ steps of gradient-ascent on $(\rho, \eta)$,  
   - project and EMA-smooth,
   - return $(\beta_t, \rho_t, \eta_t)` and dependence-corrected weights $w_i$.

3. Compute:

   $$
   m_t = \sum_i w_i g_i,
   \qquad
   A_i = w_i (g_i - m_t),
   $$

   and broadcast $A_i$ over tokens with `response_mask`.

4. Optionally apply final std-normalization (GRPO-style).

This matches the **EB–CAPO** workflow in your latest algorithms section.

---

## Tests

The test suite is designed to be:

- **unit-level** (for EB math), and  
- **algorithmically aligned** with the LaTeX text.

Key tests:

- `tests/test_eb_toy.py`  
  - Checks the **3-trajectory toy example**:
    - $L = [16, 64, 256]$,
    - $g = [0.8, 1.0, 1.1]$,
    - for $\beta = 1$ and $\beta = 0.5$, verifies that:
      - $w_i(\beta)$ sum to 1,
      - $m(\beta)$ matches the approximate values in the text (`≈ 0.846` and `≈ 0.892`).

- `tests/test_eb_lite.py`  
  - Generates synthetic trajectories with a known length exponent $\beta_{\star}$,  
  - Uses `eb_lite_fit_beta_and_weights` to estimate $\hat{\beta}$,  
  - Asserts that $\hat{\beta}$ is within a reasonable band of $\beta_{\star}$ and that weights sum to 1.

- `tests/test_acf_moment.py`  
  - Generates AR(1) increments with known $\rho_{\star}$,  
  - Calls `acf_moment_estimate`,  
  - Checks that $\hat{\rho}$ is close to $\rho_{\star}$ and $\hat{\eta} ≥ 0$.

Run tests via:

```bash
pytest
```

---

## Development Notes

- All EB code is written with **explicit notation** mirroring the paper:
  - $\omega_i$, $w_i$, $m$, $e_i$, $\Lambda_{\omega}$, $\mathrm{RSS}_{\omega}$,
  - $s(L; \rho, k, \eta)$,
  - $(\beta, \rho, \eta)$.
- Functions in `capo/eb_core.py` contain docstrings that explicitly reference the corresponding algorithms in the LaTeX:

  - EB-lite → Algorithm `EB-lite`.  
  - ACF-moment → Algorithm `ACF-moment`.  
  - k-banded weights → Algorithm `kband-weights`.  
  - Joint EB update → Algorithm `joint-eb-kband`.

- Advantage estimators are **thin, VERL-specific wrappers** on top of these EB utilities.  

If you modify notation or the algorithms in your LaTeX, the mapping from equation → function is intended to be explicit and local, so you can keep code and text synchronized.

---

## How to Extend

A few natural extensions:

- **Alternative priors** on $(\beta, \rho, \eta)$:  
  Currently, prior gradients `dlog_pi_*` are scalar inputs. You can replace them with learned or adaptive priors if you want.

- **Different dependence families**:  
  You can implement alternative $s(L; \xi)$ in `eb_core.py` and swap them into `eb_statistics` / `joint_eb_update_kband`.

- **Different reward shaping**:  
  The CAPO reward manager is written to be modular:
  - you can change step segmentation,
  - adjust how per-step penalties are distributed over tokens,
  - or plug in richer GenPRM annotations.

- **Direct loss integration**:  
  If you want, you can expose EB weights `w_i` to other parts of the training loop (e.g., teacher/student losses, replay buffers) by logging or storing them in `DataProto`.

If you want to wire any of these into VERL in a custom way, `capo/verl_integration` is the right place to begin.