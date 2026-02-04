"""CAPO paper entrypoint.

This file is intentionally thin: it wires configuration (Hydra) into the
VERL RayPPO trainer.

Design constraints:
- All algorithmic variation (baseline vs CAPO variants) is controlled by config
  overrides, not by editing Python.
- Logging is reproducible: if `trainer.local_metrics_path` is set, metrics are
  appended to a JSONL file that the paper artifact builder consumes.

Key config switches
-------------------
- `algorithm.adv_estimator`: selects the advantage estimator. CAPO options are:
    - `capo`
    - `capo_eb_lite`
    - `capo_eb`

The CAPO estimators are in `capo.verl_integration.adv_estimators`.
"""

from __future__ import annotations

import hydra
import ray

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@hydra.main(config_path="config", config_name="capo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # avoid scheduling on the Ray head if possible
class TaskRunner:
    def run(self, config):
        """Build and execute a RayPPOTrainer run.

        Notes
        -----
        - This entrypoint is intentionally light on logic. Any experimental
          variation should be done via Hydra config overrides (see
          `experiments/capo_paper/recipe/capo/scripts/*`).
        - We set RNG seeds here (before any model construction) to improve
          reproducibility across Ray workers.
        """

        import os
        import random
        from pprint import pprint

        import numpy as np
        import torch
        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        seed = int(getattr(config, "seed", 1))
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # download/checkpoint path abstraction (HDFS/local supported by VERL)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer / processor
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        # choose worker classes by strategy
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError(
                f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # reward model (optional)
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Reward manager. We rely on VERL's loader, but pass optional kwargs
        # defensively (reward managers have different ctor signatures).
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))

        # These kwargs are used by some reward managers (e.g., DAPO-style
        # overlong-buffer penalties) but are ignored by others.
        reward_kwargs.setdefault("max_resp_len", config.data.max_response_length)
        overlong_cfg = config.reward_model.get("overlong_buffer", None)
        if overlong_cfg is not None:
            reward_kwargs.setdefault("overlong_buffer_cfg", overlong_cfg)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **reward_kwargs
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **reward_kwargs
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
