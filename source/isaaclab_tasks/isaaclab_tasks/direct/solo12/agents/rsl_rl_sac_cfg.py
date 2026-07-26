# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SAC configuration for Solo12 following arXiv:2605.24975."""

import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOffPolicyRunnerCfg,
    RslRlSacActorModelCfg,
    RslRlSacAlgorithmCfg,
    RslRlSacCriticModelCfg,
)
from rsl_rl_sac.algorithms import SAC


class Solo12SAC(SAC):
    """SAC with action bounds matching Solo12's direct position-action mapping."""

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        """Construct SAC using the task's state-dependent log-std experiment toggle."""
        unwrapped = getattr(env, "unwrapped", env)
        state_dependent_std = bool(unwrapped.cfg.learnable_logstd_as_network_output)
        cfg["actor"]["state_dependent_std"] = state_dependent_std
        mode = "network output (state-dependent)" if state_dependent_std else "parameter (state-independent)"
        print(f"Solo12 SAC: log_std is learned as a {mode}.")
        return SAC.construct_algorithm(obs, env, cfg, device)

    @staticmethod
    def _compute_action_scaling(env, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        unwrapped = getattr(env, "unwrapped", env)
        limits = unwrapped._joint_soft_pos_limits[0].to(device)
        center = unwrapped._q_offset_action_and_obs.to(device)
        if center.ndim > 1:
            center = center[0]
        scale = torch.as_tensor(unwrapped.cfg.action_scale, device=device)
        if torch.any(scale <= 0):
            raise ValueError("Solo12 SAC requires a strictly positive action_scale.")
        upper = (limits[:, 1] - center) / scale
        lower = (center - limits[:, 0]) / scale
        if not torch.isfinite(upper).all() or not torch.isfinite(lower).all():
            raise ValueError("Solo12 SAC computed non-finite action bounds from the task soft joint limits.")
        if torch.any(upper <= 0) or torch.any(lower <= 0):
            raise ValueError("Solo12 SAC action center must lie strictly inside every soft joint limit.")
        print(
            "Solo12 SAC: action bounds use q_offset_action_and_obs, env.action_scale, "
            f"and a {unwrapped.cfg.joint_soft_limit_delta:g}-degree joint-limit margin."
        )
        print(f"  lower magnitudes: {lower}")
        print(f"  upper magnitudes: {upper}")
        return upper, lower


# RSL-RL-SAC 4.0.1 calls ``SAC._compute_action_scaling`` explicitly inside
# construct_algorithm rather than dispatching through the configured subclass.
# This module is imported only for the Solo12 SAC entry point, so install the
# task-specific bound computation for this process.
SAC._compute_action_scaling = staticmethod(Solo12SAC._compute_action_scaling)


@configclass
class Solo12SACRunnerCfg(RslRlOffPolicyRunnerCfg):
    """Paper-style SAC defaults adapted to the Solo12 direct task."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    log_interval = 1
    start_training = 1
    experiment_name = "solo12_rsl_rl_sac_vel_cmd_runs"
    run_name = "Solo12 SAC"
    logger = "wandb"
    wandb_project = "borinotIsaacLab"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlSacActorModelCfg(
        hidden_dims=[512, 256, 128],
        activation="swish",
        obs_normalization=True,
        layer_norm=False,
        init_noise_std=0.15,
        state_dependent_std=True,
        log_std_min=-20.0,
        log_std_max=2.0,
    )
    critic = RslRlSacCriticModelCfg(
        hidden_dims=[512, 256, 128],
        activation="swish",
        obs_normalization=True,
        layer_norm=False,
    )
    algorithm = RslRlSacAlgorithmCfg(
        class_name=(
            "isaaclab_tasks.direct.solo12.agents.rsl_rl_sac_cfg:Solo12SAC"
        ),
        replay_buffer_size=int(5.0e6),
        num_learning_epochs=1,
        num_mini_batches=200,
        mini_batch_size=8192,
        actor_learning_rate=2.0e-4,
        critic_learning_rate=2.0e-4,
        alpha_learning_rate=2.0e-5,
        gamma=0.97,
        tau=0.003,
        alpha=0.001,
        auto_alpha=True,
        target_entropy_scale=0.167,
        max_grad_norm=1.0,
        policy_frequency=1,
        n_steps=5,
    )
