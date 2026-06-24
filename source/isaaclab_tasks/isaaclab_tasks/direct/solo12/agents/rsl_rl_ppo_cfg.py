# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

_SOLO12_AGENTS_DIR = Path(__file__).resolve().parent
_ISAACLAB_ROOT = _SOLO12_AGENTS_DIR.parents[5]
_BORINOT_SKRL_SCRIPTS_DIR = _ISAACLAB_ROOT / "source" / "scripts" / "skrl"
if str(_BORINOT_SKRL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BORINOT_SKRL_SCRIPTS_DIR))

import solo12_symmetry
from isaaclab_tasks.direct.solo12.solo12_env_cfg import (
    Solo12BaseImuStudentRlEnvCfg,
    Solo12BaseImuTeacherEnvCfg,
)

from .base_imu_actor_critic import Solo12BaseImuStudentActorCritic, Solo12BaseImuTeacherActorCritic


rsl_rl_on_policy_runner.Solo12BaseImuTeacherActorCritic = Solo12BaseImuTeacherActorCritic
rsl_rl_on_policy_runner.Solo12BaseImuStudentActorCritic = Solo12BaseImuStudentActorCritic

_SOLO12_BASE_IMU_TEACHER_CFG = Solo12BaseImuTeacherEnvCfg()
_SOLO12_BASE_IMU_STUDENT_CFG = Solo12BaseImuStudentRlEnvCfg()


@configclass
class RslRlPpoSolo12BaseImuTeacherCfg(RslRlPpoActorCriticCfg):
    class_name: str = "Solo12BaseImuTeacherActorCritic"
    teacher_encoder_obs_dim: int = _SOLO12_BASE_IMU_TEACHER_CFG.teacher_encoder_obs_dim
    teacher_latent_dim: int = _SOLO12_BASE_IMU_TEACHER_CFG.teacher_latent_dim
    teacher_encoder_hidden_dims: list[int] = _SOLO12_BASE_IMU_TEACHER_CFG.teacher_encoder_hidden_dims
    command_dim: int = 3


@configclass
class RslRlPpoSolo12BaseImuStudentCfg(RslRlPpoActorCriticCfg):
    class_name: str = "Solo12BaseImuStudentActorCritic"
    history_len: int = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_history_length
    history_sample_dim: int = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_history_sample_dim
    teacher_critic_obs_dim: int = _SOLO12_BASE_IMU_STUDENT_CFG.teacher_critic_obs_dim
    command_dim: int = 3
    tcn_channels: int = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_tcn_channels
    tcn_latent_dim: int = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_tcn_latent_dim
    tcn_kernel_size: int = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_tcn_kernel_size
    tcn_activation: str = _SOLO12_BASE_IMU_STUDENT_CFG.base_imu_tcn_activation
    feed_history_encoding_to_critic: bool = _SOLO12_BASE_IMU_STUDENT_CFG.feed_history_encoding_to_critic


@configclass
class Solo12PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 45573
    save_interval = 469
    experiment_name = "solo12_rsl_rl_vel_cmd_runs"
    run_name = "[ClusterIRI]-DR remove_root_lin_vel_b_from_obs=True"
    logger = "wandb"
    wandb_project = "borinotIsaacLab"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[x for x in [256, 128, 64]],
        critic_hidden_dims=[x for x in [256, 128, 64]],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Solo12PPORunnerWithSymmetryCfg(Solo12PPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=solo12_symmetry.compute_symmetric_observations_actions,
        ),
    )


@configclass
class Solo12BaseImuTeacherPPORunnerCfg(Solo12PPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12 base IMU teacher encoder"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoSolo12BaseImuTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class Solo12BaseImuStudentRlPPORunnerCfg(Solo12PPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12 base IMU student RL TCN"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoSolo12BaseImuStudentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
