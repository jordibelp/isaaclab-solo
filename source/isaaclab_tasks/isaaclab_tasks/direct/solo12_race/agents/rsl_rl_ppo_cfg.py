# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

from isaaclab_tasks.direct.solo12_race.solo12_race_env_cfg import Solo12RaceEnvCfg, Solo12RaceParamsConditionedEnvCfg

from .env_params_conditioned_encoder_actor import EnvParamsConditionedEncoderActor
from .env_params_conditioned_actor import EnvParamsConditionedActor
from .imu_tcn_actor_critic import ActorCriticFootImuTcn
from .shared_actor_critic import SharedActorCritic


rsl_rl_on_policy_runner.SharedActorCritic = SharedActorCritic
rsl_rl_on_policy_runner.ActorCriticFootImuTcn = ActorCriticFootImuTcn
rsl_rl_on_policy_runner.EnvParamsConditionedActor = EnvParamsConditionedActor
rsl_rl_on_policy_runner.EnvParamsConditionedEncoderActor = EnvParamsConditionedEncoderActor

_SOLO12_RACE_ENV_CFG = Solo12RaceEnvCfg()
_SOLO12_RACE_PARAMS_ENV_CFG = Solo12RaceParamsConditionedEnvCfg()

shared_networks_default = False

@configclass
class RslRlPpoSharedActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SharedActorCritic"
    shared_networks: bool = shared_networks_default


@configclass
class RslRlPpoActorCriticFootImuTcnCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFootImuTcn"
    shared_networks: bool = shared_networks_default
    current_obs_dim: int = _SOLO12_RACE_ENV_CFG.proprio_observation_dim
    history_name: str = "foot-IMU"
    imu_history_len: int = _SOLO12_RACE_ENV_CFG.foot_imu_history_length
    imu_dim: int = _SOLO12_RACE_ENV_CFG.foot_imu_obs_dim
    tcn_channels: int = _SOLO12_RACE_ENV_CFG.foot_imu_tcn_channels
    tcn_latent_dim: int = _SOLO12_RACE_ENV_CFG.foot_imu_tcn_latent_dim
    tcn_kernel_size: int = _SOLO12_RACE_ENV_CFG.foot_imu_tcn_kernel_size
    tcn_activation: str = _SOLO12_RACE_ENV_CFG.foot_imu_tcn_activation


@configclass
class RslRlPpoActorCriticJointStateTcnCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFootImuTcn"
    shared_networks: bool = shared_networks_default
    current_obs_dim: int = _SOLO12_RACE_ENV_CFG.proprio_observation_dim
    history_name: str = "joint-state"
    history_len: int = _SOLO12_RACE_ENV_CFG.joint_state_history_length
    history_dim: int = _SOLO12_RACE_ENV_CFG.joint_state_history_obs_dim
    tcn_channels: int = _SOLO12_RACE_ENV_CFG.joint_state_tcn_channels
    tcn_latent_dim: int = _SOLO12_RACE_ENV_CFG.joint_state_tcn_latent_dim
    tcn_kernel_size: int = _SOLO12_RACE_ENV_CFG.joint_state_tcn_kernel_size
    tcn_activation: str = _SOLO12_RACE_ENV_CFG.joint_state_tcn_activation


@configclass
class RslRlPpoActorCriticJointStateImuTcnCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFootImuTcn"
    shared_networks: bool = shared_networks_default
    current_obs_dim: int = _SOLO12_RACE_ENV_CFG.proprio_observation_dim
    history_name: str = "joint-state + foot-IMU"
    history_len: int = _SOLO12_RACE_ENV_CFG.joint_imu_history_length
    history_dim: int = _SOLO12_RACE_ENV_CFG.joint_imu_history_obs_dim
    tcn_channels: int = _SOLO12_RACE_ENV_CFG.joint_imu_tcn_channels
    tcn_latent_dim: int = _SOLO12_RACE_ENV_CFG.joint_imu_tcn_latent_dim
    tcn_kernel_size: int = _SOLO12_RACE_ENV_CFG.joint_imu_tcn_kernel_size
    tcn_activation: str = _SOLO12_RACE_ENV_CFG.joint_imu_tcn_activation


@configclass
class RslRlPpoEnvParamsConditionedActorCfg(RslRlPpoActorCriticCfg):
    class_name: str = "EnvParamsConditionedActor"
    shared_networks: bool = shared_networks_default
    current_obs_dim: int = _SOLO12_RACE_PARAMS_ENV_CFG.base_observation_dim
    env_params_dim: int = _SOLO12_RACE_PARAMS_ENV_CFG.gt_env_params_obs_dim


@configclass
class RslRlPpoEnvParamsConditionedEncoderActorCfg(RslRlPpoActorCriticCfg):
    class_name: str = "EnvParamsConditionedEncoderActor"
    shared_networks: bool = shared_networks_default
    current_obs_dim: int = _SOLO12_RACE_PARAMS_ENV_CFG.base_observation_dim
    env_params_dim: int = _SOLO12_RACE_PARAMS_ENV_CFG.gt_env_params_obs_dim
    env_params_encoder_hidden_dims: list[int] = [64, 32]
    env_params_latent_dim: int = 8
    env_params_encoder_activation: str = "elu"


@configclass
class Solo12RacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 100
    experiment_name = "solo12_rsl_rl_race_env_runs"
    run_name = "[ClusterIRI]-Solo12Race - reward_progress_scale=100; friction randomization"
    logger = "wandb"
    wandb_project = "borinotIsaacLab"
    policy = RslRlPpoSharedActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.9965,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Solo12RaceIMUPPORunnerCfg(Solo12RacePPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12RaceIMU - foot IMU TCN; reward_progress_scale=100; friction randomization"
    policy = RslRlPpoActorCriticFootImuTcnCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class Solo12RaceJointStateTcnPPORunnerCfg(Solo12RacePPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12Race - joint-state TCN; reward_progress_scale=100; friction randomization"
    policy = RslRlPpoActorCriticJointStateTcnCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class Solo12RaceJointStateImuTcnPPORunnerCfg(Solo12RacePPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12Race - joint-state + foot IMU TCN; reward_progress_scale=100; friction randomization"
    policy = RslRlPpoActorCriticJointStateImuTcnCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class Solo12RaceParamsConditionedPPORunnerCfg(Solo12RacePPORunnerCfg):
    run_name = "[ClusterIRI]-Solo12Race - params-conditioned actor; reward_progress_scale=100; friction randomization"
    policy = RslRlPpoEnvParamsConditionedActorCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class Solo12RaceParamsConditionedEncPPORunnerCfg(Solo12RacePPORunnerCfg):
    run_name = (
        "[ClusterIRI]-Solo12Race - params-conditioned encoder actor; "
        "reward_progress_scale=100; friction randomization"
    )
    policy = RslRlPpoEnvParamsConditionedEncoderActorCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
