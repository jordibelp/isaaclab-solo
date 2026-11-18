# SPDX-License-Identifier: BSD-3-Clause
from isaaclab_tasks.direct.borinot_cmd_vel.playback import BorinotCmdVelPlayEnv
from isaaclab_tasks.direct.borinot_goto.playback import BorinotGoToPlayEnv
from .env import GoToAdaptiveEnv, VelocityAdaptiveEnv


class GoToPlayEnv(GoToAdaptiveEnv, BorinotGoToPlayEnv):
    pass


class VelocityPlayEnv(VelocityAdaptiveEnv, BorinotCmdVelPlayEnv):
    pass


def configure_perception(checkpoint, cfg, policy):
    from solo12_imu_pipeline import apply_observation_contract, mlp_shape
    infos = checkpoint["infos"]
    apply_observation_contract(cfg, infos["solo12_imu_latent"])
    state = checkpoint["model_state_dict"]
    policy.actor_hidden_dims = mlp_shape(state, "actor")[1]
    policy.critic_hidden_dims = mlp_shape(state, "critic")[1]
    policy.priv_critic_includes_perception_encoding = infos["solo12_imu_critic"]["priv_critic_includes_perception_encoding"]
    if cfg.policy_model == "base_imu_teacher":
        _, policy.teacher_encoder_hidden_dims, policy.teacher_latent_dim = mlp_shape(state, "teacher_encoder")
        policy.teacher_latent_activation = cfg.teacher_latent_activation
    else:
        policy.history_len = cfg.base_imu_history_length
        conv = state["actor_history_encoder.conv_stack.0.conv.weight"]
        policy.tcn_channels, _, policy.tcn_kernel_size = conv.shape
        policy.tcn_latent_dim = state["actor_history_encoder.to_latent.1.weight"].shape[0]
        policy.apply_tanh_to_latent = cfg.apply_tanh_to_latent
    policy.actor_obs_normalization = any(k.startswith("actor_obs_normalizer.") for k in state)
    policy.critic_obs_normalization = any(k.startswith("critic_obs_normalizer.") for k in state)
    policy.noise_std_type = "log" if "log_std" in state else "scalar"
