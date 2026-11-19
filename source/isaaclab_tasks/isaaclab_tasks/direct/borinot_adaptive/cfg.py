# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg
from isaaclab_tasks.direct.borinot_goto.borinot_goto_env_cfg import BorinotGoToEnvCfg


@configclass
class AdaptiveConfig:
    use_asym_actor_critic = False  # adaptive tasks construct their own actor and critic streams
    policy_model = "base_imu_teacher"
    encoder_hz: int = 0  # Playback only; zero follows policy_hz.
    base_imu_history_policy_steps = 15
    base_imu_history_sample_dim = 28
    base_imu_tcn_channels = 64
    base_imu_tcn_kernel_size = 5
    base_imu_tcn_activation = "elu"
    base_imu_tcn_latent_dim = 32
    teacher_encoder_obs_dim = 66
    teacher_encoder_hidden_dims = [256, 128, 64]
    teacher_latent_dim = 32
    apply_tanh_to_latent = True
    priv_critic_includes_perception_encoding = False
    priv_critic_detached_perception_encoding = True
    feed_history_encoding_to_critic = False
    # std per axis: gyro, linear acceleration, joint position/velocity, attitude, height
    sensor_std = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def configure_control(self):
        super().configure_control()
        self.refresh_base_imu_dimensions()

    def refresh_runtime_dependent_config(self):
        self.configure_control()

    def refresh_base_imu_dimensions(self):
        import math
        if self.include_prev_action_steps > 1:
            raise ValueError("Adaptive Borinot uses its own history; set include_prev_action_steps to 0 or 1.")
        if (self.control_mode not in ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control")
                or self.action_space != 8 or self.hold_arm or self.sac_action_mapping
                or getattr(self, "remove_time_from_actor_obs", False)):
            raise ValueError("Adaptive Borinot requires PPO with six thrust actions, two scalar joint torque or PD position actions, and the complete command.")
        if self.base_imu_history_policy_steps < 1 or len(self.sensor_std) != 6 or not all(
                math.isfinite(v) and v >= 0 for v in self.sensor_std):
            raise ValueError("Invalid adaptive history or sensor noise.")
        self.teacher_latent_activation = "tanh" if self.apply_tanh_to_latent else "identity"
        self.base_imu_history_length = self.base_imu_history_policy_steps * round(self.physics_hz / self.policy_hz) + 1
        self.base_imu_history_flat_dim = self.base_imu_history_length * self.base_imu_history_sample_dim
        self.command_dim = self.base_imu_history_sample_dim + (5 if getattr(self, "goal_conditioned", False) else 4)
        self.teacher_critic_obs_dim = self.teacher_encoder_obs_dim + self.command_dim
        self.state_space = self.teacher_critic_obs_dim
        self.observation_space = {
            "base_imu_teacher": self.state_space,
            "base_imu_student_rl": self.base_imu_history_flat_dim + self.command_dim,
            "base_imu_student_dagger": self.state_space + self.base_imu_history_flat_dim + self.command_dim,
        }[self.policy_model]

    def adaptation_contract(self):
        contract = dict(family="borinot-direct-v2", command_dim=self.command_dim,
                    physics_hz=self.physics_hz, policy_hz=self.policy_hz, sensor_std=list(self.sensor_std),
                    base_imu_history_policy_steps=self.base_imu_history_policy_steps,
                    rotor_action_scale=self.rotor_action_scale, action_clamp=self.action_clamp,
                    remove_thrust_clipping=self.remove_thrust_clipping,
                    apply_tanh_to_latent=self.apply_tanh_to_latent)
        # Legacy contracts omit the mode and always mean direct torque.
        if self.control_mode != "thrust_and_torque_control":
            contract["control_mode"] = self.control_mode
        return contract


@configclass
class GoToTeacherCfg(AdaptiveConfig, BorinotGoToEnvCfg):
    control_mode = "thrust_and_torque_control"


@configclass
class VelocityTeacherCfg(AdaptiveConfig, BorinotCmdVelEnvCfg):
    control_mode = "thrust_and_torque_control"


@configclass
class GoToDaggerCfg(GoToTeacherCfg):
    policy_model = "base_imu_student_dagger"


@configclass
class GoToStudentCfg(GoToTeacherCfg):
    policy_model = "base_imu_student_rl"


@configclass
class VelocityDaggerCfg(VelocityTeacherCfg):
    policy_model = "base_imu_student_dagger"


@configclass
class VelocityStudentCfg(VelocityTeacherCfg):
    policy_model = "base_imu_student_rl"
