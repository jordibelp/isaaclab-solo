# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from .cfg import AdaptiveConfig
from isaaclab_tasks.direct.borinot_cmd_vel.agents.rsl_rl_ppo_cfg import BorinotCmdVelPPORunnerCfg
from isaaclab_tasks.direct.borinot_goto.agents.rsl_rl_ppo_cfg import BorinotGoToPPORunnerCfg
from isaaclab_tasks.direct.solo12.agents.rsl_rl_ppo_cfg import Solo12BaseImuTeacherPPORunnerCfg, Solo12BaseImuStudentRlPPORunnerCfg


# configclass stores inferred fields on instances, not on the class.
_ADAPTIVE_DEFAULTS = AdaptiveConfig()


def perception_policy(student, bypass):
    common = dict(command_dim=bypass, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128])
    if student:
        return Solo12BaseImuStudentRlPPORunnerCfg().policy.replace(
            history_len=31, history_sample_dim=_ADAPTIVE_DEFAULTS.base_imu_history_sample_dim,
            teacher_critic_obs_dim=_ADAPTIVE_DEFAULTS.teacher_encoder_obs_dim + bypass, **common)
    return Solo12BaseImuTeacherPPORunnerCfg().policy.replace(
        teacher_encoder_obs_dim=_ADAPTIVE_DEFAULTS.teacher_encoder_obs_dim, teacher_latent_activation="tanh", **common)


@configclass
class GoToTeacherRunner(BorinotGoToPPORunnerCfg):
    experiment_name = "borinot_goto_teacher"
    policy = perception_policy(False, _ADAPTIVE_DEFAULTS.base_imu_history_sample_dim + 5)


@configclass
class GoToStudentRunner(BorinotGoToPPORunnerCfg):
    experiment_name = "borinot_goto_student_rl"
    policy = perception_policy(True, _ADAPTIVE_DEFAULTS.base_imu_history_sample_dim + 5)
    algorithm = BorinotGoToPPORunnerCfg().algorithm.replace(learning_rate=2e-4)


@configclass
class VelocityTeacherRunner(BorinotCmdVelPPORunnerCfg):
    experiment_name = "borinot_cmd_vel_teacher"
    policy = perception_policy(False, _ADAPTIVE_DEFAULTS.base_imu_history_sample_dim + 4)


@configclass
class VelocityStudentRunner(BorinotCmdVelPPORunnerCfg):
    experiment_name = "borinot_cmd_vel_student_rl"
    policy = perception_policy(True, _ADAPTIVE_DEFAULTS.base_imu_history_sample_dim + 4)
    algorithm = BorinotCmdVelPPORunnerCfg().algorithm.replace(learning_rate=2e-4)
