# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12_race.solo12_race_env import Solo12RaceEnv
from isaaclab_tasks.direct.solo12_race.solo12_race_env_cfg import (
    Solo12RaceJointStateImuTcnEnvCfg,
    Solo12RaceJointStateImuTcnEvalCameraEnvCfg,
    Solo12RaceJointStateTcnEnvCfg,
    Solo12RaceJointStateTcnEvalCameraEnvCfg,
    Solo12RaceParamsDaggerJointStateImuTcnEnvCfg,
    Solo12RaceParamsDaggerJointStateTcnEnvCfg,
)


def test_student_and_eval_configs_declare_asymmetric_policy_and_critic_spaces():
    for cfg_type, history_sample_dim in (
        (Solo12RaceJointStateTcnEnvCfg, 24),
        (Solo12RaceJointStateTcnEvalCameraEnvCfg, 24),
        (Solo12RaceJointStateImuTcnEnvCfg, 48),
        (Solo12RaceJointStateImuTcnEvalCameraEnvCfg, 48),
    ):
        cfg = cfg_type()
        history_len = cfg.joint_state_history_length

        assert cfg.asymmetric_actor_critic is True
        assert cfg.observation_space == cfg.base_observation_dim + history_len * history_sample_dim
        assert cfg.state_space == cfg.base_observation_dim + cfg.privileged_env_params_obs_dim
        assert cfg.privileged_env_params_obs_dim == 16


def test_asymmetric_spaces_refresh_after_runtime_overrides():
    cfg = Solo12RaceJointStateTcnEnvCfg()
    cfg.remove_c_close_vectors_from_observation = True
    cfg.joint_state_history_policy_steps = 3

    cfg.__post_init__()

    assert cfg.base_observation_dim == 57
    assert cfg.joint_state_history_length == cfg.decimation * 3
    assert cfg.observation_space == 57 + cfg.joint_state_history_length * 24
    assert cfg.state_space == 57 + 16


def test_dagger_configs_keep_privileged_labels_and_history_in_policy_only():
    for cfg_type, history_sample_dim in (
        (Solo12RaceParamsDaggerJointStateTcnEnvCfg, 24),
        (Solo12RaceParamsDaggerJointStateImuTcnEnvCfg, 48),
    ):
        cfg = cfg_type()

        assert cfg.asymmetric_actor_critic is False
        assert cfg.observation_space == (
            cfg.base_observation_dim
            + cfg.gt_env_params_obs_dim
            + cfg.joint_state_history_length * history_sample_dim
        )
        assert cfg.gt_env_params_obs_dim == 16
        assert cfg.state_space == 0


def _make_observation_stub() -> Solo12RaceEnv:
    env = object.__new__(Solo12RaceEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device="cpu")
    env.cfg = SimpleNamespace(
        include_root_lin_vel_b_obs=True,
        base_lin_vel_noise=(0.0, 0.0),
        base_ang_vel_noise=(0.0, 0.0),
        projected_gravity_noise=(0.0, 0.0),
        joint_pos_noise=(0.0, 0.0),
        joint_vel_noise=(0.0, 0.0),
        remove_c_close_vectors_from_observation=False,
        include_forces_to_gt_obs=False,
        include_mu_coefs_to_gt_obs=False,
        include_foot_imu_obs=False,
        include_joint_state_history_obs=True,
        asymmetric_actor_critic=True,
    )
    env._joint_ids = list(range(12))
    env._actions = torch.arange(12, dtype=torch.float).repeat(2, 1)
    env._previous_actions = torch.zeros_like(env._actions)
    env._joint_state_history = torch.full((2, 2, 24), 333.0)
    env._foot_imu_history = torch.empty(2, 0, 24)
    identity_quat = torch.zeros(2, 4)
    identity_quat[:, 0] = 1.0
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
            root_quat_w=identity_quat,
            root_pos_w=torch.zeros(2, 3),
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.zeros(2, 3),
        )
    )
    gate_pillars = torch.zeros(2, 2, 3)
    env._get_current_gate_data = lambda: (torch.zeros(2, 3), gate_pillars, torch.zeros(2, dtype=torch.long))
    env._get_following_gate_pillars = lambda target_idx: gate_pillars
    env._get_closest_pillar_vectors_b = lambda root_quat_w: torch.zeros(2, 2, 3)
    env._maybe_corrupt = lambda value, noise_range: value
    env._get_gt_foot_contact_forces_obs = lambda root_quat_w: torch.full((2, 12), 111.0)
    env._get_gt_patch_mu_obs = lambda: torch.full((2, 4), 222.0)
    return env


def test_asymmetric_observation_tensor_layout_keeps_privilege_out_of_policy():
    env = _make_observation_stub()

    observations = env._get_observations()

    assert set(observations) == {"policy", "critic"}
    assert observations["policy"].shape == (2, 63 + 2 * 24)
    assert observations["critic"].shape == (2, 63 + 16)
    torch.testing.assert_close(observations["critic"][:, :63], observations["policy"][:, :63])
    torch.testing.assert_close(observations["critic"][:, 63:75], torch.full((2, 12), 111.0))
    torch.testing.assert_close(observations["critic"][:, 75:79], torch.full((2, 4), 222.0))
    torch.testing.assert_close(observations["policy"][:, 63:], torch.full((2, 48), 333.0))
    assert not torch.any(observations["policy"] == 111.0)
    assert not torch.any(observations["policy"] == 222.0)
