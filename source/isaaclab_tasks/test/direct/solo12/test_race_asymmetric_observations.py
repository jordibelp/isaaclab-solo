# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import pytest
import torch

from isaaclab_tasks.direct.solo12_race.agents.rsl_rl_ppo_cfg import (
    Solo12RacePPORunnerCfg,
    Solo12RaceParamsConditionedEncPPORunnerCfg,
    configure_race_actor_critic,
)
from isaaclab_tasks.direct.solo12_race.solo12_race_env import Solo12RaceEnv
from isaaclab_tasks.direct.solo12_race.solo12_race_env_cfg import (
    Solo12RaceEnvCfg,
    Solo12RaceJointStateImuTcnEnvCfg,
    Solo12RaceJointStateImuTcnEvalCameraEnvCfg,
    Solo12RaceJointStateTcnEnvCfg,
    Solo12RaceJointStateTcnEvalCameraEnvCfg,
    Solo12RaceParamsConditionedEncEnvCfg,
    Solo12RaceParamsDaggerJointStateImuTcnEnvCfg,
    Solo12RaceParamsDaggerJointStateTcnEnvCfg,
)


@pytest.mark.parametrize("legacy_obs", [False, True])
def test_robust_hydra_switch_wires_encoder_and_refreshes_spaces(legacy_obs):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from isaaclab.utils import replace_strings_with_slices
    from isaaclab_tasks.utils.hydra import register_task_to_hydra

    task = "Isaac-Solo12-Race-Direct-v0"
    env_cfg, agent_cfg = register_task_to_hydra(task, "rsl_rl_cfg_entry_point")
    with initialize(version_base="1.3", config_path=None):
        cfg = compose(
            config_name=task,
            overrides=[
                "agent.policy.asymmetric_actor_critic=True",
                f"env.remove_c_close_vectors_from_observation={legacy_obs}",
            ],
        )
    cfg_dict = replace_strings_with_slices(OmegaConf.to_container(cfg, resolve=True))
    env_cfg.from_dict(cfg_dict["env"])
    agent_cfg.from_dict(cfg_dict["agent"])
    configure_race_actor_critic(env_cfg, agent_cfg)
    assert env_cfg.observation_space == (57 if legacy_obs else 63)
    assert env_cfg.state_space == env_cfg.observation_space + 16
    assert agent_cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    assert not env_cfg.include_forces_to_gt_obs and not env_cfg.include_mu_coefs_to_gt_obs
    teacher = Solo12RaceParamsConditionedEncPPORunnerCfg().policy
    for field in (
        "env_params_dim", "env_params_encoder_hidden_dims", "env_params_latent_dim", "env_params_encoder_activation"
    ):
        assert getattr(agent_cfg.policy, field) == getattr(teacher, field)


def test_robust_switch_leaves_baseline_and_teacher_configs_unchanged():
    for env_cfg, agent_cfg in (
        (Solo12RaceEnvCfg(), Solo12RacePPORunnerCfg()),
        (Solo12RaceParamsConditionedEncEnvCfg(), Solo12RaceParamsConditionedEncPPORunnerCfg()),
    ):
        before = (env_cfg.to_dict(), agent_cfg.to_dict())
        configure_race_actor_critic(env_cfg, agent_cfg)
        assert (env_cfg.to_dict(), agent_cfg.to_dict()) == before


@pytest.mark.parametrize(
    "field", ["include_mu_coefs_to_gt_obs", "include_forces_to_gt_obs", "include_joint_state_history_obs"]
)
def test_robust_switch_rejects_changed_actor_information(field):
    env_cfg, agent_cfg = Solo12RaceEnvCfg(), Solo12RacePPORunnerCfg()
    agent_cfg.policy.asymmetric_actor_critic = True
    setattr(env_cfg, field, True)
    with pytest.raises(ValueError, match="robust asymmetric"):
        configure_race_actor_critic(env_cfg, agent_cfg)


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
