# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import (
    PRIVILEGED_OBS_GROUP,
    Solo12BaseImuStudentRlEnvCfg,
    Solo12TwoFeetEnvCfg,
    route_privileged_obs_to_critic,
)


def two_feet_cfg(**overrides) -> Solo12TwoFeetEnvCfg:
    cfg = Solo12TwoFeetEnvCfg()
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.refresh_runtime_dependent_config()
    return cfg


def test_defaults_keep_the_mocap_observation():
    cfg = two_feet_cfg()

    assert cfg.base_obs_from_imu is False
    assert cfg.remove_root_lin_vel_b_from_actor_obs is False
    assert cfg.include_lin_acc_in_imu_obs is True
    assert cfg.observation_space == 48
    assert cfg.has_privileged_critic_obs() is False


@pytest.mark.parametrize(
    "overrides, actor_obs_dim, privileged_critic",
    [
        ({"remove_root_lin_vel_b_from_actor_obs": True}, 45, True),
        ({"remove_root_lin_vel_b_from_obs": True}, 45, False),
        ({"remove_root_lin_vel_b_from_obs": True, "remove_root_lin_vel_b_from_actor_obs": True}, 45, False),
        ({"base_obs_from_imu": True}, 51, False),
        ({"base_obs_from_imu": True, "include_lin_acc_in_imu_obs": False}, 48, False),
        ({"base_obs_from_imu": True, "remove_root_lin_vel_b_from_actor_obs": True}, 48, True),
        (
            {
                "base_obs_from_imu": True,
                "include_lin_acc_in_imu_obs": False,
                "remove_root_lin_vel_b_from_actor_obs": True,
            },
            45,
            True,
        ),
    ],
)
def test_actor_observation_size_and_privileged_critic(overrides, actor_obs_dim, privileged_critic):
    cfg = two_feet_cfg(**overrides)

    assert cfg.observation_space == actor_obs_dim
    assert cfg.has_privileged_critic_obs() is privileged_critic


@pytest.mark.parametrize("option", ["base_obs_from_imu", "remove_root_lin_vel_b_from_actor_obs"])
def test_new_observation_options_require_the_simple_mlp_layout(option):
    cfg = Solo12BaseImuStudentRlEnvCfg()
    setattr(cfg, option, True)

    with pytest.raises(ValueError, match="simple_mlp"):
        cfg.refresh_runtime_dependent_config()


@pytest.mark.parametrize("field, value", [("imu_ekf_processed_inputs", False), ("use_rotMat_on_imu_encoder", True)])
def test_imu_base_observation_uses_the_odri_imu_signals(field, value):
    with pytest.raises(ValueError, match="odri/imu"):
        two_feet_cfg(base_obs_from_imu=True, **{field: value})


@pytest.mark.parametrize(
    "obs_groups, actor_set",
    [
        ({"actor": ["policy"], "critic": ["policy"]}, "actor"),  # SAC
        ({"policy": ["policy"], "critic": ["policy"]}, "policy"),  # PPO
    ],
)
def test_privileged_group_is_routed_to_the_critic_only(obs_groups, actor_set):
    cfg = two_feet_cfg(remove_root_lin_vel_b_from_actor_obs=True)
    agent_cfg = SimpleNamespace(obs_groups=dict(obs_groups))

    route_privileged_obs_to_critic(cfg, agent_cfg)
    route_privileged_obs_to_critic(cfg, agent_cfg)

    assert agent_cfg.obs_groups["critic"] == ["policy", PRIVILEGED_OBS_GROUP]
    assert agent_cfg.obs_groups[actor_set] == ["policy"]


def test_routing_is_a_noop_without_a_privileged_critic():
    agent_cfg = SimpleNamespace(obs_groups={"actor": ["policy"], "critic": ["policy"]})

    route_privileged_obs_to_critic(two_feet_cfg(base_obs_from_imu=True), agent_cfg)
    route_privileged_obs_to_critic(SimpleNamespace(), agent_cfg)

    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}


def make_obs_env(cfg: Solo12TwoFeetEnvCfg) -> Solo12Env:
    """Return a Solo12Env shell with distinct, recognizable values in every observed signal."""
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[1.0, 2.0, 3.0]]),
            root_ang_vel_b=torch.tensor([[4.0, 5.0, 6.0]]),
            projected_gravity_b=torch.tensor([[7.0, 8.0, 9.0]]),
            joint_pos=torch.full((1, 12), 0.5),
            joint_vel=torch.full((1, 12), 0.25),
        )
    )
    env._joint_ids = slice(None)
    env._q_offset_action_and_obs = torch.zeros(1, 12)
    env._commands = torch.tensor([[0.1, 0.2, 0.3]])
    env._actions = torch.full((1, 12), -1.0)
    env._applied_actions = env._actions
    # gyro, gravity-free acceleration, projected gravity: the order _read_base_imu_sample returns.
    env._read_base_imu_sample = lambda corrupt: torch.tensor([[40.0, 50.0, 60.0, 10.0, 11.0, 12.0, 70.0, 80.0, 90.0]])
    return env


@pytest.mark.parametrize(
    "overrides, expected_base_terms",
    [
        ({}, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ({"remove_root_lin_vel_b_from_actor_obs": True}, [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ({"base_obs_from_imu": True}, [1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 10.0, 11.0, 12.0]),
        (
            {
                "base_obs_from_imu": True,
                "include_lin_acc_in_imu_obs": False,
                "remove_root_lin_vel_b_from_actor_obs": True,
            },
            [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        ),
    ],
)
def test_simple_mlp_observation_layout(overrides, expected_base_terms):
    cfg = two_feet_cfg(enable_observation_corruption=False, **overrides)
    env = make_obs_env(cfg)

    observations = env._get_observations()

    expected_policy = torch.cat(
        (
            torch.tensor([expected_base_terms]),
            env._commands,
            torch.full((1, 12), 0.5),
            torch.full((1, 12), 0.25),
            torch.full((1, 12), -1.0),
        ),
        dim=-1,
    )
    assert expected_policy.shape[-1] == cfg.observation_space
    torch.testing.assert_close(observations["policy"], expected_policy)
    if cfg.has_privileged_critic_obs():
        assert set(observations) == {"policy", PRIVILEGED_OBS_GROUP}
        torch.testing.assert_close(observations[PRIVILEGED_OBS_GROUP], torch.tensor([[1.0, 2.0, 3.0]]))
    else:
        assert set(observations) == {"policy"}
