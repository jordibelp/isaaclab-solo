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
    route_critic_obs,
)

SHARED_WITH_LIN_VEL = ["policy", PRIVILEGED_OBS_GROUP]


def two_feet_cfg(**overrides) -> Solo12TwoFeetEnvCfg:
    cfg = Solo12TwoFeetEnvCfg()
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.refresh_runtime_dependent_config()
    return cfg


def test_defaults_keep_the_noisy_mocap_observation_and_a_shared_critic():
    cfg = two_feet_cfg()

    assert cfg.base_obs_from_imu is False
    assert cfg.remove_root_lin_vel_b_from_actor_obs is False
    assert cfg.include_lin_acc_steps_in_imu_obs == 0
    assert cfg.noisy_actor_obs is True
    assert cfg.noisy_critic_obs is True
    assert not hasattr(cfg, "enable_observation_corruption")
    assert cfg.observation_space == 48
    assert cfg.critic_obs_groups() is None


@pytest.mark.parametrize(
    "overrides, actor_obs_dim, critic_groups",
    [
        ({"remove_root_lin_vel_b_from_actor_obs": True}, 45, SHARED_WITH_LIN_VEL),
        ({"remove_root_lin_vel_b_from_obs": True}, 45, None),
        ({"remove_root_lin_vel_b_from_obs": True, "remove_root_lin_vel_b_from_actor_obs": True}, 45, None),
        ({"base_obs_from_imu": True}, 48, None),
        ({"base_obs_from_imu": True, "include_lin_acc_steps_in_imu_obs": 1}, 51, None),
        ({"base_obs_from_imu": True, "remove_root_lin_vel_b_from_actor_obs": True}, 45, SHARED_WITH_LIN_VEL),
        (
            {"base_obs_from_imu": True, "include_lin_acc_steps_in_imu_obs": 4, "noisy_critic_obs": False},
            60,
            ["critic"],
        ),
        (
            {
                "base_obs_from_imu": True,
                "include_lin_acc_steps_in_imu_obs": 4,
                "remove_root_lin_vel_b_from_actor_obs": True,
            },
            57,
            SHARED_WITH_LIN_VEL,
        ),
        ({"noisy_critic_obs": False}, 48, ["critic"]),
        ({"noisy_critic_obs": False, "remove_root_lin_vel_b_from_actor_obs": True}, 45, ["critic"]),
        # Play turns actor noise off; the critic inputs must keep the layout the critic was trained with.
        ({"noisy_critic_obs": False, "noisy_actor_obs": False}, 48, ["critic"]),
        ({"noisy_actor_obs": False, "remove_root_lin_vel_b_from_actor_obs": True}, 45, SHARED_WITH_LIN_VEL),
    ],
)
def test_actor_observation_size_and_critic_groups(overrides, actor_obs_dim, critic_groups):
    cfg = two_feet_cfg(**overrides)

    assert cfg.observation_space == actor_obs_dim
    assert cfg.critic_obs_groups() == critic_groups


@pytest.mark.parametrize(
    "option, value",
    [("base_obs_from_imu", True), ("remove_root_lin_vel_b_from_actor_obs", True), ("noisy_critic_obs", False)],
)
def test_new_observation_options_require_the_simple_mlp_layout(option, value):
    cfg = Solo12BaseImuStudentRlEnvCfg()
    setattr(cfg, option, value)

    with pytest.raises(ValueError, match="simple_mlp"):
        cfg.refresh_runtime_dependent_config()


def test_actor_noise_switch_works_for_every_policy_model():
    cfg = Solo12BaseImuStudentRlEnvCfg()
    cfg.noisy_actor_obs = False

    cfg.refresh_runtime_dependent_config()

    assert cfg.critic_obs_groups() is None


@pytest.mark.parametrize(
    "overrides",
    [{"include_lin_acc_steps_in_imu_obs": 2}, {"base_obs_from_imu": True, "include_lin_acc_steps_in_imu_obs": -1}],
)
def test_acceleration_steps_must_be_non_negative_and_need_the_imu(overrides):
    with pytest.raises(ValueError, match="include_lin_acc_steps_in_imu_obs"):
        two_feet_cfg(**overrides)


@pytest.mark.parametrize("field, value", [("imu_ekf_processed_inputs", False), ("use_rotMat_on_imu_encoder", True)])
def test_imu_base_observation_uses_the_odri_imu_signals(field, value):
    with pytest.raises(ValueError, match="odri/imu"):
        two_feet_cfg(base_obs_from_imu=True, **{field: value})


@pytest.mark.parametrize(
    "overrides, critic_groups",
    [({"remove_root_lin_vel_b_from_actor_obs": True}, SHARED_WITH_LIN_VEL), ({"noisy_critic_obs": False}, ["critic"])],
)
@pytest.mark.parametrize(
    "obs_groups, actor_set",
    [
        ({"actor": ["policy"], "critic": ["policy"]}, "actor"),  # SAC
        ({"policy": ["policy"], "critic": ["policy"]}, "policy"),  # PPO
    ],
)
def test_critic_groups_are_routed_to_the_critic_only(overrides, critic_groups, obs_groups, actor_set):
    cfg = two_feet_cfg(**overrides)
    agent_cfg = SimpleNamespace(obs_groups=dict(obs_groups))

    route_critic_obs(cfg, agent_cfg)
    route_critic_obs(cfg, agent_cfg)

    assert agent_cfg.obs_groups["critic"] == critic_groups
    assert agent_cfg.obs_groups[actor_set] == ["policy"]


def test_routing_keeps_the_agent_default_for_a_shared_critic():
    agent_cfg = SimpleNamespace(obs_groups={"actor": ["policy"], "critic": ["policy"]})

    route_critic_obs(two_feet_cfg(base_obs_from_imu=True), agent_cfg)
    route_critic_obs(Solo12BaseImuStudentRlEnvCfg(), agent_cfg)
    route_critic_obs(SimpleNamespace(), agent_cfg)

    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}


# Constant "noise": every corrupted value is shifted by exactly NOISE, so tests can see which terms are noisy.
NOISE = 5.0
LIN_VEL, ANG_VEL, GRAVITY = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
IMU_GYRO, IMU_GRAVITY = [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]


def imu_acc(steps: int) -> list[float]:
    """Noise-free physics-rate accelerations, oldest first: [10, 11, 12], [13, 14, 15], ..."""
    return [10.0 + value for value in range(3 * steps)]


def make_obs_env(cfg: Solo12TwoFeetEnvCfg) -> Solo12Env:
    """Return a Solo12Env shell with distinct, recognizable values in every observed signal."""
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([LIN_VEL]),
            root_ang_vel_b=torch.tensor([ANG_VEL]),
            projected_gravity_b=torch.tensor([GRAVITY]),
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
    imu_sample = torch.tensor([IMU_GYRO + [-1.0, -1.0, -1.0] + IMU_GRAVITY])
    env._read_base_imu_sample = lambda corrupt: imu_sample + (NOISE if corrupt else 0.0)
    clean_acc = torch.tensor(imu_acc(cfg.include_lin_acc_steps_in_imu_obs)).reshape(1, -1, 3)
    env._imu_lin_acc_history = torch.stack((clean_acc + NOISE, clean_acc), dim=2)
    return env


def expected_obs(env: Solo12Env, base_terms: list[float], noisy: bool) -> torch.Tensor:
    noise = NOISE if noisy else 0.0
    return torch.cat(
        (
            torch.tensor([base_terms]) + noise,
            env._commands,
            torch.full((1, 12), 0.5 + noise),
            torch.full((1, 12), 0.25 + noise),
            env._actions,
        ),
        dim=-1,
    )


@pytest.mark.parametrize(
    "overrides, actor_base, critic_group, critic_base, critic_noisy",
    [
        ({}, LIN_VEL + ANG_VEL + GRAVITY, None, None, None),
        ({"remove_root_lin_vel_b_from_actor_obs": True}, ANG_VEL + GRAVITY, PRIVILEGED_OBS_GROUP, LIN_VEL, True),
        ({"base_obs_from_imu": True}, LIN_VEL + IMU_GYRO + IMU_GRAVITY, None, None, None),
        (
            {"base_obs_from_imu": True, "include_lin_acc_steps_in_imu_obs": 1},
            LIN_VEL + IMU_GYRO + IMU_GRAVITY + imu_acc(1),
            None,
            None,
            None,
        ),
        (
            {
                "base_obs_from_imu": True,
                "include_lin_acc_steps_in_imu_obs": 4,
                "remove_root_lin_vel_b_from_actor_obs": True,
                "noisy_critic_obs": False,
            },
            IMU_GYRO + IMU_GRAVITY + imu_acc(4),
            "critic",
            LIN_VEL + IMU_GYRO + IMU_GRAVITY + imu_acc(4),
            False,
        ),
        (
            {"base_obs_from_imu": True, "remove_root_lin_vel_b_from_actor_obs": True, "noisy_critic_obs": False},
            IMU_GYRO + IMU_GRAVITY,
            "critic",
            LIN_VEL + IMU_GYRO + IMU_GRAVITY,
            False,
        ),
        (
            {"remove_root_lin_vel_b_from_obs": True, "noisy_critic_obs": False},
            ANG_VEL + GRAVITY,
            "critic",
            ANG_VEL + GRAVITY,
            False,
        ),
    ],
)
def test_actor_and_critic_observations(overrides, actor_base, critic_group, critic_base, critic_noisy):
    noise = (NOISE, NOISE)
    cfg = two_feet_cfg(
        base_lin_vel_noise=noise,
        base_ang_vel_noise=noise,
        projected_gravity_noise=noise,
        joint_pos_noise=noise,
        joint_vel_noise=noise,
        **overrides,
    )
    env = make_obs_env(cfg)

    observations = env._get_observations()

    assert observations["policy"].shape[-1] == cfg.observation_space
    torch.testing.assert_close(observations["policy"], expected_obs(env, actor_base, noisy=True))
    if critic_group is None:
        assert set(observations) == {"policy"}
        return
    assert set(observations) == {"policy", critic_group}
    if critic_group == PRIVILEGED_OBS_GROUP:
        expected_critic = torch.tensor([critic_base]) + (NOISE if critic_noisy else 0.0)
    else:
        expected_critic = expected_obs(env, critic_base, noisy=critic_noisy)
    torch.testing.assert_close(observations[critic_group], expected_critic)


def test_actor_noise_switch_also_silences_the_shared_privileged_velocity():
    noise = (NOISE, NOISE)
    cfg = two_feet_cfg(
        noisy_actor_obs=False,
        base_lin_vel_noise=noise,
        joint_vel_noise=noise,
        remove_root_lin_vel_b_from_actor_obs=True,
    )
    env = make_obs_env(cfg)

    observations = env._get_observations()

    torch.testing.assert_close(observations["policy"], expected_obs(env, ANG_VEL + GRAVITY, noisy=False))
    torch.testing.assert_close(observations[PRIVILEGED_OBS_GROUP], torch.tensor([LIN_VEL]))


def test_each_physics_step_appends_observed_and_clean_accelerations_oldest_first():
    cfg = two_feet_cfg(base_obs_from_imu=True, include_lin_acc_steps_in_imu_obs=3)
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    env._base_imu_enabled = True
    env._base_imu_history_len = 0
    env._imu_lin_acc_history = torch.zeros(1, 3, 2, 3)
    env._update_base_imu_bias_random_walk = lambda: None

    for step in range(4):
        sample = torch.tensor([[0.0, 0.0, 0.0] + [float(step)] * 3 + [0.0, 0.0, 0.0]])
        env._read_base_imu_sample = lambda corrupt, sample=sample: sample + (NOISE if corrupt else 0.0)
        env._record_base_imu_history_sample()

    newest_three = torch.tensor([[1.0] * 3, [2.0] * 3, [3.0] * 3])
    torch.testing.assert_close(env._imu_lin_acc_history[0, :, 1], newest_three)
    torch.testing.assert_close(env._imu_lin_acc_history[0, :, 0], newest_three + NOISE)
