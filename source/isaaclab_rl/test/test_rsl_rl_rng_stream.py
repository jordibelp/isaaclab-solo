# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_rl.rsl_rl.vecenv_wrapper as wrapper_mod
from isaaclab.utils.seed import RngStream, configure_seed


class _FakeCfg:
    is_finite_horizon = False


class _FakeEnv:
    def __init__(self):
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 100
        self.cfg = _FakeCfg()
        self.render_mode = None
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        policy_space = gym.vector.utils.batch_space(
            gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)), self.num_envs
        )
        self.observation_space = gym.spaces.Dict({"policy": policy_space})
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int = -1) -> int:
        return configure_seed(seed)

    def reset(self):
        return {"policy": torch.rand(self.num_envs, 3)}, {}

    def _get_observations(self):
        return {"policy": torch.rand(self.num_envs, 3)}

    def step(self, actions: torch.Tensor):
        return (
            {"policy": torch.rand(self.num_envs, 3)},
            torch.rand(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def close(self):
        pass


def _run_wrapper_sequence(agent_draw_count: int):
    configure_seed(123)
    env_stream = RngStream.from_seed(999)
    torch.rand(agent_draw_count)

    env = wrapper_mod.RslRlVecEnvWrapper(_FakeEnv(), rng_stream=env_stream)
    obs_before = env.get_observations()["policy"].clone()
    torch.rand(agent_draw_count + 17)
    env.randomize_episode_length_buf()
    episode_lengths = env.episode_length_buf.clone()
    torch.rand(agent_draw_count + 31)
    obs_after, rewards, dones, extras = env.step(torch.zeros(env.num_envs, env.num_actions))
    return obs_before, episode_lengths, obs_after["policy"].clone(), rewards.clone(), dones.clone(), extras


def test_rsl_rl_wrapper_uses_environment_rng_stream(monkeypatch):
    monkeypatch.setattr(wrapper_mod, "ManagerBasedRLEnv", _FakeEnv)

    small_agent = _run_wrapper_sequence(1)
    large_agent = _run_wrapper_sequence(10_000)

    for actual, expected in zip(large_agent[:-1], small_agent[:-1], strict=True):
        torch.testing.assert_close(actual, expected)
    assert large_agent[-1].keys() == small_agent[-1].keys()
    torch.testing.assert_close(large_agent[-1]["time_outs"], small_agent[-1]["time_outs"])
