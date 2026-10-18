# SPDX-License-Identifier: BSD-3-Clause
"""Goal termination is optional and independent of scoring and safety resets."""
import pytest
import torch

from test_continuous import env as continuous_env
from test_control import config


def test_goal_reset_defaults_to_false():
    assert config().reset_on_goal is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("at_time_limit", [False, True])
def test_goal_reset_switch_keeps_reward_and_partial_env_isolation(continuous_env, enabled, at_time_limit):
    env = continuous_env
    env.cfg.reset_on_goal = enabled
    env._flags["goal"][2] = True
    env._episode_steps[2] = env.max_episode_length if at_time_limit else 41
    root = env.robot.data.root_pos_w.clone()
    controller = env._rate_integral.clone()
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [False, False, enabled] + [False] * 6
    assert truncated.tolist() == [False, False, at_time_limit and not enabled] + [False] * 6
    reward = env._get_rewards()
    assert reward.tolist() == [0., 0., env.cfg.rwd_scale_goal] + [0.] * 6
    assert env._outcome[2] == 3 and env.outcome_counts[3] == 1
    assert env.extras["goalkeeper_shot_finished"].tolist() == [False, False, True] + [False] * 6
    assert env._ball_active.tolist() == [True, True, False] + [True] * 6
    # Rewards retire the shot; DirectRLEnv performs any requested robot reset later.
    assert torch.equal(env.robot.data.root_pos_w, root)
    assert torch.equal(env._rate_integral, controller)


@pytest.mark.parametrize("enabled", [False, True])
def test_goal_reset_does_not_depend_on_selected_outcome_or_reward(continuous_env, enabled):
    env = continuous_env
    env.cfg.reset_on_goal = enabled
    env.cfg.rwd_scale_goal = 0.
    env._flags["goal"][:2] = True
    env._flags["base"][0] = True  # base hit wins the outcome priority
    terminated, _ = env._get_dones()
    assert terminated.tolist() == [enabled, enabled] + [False] * 7
    assert env._outcome[:2].tolist() == [1, 3]
    assert env._get_rewards()[1] == 0.
