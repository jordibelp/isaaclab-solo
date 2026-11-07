# SPDX-License-Identifier: BSD-3-Clause
"""Minimum-time task contracts, using the real task methods without a simulator."""
import ast
import importlib

import pytest
import torch

from test_goto import FOLDER, MODES, config, fake_env
from test_wrench import wrench


def min_time_config(mode=MODES[0], **overrides):
    source = ast.parse((FOLDER / "borinot_goto_env_cfg.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "BorinotGoToMinTimeEnvCfg")
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign)}
    if mode in ("model_wrench_indi", "model_wrench_yawless_indi"):
        module = importlib.import_module(wrench.__package__ + ".borinot_model_wrench_indi")
        overrides = {"control_mode": mode, "model_wrench_indi": module.ModelWrenchIndiCfg(), **overrides}
        mode = "wrench"  # Supply the actual hybrid config before its inherited validation runs.
    return config(mode, **(values | overrides))


def test_min_time_defaults_are_separate_from_the_deadline_task():
    base, minimum = config(), min_time_config()
    assert not base.reset_at_success and base.per_step_penalty_weight == 0.0
    assert minimum.reset_at_success and minimum.per_step_penalty_weight == -0.015
    assert not minimum.remove_time_from_actor_obs
    for name in ("goal_half_extents_m", "goal_deadline_range_s", "goal_success_tolerance_m",
                 "goal_success_lin_vel_tolerance_m_s", "goal_success_ang_vel_tolerance_rad_s",
                 "use_terminal_success_bonus", "rwd_scale_terminal_success"):
        assert getattr(minimum, name) == getattr(base, name)


@pytest.mark.parametrize("at_deadline", [False, True])
def test_success_ends_immediately_and_safety_failure_always_wins(at_deadline):
    env = fake_env(min_time_config())
    env._episode_steps[:] = env._goal_deadline_steps if at_deadline else 7
    env._goal_position_w.copy_(env.robot.data.root_pos_w)
    env.robot.data.root_lin_vel_w.zero_()
    env.robot.data.root_ang_vel_w.zero_()
    env.safety_failures[1] = True
    done, timeout = env._get_dones()
    assert done.tolist() == [True, True] and not timeout.any()
    assert env._goal_success.tolist() == [True, False]
    assert env._goal_deadline_reached.tolist() == [at_deadline, at_deadline]


def test_min_time_still_requires_arrival_and_stop_and_keeps_the_deadline_cap():
    env = fake_env(min_time_config())
    env._goal_position_w.copy_(env.robot.data.root_pos_w)
    env.robot.data.root_lin_vel_w[:] = torch.tensor([.1, 0., 0.])
    env.robot.data.root_ang_vel_w.zero_()
    env._episode_steps[:] = 1
    assert not env._get_dones()[0].any()
    env.robot.data.root_lin_vel_w.zero_()
    env.robot.data.root_ang_vel_w[:] = torch.tensor([0., 0., .2])
    assert not env._get_dones()[0].any()
    env._episode_steps.copy_(env._goal_deadline_steps)
    done, timeout = env._get_dones()
    assert done.all() and not timeout.any() and not env._goal_success.any()


def test_early_success_receives_terminal_terms_and_the_last_step_cost():
    env = fake_env(min_time_config(use_terminal_success_bonus=True, rwd_scale_terminal_success=5.0))
    env._episode_steps[:] = 9
    env._goal_position_w.copy_(env.robot.data.root_pos_w + torch.tensor([.03, 0., 0.]))
    env.robot.data.root_lin_vel_w[:] = torch.tensor([.04, 0., 0.])
    env.robot.data.root_ang_vel_w[:] = torch.tensor([0., .08, 0.])
    env._previous_goal_distance.copy_((env._goal_position_w - env.robot.data.root_pos_w).norm(dim=-1))
    assert env._get_dones()[0].all() and not env._goal_deadline_reached.any()
    reward = env._get_rewards()
    terms = env.extras["reward_terms"]
    for name, expected in (("terminal_distance", -.3), ("terminal_linear_vel", -.0016),
                           ("terminal_ang_vel", -.0064), ("terminal_success", 5.), ("time", -.015)):
        torch.testing.assert_close(terms[name], torch.full_like(terms[name], expected))
    torch.testing.assert_close(reward, torch.full_like(reward, 4.677))
    assert env.extras["goal"]["success"].all()
    env._update_curriculum(torch.tensor([0, 1]))
    assert env.extras["log"]["Episode/success_count"] == 2
    assert env.extras["log"]["Goal/deadline_count"] == 0
    assert env.extras["log"]["Episode/duration_s"] == pytest.approx(.09)


@pytest.mark.parametrize("step_dt", [.01, .02])
def test_time_cost_is_per_policy_step_not_scaled_by_dt(step_dt):
    env = fake_env(min_time_config())
    env.step_dt = step_dt
    for _ in range(100):
        env._get_rewards()
    torch.testing.assert_close(env._episode_terms["time"], torch.full((2,), -1.5, dtype=torch.float64))
    env.cfg.per_step_penalty_weight = 0.0
    env._get_rewards()
    assert not env.extras["reward_terms"]["time"].any()


@pytest.mark.parametrize("mode", MODES + ("model_wrench", "model_wrench_yawless", "model_wrench_indi",
                                        "model_wrench_yawless_indi"))
@pytest.mark.parametrize("acc,gravity,height", [(False, False, True), (True, True, False)])
def test_hiding_both_clocks_changes_only_actor_inputs(mode, acc, gravity, height):
    cfg = min_time_config(mode, remove_time_from_actor_obs=True, obs_include_base_lin_acc=acc,
                          obs_include_base_lin_acc_steps=3, projected_gravity_in_obs=gravity,
                          include_base_height=height)
    env = fake_env(cfg)
    first = env._get_observations()
    assert first["policy"].shape == (2, cfg.observation_space)
    assert first["critic"].shape == (2, cfg.state_space)
    assert cfg.state_space == cfg.observation_space + 2
    env._episode_steps += 13
    second = env._get_observations()
    torch.testing.assert_close(first["policy"], second["policy"], rtol=0, atol=0)
    changed = (first["critic"] != second["critic"]).any(0)
    assert changed.sum() == 2  # Independently identify both clocks by changing only time.
    torch.testing.assert_close(first["policy"], first["critic"][:, ~changed], rtol=0, atol=0)
    size = cfg.observation_space
    cfg.configure_control()
    assert cfg.observation_space == size and cfg.state_space == size + 2
    cfg.remove_time_from_actor_obs = False
    cfg.configure_control()
    assert cfg.observation_space == cfg.state_space == size + 2


@pytest.mark.parametrize("name,value", [("reset_at_success", 1), ("remove_time_from_actor_obs", "False"),
                                      ("per_step_penalty_weight", .01),
                                      ("per_step_penalty_weight", float("nan")),
                                      ("per_step_penalty_weight", float("inf"))])
def test_invalid_min_time_settings_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        min_time_config(**{name: value})


def test_live_player_keeps_minimum_time_episode_resets():
    from test_playback import play_env

    env = play_env(reset_at_success=True, per_step_penalty_weight=-.015)
    assert env.reset_at_deadline
    env._goal_position_w.copy_(env.robot.data.root_pos_w)
    env.robot.data.root_lin_vel_w.zero_()
    env.robot.data.root_ang_vel_w.zero_()
    env._episode_steps[:] = 5
    done, timeout = env._get_dones()
    assert done.all() and not timeout.any() and env._goal_success.all()
    env._get_rewards()
    assert env.play_totals["goal_successes"] == 2 and env.play_totals["holds"] == 0


@pytest.mark.parametrize("mode", ["wrench", "thrust_and_PD_joint_pos_control"])
def test_hidden_clocks_keep_sincos_actions_and_controller_memory(mode):
    cfg = min_time_config(mode, remove_time_from_actor_obs=True, use_sin_cos_theta=True)
    env = fake_env(cfg)
    output = env._get_observations()
    assert output["policy"].shape[-1] == cfg.observation_space
    assert output["critic"].shape[-1] == cfg.state_space == cfg.observation_space + 2
    env._episode_steps += 1
    assert torch.equal(output["policy"], env._get_observations()["policy"])
