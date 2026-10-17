# SPDX-License-Identifier: BSD-3-Clause
"""Validate command-rate rewards and reset boundaries through real simulator steps."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.curriculum = False
    cfg.seed = 230999
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    for name in vars(cfg):
        if name.startswith("rwd_scale_"):
            setattr(cfg, name, 0.0)
    cfg.rwd_scale_action_rate = -0.001
    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    report = {"mode": cfg.control_mode, "sac_mapping": cfg.sac_action_mapping, "checks": [], "costs": []}

    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        assert condition, name

    original_reward = env._get_rewards
    previous = torch.zeros_like(env._actuator_commands)

    def traced_reward():
        current = env._actuator_commands.clone()
        effort = env.robot.data.computed_torque[:, env.joint_ids].clamp(-cfg.effort_lim_sim, cfg.effort_lim_sim)
        current[:, 6:] = effort / (2 * cfg.effort_lim_sim)
        # Scalar reference, independent of the production tensor reductions.
        delta = (current - previous).cpu().tolist()
        cost = torch.tensor([sum(v*v for v in row[:6])/6 + sum(v*v for v in row[6:])/2
                             for row in delta], device=env.device)
        reward = original_reward()
        check("reward_matches_group_averaged_physical_command_changes",
              torch.allclose(reward, cfg.rwd_scale_action_rate * cost * env.step_dt, atol=1e-10, rtol=2e-5))
        check("stored_effort_is_post_clipping", torch.allclose(env._actuator_commands[:, 6:], current[:, 6:]))
        report["costs"].append(cost.cpu().tolist())
        return reward

    env._get_rewards = traced_reward

    def step(actions):
        previous.copy_(env._actuator_commands)
        obs, reward, terminated, truncated, info = env.step(actions)
        check("finite_real_policy_step", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
        return reward, terminated, truncated, info

    def baseline():
        value = torch.zeros_like(env._actuator_commands)
        value[:, :6] = 1 / cfg.thrust_to_weight
        return value

    try:
        env.reset()
        check("reset_current_and_previous_are_hover_and_zero_torque",
              torch.equal(env._actuator_commands, baseline()) and torch.equal(env._previous_actuator_commands, baseline()))
        actions = torch.zeros(4, cfg.action_space, device=env.device)
        reward, *_ = step(actions)
        torque_mode = cfg.control_mode == "thrust_and_torque_control" and not cfg.hold_arm
        if torque_mode:
            check("neutral_first_step_has_no_artificial_reset_jump", bool((reward.abs() < 1e-15).all()))
        if torque_mode and not cfg.sac_action_mapping:
            actions.fill_(3.0)
            step(actions)
            actions.fill_(8.0)
            reward, *_ = step(actions)
            check("positive_saturation_excursion_has_zero_cost", not reward.any())
            check("rotor_command_is_before_lag", not torch.allclose(env._rotor_thrust / (env._hover_per_rotor * cfg.thrust_to_weight), env._actuator_commands[:, :6]))
            actions.fill_(-3.0)
            reward, *_ = step(actions)
            check("full_span_change_in_both_groups_costs_two", torch.allclose(reward, torch.full_like(reward, -0.001 * 2 * env.step_dt), atol=1e-10, rtol=1e-6))
            actions.fill_(-8.0)
            reward, *_ = step(actions)
            check("negative_saturation_excursion_has_zero_cost", not reward.any())
        else:
            actions[:, -2:] = torch.tensor([0.3, -0.4], device=env.device)
            for _ in range(8):
                step(actions)

        before = env._actuator_commands.clone()
        before_previous = env._previous_actuator_commands.clone()
        ids = torch.tensor([0, 2], device=env.device)
        env._reset_idx(ids)
        check("partial_reset_reinitializes_selected_histories",
              torch.equal(env._actuator_commands[ids], baseline()[ids]) and torch.equal(env._previous_actuator_commands[ids], baseline()[ids]))
        check("partial_reset_preserves_other_histories", torch.equal(env._actuator_commands[[1, 3]], before[[1, 3]])
              and torch.equal(env._previous_actuator_commands[[1, 3]], before_previous[[1, 3]]))
        before = env._actuator_commands.clone()
        env._launch_ball(ids)
        check("ball_only_reset_preserves_command_history", torch.equal(env._actuator_commands, before))
        actions.zero_()
        reward, *_ = step(actions)
        if torque_mode:
            check("selected_neutral_first_step_has_no_reset_jump", bool((reward[ids].abs() < 1e-15).all()))
        env._episode_steps[:] = env.max_episode_length - 1
        _, _, truncated, _ = step(actions)
        check("terminal_interval_checked_before_auto_reset", truncated.all())
        check("automatic_reset_reinitializes_both_histories", torch.equal(env._actuator_commands, baseline())
              and torch.equal(env._previous_actuator_commands, baseline()))

        # Disabling this one reward leaves all rewards at zero in this fixture.
        env._get_rewards = original_reward
        cfg.rwd_scale_action_rate = 0.0
        for _ in range(32):
            reward, *_ = step(actions)
            assert not reward.any()
        check("disabled_weight_has_no_reward_effect", True)
        report["passed"] = True
        print("ACTION_RATE_VALIDATION=" + json.dumps(report), flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
