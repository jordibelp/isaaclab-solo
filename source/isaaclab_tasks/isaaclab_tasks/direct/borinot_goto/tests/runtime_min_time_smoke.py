# SPDX-License-Identifier: BSD-3-Clause
"""Registered Hydra/PhysX proof of minimum-time terminal and observation semantics."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-goto-minTime")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
    cfg.include_events_randomization = False
    cfg.forces_applied_to_base_curriculum = [0.0]
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    cfg.reset_rpy_noise_degs = (0.0, 0.0, 0.0)
    cfg.use_terminal_success_bonus, cfg.rwd_scale_terminal_success = True, 5.0
    cfg.curriculum_min_steps = 1
    cfg.curriculum_min_episodes = cfg.curriculum_window_episodes = 1
    cfg.configure_control()
    wrapped = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    env = wrapped.unwrapped
    checks = []

    def check(name, passed):
        passed = bool(passed)
        checks.append({"name": name, "passed": passed})
        assert passed, name

    try:
        obs = wrapped.get_observations()
        check("separate_task_defaults", cfg.reset_at_success and cfg.per_step_penalty_weight == -.015
              and agent_cfg.experiment_name == "borinot_goto_min_time" and agent_cfg.algorithm.gamma == 1.0)
        check("actor_critic_spaces", obs["policy"].shape == (4, cfg.observation_space)
              and obs["critic"].shape == (4, cfg.state_space)
              and cfg.state_space - cfg.observation_space == 2 * cfg.remove_time_from_actor_obs)
        # Identical physical state, different clock: actor must not see either clock when hidden.
        before = env._get_observations()
        env._episode_steps[:] = 3
        after = env._get_observations()
        changed = (before["critic"] != after["critic"]).any(0)
        check("both_critic_clocks_retained", changed.sum() == 2)
        if cfg.remove_time_from_actor_obs:
            check("actor_has_no_clock", torch.equal(before["policy"], after["policy"])
                  and torch.equal(before["policy"], before["critic"][:, ~changed]))
        env.reset()
        root = env.robot.data.root_state_w.clone()
        root[:, 7:] = 0.0
        root[2, 2] = cfg.flight_height_range_m[1] + 1.0
        env.robot.write_root_state_to_sim(root)
        env._goal_position_w.copy_(root[:, :3])
        env._goal_position_w[0, 0] += .02
        env._goal_position_w[[1, 3], 0] += .5
        env._goal_deadline_steps[:] = 100
        env._goal_deadline_steps[3] = 1
        env._episode_steps.zero_()
        env._previous_goal_distance.copy_((env._goal_position_w - env.robot.data.root_pos_w).norm(dim=-1))
        obs, reward, done, info = wrapped.step(torch.zeros(4, cfg.action_space, device=env.device))
        check("mixed_automatic_resets", done.tolist() == [True, False, True, True])
        check("safety_failure_beats_spatial_success", info["goal"]["success"].tolist() == [True, False, False, False]
              and info["goal"]["failed"].tolist() == [False, False, True, False])
        check("early_success_not_deadline_or_bootstrapped_timeout", not info["goal"]["deadline_reached"][0]
              and info["goal"]["deadline_reached"][3] and "time_outs" not in info)
        terms = info["reward_terms"]
        check("time_cost_on_every_step_including_terminal", torch.allclose(terms["time"], torch.full_like(reward, -.015)))
        check("success_bonus_on_that_step", terms["terminal_success"].tolist() == [5., 0., 0., 0.])
        check("early_terminal_distance_uses_final_not_reset_state", torch.allclose(
            terms["terminal_distance"][0], cfg.rwd_scale_terminal_distance * info["goal"]["distance_m"][0]))
        check("failure_cost_retained", terms["terminal_distance"][2] <= -100.)
        final = info["time_outs_obs"]["critic"]
        check("early_terminal_velocity_costs", torch.allclose(
            terms["terminal_linear_vel"][0], cfg.rwd_scale_terminal_linear_vel * final[0, :3].square().sum(), atol=1e-5)
              and torch.allclose(terms["terminal_ang_vel"][0],
                                 cfg.rwd_scale_terminal_ang_vel * final[0, 3:6].square().sum(), atol=1e-5))
        check("reward_sum", torch.allclose(reward, sum(terms.values())))
        check("new_episode_clock_after_reset", env._episode_steps.tolist() == [0, 1, 0, 0]
              and final[0, -1] > 0 and obs["critic"][0, -1] > 0)
        check("early_success_logged_for_curriculum", info["log"]["Episode/success_count"] == 1
              and info["log"]["Goal/early_success_count"] == 1
              and torch.isclose(info["log"]["Goal/success_duration_s"], torch.tensor(env.step_dt, device=env.device)))
        check("terminal_time_accumulator_cleared", env._episode_terms["time"].tolist()[0] == 0.0
              and abs(float(env._episode_terms["time"][1]) + .015) < 1e-7)
        kept = torch.tensor([0, 2, 3], device=env.device)
        buffers = (env._goal_position_w, env._goal_deadline_steps, env._episode_steps, env._episode_terms["time"])
        snapshots = [value[kept].clone() for value in buffers]
        env._reset_idx(torch.tensor([1], device=env.device))
        check("partial_reset_isolation", all(torch.equal(value[kept], old) for value, old in zip(buffers, snapshots))
              and env._episode_terms["time"][1] == 0)
        receipt = {"task": args.task, "remove_time_from_actor_obs": cfg.remove_time_from_actor_obs,
                   "actor_dimension": cfg.observation_space, "critic_dimension": cfg.state_space,
                   "checks": checks, "passed": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print("BORINOT_MIN_TIME_RUNTIME " + json.dumps(receipt), flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
