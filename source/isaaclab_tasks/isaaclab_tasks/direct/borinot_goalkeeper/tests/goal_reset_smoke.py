# SPDX-License-Identifier: BSD-3-Clause
"""Verify a real goal through Hydra, playback scoring and automatic partial resets."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
sys.argv = [sys.argv[0]] + overrides
app = AppLauncher(args).app

import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.borinot_goalkeeper.playback import BorinotGoalkeeperPlayEnv
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.seed = 53
    cfg.curriculum = cfg.reset_grid_enabled = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
    cfg.rwd_scale_energy = 0.
    env = BorinotGoalkeeperPlayEnv(cfg, keep_balls=2)
    report = {"reset_on_goal": cfg.reset_on_goal, "device": env.device, "checks": []}

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(f"CHECK {name}: {bool(passed)}", flush=True)
        assert passed, name

    try:
        env.reset()
        # Place the keeper away from its reset pose, and send only env 0's
        # ball through the goal, clear of the robot. Env 1 must keep running.
        root = env.robot.data.root_state_w.clone()
        root[:, 1] += .2
        env.robot.write_root_state_to_sim(root)
        ball = env.ball.data.root_state_w.clone()
        ball[:, :3] = torch.tensor([2., .8, 1.2], device=env.device) + env.scene.env_origins
        ball[0, 0] = env.scene.env_origins[0, 0] - .35
        ball[:, 7:] = 0.
        ball[0, 7] = -10.
        env.ball.write_root_state_to_sim(ball)
        env.ball_contact.reset()
        env._previous_ball.copy_(ball[:, :3] - env.scene.env_origins)
        env._episode_steps[:] = torch.tensor([12, 21], device=env.device)
        env._next_launch_steps[:] = 10000
        env._rate_integral.fill_(.37)
        writes = []
        original_write = env.robot.write_root_state_to_sim

        def track_write(state, env_ids=None):
            writes.append(env_ids.cpu().tolist())
            return original_write(state, env_ids)

        env.robot.write_root_state_to_sim = track_write
        actions = torch.zeros((2, cfg.action_space), device=env.device)
        _, _, terminated, truncated, info = env.step(actions)
        check("real_goal_detected_only_in_env0", info["goalkeeper_outcome"].tolist() == [3, 0])
        check("goal_counted_once", env.outcome_counts[3] == 1)
        check("goal_penalty_unchanged", info["goalkeeper_reward_terms"]["outcome"].tolist() == [cfg.rwd_scale_goal, 0.])
        check("termination_follows_hydra_flag", terminated.tolist() == [cfg.reset_on_goal, False])
        check("goal_is_not_time_truncation", not truncated.any())
        check("only_selected_robot_can_reset", writes == ([[0]] if cfg.reset_on_goal else []))
        check("episode_clocks_follow_partial_reset", env._episode_steps.tolist() == [0 if cfg.reset_on_goal else 13, 22])
        expected_integral = torch.full_like(env._rate_integral, .37)
        if cfg.reset_on_goal:
            expected_integral[0] = 0.
        check("controller_preserved_or_cleared_as_requested", torch.equal(env._rate_integral, expected_integral))
        check("robot_position_preserved_or_reset", abs(float(
            env.robot.data.root_pos_w[0, 1] - env.scene.env_origins[0, 1]) - (0. if cfg.reset_on_goal else .2)) < 1e-4)
        check("fresh_shot_only_after_full_reset", env._ball_active.tolist() == [cfg.reset_on_goal, True])
        check("other_env_launch_timer_is_untouched", env._next_launch_steps[1] == 9999)
        check("retired_goal_ball_kept_once", env._history_used[0].sum() == 1)
        env.step(actions)
        check("goal_not_scored_again_next_step", env.outcome_counts[3] == 1)
    finally:
        env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


try:
    main()
finally:
    app.close()
