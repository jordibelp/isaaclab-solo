# SPDX-License-Identifier: BSD-3-Clause
"""Real-physics checks for the playback-only ball pool and home reset."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--keep-balls", type=int, default=5)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg
from isaaclab_tasks.direct.borinot_goalkeeper.playback import BorinotGoalkeeperPlayEnv


def main():
    cfg = BorinotGoalkeeperEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.seed = 53
    cfg.curriculum = False
    cfg.reset_grid_enabled = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
    cfg.hold_arm = True
    cfg.reset_robot_on_robot_crash = cfg.reset_robot_on_tilt_threshold = False
    cfg.shot_resampling_time_range_s = (100., 100.)
    env = BorinotGoalkeeperPlayEnv(cfg, keep_balls=args.keep_balls)
    checks = []

    def check(name, passed):
        passed = bool(passed)
        checks.append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    ids = torch.tensor([0], device=env.device)
    actions = torch.zeros((2, cfg.action_space), device=env.device)
    try:
        obs, _ = env.reset()
        check("unchanged_actor_critic_shapes", obs["policy"].shape == (2, 44) and obs["critic"].shape == (2, 56))
        check("bounded_pool", len(env.history_balls) == args.keep_balls)
        untouched = env.ball.data.root_state_w[1].clone()
        # Tag twelve trajectories by position; this wraps the five-ball pool.
        states = []
        for i in range(12):
            state = env.ball.data.root_state_w[ids].clone()
            state[:, :3] = torch.tensor([1. + .01*i, .8, 1.8], device=env.device)
            state[:, 7:] = torch.tensor([1., .2, .4, .1, .2, .3], device=env.device)
            env.ball.write_root_state_to_sim(state, ids)
            states.append(state.clone())
            env._ball_active[ids] = False
            env._park_ball(ids)
            if args.keep_balls:
                newest = (int(env._history_cursor[0]) - 1) % args.keep_balls
                history = env.history_balls[newest]
                check(f"exact_post_contact_state_{i}", torch.equal(history.data.root_state_w[ids], state))
                check(f"randomized_mass_{i}", torch.allclose(history.root_physx_view.get_masses()[ids.cpu()],
                      env._ball_masses[ids.cpu()], rtol=1e-6, atol=1e-8))
                check(f"randomized_inertia_{i}", torch.allclose(history.root_physx_view.get_inertias()[ids.cpu()],
                      env._ball_inertias[ids.cpu()], rtol=1e-6, atol=1e-9))
                check(f"randomized_material_{i}", torch.equal(history.root_physx_view.get_material_properties()[ids.cpu()], env._ball_materials[ids.cpu()]))
                check(f"bounded_visible_after_result_{i}", env._history_used[0].sum() <= args.keep_balls)
            if i < 11:
                env._launch_ball(ids)
                check(f"room_for_live_shot_{i}", env._history_used[0].sum() + 1 <= max(args.keep_balls, 1))
        check("partial_updates_leave_env1_untouched", torch.equal(env.ball.data.root_state_w[1], untouched)
              and not env._history_used[1].any())
        if args.keep_balls:
            actual = sorted(round(float(ball.data.root_pos_w[0, 0]), 4) for i, ball in enumerate(env.history_balls)
                            if env._history_used[0, i])
            expected = [round(float(state[0, 0]), 4) for state in states[-args.keep_balls:]]
            check("only_newest_n_trajectories_retained", actual == expected)

            ghost = env.history_balls[(int(env._history_cursor[0]) - 1) % args.keep_balls]
            # Traverse the body on the same line that makes a live ball hit it.
            state = ghost.data.root_state_w[ids].clone()
            state[:, :3] = env.robot.data.root_pos_w[ids] + torch.tensor([.38, 0., 0.], device=env.device)
            state[:, 7:] = 0.
            state[:, 7] = -3.
            state[:, 9] = .62
            ghost.write_root_state_to_sim(state, ids)
            env._next_launch_steps[:] = 10000
            env._ball_active[:] = False
            for _ in range(25):
                env.step(actions)
            check("old_ball_passes_robot_without_impulse", abs(float(ghost.data.root_lin_vel_w[0, 0]) + 3.) < 1e-4)
            check("old_ball_cannot_score", env.outcome_counts.sum() == 0)

            # Cross an old ball and a live ball head-on away from the robot.
            state = ghost.data.root_state_w[ids].clone()
            state[:, :3] = torch.tensor([1., .8, 1.8], device=env.device) + env.scene.env_origins[ids]
            state[:, 7:] = 0.
            state[:, 7] = 2.
            ghost.write_root_state_to_sim(state, ids)
            live_state = state.clone()
            live_state[:, 0] += .15
            live_state[:, 7] = -2.
            env.ball.write_root_state_to_sim(live_state, ids)
            env._previous_ball[ids] = live_state[:, :3] - env.scene.env_origins[ids]
            env._ball_active[ids] = True
            for _ in range(8):
                env.step(actions)
            check("old_ball_does_not_deflect_live_ball", abs(float(env.ball.data.root_lin_vel_w[0, 0]) + 2.) < 1e-4
                  and abs(float(ghost.data.root_lin_vel_w[0, 0]) - 2.) < 1e-4)
            env._ball_active[ids] = False
            super(BorinotGoalkeeperPlayEnv, env)._park_ball(ids)

            # A physical floor bounce must reverse downward velocity. Keep the
            # live ball parked, so this checks only the retired ball's physics.
            state = ghost.data.root_state_w[ids].clone()
            state[:, :3] = torch.tensor([1., .8, .08], device=env.device) + env.scene.env_origins[ids]
            state[:, 7:] = 0.
            state[:, 9] = -1.
            ghost.write_root_state_to_sim(state, ids)
            bounced = False
            for _ in range(12):
                env.step(actions)
                bounced |= float(ghost.data.root_lin_vel_w[0, 2]) > .1
            check("old_ball_still_bounces_on_floor", bounced)
            state[:, :3] = torch.tensor([-.2, .8, 1.], device=env.device) + env.scene.env_origins[ids]
            state[:, 7:] = 0.
            state[:, 7] = -3.
            ghost.write_root_state_to_sim(state, ids)
            bounced = False
            for _ in range(15):
                env.step(actions)
                bounced |= float(ghost.data.root_lin_vel_w[0, 0]) > .1
            check("old_ball_still_bounces_on_goal", bounced)

        # Home reset must override stage-grid/noise and clear all dynamics and
        # controller targets. It is not an extra shot result.
        cfg.reset_grid_enabled = True
        env.shot_level = len(cfg.shot_half_extents_yz) - 1
        cfg.reset_xy_noise_m = cfg.reset_z_noise_m = .01
        env.actions.fill_(.7)
        env._rate_integral.fill_(.5)
        before = env.outcome_counts.clone()
        env.reset_borinot()
        check("button_resets_exact_home_not_random_grid", torch.allclose(
            env.robot.data.root_pos_w-env.scene.env_origins, env._home.expand(2, -1), atol=1e-6))
        check("button_clears_velocity_and_controller", torch.count_nonzero(env.robot.data.root_vel_w) == 0
              and torch.count_nonzero(env.actions) == 0 and torch.count_nonzero(env._rate_integral) == 0)
        check("reset_does_not_change_score", torch.equal(env.outcome_counts, before))
        check("reset_starts_fresh_live_shot", env._ball_active.all() and (env._shot_steps == 0).all())
    finally:
        env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"keep_balls": args.keep_balls, "device": args.device, "checks": checks}, indent=2))


try:
    main()
finally:
    app.close()
