# SPDX-License-Identifier: BSD-3-Clause
"""Independent live momentum, moment, and curriculum checks; optional rendered scene."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--image", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.image:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg


def main():
    cfg = BorinotCmdVelEnvCfg()
    cfg.scene.num_envs, cfg.sim.device = 4, args.device
    cfg.physics_hz = cfg.policy_hz = 200
    cfg.seed, cfg.include_events_randomization, cfg.curriculum = 42, False, False
    cfg.reset_xy_noise_m, cfg.standing_env_prob = 0.0, 1.0
    cfg.viewer.origin_type, cfg.viewer.env_index = "asset_root", 0
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye, cfg.viewer.lookat = (3.0, 3.0, 1.5), (0.0, 0.0, 0.0)
    cfg.viewer.resolution = (1280, 720)
    env = gym.make("Borinot-cmd-vel", cfg=cfg, render_mode="rgb_array" if args.image else None).unwrapped
    report = {"checks": []}

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(f"CHECK {name}: {bool(passed)}", flush=True)
        assert passed, name

    def momentum_velocity():
        masses = env.robot.data.default_mass.to(env.device)
        return (masses[..., None] * env.robot.data.body_lin_vel_w).sum(1) / masses.sum(1, keepdim=True)

    try:
        env.reset()
        action = torch.zeros(4, cfg.action_space, device=env.device)
        env._base_push_steps_until_next[:] = 10000
        env._base_push_steps_left[:] = 1
        env._base_push_force_z_w[:] = torch.tensor([5., -5., 0., 0.], device=env.device)
        env._base_push_points_b.copy_(env._base_com_b)
        before = momentum_velocity().clone()
        env.step(action)
        change = momentum_velocity() - before
        expected = 5 * env.step_dt / env.robot_mass_kg
        observed = change[[0, 1]] - change[2]
        target = torch.tensor([[0., 0., expected], [0., 0., -expected]], device=env.device)
        report["delta_v_world_m_s"] = observed.cpu().tolist()
        report["expected_delta_v_z_m_s"] = expected
        check("physx_momentum_matches_vertical_force", torch.allclose(observed, target, atol=1e-5, rtol=0))

        env.reset()
        env._base_push_steps_until_next[:] = 10000
        env._base_push_steps_left[:] = 1
        env._base_push_force_z_w[:] = torch.tensor([5., 5., 0., 0.], device=env.device)
        env._base_push_points_b.copy_(env._base_com_b)
        env._base_push_points_b[0, 0] += .1
        env.step(action)
        delta_rate = env.robot.data.root_ang_vel_b[0] - env.robot.data.root_ang_vel_b[1]
        report["offset_minus_center_body_rate_rad_s"] = delta_rate.cpu().tolist()
        check("offset_vertical_force_physically_pitches_base", delta_rate[1] < -.001)

        env.reset()
        cfg.curriculum, cfg.curriculum_min_episodes = True, 1
        env._curriculum_window.zero_()
        env._window_cursor = env._stage_episodes = 0
        env.common_step_counter = cfg.curriculum_min_steps
        env._episode_steps[0], env._episode_xy_integral[0] = env.max_episode_length, cfg.episode_length_s
        env._reset_idx(torch.tensor([0], device=env.device))
        check("stage_zero_promotes_to_five_newton_limit", env.force_level == 1 and env._episode_stage[0] == 1
              and (env._episode_stage[1:] == 0).all())
        env._base_push_steps_until_next[:] = 0
        env._update_pushes()
        check("new_stage_pushes_old_episodes_stay_force_free", 0 < env._base_push_force_z_w[0].abs() <= 5
              and not env._base_push_force_z_w[1:].any())
        env.common_step_counter += cfg.curriculum_min_steps
        env._episode_steps[1], env._episode_xy_integral[1] = env.max_episode_length, cfg.episode_length_s
        env._reset_idx(torch.tensor([1], device=env.device))
        check("old_stage_success_cannot_promote_again", env.force_level == 1 and env._stage_episodes == 0)
        env._episode_steps[0], env._episode_xy_integral[0] = env.max_episode_length, cfg.episode_length_s
        env._reset_idx(torch.tensor([0], device=env.device))
        check("current_stage_success_promotes_to_eight_newton_limit", env.force_level == 2
              and env._episode_stage[0] == 2 and env._force_limits[2] == 8)

        for key, value in (("tracking_lin_std", 0.0), ("command_lin_vel_z_range", (1., -1.)),
                           ("base_push_duration_range_s", (.5, 3.)), ("curriculum_start_level", 9),
                           ("forces_applied_to_base_curriculum", (0., -1.)), ("policy_hz", 70)):
            invalid = cfg.copy()
            setattr(invalid, key, value)
            try:
                invalid.configure_control()
            except ValueError:
                check(f"invalid_{key}_rejected", True)
            else:
                check(f"invalid_{key}_rejected", False)

        if args.image:
            env.reset()
            import numpy as np
            from PIL import Image
            frame = None
            for _ in range(15):
                env.sim.render()
                frame = env.render(recompute=True)
            check("rendered_frame_has_scene_content", frame is not None and frame.shape == (720, 1280, 3)
                  and float(np.std(frame)) > 5)
            args.image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(args.image)
            report["scene_image"] = str(args.image)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASSED {len(report['checks'])} checks", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
