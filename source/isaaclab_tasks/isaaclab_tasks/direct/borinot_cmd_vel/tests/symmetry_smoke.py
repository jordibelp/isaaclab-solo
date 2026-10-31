# SPDX-License-Identifier: BSD-3-Clause
"""Live matched-state world-heading equivariance; one controller per process."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel")
parser.add_argument("--mode", default="indi")
parser.add_argument("--sac", action="store_true")
parser.add_argument("--pairs", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, extra = parser.parse_known_args()
sys.argv = [sys.argv[0]]+extra
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz
from isaaclab_tasks.direct.borinot.symmetry import BorinotSymmetry, compute_symmetric_observations_actions
from isaaclab_tasks.utils.hydra import hydra_task_config
from tensordict import TensorDict


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent):
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.seed, cfg.sim.device = 2, 6., 42, args.device
    cfg.flight_xy_limit_m = 1.
    cfg.control_mode, cfg.sac_action_mapping = args.mode, args.sac
    cfg.predict_yaw_as_sincos = cfg.predict_thigh_joint_pos_as_sincos = cfg.predict_calf_joint_pos_as_sincos = args.pairs
    cfg.obs_include_base_lin_acc, cfg.include_events_randomization = True, False
    cfg.reset_xy_noise_m, cfg.reset_z_noise_m, cfg.reset_rpy_noise_degs = 0., 0., (0., 0., 0.)
    cfg.curriculum, cfg.actuator_delay, cfg.reset_grid_enabled = False, .01, False
    cfg.configure_control()
    transform = BorinotSymmetry(cfg)
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"mode": args.mode, "sac": args.sac, "pairs": args.pairs, "transforms": transform.transforms,
              "rotor_audit": transform.audit, "checks": [], "max_observation_error": 0., "max_reward_error": 0.}

    def check(name, condition):
        assert bool(condition), (name, report)
        report["checks"].append(name)

    try:
        env.reset()
        device = env.device
        q = quat_from_euler_xyz(torch.tensor([.2], device=device), torch.tensor([-.15], device=device), torch.tensor([.1], device=device))
        q = torch.cat((q, quat_mul(q.new_tensor([[0., 0., 0., 1.]]), q)))
        pose = torch.zeros(2, 7, device=device)
        pose[:, :3] = torch.tensor([[.1, -.2, 1.5], [-.1, .2, 1.5]], device=device)+env.scene.env_origins
        pose[:, 3:] = q
        velocity = torch.cat((quat_apply(q, q.new_tensor([[.1, -.12, .03]]).expand(2, -1)),
                              quat_apply(q, q.new_tensor([[.1, .15, -.2]]).expand(2, -1))), -1)
        env.robot.write_root_pose_to_sim(pose)
        env.robot.write_root_velocity_to_sim(velocity)
        env.robot.write_joint_state_to_sim(q.new_tensor([[.2, -.3]]).expand(2, -1),
                                           q.new_tensor([[-.1, .15]]).expand(2, -1), joint_ids=env.joint_ids)
        env.scene.update(env.physics_dt)
        env._commands[:] = q.new_tensor([.1, -.2, .05, .15])
        env._command_steps_left.fill_(100000)
        env._base_push_steps_until_next.fill_(100000)
        actions = torch.zeros(1, cfg.action_space, device=device)
        layout = transform.action_slices
        if "acceleration" in layout:
            actions[:, layout["acceleration"]] = q.new_tensor([.1, -.15, .05])
        if "rates" in layout:
            actions[:, layout["rates"]] = q.new_tensor([.05, -.03, .07])
        if "yaw" in layout:
            sl = layout["yaw"]
            actions[:, sl] = q.new_tensor([.4]) if sl.stop-sl.start == 1 else q.new_tensor([.921060994, .389418342])
        if args.mode in ("cbrt", "cbrt_indi") and not args.sac:
            actions[:, layout["collective"]] = env._hover_total
        actions = torch.cat((actions, transform.actions(actions, "world_yaw_180")))
        # Match hidden delayed commands as well as the visible last-action state.
        env.actions.copy_(actions)
        if env._actuator_delay._queue is not None:
            env._actuator_delay._queue[:] = actions
        td, augmented = compute_symmetric_observations_actions(env, TensorDict(env._get_observations(), [2]), actions)
        check("resolved_TensorDict_and_action_sizes", td.batch_size == (4,) and augmented.shape == (4, cfg.action_space))
        for step in range(16):
            obs, rewards, terminated, truncated, info = env.step(actions)
            check("finite_step", torch.isfinite(rewards).all() and all(torch.isfinite(x).all() for x in obs.values()))
            check("no_reset_in_matched_fixture", not terminated.any() and not truncated.any())
            predicted = transform.observations(obs["policy"][:1], "world_yaw_180")
            error = float((predicted-obs["policy"][1:]).abs().max())
            reward_error = float((rewards[0]-rewards[1]).abs())
            report["max_observation_error"] = max(report["max_observation_error"], error)
            report["max_reward_error"] = max(report["max_reward_error"], reward_error)
        check("world_rotation_matches_measured_next_state", report["max_observation_error"] < .005)
        check("world_rotation_preserves_reward", report["max_reward_error"] < 2e-5)
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2))
        print("SYMMETRY_OK="+json.dumps({k:v for k,v in report.items() if k not in ("checks", "rotor_audit")}), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
