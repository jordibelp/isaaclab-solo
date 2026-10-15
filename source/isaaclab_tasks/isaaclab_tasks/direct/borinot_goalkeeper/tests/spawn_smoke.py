# SPDX-License-Identifier: BSD-3-Clause
"""Actual staged/partial/automatic resets and PhysX spawn-position readback."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
sys.argv = [sys.argv[0]] + overrides
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_goalkeeper.task_math import validate_reset_grid


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 84
    cfg.curriculum = False
    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    original_target = (cfg.keeper_plane_x, *cfg.target_center_yz)
    report = {"checks": [], "config": {name:getattr(cfg,name) for name in (
        "robot_home", "reset_grid_enabled", "reset_grid_start_level", "reset_grid_y_offsets_m",
        "reset_grid_z_offsets_m", "reset_xy_noise_m", "reset_z_noise_m", "shot_half_extents_yz")}}
    report["config"]["enabled_self_collisions"] = cfg.enabled_self_collisions

    def check(name, condition):
        passed = bool(condition)
        report["checks"].append({"name":name,"passed":passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    def readback(grid):
        env.sim.forward()
        env.scene.update(env.physics_dt)
        local = env.robot.root_physx_view.get_root_transforms()[:, :3] - env.scene.env_origins
        centers = torch.tensor(cfg.robot_home, device=env.device).expand(env.num_envs,-1).clone()
        if grid:
            ids = env.spawn_grid_index
            y = torch.tensor(cfg.reset_grid_y_offsets_m, device=env.device)
            z = torch.tensor(cfg.reset_grid_z_offsets_m, device=env.device)
            centers[:,1] += y[ids % len(y)]
            centers[:,2] += z[ids // len(y)]
            check("grid_indices_valid", ((ids >= 0) & (ids < len(y)*len(z))).all())
        else:
            check("centre_has_no_grid_index", (env.spawn_grid_index == -1).all())
        tolerance = torch.tensor([cfg.reset_xy_noise_m,cfg.reset_xy_noise_m,cfg.reset_z_noise_m],device=env.device)+5e-5
        check("physx_positions_match_selected_centres_and_noise", ((local-centers).abs() <= tolerance).all())
        check("no_spawn_outside_allowed_area", ((local >= env._area_min) & (local <= env._area_max)).all())
        check("reset_orientation_unchanged", torch.allclose(env.robot.data.root_quat_w,
              torch.tensor(cfg.robot_home_quat,device=env.device).expand(env.num_envs,-1),atol=1e-6))
        check("reset_velocities_zero", not env.robot.data.root_vel_w.any())
        ball = env.ball.data.root_pos_w - env.scene.env_origins
        velocity = env.ball.data.root_lin_vel_w
        time = (original_target[0]-ball[:,0])/velocity[:,0]
        at_plane = ball + velocity*time[:,None]
        at_plane[:,2] -= .5*9.81*time.square()
        extent = torch.tensor(cfg.shot_half_extents_yz[env.shot_level],device=env.device)
        centre = torch.tensor(original_target[1:],device=env.device)
        check("ball_aim_stays_in_fixed_world_target_rectangle", ((at_plane[:,1:]-centre).abs() <= extent+5e-5).all())
        return local

    try:
        env.reset()
        articulation_flags = {str(p.GetPath()):p.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
                              for p in env.sim.stage.Traverse()
                              if str(p.GetPath()).startswith("/World/envs/env_0/Robot")
                              and p.HasAttribute("physxArticulation:enabledSelfCollisions")}
        check("hydra_self_collision_flag_reaches_spawn_config", cfg.robot.spawn.articulation_props.enabled_self_collisions == cfg.enabled_self_collisions)
        check("spawned_articulation_has_requested_self_collision_flag", articulation_flags and all(v == cfg.enabled_self_collisions for v in articulation_flags.values()))
        report["articulation_self_collision_flags"] = articulation_flags
        check("no_observation_shape_change", env._get_observations()["policy"].shape == (env.num_envs,cfg.observation_space))
        # Both stages immediately before/at the configured activation boundary.
        for level in (0, max(0,cfg.reset_grid_start_level-1), cfg.reset_grid_start_level, len(cfg.shot_half_extents_yz)-1):
            env.shot_level = level
            env.reset()
            readback(cfg.reset_grid_enabled and level >= cfg.reset_grid_start_level)
        env.shot_level = cfg.reset_grid_start_level
        counts = torch.zeros(len(cfg.reset_grid_y_offsets_m)*len(cfg.reset_grid_z_offsets_m),device=env.device,dtype=torch.long)
        for _ in range(12):
            env.reset()
            counts += torch.bincount(env.spawn_grid_index,minlength=len(counts))
        check("all_spawn_cells_appear_in_real_resets", (counts > 0).all())
        report["grid_counts"] = counts.cpu().tolist()

        # Reset a non-contiguous subset and read positions directly from PhysX.
        before = env.robot.root_physx_view.get_root_transforms().clone()
        before_ball = env.ball.root_physx_view.get_transforms().clone()
        before_indices = env.spawn_grid_index.clone()
        chosen = torch.tensor([0,3,7],device=env.device)
        keep = torch.ones(env.num_envs,dtype=torch.bool,device=env.device);keep[chosen]=False
        env._reset_idx(chosen)
        env.sim.forward()
        check("partial_reset_keeps_other_robot_poses", torch.equal(env.robot.root_physx_view.get_root_transforms()[keep],before[keep]))
        check("partial_reset_keeps_other_ball_poses", torch.equal(env.ball.root_physx_view.get_transforms()[keep],before_ball[keep]))
        check("partial_reset_keeps_other_spawn_indices", torch.equal(env.spawn_grid_index[keep],before_indices[keep]))
        readback(True)

        # Promotion must change the same reset batch's spawn distribution and episode stage.
        # Take a real step first so completed-episode logging has its power packet.
        env.step(torch.zeros(env.num_envs,cfg.action_space,device=env.device))
        cfg.curriculum = True
        cfg.curriculum_min_steps = cfg.curriculum_min_episodes = 1
        cfg.curriculum_success_threshold = 0.0
        env.shot_level = cfg.reset_grid_start_level-1
        env._episode_stage[:] = env.shot_level
        env._stage_episodes = env._window_cursor = 0
        env._curriculum_window.zero_()
        env._stage_start_step = 0
        env.common_step_counter = 10
        env._shot_steps[:] = 1
        env._outcome[:] = 7
        env._reset_idx(env.robot._ALL_INDICES)
        check("promotion_happens_before_spawn_sampling", env.shot_level == cfg.reset_grid_start_level and (env.spawn_grid_index >= 0).all())
        check("promoted_episodes_record_new_stage", (env._episode_stage == env.shot_level).all())
        check("promotion_clears_stage_pure_checkpoint_gate", env.curriculum_stage_pure_episodes == 0)
        readback(True)

        cfg.curriculum = False
        env._shot_steps[:] = env.max_episode_length-1
        obs,reward,terminated,truncated,info = env.step(torch.zeros(env.num_envs,cfg.action_space,device=env.device))
        check("deadline_triggers_real_automatic_reset", terminated.all() and not truncated.any())
        check("automatic_reset_uses_grid_and_clears_episode_clock", (env.spawn_grid_index >= 0).all() and not env._shot_steps.any())
        check("finite_post_reset_observations", all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all())
        readback(True)

        cfg.reset_grid_enabled = False
        env.reset()
        readback(False)
        cfg.reset_grid_enabled = True
        cfg.reset_grid_y_offsets_m = (-.4,0.,.4)
        cfg.reset_grid_z_offsets_m = (-.3,0.,.3)
        cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
        validate_reset_grid(cfg)
        env.reset()
        readback(True)
        check("shot_target_remains_fixed", (cfg.keeper_plane_x,*cfg.target_center_yz) == original_target)
        report["passed"] = True
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        env.close()


try:
    main()
finally:
    app.close()
