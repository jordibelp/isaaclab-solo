# SPDX-License-Identifier: BSD-3-Clause
"""Live Hydra/PhysX integration for INDI in either goalkeeper task (one process per task)."""
import argparse
import inspect
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device, cfg.seed, cfg.curriculum = args.device, 42, False
    cfg.reset_grid_enabled = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "config": cfg.to_dict(), "checks": []}

    def check(name, condition):
        passed = bool(condition)
        report['checks'].append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    try:
        obs, _ = env.reset()
        check("hydra_nine_actions", cfg.control_mode == "indi" and env.action_space.shape[-1] == cfg.action_space == 9)
        check("hydra_nested_controller_cfg", cfg.indi.filter_cutoff_hz == env._indi.cfg.filter_cutoff_hz)
        check("hydra_tilt_method", cfg.indi_tilt_method in ("geom", "method_1", "method_2")
              and env._indi.task.indi_tilt_method == cfg.indi_tilt_method)
        from isaaclab_tasks.controllers.borinot_indi import IndiController, thrust_attitude, attitude_acceleration
        from isaaclab_tasks.direct.borinot_goalkeeper.indi import IndiController as legacy_controller
        check("shared_controller_identity", type(env._indi) is IndiController and legacy_controller is IndiController)
        check("shared_source_is_snapshotted", inspect.getsourcefile(IndiController) in env.get_code_snapshot_files())
        check("physics_policy_timing", abs(env.physics_dt-1/cfg.physics_hz) < 1e-10
              and abs(env.step_dt-1/cfg.policy_hz) < 1e-10)
        simple = 'Simple' in args.task
        velocity = args.task in ("Borinot-cmd-vel", "Borinot-Cmd-Vel-Direct-v0")
        actor_size, critic_size = (43, 43) if velocity else (53, 63 if simple else 65)
        check("actor_critic_dimensions", obs['policy'].shape == (args.num_envs, actor_size)
              and obs['critic'].shape == (args.num_envs, critic_size))
        view = env.robot.root_physx_view
        check("no_double_pd", not view.get_dof_stiffnesses()[:, env.joint_ids].any()
              and not view.get_dof_dampings()[:, env.joint_ids].any())
        check("effort_limit_preserved", torch.allclose(view.get_dof_max_forces()[:, env.joint_ids],
              torch.full((args.num_envs, 2), cfg.effort_lim_sim)))
        check("armature_preserved", torch.allclose(view.get_dof_armatures()[:, env.joint_ids],
              torch.full((args.num_envs, 2), .00036207)))
        report['nominal_inertia_kg_m2'] = env._indi.inertia.cpu().tolist()
        check("nominal_not_randomized_inertia", torch.allclose(env._indi.inertia,
              env.robot.data.default_inertia[0, env.base_id].reshape(3, 3).to(env.device)*cfg.indi.inertia_scale))
        check("reset_controller_observation", torch.allclose(env._indi.observation(),
              torch.tensor([1.]*6+[0.]*2, device=env.device).expand(args.num_envs, -1), atol=1e-6))
        # Exercise the live controller object at large, tiny and pi errors against
        # an independent SciPy oracle before resetting its state for the rollout.
        reference = torch.zeros(args.num_envs, 9, device=env.device)
        _, desired = thrust_attitude(reference[:, :3],
            torch.full((args.num_envs,), env._indi.home_yaw, device=env.device), env._indi.mass)
        errors = Rotation.from_rotvec(np.resize(np.array([
            [0., 0., 0.], [1.2, -.8, .9], [np.pi-1e-5, 0., 0.], [0., 0., 1e-7]]), (args.num_envs, 3)))
        desired_scipy = Rotation.from_matrix(desired.cpu().numpy())
        current = desired_scipy * errors.inv()
        test_rates = torch.zeros(args.num_envs, 3, device=env.device)
        commands, _ = env._indi.step(reference,
            torch.tensor(current.as_quat(scalar_first=True), dtype=torch.float32, device=env.device),
            test_rates, test_rates[:, :2], test_rates[:, :2])
        tau_z = np.zeros((args.num_envs, 3))
        if cfg.indi_tilt_method == "geom":
            z = errors.as_matrix()[:, :, 2]
            cross = np.column_stack((-z[:, 1], z[:, 0], np.zeros(args.num_envs)))
            sine = np.linalg.norm(cross, axis=-1)
            angle = np.arctan2(sine, z[:, 2])
            scale = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 1e-14)
            tilt = cross*scale[:, None]
            tau_z = (Rotation.from_rotvec(tilt).inv()*errors).as_rotvec()
        else:
            tau_z[:, 2] = errors.as_rotvec()[:, 2]
            tilt = ((errors*Rotation.from_rotvec(-tau_z)).as_rotvec()
                    if cfg.indi_tilt_method == "method_1" else errors.as_rotvec()-tau_z)
        expected = cfg.indi.tilt_gain*tilt + cfg.indi.yaw_gain*tau_z
        error_max = np.max(np.abs(env._indi.alpha_desired.cpu().numpy()-expected))
        report['tangent_alpha_max_error_rad_s2'] = float(error_max)
        check("live_tangent_feedback_matches_scipy", error_max < 3e-4)
        check("large_error_commands_bounded", torch.isfinite(commands).all()
              and commands.min() >= 0 and commands.max() <= env._indi.allocator.cap+1e-5)
        env._indi.reset(slice(None))
        calls = []
        original_step = env._indi.step

        def traced_step(actions, quaternion, rates, joint_pos, joint_vel):
            before = env._indi.joint_integral.clone()
            commands, efforts = original_step(actions, quaternion, rates, joint_pos, joint_vel)
            u = policy_actions.clamp(-1, 1) if cfg.sac_action_mapping else policy_actions
            acceleration = u[:, :3]*env._indi.accel_scale if cfg.sac_action_mapping else u[:, :3]
            reference_rates = u[:, 3:6]*env._indi.rate_scale if cfg.sac_action_mapping else u[:, 3:6]
            heading = env._indi.home_yaw + u[:, 6]*(np.deg2rad(cfg.indi.heading_range_degs) if cfg.sac_action_mapping else 1.)
            _, desired_rotation = thrust_attitude(acceleration, heading, env._indi.mass)
            expected_alpha = attitude_acceleration(quaternion, desired_rotation, rates, reference_rates,
                                                   cfg.indi, cfg.indi_tilt_method)
            target = u[:, 7:9]*env._indi.joint_scale if cfg.sac_action_mapping else u[:, 7:9]
            if cfg.hold_arm:
                target = torch.zeros_like(target)
            error = target-joint_pos
            integral = (before+env.physics_dt*error).clamp(-cfg.indi.joint_integral_limit, cfg.indi.joint_integral_limit)
            effort = cfg.kp*error-cfg.kd*joint_vel+cfg.indi.joint_ki*integral
            frozen = ((effort > cfg.effort_lim_sim) & (error > 0)) | ((effort < -cfg.effort_lim_sim) & (error < 0))
            expected_integral = torch.where(frozen, before, integral)
            expected = (cfg.kp*error-cfg.kd*joint_vel+cfg.indi.joint_ki*expected_integral).clamp(-cfg.effort_lim_sim, cfg.effort_lim_sim)
            calls.append((torch.allclose(efforts, expected, atol=1e-6), commands.min(), commands.max(),
                          torch.equal(actions, u), torch.allclose(env._indi.alpha_desired, expected_alpha, atol=1e-5)))
            return commands, efforts

        env._indi.step = traced_step
        policy_actions = torch.tensor([24., -32., 48., 6., -7., 9., 4.2, .01, -.01],
                                      device=env.device).expand(args.num_envs, -1).clone()
        report['policy_actions'] = policy_actions.cpu().tolist()
        report['policy_action_units'] = 'normalized SAC' if cfg.sac_action_mapping else 'physical PPO'
        max_residual = torch.zeros(4, device=env.device)
        for _ in range(16):
            calls.clear()
            obs, reward, terminated, truncated, info = env.step(policy_actions)
            check("finite_rollout", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            check("fresh_pid_every_physics_step", len(calls) == cfg.decimation and all(item[0] for item in calls))
            check("ppo_raw_or_sac_clipped_references_reach_llc", all(item[3] for item in calls))
            check("ppo_si_or_sac_scaled_units_match_feedback", all(item[4] for item in calls))
            check("bounded_rotors", all(item[1] >= 0 and item[2] <= env._indi.allocator.cap+1e-5 for item in calls))
            check("bounded_joint_efforts", env.robot.data.joint_effort_target[:, env.joint_ids].abs().max() <= cfg.effort_lim_sim)
            max_residual = torch.maximum(max_residual, env._indi.allocation_residual.abs().max(0).values)
        report['max_eq16_residual_N_Nm'] = max_residual.cpu().tolist()
        check("software_pid_target_reaches_physx", torch.allclose(
              env.robot.data.applied_torque[:, env.joint_ids], env.robot.data.joint_effort_target[:, env.joint_ids], atol=1e-6))
        ids = torch.tensor([1], device=env.device)
        remaining = torch.tensor([i for i in range(args.num_envs) if i != 1], device=env.device)
        buffers = (env._indi.rotor_speed, env._indi.command_speed, env._indi.gyro_filter.z1,
                   env._indi.gyro_filter.z2, env._indi.rotor_filter.z1, env._indi.rotor_filter.z2,
                   env._indi.filtered_rates, env._indi.filtered_speeds, env._indi.joint_integral)
        saved = [buffer[remaining].clone() for buffer in buffers]
        env._reset_idx(ids)
        check("partial_reset_isolation", all(torch.equal(buffer[remaining], old) for buffer, old in zip(buffers, saved)))
        check("partial_reset_steady_filters", not env._indi.joint_integral[ids].any()
              and not env._indi.angular_acceleration[ids].any()
              and torch.allclose(env._indi.observation()[ids], torch.tensor([[1.]*6+[0.]*2], device=env.device), atol=1e-6))
        check("partial_reset_clears_effort", not env.robot.data.joint_effort_target[ids][:, env.joint_ids].any())
        # The standard timeout path must reset controller state too, not only explicit resets.
        env.episode_length_buf[:] = env.max_episode_length-1
        env._episode_steps[:] = env.max_episode_length-1
        policy_actions.zero_()
        _, _, _, truncated, _ = env.step(policy_actions)
        check("automatic_timeout_reset", truncated.all() and not env._indi.joint_integral.any()
              and not env._indi.angular_acceleration.any())
        report['passed'] = all(item['passed'] for item in report['checks'])
    finally:
        # Raw configs may contain callable event terms; keep JSON as an evidence snapshot only.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str)+"\n")
        env.close()


try:
    main()
finally:
    app.close()
