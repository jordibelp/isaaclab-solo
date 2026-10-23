# SPDX-License-Identifier: BSD-3-Clause
"""Live Hydra/PhysX checks. Launch one process per control mode."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--output", required=True, type=Path)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "rsl_rl_sac_vendor"))

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env import BorinotGoalkeeperEnv
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, args.agent)
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, args.device, 42
    cfg.curriculum = False
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(cfg)
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "agent": args.agent, "control_mode": cfg.control_mode, "checks": []}

    def check(name, passed):
        passed = bool(passed)
        report["checks"].append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    try:
        obs, _ = env.reset()
        view = env.robot.root_physx_view
        check("registered_alias", gym.spec("Borinot-Cmd-Vel-Direct-v0").entry_point == gym.spec("Borinot-cmd-vel").entry_point)
        check("spaces_after_hydra", obs["policy"].shape == obs["critic"].shape == (8, cfg.observation_space)
              and env.action_space.shape == (8, cfg.action_space))
        check("same_actor_critic", torch.equal(obs["policy"], obs["critic"]))
        check("spawn_1_5m", torch.allclose(env.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
                                        torch.full((8,), cfg.robot_home[2], device=env.device)))
        nominal = torch.tensor(cfg.robot_home_quat, device=env.device).expand(8, -1)
        inverse = nominal.clone()
        inverse[:, 1:] *= -1
        relative = matrix_from_quat(quat_mul(inverse, env.robot.data.root_quat_w))
        reset_rpy = torch.stack((torch.atan2(relative[:, 2, 1], relative[:, 2, 2]),
            torch.asin((-relative[:, 2, 0]).clamp(-1, 1)), torch.atan2(relative[:, 1, 0], relative[:, 0, 0])), -1)
        bounds = torch.deg2rad(torch.tensor(cfg.reset_rpy_noise_degs, device=env.device))
        check("spawn_orientation_bounds", (reset_rpy.abs() <= bounds + 1e-6).all())
        check("spawn_unit_quaternions", torch.allclose(env.robot.data.root_quat_w.norm(dim=-1), torch.ones(8, device=env.device), atol=1e-6))
        check("spawn_velocity_zero", not env.robot.data.root_vel_w.any())
        if any(cfg.reset_rpy_noise_degs):
            axes = bounds > 0
            check("spawn_orientation_diversity", (reset_rpy[:, axes].std(0) > bounds[axes] * .05).all())
        else:
            check("zero_noise_exact_nominal", torch.equal(env.robot.data.root_quat_w, nominal))
        check("no_goal_or_ball", not env.scene.rigid_objects)
        check("four_filtered_robot_contact_sensors", len(env._body_contact_sensors) == 4
              and len(env.scene.sensors) == 4)
        check("physics_policy_clock", abs(env.physics_dt-1/cfg.physics_hz) < 1e-10
              and abs(env.step_dt-1/cfg.policy_hz) < 1e-10)
        use_pd = cfg.control_mode not in ("thrust_and_torque_control", "indi") or cfg.hold_arm
        check("live_gains", torch.allclose(view.get_dof_stiffnesses()[:, env.joint_ids], torch.full((8, 2), cfg.kp if use_pd else 0.))
              and torch.allclose(view.get_dof_dampings()[:, env.joint_ids], torch.full((8, 2), cfg.kd if use_pd else 0.)))
        check("live_effort_limits", torch.allclose(view.get_dof_max_forces()[:, env.joint_ids], torch.full((8, 2), cfg.effort_lim_sim)))
        check("armature_preserved", torch.allclose(view.get_dof_armatures()[:, env.joint_ids], torch.full((8, 2), .00036207)))
        masses, coms, friction = view.get_masses().clone(), view.get_coms().clone(), view.get_dof_friction_properties().clone()
        paddle = env.robot.find_bodies("paddle")[0][0]
        for tag, body in (("add_base_mass", env.base_id), ("racket_mass", paddle)):
            ratio = masses[:, body] / env.robot.data.default_mass[:, body].cpu()
            enabled = cfg.include_events_randomization and cfg.event_randomization[tag]
            check(f"{tag}_physical", ((ratio >= .9-1e-6).all() and (ratio <= 1.2+1e-6).all()
                  and ratio.std() > .001) if enabled else torch.equal(ratio, torch.ones_like(ratio)))
        check("nominal_hover_calibration", abs(env._hover_per_rotor - float(env.robot.data.default_mass[0].sum())*9.81/6) < 1e-7)
        enabled_friction = cfg.include_events_randomization and cfg.event_randomization["joint_friction"]
        check("joint_friction_physical", ((friction >= .01-1e-7).all() and (friction <= 1).all())
              if enabled_friction else not friction.any())
        base_delta = env._base_com_offset
        enabled_com = cfg.include_events_randomization and cfg.event_randomization["base_com"]
        check("base_com_physical", ((base_delta.abs() <= .010001).all() and base_delta.std(0).min() > .001)
              if enabled_com else not base_delta.any())

        # A tilted body must still receive a purely WORLD-Z disturbance.
        root = env.robot.data.root_state_w.clone()
        angles = torch.tensor([.3, -.4, .7], device=env.device).expand(8, -1)
        root[:, 3:7] = quat_from_euler_xyz(*angles.unbind(-1))
        root[:, 7:] = 0.
        env.robot.write_root_state_to_sim(root)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.actions.zero_()
        env._base_push_force_z_w[:] = torch.tensor([5., -5., 0., 3., -3., 0., 2., -2.], device=env.device)
        env._base_push_points_b[:] = torch.tensor([.08, -.04, .03], device=env.device)
        composer = env.robot.permanent_wrench_composer
        BorinotGoalkeeperEnv._apply_action(env)
        baseline_f = composer.composed_force_as_torch[:, env.base_id].clone()
        baseline_t = composer.composed_torque_as_torch[:, env.base_id].clone()
        # Freeze rotor lag for this composition-only comparison; restore it below.
        alpha = env._motor_alpha
        env._motor_alpha = 0.
        if cfg.control_mode == "indi":
            # INDI has its own physics-rate state. Capture the shared-control
            # baseline in the same call instead of advancing that state twice.
            saved_add = composer.add_forces_and_torques
            captured = {}
            def capture_add(**kwargs):
                captured["f"] = composer.composed_force_as_torch[:, env.base_id].clone()
                captured["t"] = composer.composed_torque_as_torch[:, env.base_id].clone()
                return saved_add(**kwargs)
            composer.add_forces_and_torques = capture_add
        env._apply_action()
        if cfg.control_mode == "indi":
            composer.add_forces_and_torques = saved_add
            baseline_f, baseline_t = captured["f"], captured["t"]
        env._motor_alpha = alpha
        delta_f = composer.composed_force_as_torch[:, env.base_id] - baseline_f
        delta_t = composer.composed_torque_as_torch[:, env.base_id] - baseline_t
        world_force = quat_apply(env.robot.data.root_quat_w, delta_f)
        expected_w = torch.zeros_like(world_force)
        expected_w[:, 2] = env._base_push_force_z_w
        check("pure_world_z_force_under_tilt", torch.allclose(world_force, expected_w, atol=4e-6, rtol=0))
        expected_t = torch.cross(env._base_push_points_b-env._base_com_b,
                                 quat_apply_inverse(env.robot.data.root_quat_w, expected_w), dim=-1)
        check("moment_about_actual_randomized_com", torch.allclose(delta_t, expected_t, atol=3e-7, rtol=0))
        env._base_push_force_z_w.zero_()
        env._apply_action()
        check("no_stale_disturbance", torch.allclose(composer.composed_force_as_torch[:, env.base_id, :2], torch.zeros(8, 2, device=env.device), atol=1e-7))

        env.reset()
        actions = torch.zeros(8, cfg.action_space, device=env.device)
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(actions)
            check("finite_step", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            check("all_substep_power_samples", env._power_samples == cfg.decimation)
            check("zero_push_stage", not env._base_push_force_z_w.any())
        check("body_velocity_obs", torch.equal(obs["policy"][:, :3], env.robot.data.root_lin_vel_b)
              and torch.equal(obs["policy"][:, 3:6], env.robot.data.root_ang_vel_b))
        offset = 3 if cfg.obs_include_base_lin_acc else 0
        check("previous_action_obs", torch.equal(obs["policy"][:, 10+offset:10+offset+cfg.action_space], env.actions))
        start = 10 + offset + cfg.action_space
        check("gravity_and_full_rotation_obs", torch.allclose(obs["policy"][:, start:start+3], env.robot.data.projected_gravity_b, atol=1e-6)
              and torch.equal(obs["policy"][:, start+3:start+12], matrix_from_quat(env.robot.data.root_quat_w).flatten(1)))
        check("command_obs", torch.equal(obs["policy"][:, start+12:start+16], env._commands))

        # Partial reset must clear selected force/controller/effort state only.
        env._base_push_force_z_w[:] = 2.
        env._base_push_steps_left[:] = 10
        remaining = torch.tensor([0, 2, 3, 4, 5, 6, 7], device=env.device)
        buffers = (env.actions, env._rotor_thrust, env._rate_integral, env._base_push_force_z_w,
                   env._base_push_steps_left, env._base_push_steps_until_next, env._commands)
        before = [v[remaining].clone() for v in buffers]
        root_before = env.robot.data.root_state_w[remaining].clone()
        old_reset_quat = env.robot.data.root_quat_w[1].clone()
        env._reset_idx(torch.tensor([1], device=env.device))
        check("partial_reset_isolation", all(torch.equal(v[remaining], old) for v, old in zip(buffers, before)))
        check("partial_reset_pose_isolation", torch.equal(env.robot.data.root_state_w[remaining], root_before))
        if any(cfg.reset_rpy_noise_degs):
            check("partial_reset_resamples_attitude", not torch.equal(env.robot.data.root_quat_w[1], old_reset_quat))
        check("partial_reset_velocity_zero", not env.robot.data.root_vel_w[1].any())
        check("partial_reset_clears_force_effort", env._base_push_force_z_w[1] == 0 and env._base_push_steps_left[1] == 0
              and not env.robot.data.joint_effort_target[1, env.joint_ids].any())
        check("randomization_survives_resets", torch.equal(masses, view.get_masses())
              and torch.equal(coms, view.get_coms()) and torch.equal(friction, view.get_dof_friction_properties()))
        if cfg.obs_include_base_lin_acc:
            check("reset_acceleration_zero", not env._get_observations()["policy"][1, 6:9].any())
        # Final interval must pay power/tracking before timeout resets its counters.
        env._episode_steps[:] = env.max_episode_length - 1
        _, reward, _, timeout, info = env.step(actions)
        check("automatic_timeout_and_energy_payment", timeout.all() and torch.isfinite(reward).all()
              and info["log"]["Episode/energy_j"] > 0 and not env._episode_steps.any())
        check("automatic_reset_clears_pushes", not env._base_push_force_z_w.any() and not env._base_push_steps_left.any())
        check("automatic_reset_unit_attitudes", torch.allclose(env.robot.data.root_quat_w.norm(dim=-1), torch.ones(8, device=env.device), atol=1e-6))
        report["reset_rpy_noise_degs"] = list(cfg.reset_rpy_noise_degs)
        report["nominal_quaternion_wxyz"] = list(cfg.robot_home_quat)
        report["initial_relative_rpy_degs"] = torch.rad2deg(reset_rpy).cpu().tolist()
        for field, value in (("reset_rpy_noise_degs", (-1., 5., 5.)), ("reset_rpy_noise_degs", (5., 5.)),
                             ("reset_rpy_noise_degs", (5., float('nan'), 5.))):
            invalid = cfg.copy()
            setattr(invalid, field, value)
            try:
                invalid.configure_control()
            except ValueError:
                check("invalid_attitude_noise_rejected", True)
            else:
                check("invalid_attitude_noise_rejected", False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASSED {len(report['checks'])} checks", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
