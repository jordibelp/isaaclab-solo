# SPDX-License-Identifier: BSD-3-Clause
"""Real Hydra overrides, actuator property readback, interval sampling, and resets."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import UsdGeom
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.curriculum = False
    # A known in-area fixture, without modifying Jordi's tuned home in source.
    cfg.robot_home = (0.0, 0.0, 1.5)
    cfg.robot.init_state.pos = cfg.robot_home
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    report = {"checks": [], "control_mode": cfg.control_mode, "physics_hz": cfg.physics_hz,
              "policy_hz": cfg.policy_hz, "decimation": cfg.decimation, "action_space": cfg.action_space,
              "observation_space": cfg.observation_space, "hold_arm": cfg.hold_arm,
              "paddle_diameter_m": cfg.paddle_diameter_m,
              "paddle_mount_delta_m": cfg.paddle_mount_delta_m,
              "effort_lim_sim": cfg.effort_lim_sim, "velocity_lim_sim": cfg.velocity_lim_sim}

    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        print(f"CHECK {name}: {bool(condition)}", flush=True)
        assert condition, name

    def properties():
        view = env.robot.root_physx_view
        return {"stiffness": view.get_dof_stiffnesses()[:, env.joint_ids],
                "damping": view.get_dof_dampings()[:, env.joint_ids],
                "armature": view.get_dof_armatures()[:, env.joint_ids],
                "effort_limit": view.get_dof_max_forces()[:, env.joint_ids],
                "velocity_limit": view.get_dof_max_velocities()[:, env.joint_ids],
                "friction": view.get_dof_friction_properties()[:, env.joint_ids, 0]}

    samples = []
    original_sample = env._sample_power

    def traced_sample():
        if env._power_samples < env._physics_samples:
            thrust = env._rotor_thrust.clone()
            torque = env.robot.data.applied_torque[:, env.joint_ids].clone()
            velocity = env.robot.data.joint_vel[:, env.joint_ids].clone()
            rotor = (cfg.km * (thrust / cfg.kf).sqrt() ** 3).sum(-1)
            samples.append(torch.stack((rotor, rotor * (1 / cfg.p_el_motor_efficiency - 1),
                                        (torque * velocity).sum(-1),
                                        cfg.arm_copper_loss_coefficient * torque.square().sum(-1)), -1))
        original_sample()

    env._sample_power = traced_sample
    try:
        obs, _ = env.reset()
        head = UsdGeom.Cylinder(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/paddle/geometry/mesh"))
        check("hydra_paddle_diameter_reaches_physics", abs(head.GetRadiusAttr().Get()*2 - cfg.paddle_diameter_m) < 1e-7)
        mount = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/paddle_fixed_joint").GetAttribute("physics:localPos0").Get()
        check("hydra_mount_delta_reaches_physics", abs(mount[1] - (cfg.paddle_offset_ee[1]-cfg.paddle_mount_delta_m)) < 1e-7)
        direct = cfg.control_mode != "PI_rotors_PD_joint_pos_control"
        use_pd = cfg.control_mode != "thrust_and_torque_control" or cfg.hold_arm
        check("spaces_follow_selected_mode", cfg.action_space == (8 if direct else 6)
              and all(value.shape == (args.num_envs, 49 + cfg.action_space) for value in obs.values())
              and env.action_space.shape[-1] == cfg.action_space)
        check("hydra_frequencies_reach_simulator", abs(env.physics_dt - 1/cfg.physics_hz) < 1e-10
              and abs(env.step_dt - 1/cfg.policy_hz) < 1e-10)
        check("deadline_uses_policy_frequency", env.max_episode_length == 3*cfg.policy_hz)
        expected = {"stiffness": cfg.joint_stiffness if use_pd else 0.0,
                    "damping": cfg.joint_damping if use_pd else 0.0, "armature": .00036207,
                    "effort_limit": cfg.effort_lim_sim, "velocity_limit": cfg.velocity_lim_sim, "friction": 0.0}
        before = properties()
        for name, value in expected.items():
            check(f"physx_{name}", torch.allclose(before[name], torch.full_like(before[name], value), atol=1e-7, rtol=1e-5))
        report["physx_properties"] = {name: value.tolist() for name, value in before.items()}
        actions = torch.zeros(args.num_envs, cfg.action_space, device=env.device)
        actions[:, -2:] = torch.tensor([.15, -.2], device=env.device)
        if direct:
            actions[:, :6] = torch.tensor([-.1, .05, 0., .1, -.05, 0.], device=env.device)
        for _ in range(8):
            samples.clear()
            obs, reward, terminated, truncated, info = env.step(actions)
            check("finite_step", torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values()))
            check("every_completed_substep_sampled", len(samples) == cfg.decimation)
            expected_power = torch.stack(samples).mean(0)
            actual_power = torch.stack([info["power_mean_w"][name] for name in
                ("rotor_mechanical", "rotor_dissipative", "arm_mechanical", "arm_copper_loss")], -1)
            check("physics_rate_power_mean", torch.allclose(actual_power, expected_power, atol=1e-5, rtol=1e-5))
        check("pi_state_only_changes_in_pi_mode", bool(env._rate_integral.any()) if not direct else not bool(env._rate_integral.any()))
        if use_pd:
            target = torch.zeros_like(actions[:, -2:]) if cfg.hold_arm else actions[:, -2:] * env._joint_scale
            check("pd_angle_targets", torch.allclose(env.robot.data.joint_pos_target[:, env.joint_ids], target))
            check("pd_has_no_feedforward_effort", not bool(env.robot.data.joint_effort_target[:, env.joint_ids].any()))
        else:
            check("signed_torque_targets", torch.allclose(env.robot.data.joint_effort_target[:, env.joint_ids],
                                                          actions[:, -2:] * cfg.effort_lim_sim))
            check("torque_power_active", bool((actual_power[:, 3] > 0).all()))
        untouched = env.robot.data.joint_effort_target[1:].clone()
        untouched_actions = env.actions[1:].clone()
        untouched_rotors = env._rotor_thrust[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_clears_effort_and_actions", not bool(env.robot.data.joint_effort_target[0].any())
              and not bool(env.actions[0].any()))
        check("partial_reset_leaves_other_commands", torch.equal(untouched, env.robot.data.joint_effort_target[1:])
              and torch.equal(untouched_actions, env.actions[1:]) and torch.equal(untouched_rotors, env._rotor_thrust[1:]))
        env._shot_steps[:] = env.max_episode_length - 1
        samples.clear()
        _, _, terminated, truncated, info = env.step(actions)
        check("real_task_deadline_no_bootstrap", bool(terminated.all()) and not bool(truncated.any()))
        check("final_substep_before_auto_reset", len(samples) == cfg.decimation
              and bool((info["power_mean_w"]["rotor_mechanical"] > 0).all()))
        check("auto_reset_clears_commands_and_power", not bool(env.actions.any())
              and not bool(env.robot.data.joint_effort_target.any()) and not bool(env._power_sum_w.any()))
        for name, value in properties().items():
            check(f"{name}_persists_after_resets", torch.equal(value, before[name]))
        report["passed"] = True
        print("CONTROL_VALIDATION=" + json.dumps(report), flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
