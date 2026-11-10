# SPDX-License-Identifier: BSD-3-Clause
"""Registered Hydra/PhysX proof of hybrid modes and completed-interval inputs."""
import argparse
import inspect
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]]+hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.controllers.borinot_angles import angle_action_slots
from isaaclab_tasks.controllers.borinot_model_wrench_indi import ModelWrenchIndiController, ModelWrenchYawlessIndiController
from isaaclab_tasks.controllers.borinot_indi import rotation_from_quat
from isaaclab_tasks.controllers.borinot_wrench import matvec
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed, cfg.curriculum = 4, args.device, 42, False
    cfg.reset_grid_enabled = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "config": cfg.to_dict(), "checks": [], "status": "incomplete"}

    def check(name, value):
        passed = bool(value)
        report["checks"].append({"name": name, "passed": passed})
        assert passed, name

    try:
        obs, _ = env.reset()
        ctrl, data, view = env._wrench, env.robot.data, env.robot.root_physx_view
        report["model"] = ctrl.model.description
        cls = ModelWrenchYawlessIndiController if "yawless" in cfg.control_mode else ModelWrenchIndiController
        check("hybrid_controller", type(ctrl) is cls)
        check("no_hip_sensor", not hasattr(env, "_wrench_thigh_id"))
        check("source_snapshot", inspect.getsourcefile(cls) in env.get_code_snapshot_files())
        check("actor_shape", obs["policy"].shape == (4, cfg.observation_space))
        check("critic_shape", obs["critic"].shape == (4, cfg.state_space))
        check("zero_implicit_pd", not view.get_dof_stiffnesses()[:, env.joint_ids].any()
              and not view.get_dof_dampings()[:, env.joint_ids].any())
        scalar = torch.zeros(4, 8 if "yawless" in cfg.control_mode else 9, device=env.device)
        scalar[:, -2:] = torch.tensor([.025, -.02], device=env.device)
        parts, start = [], 0
        for slot in angle_action_slots(cfg):
            value = scalar[:, slot]
            if cfg.sac_action_mapping and slot >= len(scalar[0])-2:
                value = value*ctrl.joint_scale[slot-(len(scalar[0])-2)]
            parts += [scalar[:, start:slot], torch.stack((value.cos(), value.sin()), -1)]
            start = slot+1
        actions = torch.cat(parts+[scalar[:, start:]], -1)
        original = ctrl.step
        raw_peaks = []
        transient_peaks = []
        last_transient = torch.zeros_like(ctrl.transient_torque)
        original_wrench = env.robot.permanent_wrench_composer.set_forces_and_torques

        def traced_wrench(**kwargs):
            expected = env._rotor_thrust @ env._allocation.T
            expected[:, 1:] += last_transient
            expected[:, 1] -= env._base_com_offset[:, 1]*expected[:, 0]
            expected[:, 2] += env._base_com_offset[:, 0]*expected[:, 0]
            check("plant_includes_spin_acceleration", torch.allclose(
                kwargs["torques"][:, 0], expected[:, 1:], atol=2e-5, rtol=2e-5))
            return original_wrench(**kwargs)

        env.robot.permanent_wrench_composer.set_forces_and_torques = traced_wrench

        def traced(action, quat, rates, q, dq, velocity_w):
            rotation = rotation_from_quat(quat)
            check("base_com_velocity_input", torch.equal(velocity_w, data.body_com_lin_vel_w[:, env.base_id]))
            valid = ctrl.measurement_ready.clone()
            dv = (velocity_w-ctrl.previous_velocity_w)/ctrl.dt
            dw = (matvec(rotation, rates)-ctrl.previous_omega_w)/ctrl.dt
            previous = ctrl.previous_rotation.clone()
            acc = torch.cat((matvec(previous.mT, dv), matvec(previous.mT, dw)), -1)
            expected_raw = matvec(previous, matvec(ctrl.previous_effective_row, acc)
                                 +ctrl.previous_effective_bias-ctrl.applied_wrench[:, 1:])
            expected_raw = torch.where(valid[:, None], expected_raw, 0.)
            speed = ctrl.rotor_speed.clone()
            commands, effort = original(action, quat, rates, q, dq, velocity_w)
            check("completed_interval_residual", torch.allclose(ctrl.raw_disturbance_w, expected_raw, atol=2e-5, rtol=2e-5))
            check("torque_correction_sign", torch.allclose(ctrl.requested_wrench[:, 1:],
                  ctrl.nominal_wrench[:, 1:]-ctrl.disturbance_b, atol=1e-6, rtol=1e-6))
            check("collective_not_corrected", torch.equal(ctrl.requested_wrench[:, :1], ctrl.nominal_wrench[:, :1]))
            check("one_speed_update", torch.allclose(ctrl.rotor_speed,
                  speed+ctrl.motor_alpha*((commands/cfg.kf).sqrt()-speed), atol=1e-5, rtol=1e-6))
            last_transient.zero_()
            last_transient[:, 2] = (((ctrl.rotor_speed-speed)/ctrl.dt)
                *torch.tensor(cfg.rotor_spin, device=env.device)*cfg.indi.rotor_spin_inertia_kg_m2).sum(-1)
            check("spin_acceleration_torque", torch.allclose(ctrl.transient_torque, last_transient,
                  atol=2e-5, rtol=2e-5))
            applied = matvec(ctrl.allocator.allocation, cfg.kf*ctrl.rotor_speed.square())
            applied[:, 1:] += last_transient
            check("applied_rotor_wrench", torch.allclose(ctrl.applied_wrench,
                  applied, atol=2e-5, rtol=2e-5))
            check("finite_correction", torch.isfinite(ctrl.disturbance_b).all())
            raw_peaks.append(float(ctrl.raw_disturbance_w.abs().max()))
            transient_peaks.append(float(last_transient.abs().max()))
            return commands, effort

        ctrl.step = traced
        for _ in range(32):
            obs, reward, _, _, _ = env.step(actions)
            check("finite_step", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            check("bounded_rotors", env._rotor_thrust.min() >= 0 and env._rotor_thrust.max() <= ctrl.allocator.cap+1e-5)
            check("bounded_arm", data.joint_effort_target[:, env.joint_ids].abs().max() <= cfg.effort_lim_sim)
        preserved = ctrl.residual_filter.z1[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_isolation", torch.equal(ctrl.residual_filter.z1[1:], preserved))
        check("no_stale_residual", not ctrl.measurement_ready[0] and not ctrl.disturbance_b[0].any())
        check("reset_spin_history", not ctrl.transient_torque[0].any()
              and torch.equal(ctrl.command_speed[0], ctrl.rotor_speed[0]))
        env._step_wrench(actions)
        check("fresh_reset_tick", not ctrl.raw_disturbance_w[0].any() and not ctrl.disturbance_b[0].any())
        env.episode_length_buf.fill_(env.max_episode_length)
        env._episode_steps.fill_(env.max_episode_length)
        if hasattr(env, "_goal_deadline_steps"):
            env._goal_deadline_steps.fill_(1)
        env.step(actions)
        check("automatic_reset", not ctrl.measurement_ready.any() and not ctrl.disturbance_b.any())
        report["max_raw_residual_nm"] = max(raw_peaks)
        report["max_spin_acceleration_torque_nm"] = max(transient_peaks)
        if cfg.indi.rotor_spin_inertia_kg_m2:
            check("nonzero_inertia_exercised", max(transient_peaks) > 1e-5)
        report["status"] = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        env.close()
    print(f"MODEL_WRENCH_INDI_RECEIPT {len(report['checks'])} checks passed", flush=True)


main()
app.close()
