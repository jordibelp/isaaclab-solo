# SPDX-License-Identifier: BSD-3-Clause
"""Real Hydra/PhysX integration; receipts, FK and bounded-command checks."""
import argparse
import inspect
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--forward-check", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]]+hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.controllers.borinot_angles import angle_action_slots
from isaaclab_tasks.controllers.borinot_model_wrench import ModelWrenchController
from isaaclab_tasks.controllers.borinot_wrench import matvec
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 4
    cfg.sim.device, cfg.seed, cfg.curriculum = args.device, 42, False
    cfg.reset_grid_enabled = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "config": cfg.to_dict(), "checks": [], "numerics": {}}

    def check(name, condition):
        passed = bool(condition)
        report["checks"].append({"name": name, "passed": passed})
        assert passed, name

    try:
        obs, _ = env.reset()
        ctrl, data, view = env._wrench, env.robot.data, env.robot.root_physx_view
        report["model"] = ctrl.model.description
        check("new_class_no_measured_hip_setup", isinstance(ctrl, ModelWrenchController) and not hasattr(env, "_wrench_thigh_id"))
        check("resolved_spaces", cfg.action_space == 9+len(angle_action_slots(cfg)) and obs["policy"].shape == (4, cfg.observation_space))
        check("critic_space", obs["critic"].shape == (4, cfg.state_space))
        check("model_snapshot", inspect.getsourcefile(ModelWrenchController) in env.get_code_snapshot_files()
              and any(p.endswith("model_description.py") for p in env.get_code_snapshot_files()))
        check("zero_implicit_pd", not view.get_dof_stiffnesses()[:, env.joint_ids].any()
              and not view.get_dof_dampings()[:, env.joint_ids].any())
        check("configured_armature", torch.allclose(view.get_dof_armatures()[:, env.joint_ids],
              ctrl.model.armature.cpu().expand(4, -1)))
        if not cfg.include_events_randomization:
            for b in ctrl.model.bodies:
                idx = env.robot.find_bodies(b["name"])[0][0]
                check(f"nominal_mass_{b['name']}", torch.allclose(view.get_masses()[:, idx], torch.full((4,), b["mass"])))
                check(f"nominal_inertia_{b['name']}", torch.allclose(view.get_inertias()[:, idx].reshape(-1, 3, 3),
                      b["inertia"].cpu().expand(4, -1, -1), atol=2e-8, rtol=2e-5))
        else:
            check("randomization_is_not_model_calibration", (view.get_masses()[:, env.base_id]-ctrl.mass.cpu()).abs().max() > .01)
        actions = torch.zeros(4, cfg.action_space, device=env.device)
        # Encode benign physical targets using the exact scalar/pair contract.
        scalar = torch.zeros(4, 9, device=env.device)
        scalar[:, 3] = 0. if cfg.sac_action_mapping else ctrl.home_yaw
        scalar[:, 7:] = torch.tensor([.025, -.02], device=env.device)
        parts, start = [], 0
        for slot in angle_action_slots(cfg):
            parts.extend((scalar[:, start:slot], torch.stack((scalar[:, slot].cos(), scalar[:, slot].sin()), -1)))
            start = slot+1
        parts.append(scalar[:, start:])
        actions = torch.cat(parts, -1)
        calls = []
        original = ctrl.step

        def traced(*values):
            before = ctrl.rotor_speed.clone()
            commands, effort = original(*values)
            expected = before+ctrl.motor_alpha*((commands/cfg.kf).sqrt()-before)
            assert torch.allclose(ctrl.rotor_speed, expected, atol=1e-5, rtol=1e-6)
            z = torch.cat((ctrl.predicted_acceleration_b, ctrl.alpha_desired, ctrl.predicted_joint_acceleration), -1)
            balance = matvec(ctrl.mass_matrix, z)+ctrl.bias
            target = torch.cat((torch.zeros(4, 2, device=env.device), ctrl.requested_wrench[:, :1],
                                ctrl.requested_wrench[:, 1:], effort), -1)
            error = float((balance-target).abs().max())
            assert error < 2e-4, error
            calls.append(error)
            return commands, effort

        ctrl.step = traced
        max_position, max_rotation = 0., 0.
        for _ in range(24):
            calls.clear()
            obs, reward, _, _, _ = env.step(actions)
            check("finite_tick", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            check("one_model_and_lag_update_per_substep", len(calls) == cfg.decimation)
            check("bounded_rotors", env._rotor_thrust.min() >= 0 and env._rotor_thrust.max() <= ctrl.allocator.cap+1e-5)
            check("bounded_joint_effort", data.joint_effort_target[:, env.joint_ids].abs().max() <= cfg.effort_lim_sim)
            base_r = matrix_from_quat(data.body_link_quat_w[:, env.base_id])
            states = ctrl.model.kinematics(data.joint_pos[:, env.joint_ids], data.joint_vel[:, env.joint_ids], data.root_ang_vel_b)
            for b, s in zip(ctrl.model.bodies, states):
                idx = env.robot.find_bodies(b["name"])[0][0]
                predicted = data.body_com_pos_w[:, env.base_id]+matvec(base_r, s["com"])
                if not cfg.include_events_randomization:
                    max_position = max(max_position, float((predicted-data.body_com_pos_w[:, idx]).abs().max()))
                max_rotation = max(max_rotation, float((base_r@s["rotation"]-matrix_from_quat(data.body_link_quat_w[:, idx])).abs().max()))
        report["numerics"].update(fk_com_max_error_m=None if cfg.include_events_randomization else max_position,
                                  fk_rotation_max_error=max_rotation)
        check("actual_link_fk_matches_model", max_position < 2e-5 and max_rotation < 2e-5)
        speed = ctrl.rotor_speed[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_isolation", torch.equal(ctrl.rotor_speed[1:], speed) and not ctrl.mass_matrix[0].any())
        env._step_wrench(actions)
        check("first_tick_uses_fresh_model_no_stale_wrench", ctrl.mass_matrix[0].any() and ctrl.ready.all())
        # A real timeout exercises each task's inherited automatic reset path.
        if hasattr(env, "_episode_steps"):
            env._episode_steps.fill_(env.max_episode_length)
        env.episode_length_buf.fill_(env.max_episode_length)
        if hasattr(env, "_goal_steps_left"):
            env._goal_steps_left.fill_(1)
        env.step(actions)
        check("automatic_reset_clears_model_buffers", not ctrl.mass_matrix.any() and not ctrl.ready.any())
        check("automatic_reset_clears_joint_targets", not data.joint_effort_target[:, env.joint_ids].any())
        if args.forward_check:
            assert not cfg.include_events_randomization
            rows = []
            # Set dt before simulator creation through Hydra. Changing it while
            # an articulation is live invalidates PhysX's tensor views.
            for dt in (env.physics_dt,):
                env._reset_idx(torch.arange(4, device=env.device))
                root = data.default_root_state.clone()
                root[:, :3] += env.scene.env_origins
                root[:, 2] = 3.
                root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
                root[:, 7:10] = torch.tensor([.2, -.1, .3], device=env.device)
                root[:, 10:13] = torch.tensor([.1, -.15, .2], device=env.device)
                env.robot.write_root_state_to_sim(root)
                q = torch.tensor([[.2, -.35]], device=env.device).expand(4, -1)
                dq = torch.tensor([[.2, -.1]], device=env.device).expand(4, -1)
                env.robot.write_joint_state_to_sim(q, dq, joint_ids=env.joint_ids)
                env.scene.update(dt)
                rotation = matrix_from_quat(data.body_link_quat_w[:, env.base_id])
                initial_v = data.body_com_lin_vel_w[:, env.base_id].clone()
                initial_w = data.root_ang_vel_b.clone()
                initial_dq = data.joint_vel[:, env.joint_ids].clone()
                m, h = ctrl.model.dynamics(data.joint_pos[:, env.joint_ids], initial_dq,
                    initial_w, matvec(rotation.mT, ctrl.gravity_w))
                thrust = torch.full((4, 6), ctrl.hover, device=env.device)
                thrust += torch.tensor([.05, -.03, .04, -.02, .01, -.01], device=env.device)
                applied = matvec(ctrl.allocator.allocation, thrust)
                effort = torch.tensor([[.03, -.02]], device=env.device).expand(4, -1)
                rhs = torch.cat((torch.zeros(4, 2, device=env.device), applied[:, :1], applied[:, 1:], effort), -1)
                predicted = torch.linalg.solve(m, (rhs-h).unsqueeze(-1)).squeeze(-1)
                force = torch.zeros(4, 1, 3, device=env.device)
                force[:, 0, 2] = applied[:, 0]
                env.robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=force, torques=applied[:, None, 1:], body_ids=[env.base_id], is_global=False)
                env.robot.set_joint_effort_target(effort, joint_ids=env.joint_ids)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt)
                observed = torch.cat((matvec(rotation.mT, (data.body_com_lin_vel_w[:, env.base_id]-initial_v)/dt),
                    (data.root_ang_vel_b-initial_w)/dt, (data.joint_vel[:, env.joint_ids]-initial_dq)/dt), -1)
                error = (observed-predicted).abs().max(0).values
                rows.append(dict(dt_s=dt, predicted=predicted[0].cpu().tolist(), observed=observed[0].cpu().tolist(),
                                 max_abs_error=error.cpu().tolist()))
            report["forward_dynamics"] = rows
            last = max(rows[-1]["max_abs_error"])
            check("physical_forward_dynamics_finite_step_error", last < .6)
        report["status"] = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        env.close()
    print(f"MODEL_WRENCH_RECEIPT {len(report['checks'])} checks passed", flush=True)


main()
app.close()
