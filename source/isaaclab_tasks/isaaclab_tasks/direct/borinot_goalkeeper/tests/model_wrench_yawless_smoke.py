# SPDX-License-Identifier: BSD-3-Clause
"""Real Hydra/PhysX yawless mode, spaces, feedback, limits and reset proof."""
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
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.controllers.borinot_angles import angle_action_slots
from isaaclab_tasks.controllers.borinot_model_wrench_yawless import ModelWrenchYawlessController, minimal_tilt_error
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
        report["actual_base_masses_kg"] = view.get_masses()[:, env.base_id].tolist()
        if cfg.include_events_randomization:
            check("randomized_plant_is_not_controller_calibration",
                  (view.get_masses()[:, env.base_id]-ctrl.model.bodies[0]["mass"]).abs().max() > .001)
        check("separate_yawless_class", type(ctrl) is ModelWrenchYawlessController)
        check("no_hip_sensor", not hasattr(env, "_wrench_thigh_id"))
        check("eight_actions_plus_joint_pairs", cfg.action_space == 8 + len(angle_action_slots(cfg)))
        check("actor_shape", obs["policy"].shape == (4, cfg.observation_space))
        check("critic_shape", obs["critic"].shape == (4, cfg.state_space))
        check("controller_source_snapshot", inspect.getsourcefile(ModelWrenchYawlessController) in env.get_code_snapshot_files())
        check("zero_implicit_pd", not view.get_dof_stiffnesses()[:, env.joint_ids].any()
              and not view.get_dof_dampings()[:, env.joint_ids].any())
        check("preserved_armature", torch.allclose(view.get_dof_armatures()[:, env.joint_ids],
              ctrl.model.armature.cpu().expand(4, -1)))
        scalar = torch.zeros(4, 8, device=env.device)
        scalar[:, 5] = .25  # A body-z rate reference, not a yaw angle.
        scalar[:, 6:] = torch.tensor([.025, -.02], device=env.device)
        physical = scalar.clone()
        if cfg.sac_action_mapping:
            physical[:, -2:] *= ctrl.joint_scale
        parts, start = [], 0
        for slot in angle_action_slots(cfg):
            parts += [scalar[:, start:slot], torch.stack((physical[:, slot].cos(), physical[:, slot].sin()), -1)]
            start = slot + 1
        actions = torch.cat(parts + [scalar[:, start:]], -1)
        calls, dynamics_errors = [], []
        original = ctrl.step

        def traced(action, quat, rates, joint_pos, joint_vel):
            before = ctrl.rotor_speed.clone()
            commands, effort = original(action, quat, rates, joint_pos, joint_vel)
            check("single_speed_lag_update", torch.allclose(ctrl.rotor_speed,
                  before + ctrl.motor_alpha*((commands/cfg.kf).sqrt()-before), atol=1e-5, rtol=1e-6))
            z = torch.cat((ctrl.predicted_acceleration_b, ctrl.alpha_desired, ctrl.predicted_joint_acceleration), -1)
            target = torch.cat((torch.zeros(4, 2, device=env.device), ctrl.requested_wrench, effort), -1)
            residual = float((matvec(ctrl.mass_matrix, z)+ctrl.bias-target).abs().max())
            dynamics_errors.append(residual)
            check("eight_coupled_dynamics_rows", residual < 3e-4)
            tilt = minimal_tilt_error(ctrl.desired_force_w, rotation_from_quat(quat))
            check("no_body_z_tilt_error", not tilt[:, 2].any())
            ref = scalar[:, 3:6]*ctrl.rate_scale if cfg.sac_action_mapping else scalar[:, 3:6]
            check("tilt_and_rate_law_only", torch.allclose(ctrl.alpha_desired,
                  cfg.wrench.tilt_gain*tilt + rates.new_tensor(cfg.wrench.rate_gain)*(ref-rates), atol=2e-5, rtol=2e-5))
            calls.append(1)
            return commands, effort

        ctrl.step = traced
        for _ in range(24):
            calls.clear()
            obs, reward, _, _, _ = env.step(actions)
            check("finite_step", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            check("physics_rate_controller", len(calls) == cfg.decimation)
            check("bounded_rotors", env._rotor_thrust.min() >= 0 and env._rotor_thrust.max() <= ctrl.allocator.cap+1e-5)
            check("bounded_arm_effort", data.joint_effort_target[:, env.joint_ids].abs().max() <= cfg.effort_lim_sim)
        speed = ctrl.rotor_speed[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_isolation", torch.equal(ctrl.rotor_speed[1:], speed) and not ctrl.mass_matrix[0].any())
        env._step_wrench(actions)
        check("fresh_first_tick", ctrl.mass_matrix[0].any() and ctrl.ready.all())
        env.episode_length_buf.fill_(env.max_episode_length)
        env._episode_steps.fill_(env.max_episode_length)
        if hasattr(env, "_goal_deadline_steps"):
            env._goal_deadline_steps.fill_(1)
        env.step(actions)
        check("automatic_reset", not ctrl.mass_matrix.any() and not ctrl.ready.any())
        check("automatic_reset_clears_joint_effort", not data.joint_effort_target[:, env.joint_ids].any())
        report["max_dynamics_residual"] = max(dynamics_errors)
        report["status"] = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        env.close()
    print(f"YAWLESS_RECEIPT {len(report['checks'])} checks passed", flush=True)


main()
app.close()
