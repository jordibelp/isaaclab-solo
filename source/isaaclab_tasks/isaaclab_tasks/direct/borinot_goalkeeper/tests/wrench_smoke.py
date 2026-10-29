# SPDX-License-Identifier: BSD-3-Clause
"""Live Hydra/PhysX measured-wrench integration (one simulator per process)."""
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
from isaaclab_tasks.controllers.borinot_wrench import WrenchController, leg_on_base_wrench, matvec
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 4
    cfg.sim.device, cfg.seed, cfg.curriculum = args.device, 42, False
    cfg.reset_grid_enabled = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "config": cfg.to_dict(), "checks": []}

    def check(name, condition):
        passed = bool(condition)
        report["checks"].append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    try:
        obs, _ = env.reset()
        ctrl, data, view = env._wrench, env.robot.data, env.robot.root_physx_view
        check("hydra_action_and_observation_sizes", env.action_space.shape[-1] == cfg.action_space == 9
              and obs["policy"].shape == (4, cfg.observation_space) and obs["critic"].shape == (4, cfg.state_space))
        check("shared_source_snapshot", inspect.getsourcefile(WrenchController) in env.get_code_snapshot_files())
        check("actual_base_mass_not_whole_robot", torch.equal(ctrl.mass.cpu(), view.get_masses()[:, env.base_id])
              and (ctrl.mass < view.get_masses().sum(-1).to(env.device)).all())
        check("actual_base_inertia", torch.equal(ctrl.inertia.cpu(), view.get_inertias()[:, env.base_id].reshape(-1, 3, 3)))
        check("zero_implicit_pd", not view.get_dof_stiffnesses()[:, env.joint_ids].any()
              and not view.get_dof_dampings()[:, env.joint_ids].any())
        check("effort_and_armature_preserved", torch.allclose(view.get_dof_max_forces()[:, env.joint_ids],
              torch.full((4, 2), cfg.effort_lim_sim)) and torch.allclose(view.get_dof_armatures()[:, env.joint_ids],
              torch.full((4, 2), .00036207)))
        actual_allocation = env._allocation.expand(4, -1, -1).clone()
        actual_allocation[:, 1] -= env._base_com_offset[:, 1:2]
        actual_allocation[:, 2] += env._base_com_offset[:, 0:1]
        check("allocator_matches_actual_com_plant", torch.equal(ctrl.allocator.allocation, actual_allocation))
        actions = torch.zeros(4, 9, device=env.device)
        actions[:, 3] = 0. if cfg.sac_action_mapping else ctrl.home_yaw
        actions[:, 7:] = torch.tensor([.05, -.04], device=env.device)
        calls, original = [], ctrl.step

        def traced(*values):
            commands, efforts = original(*values)
            act, _, _, q, dq, _, _ = values
            target = act[:, 7:9]*ctrl.joint_scale if cfg.sac_action_mapping else act[:, 7:9]
            if cfg.hold_arm:
                target = torch.zeros_like(target)
            expected = (cfg.kp*(target-q)-cfg.kd*dq).clamp(-cfg.effort_lim_sim, cfg.effort_lim_sim)
            calls.append(torch.allclose(efforts, expected, atol=1e-6))
            return commands, efforts

        ctrl.step = traced
        for _ in range(20):
            calls.clear()
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            assert len(calls) == cfg.decimation and all(calls)
            assert env._rotor_thrust.min() >= 0 and env._rotor_thrust.max() <= ctrl.allocator.cap+1e-5
        check("finite_rollout_fresh_pd_each_substep_bounded_rotors", True)
        check("explicit_effort_reaches_physx", torch.allclose(data.applied_torque[:, env.joint_ids],
              data.joint_effort_target[:, env.joint_ids], atol=1e-6))
        # Last completed sample must be read directly from the named thigh body.
        raw = data.body_incoming_joint_wrench_b[:, env._wrench_thigh_id].clone()
        env._step_wrench(actions)
        f, m = leg_on_base_wrench(raw, matrix_from_quat(data.body_link_quat_w[:, env._wrench_thigh_id]),
            data.body_link_pos_w[:, env._wrench_thigh_id], env._hip_rotation_child, env._hip_position_child,
            matrix_from_quat(data.body_link_quat_w[:, env.base_id]), data.body_com_pos_w[:, env.base_id])
        check("exact_requested_api_and_body_feed_controller", torch.equal(ctrl.external_force_w, f)
              and torch.equal(ctrl.external_moment_b, m) and f.abs().max() > .1)
        other = ctrl.rotor_speed[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_isolation", torch.equal(ctrl.rotor_speed[1:], other) and not ctrl.ready[0])
        check("reset_clears_effort", not data.joint_effort_target[0, env.joint_ids].any())
        data.body_incoming_joint_wrench_b[0, env._wrench_thigh_id] = 1e6
        commands, _ = env._step_wrench(actions)
        check("reset_rejects_stale_wrench", torch.allclose(commands[0], torch.full_like(commands[0], ctrl.reset_thrust))
              and not ctrl.external_force_w[0].any() and not ctrl.external_moment_b[0].any())
        # References outside [-1,1] must remain physical PPO; SAC must clamp them.
        actions[:, :3], actions[:, 4:7] = 12., 4.
        env._pre_physics_step(actions)
        expected = actions.clamp(-1, 1) if cfg.sac_action_mapping else actions
        check("ppo_si_or_sac_normalized_prephysics", torch.equal(env.actions, expected))
        # Force a real timeout through env.step rather than calling reset directly.
        env._episode_steps.fill_(env.max_episode_length)
        _, _, _, truncated, _ = env.step(torch.zeros_like(actions))
        check("automatic_timeout_reset", truncated.all() and not ctrl.ready.any())
        check("automatic_reset_clears_efforts", not data.joint_effort_target[:, env.joint_ids].any())
        report["base_mass_kg"] = ctrl.mass.cpu().tolist()
        report["last_external_force_world_n"] = f.cpu().tolist()
        report["last_external_moment_base_nm"] = m.cpu().tolist()
        report["action_space"] = cfg.action_space
        report["actor_critic_sizes"] = [cfg.observation_space, cfg.state_space]
    finally:
        args.output.write_text(json.dumps(report, default=str, indent=2)+"\n")
        env.close()


try:
    main()
finally:
    app.close()
