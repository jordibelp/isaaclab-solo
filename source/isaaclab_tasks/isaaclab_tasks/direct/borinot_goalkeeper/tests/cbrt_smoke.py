# SPDX-License-Identifier: BSD-3-Clause
"""Live Hydra/PhysX six-action controller checks; one task/mode per process."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import traceback

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
from isaaclab_tasks.controllers.borinot_cbrt import CbrtController
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.seed, cfg.curriculum = 4, 42, False
    cfg.sim.device, cfg.reset_grid_enabled = args.device, False
    env = gym.make(args.task, cfg=cfg).unwrapped
    report = {"task": args.task, "config": cfg.to_dict(), "checks": []}

    def check(name, value):
        ok = bool(value)
        report["checks"].append({"name": name, "passed": ok})
        print(f"CHECK {name}: {ok}", flush=True)
        assert ok, name

    try:
        obs, _ = env.reset()
        c, d, view = env._cbrt, env.robot.data, env.robot.root_physx_view
        check("canonical_mode_and_six_actions", cfg.control_mode in ("cbrt", "cbrt_indi")
              and cfg.action_space == env.action_space.shape[-1] == 6)
        check("resolved_observation_sizes", obs["policy"].shape == (4, cfg.observation_space)
              and obs["critic"].shape == (4, cfg.state_space))
        check("nested_gains_reach_controller", c.cfg is cfg.cbrt and tuple(c.rate_gain.tolist()) == tuple(cfg.cbrt.rate_gain))
        check("nominal_not_privileged_base_inertia", torch.allclose(c.inertia,
              d.default_inertia[0, env.base_id].reshape(3,3).to(env.device)*cfg.cbrt.inertia_scale))
        check("source_snapshot_includes_controller_and_shared_solver", inspect.getsourcefile(CbrtController) in env.get_code_snapshot_files()
              and any(path.endswith("borinot_wrench.py") for path in env.get_code_snapshot_files()))
        check("zero_implicit_gains", not view.get_dof_stiffnesses()[:,env.joint_ids].any()
              and not view.get_dof_dampings()[:,env.joint_ids].any())
        check("effort_velocity_and_armature", torch.allclose(view.get_dof_max_forces()[:,env.joint_ids], torch.full((4,2),cfg.effort_lim_sim))
              and torch.allclose(view.get_dof_max_velocities()[:,env.joint_ids], torch.full((4,2),cfg.velocity_lim_sim))
              and torch.allclose(view.get_dof_armatures()[:,env.joint_ids], torch.full((4,2),.00036207)))
        actions = torch.tensor([[.1,-.15,.2,0.,.04,-.03]],device=env.device).repeat(4,1)
        if not cfg.sac_action_mapping:
            actions[:,3] = c.hover_total
        calls, allocation_calls = [], []
        original, allocate = c.step, c.allocator.thrust

        def allocation_trace(wrench):
            allocation_calls.append(wrench.clone())
            return allocate(wrench)

        def trace(action, rates, q, dq):
            before = c.rotor_speed.clone()
            command, effort = original(action,rates,q,dq)
            u = action.clamp(-1,1) if cfg.sac_action_mapping else action
            reference = u[:,:3]*c.rate_scale if cfg.sac_action_mapping else u[:,:3]
            target = u[:,4:6]*c.joint_scale if cfg.sac_action_mapping else u[:,4:6]
            if cfg.hold_arm:
                target = torch.zeros_like(target)
            expected_pd = (cfg.kp*(target-q)-cfg.kd*dq).clamp(-cfg.effort_lim_sim,cfg.effort_lim_sim)
            calls.append(bool(torch.allclose(effort,expected_pd,atol=1e-6)
                and torch.allclose(c.alpha_desired,c.rate_gain*(reference-rates),atol=2e-6)
                and torch.allclose(c.rotor_speed,before+c.motor_alpha*((command/cfg.kf).sqrt()-before),atol=1e-5)))
            return command,effort

        c.step, c.allocator.thrust = trace, allocation_trace
        for _ in range(20):
            calls.clear(); allocation_calls.clear()
            obs,reward,terminated,truncated,_ = env.step(actions)
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            assert len(calls)==cfg.decimation and all(calls)
            assert len(allocation_calls)==cfg.decimation*(2 if c.use_indi else 1)
            assert env._rotor_thrust.min()>=0 and env._rotor_thrust.max()<=c.allocator.cap+1e-5
        check("finite_fresh_rate_feedback_pd_speed_lag_and_stage_count", True)
        check("explicit_effort_reaches_physx", torch.allclose(d.applied_torque[:,env.joint_ids],d.joint_effort_target[:,env.joint_ids],atol=1e-6))
        other = c.rotor_speed[1:].clone()
        env._reset_idx(torch.tensor([0],device=env.device))
        check("partial_reset_isolated_and_effort_cleared", torch.equal(c.rotor_speed[1:],other)
              and not c.ready[0] and not d.joint_effort_target[0,env.joint_ids].any())
        if c.use_indi:
            check("filter_reset_zero_accel_and_steady_hover", not c.angular_acceleration[0].any()
                  and torch.allclose(c.rotor_filter.update(c.rotor_speed)[0],c.rotor_speed[0],atol=1e-4))
        actions[:,:3],actions[:,3] = 5.,30.
        env._pre_physics_step(actions)
        check("ppo_raw_or_sac_clipped_reference_gate", torch.equal(env.actions,actions.clamp(-1,1) if cfg.sac_action_mapping else actions))
        env._episode_steps.fill_(env.max_episode_length)
        _,_,_,truncated,_ = env.step(torch.zeros_like(actions))
        check("automatic_timeout_clears_controller_and_effort", truncated.all() and not c.ready.any()
              and not d.joint_effort_target[:,env.joint_ids].any())
        report.update(passed=True, actor_critic_sizes=[cfg.observation_space,cfg.state_space],
                      controller=cfg.control_mode, allocation_stages=2 if c.use_indi else 1)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"],flush=True)
        raise
    finally:
        args.output.write_text(json.dumps(report,default=str,indent=2)+"\n")
        env.close()


try:
    main()
finally:
    app.close()
