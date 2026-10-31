# SPDX-License-Identifier: BSD-3-Clause
"""Registered-task Hydra/PhysX checks for optional angle action pairs."""
import argparse
import copy
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", default="wrench")
parser.add_argument("--sac", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_sac_cfg import BorinotGoalkeeperSAC


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(base_cfg, agent_cfg):
    modes = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control",
             "PI_rotors_PD_joint_pos_control", "indi", "wrench", "cbrt", "cbrt_indi")
    legacy = not hasattr(base_cfg, "control_mode")
    # Isaac Sim teardown/recreation can stall. Use one registered env per process.
    cases = [("legacy", False)] if legacy else [(args.mode, args.sac)]
    report = {"task": args.task, "cases": []}
    for mode, sac in cases:
        cfg = copy.deepcopy(base_cfg)
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
        cfg.predict_yaw_as_sincos = cfg.predict_thigh_joint_pos_as_sincos = cfg.predict_calf_joint_pos_as_sincos = True
        if not legacy:
            cfg.control_mode, cfg.sac_action_mapping = mode, sac
            cfg.curriculum, cfg.reset_grid_enabled, cfg.include_events_randomization = False, False, False
            cfg.action_clamp, cfg.actuator_delay = 1., .01
            cfg.rwd_scale_action_limits = -1.
            cfg.configure_control()
        env = gym.make(args.task, cfg=cfg).unwrapped
        checks = []

        def check(name, value):
            assert bool(value), (args.task, mode, sac, name)
            checks.append(name)

        try:
            obs, _ = env.reset()
            check("resolved_policy_and_action_spaces", obs["policy"].shape[-1] == cfg.observation_space
                  and env.action_space.shape[-1] == cfg.action_space)
            if not legacy:
                check("resolved_critic_space", obs["critic"].shape[-1] == cfg.state_space)
                check("source_snapshot", any(p.endswith("borinot_angles.py") for p in env.get_code_snapshot_files()))
                if sac:
                    upper, lower = BorinotGoalkeeperSAC._compute_action_scaling(env, env.device)
                    check("sac_expanded_bounds", upper.shape == lower.shape == (cfg.action_space,)
                          and (upper==1).all() and (lower==1).all())
            actions = torch.zeros(4, cfg.action_space, device=env.device)
            joint_start = 6 if legacy or mode.startswith("thrust_and_") else (4 if mode.startswith("cbrt") or mode==modes[2] else 8)
            targets = actions.new_tensor([.25, -.45])
            if legacy or mode != modes[0]:
                radius = .3 if legacy else 50.
                actions[:, joint_start:joint_start+4] = radius*torch.stack((targets.cos(), targets.sin()), -1).flatten()
            if mode in ("indi", "wrench"):
                start = 6 if mode=="indi" else 3
                actions[:, start:start+2] = 50.*actions.new_tensor([.6]).cos().repeat(2)
                actions[:, start+1] = 50.*actions.new_tensor(.6).sin()
            if mode.startswith("cbrt") and not sac:
                actions[:, 3] = env._hover_total
            captured = {}
            setter = env.robot.set_joint_position_target

            def position_trace(value, **kwargs):
                captured["target"] = value.clone()
                return setter(value, **kwargs)

            env.robot.set_joint_position_target = position_trace
            controller = getattr(env, "_"+mode, None) if mode in ("indi", "wrench") else (
                getattr(env, "_cbrt", None) if mode.startswith("cbrt") else None)
            if controller is not None:
                original = controller.step

                def control_trace(*values):
                    # Inputs are tensors owned by the simulator; capture before stepping.
                    position, velocity = (values[3], values[4]) if mode in ("indi", "wrench") else (values[2], values[3])
                    position, velocity = position.clone(), velocity.clone()
                    result = original(*values)
                    desired = cfg.kp*(targets-position)-cfg.kd*velocity
                    if mode=="indi":
                        desired += controller.cfg.joint_ki*controller.joint_integral
                    captured["pd_matches"] = torch.allclose(controller.requested_joint_effort, desired, atol=2e-5, rtol=1e-5)
                    return result

                controller.step = control_trace
            for _ in range(12):
                obs, reward, terminated, truncated, info = env.step(actions)
                check("finite_step", torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            if controller is not None:
                check("decoded_angles_reach_software_pd_pid", captured.get("pd_matches", False))
            elif legacy or mode in (modes[1], modes[2]):
                check("decoded_angles_reach_implicit_pd", torch.allclose(captured["target"], targets.expand(4, -1), atol=1e-6))
            if not legacy:
                check("newest_raw_action_preserved_with_delay", torch.equal(env.actions, actions))
                check("pair_radius_not_penalized", not env._action_limit_costs[:, 2].any())
                env._reset_idx(torch.tensor([1, 3], device=env.device))
                check("partial_reset_clears_only_selected_actions", not env.actions[[1, 3]].any()
                      and torch.equal(env.actions[[0, 2]], actions[[0, 2]]))
            report["cases"].append({"mode": mode, "sac": sac, "actions": cfg.action_space,
                                    "policy": cfg.observation_space, "checks": len(checks)})
            args.output.write_text(json.dumps(report, indent=2))
            print("ANGLE_CASE_OK="+json.dumps(report["cases"][-1]), flush=True)
        finally:
            env.close()
    report["passed"] = len(report["cases"]) == len(cases)
    args.output.write_text(json.dumps(report, indent=2))


try:
    main()
finally:
    app.close()
