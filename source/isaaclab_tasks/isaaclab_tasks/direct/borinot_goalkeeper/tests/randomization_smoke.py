# SPDX-License-Identifier: BSD-3-Clause
"""Read live PhysX properties for both tasks, Hydra overrides, and single-term ablations."""
import argparse
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    report = {"task": args.task, "hydra_toggles": cfg.event_randomization.copy(), "checks": []}
    tags = tuple(cfg.event_randomization)

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(f"CHECK {name}: {bool(passed)}", flush=True)
        assert passed, name

    # Run one environment per process; reusing Isaac Sim after teardown can hang.
    cfg.scene.num_envs = 16
    cfg.sim.device = args.device
    cfg.seed = 211021
    cfg.curriculum = cfg.reset_grid_enabled = False
    expected = {tag: cfg.include_events_randomization and cfg.events is not None and enabled
                for tag, enabled in cfg.event_randomization.items()}
    env = gym.make(args.task, cfg=cfg).unwrapped

    def snapshot():
        view = env.robot.root_physx_view
        return {"mass": view.get_masses().clone(), "inertia": view.get_inertias().clone(),
                "com": view.get_coms().clone(), "friction": view.get_dof_friction_properties().clone(),
                "stiffness": view.get_dof_stiffnesses().clone(), "damping": view.get_dof_dampings().clone(),
                "armature": view.get_dof_armatures().clone(), "limits": view.get_dof_max_forces().clone()}

    try:
        current = snapshot()
        nominal_com = getattr(env, "_nominal_base_com", current["com"][0, env.base_id, :3]).to(env.device)
        nominal_hover = float(env.robot.data.default_mass[0].sum()) * 9.81 / 6
        rotor_pos = torch.tensor(cfg.rotor_pos_b, device=env.device) - nominal_com
        nominal_allocation = torch.stack((torch.ones(6, device=env.device), rotor_pos[:, 1], -rotor_pos[:, 0],
                                         torch.tensor(cfg.rotor_spin, device=env.device) * cfg.km_over_kf))
        baseline = {name: value.clone() for name, value in current.items()}
        baseline["mass"] = env.robot.data.default_mass.cpu().clone()
        baseline["inertia"] = env.robot.data.default_inertia.cpu().clone()
        baseline["com"][:, env.base_id, :3] = nominal_com.cpu()
        baseline["friction"].zero_()  # Both configs explicitly use nominal friction=0.
        check(f"fixed nominal thrust scaling", abs(env._hover_per_rotor - nominal_hover) < 1e-7)
        check(f"fixed nominal PI mixer", torch.equal(env._allocation, nominal_allocation))
        active = set(env.event_manager.active_terms.get("startup", [])) if cfg.events else set()
        check(f"only requested events registered", active == {tag for tag in tags if expected[tag]})
        for tag, body in (("add_base_mass", env.base_id), ("racket_mass", env.paddle_id)):
            ratio = current["mass"][:, body] / baseline["mass"][:, body]
            if expected[tag]:
                check(f"{tag} scale and diversity", (ratio >= .9 - 1e-6).all()
                      and (ratio <= 1.2 + 1e-6).all() and ratio.std() > .01)
                check(f"{tag} inertia follows mass", torch.allclose(
                    current["inertia"][:, body], baseline["inertia"][:, body] * ratio[:, None], atol=1e-9))
            else:
                check(f"{tag} nominal", torch.equal(current["mass"][:, body], baseline["mass"][:, body]))
        others = [i for i in range(env.robot.num_bodies) if i not in (env.base_id, env.paddle_id)]
        check(f"other masses unchanged", torch.equal(current["mass"][:, others], baseline["mass"][:, others]))
        delta = current["com"] - baseline["com"]
        if expected["base_com"]:
            xyz = delta[:, env.base_id, :3]
            check(f"CoM within 1 cm on every axis", (xyz.abs() <= .010001).all() and (xyz.std(0) > .001).all())
        else:
            check(f"nominal CoM", not delta.any())
        others = [i for i in range(env.robot.num_bodies) if i != env.base_id]
        check(f"other CoMs unchanged", not delta[:, others].any())
        if expected["joint_friction"]:
            friction = current["friction"][:, env.joint_ids]
            check(f"live static/dynamic/viscous friction", (friction >= .01 - 1e-7).all()
                  and (friction <= 1.0).all() and (friction[..., 1] <= friction[..., 0]).all()
                  and friction[..., 0].std() > .01)
        else:
            check(f"nominal joint friction", torch.equal(current["friction"], baseline["friction"]))
        for name in ("armature", "limits"):
            check(f"{name} unchanged", torch.equal(current[name], baseline[name]))
        # Independently sum r cross F about each sampled CoM. The mixer stays nominal.
        thrust = torch.arange(1, 7, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
        env._rotor_thrust.copy_(thrust)
        cfg.motor_time_constant_s = 1.0
        env._motor_alpha = 0.0
        captured = {}
        composer = env.robot.permanent_wrench_composer
        original = composer.set_forces_and_torques

        def capture(**kwargs):
            captured.update(kwargs)

        composer.set_forces_and_torques = capture
        env._apply_action()
        composer.set_forces_and_torques = original
        arms = torch.tensor(cfg.rotor_pos_b, device=env.device)[None] - current["com"][:, env.base_id, :3].to(env.device)[:, None]
        forces = torch.zeros(env.num_envs, 6, 3, device=env.device)
        forces[..., 2] = thrust
        torque = torch.cross(arms, forces, dim=-1).sum(1)
        torque[:, 2] += (thrust * torch.tensor(cfg.rotor_spin, device=env.device) * cfg.km_over_kf).sum(1)
        check(f"individual CoM rotor moments", torch.allclose(captured["torques"][:, 0], torque, atol=1e-6))
        env.reset()
        env._reset_idx(torch.tensor([1, 3], device=env.device))
        obs, reward, _, _, _ = env.step(torch.zeros(env.num_envs, cfg.action_space, device=env.device))
        check(f"finite observations and rewards", torch.isfinite(reward).all()
              and all(torch.isfinite(value).all() for value in obs.values()))
        after = snapshot()
        check(f"parameters survive full/partial resets and steps", all(
            torch.equal(current[name], after[name]) for name in current))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASSED {len(report['checks'])} checks", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
