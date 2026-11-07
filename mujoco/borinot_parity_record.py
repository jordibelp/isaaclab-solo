#!/usr/bin/env python3
"""Record the same open-loop Borinot motion in Isaac Sim or MuJoCo, then compare them.

Both backends run this file unchanged through the Isaac Lab API: the go-to task's robot,
a scripted base wrench (body frame, at the centre of mass) and a software joint PD toward
sinusoidal targets, like the wrench controllers. Policies and task logic are not involved.

  ./isaaclab.sh -p mujoco/borinot_parity_record.py --backend isaac  --out /tmp/isaac.npz
  ./isaaclab.sh -p mujoco/borinot_parity_record.py --backend mujoco --out /tmp/mujoco.npz
  ./isaaclab.sh -p mujoco/borinot_parity_record.py --compare /tmp/isaac.npz /tmp/mujoco.npz
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--backend", choices=("isaac", "mujoco"))
parser.add_argument("--out", type=Path)
parser.add_argument("--steps", type=int, default=200, help="Physics steps at 200 Hz.")
parser.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "CANDIDATE"))
args, _ = parser.parse_known_args()

SIGNALS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_b", "joint_pos", "joint_vel",
           "hip_wrench", "base_lin_acc_w")


def compare(reference, candidate):
    a, b = np.load(reference), np.load(candidate)
    if "masses" in a and "masses" in b:
        order = [list(b["body_names"]).index(name) for name in a["body_names"]]
        # CoM positions only: each engine picks its own principal-axis quaternion for the same tensor.
        for key, cut in (("masses", ...), ("coms", np.s_[..., :3]), ("inertias", ...)):
            reference, candidate = a[key][cut], b[key][order][cut]
            print(f"{key:9s} max |diff| {np.abs(reference - candidate).max():.2e}  reference:",
                  np.array2string(reference, precision=6, suppress_small=True).replace("\n", ""))
    print(f"{'signal':16s} {'max |ref|':>10s} {'max |diff|':>11s} {'at step':>8s}  diff at steps 10/50/100/end")
    for name in SIGNALS:
        diff = np.abs(a[name] - b[name]).max(-1)
        marks = [diff[min(i, len(diff) - 1)] for i in (10, 50, 100, len(diff) - 1)]
        print(f"{name:16s} {np.abs(a[name]).max():10.4f} {diff.max():11.5f} {int(diff.argmax()):8d}  "
              + " ".join(f"{m:.5f}" for m in marks))


if args.compare:
    compare(*args.compare)
    sys.exit(0)
if args.backend is None or args.out is None:
    parser.error("--backend and --out are required to record.")

if args.backend == "mujoco":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from borinot_sim2sim import backend, bootstrap

    bootstrap.bootstrap()
    backend.OPTIONS.update(headless=True, substeps=1)
    app = None
else:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

import torch  # noqa: E402

from isaaclab_tasks.direct.borinot_goto.borinot_goto_env import BorinotGoToEnv  # noqa: E402
from isaaclab_tasks.direct.borinot_goto.borinot_goto_env_cfg import BorinotGoToEnvCfg  # noqa: E402

cfg = BorinotGoToEnvCfg()
cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, "cpu", 0
cfg.include_events_randomization = False
cfg.control_mode = "wrench"
env = BorinotGoToEnv(cfg)
env.reset()
robot, dt = env.robot, env.physics_dt
base, thigh = robot.find_bodies("base")[0][0], robot.find_bodies("flying_arm_2__link1")[0][0]
root = robot.data.default_root_state.clone()
root[:, :3] = torch.tensor((0.0, 0.0, 2.0))
root[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0))
root[:, 7:] = 0.0
robot.write_root_state_to_sim(root)
robot.write_joint_state_to_sim(torch.zeros(1, 2), torch.zeros(1, 2))
robot.data.body_incoming_joint_wrench_b  # MuJoCo measures joint wrenches once they are read
env.scene.write_data_to_sim()
env.sim.forward()
env.scene.update(dt)
hover = env._hover_total
log = {name: [] for name in SIGNALS}
for k in range(args.steps):
    t = k * dt
    force = torch.tensor([[[0.4 * math.sin(2 * math.pi * 0.7 * t), -0.3 * math.cos(2 * math.pi * 0.5 * t),
                            hover * (1.0 + 0.05 * math.sin(2 * math.pi * 1.1 * t))]]])
    torque = torch.tensor([[[0.010 * math.sin(2 * math.pi * 0.9 * t), 0.008 * math.sin(2 * math.pi * 1.3 * t + 1),
                             0.006 * math.cos(2 * math.pi * 0.6 * t)]]])
    # Ramped targets: the arm starts at rest, as in flight (no 5 ms whip to the torque limit).
    ramp = 1.0 - math.exp(-t / 0.15)
    target = ramp * torch.tensor([[0.6 * math.sin(2 * math.pi * 0.8 * t),
                                   -0.8 * math.sin(2 * math.pi * 0.6 * t + 0.5)]])
    effort = (cfg.kp * (target - robot.data.joint_pos[:, env.joint_ids])
              - cfg.kd * robot.data.joint_vel[:, env.joint_ids]).clamp(-cfg.effort_lim_sim, cfg.effort_lim_sim)
    robot.permanent_wrench_composer.set_forces_and_torques(forces=force, torques=torque, body_ids=[base],
                                                           is_global=False)
    robot.set_joint_effort_target(effort, joint_ids=env.joint_ids)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt)
    data = robot.data
    values = {"root_pos_w": data.root_pos_w, "root_quat_w": data.root_quat_w, "root_lin_vel_w": data.root_lin_vel_w,
              "root_ang_vel_b": data.root_ang_vel_b, "joint_pos": data.joint_pos, "joint_vel": data.joint_vel,
              "hip_wrench": data.body_incoming_joint_wrench_b[:, thigh], "base_lin_acc_w": data.body_lin_acc_w[:, base]}
    for name, value in values.items():
        log[name].append(value[0].detach().cpu().double().numpy().copy())
view = robot.root_physx_view  # mass properties as the physics engine holds them
properties = {"body_names": np.array(robot.body_names), "masses": view.get_masses()[0].double().numpy(),
              "coms": view.get_coms()[0].double().numpy(), "inertias": view.get_inertias()[0].double().numpy()}
np.savez(args.out, **{name: np.array(values) for name, values in log.items()}, **properties)
print(f"PARITY_RECORDED backend={args.backend} steps={args.steps} out={args.out}", flush=True)
env.close()
if app is not None:
    app.close()
