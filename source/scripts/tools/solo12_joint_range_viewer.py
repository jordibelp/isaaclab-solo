# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive range-of-motion viewer for the Solo12 direct tasks.

This script spawns only the Solo12 articulation (no ground plane, no gravity) and gives you one
slider per joint, in **degrees**. The slider range is the hard joint limit that the task writes
into the simulator, so you cannot drag a joint anywhere the policy could not go either.

It reads every limit from the task config, so there are no duplicated numbers here:

    hard limits   <- cfg.joint_physical_limit_{hip,thigh,calf}
                     or cfg.joint_physical_limit_{front,rear}_thigh when asymmetric limits are on
    soft limits   <- hard limits moved inward by cfg.joint_soft_limit_{type}_delta
    standing pose <- cfg.initial_joint_pos_by_name[cfg.initial_position]

The panel shows, per joint, how far it can still travel up and down from the standing pose. That
is the number the backflip notes refer to when they say the front thigh has only ~12 degrees above
the standing pose.

Usage:

    ./isaaclab.sh -p source/scripts/tools/solo12_joint_range_viewer.py
    ./isaaclab.sh -p source/scripts/tools/solo12_joint_range_viewer.py --asymmetric
    ./isaaclab.sh -p source/scripts/tools/solo12_joint_range_viewer.py --asymmetric --initial_position two_feet

Needs the Isaac Sim GUI. Do not run it headless.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Solo12 joint range-of-motion viewer.")
parser.add_argument(
    "--task_cfg",
    type=str,
    default="backflip",
    choices=["backflip", "solo12"],
    help="Which direct task config to read the joint limits and standing pose from.",
)
parser.add_argument(
    "--asymmetric",
    action="store_true",
    help="Start with use_asymmetric_thigh_limits=True. You can also toggle it live in the panel.",
)
parser.add_argument(
    "--initial_position",
    type=str,
    default=None,
    help="Override cfg.initial_position (rigid, flexed, crab, safe, two_feet).",
)
parser.add_argument(
    "--use_pd",
    action="store_true",
    help="Drive the joints with the task PD actuator instead of teleporting them. Shows what the "
    "motors can actually hold, not the pure kinematic range.",
)
parser.add_argument("--dt", type=float, default=0.005, help="Physics dt.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    raise ValueError("This script needs the Isaac Sim GUI because it creates sliders. Run without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.ui as ui
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

if args_cli.task_cfg == "backflip":
    from isaaclab_tasks.direct.solo12_backflip.solo12_backflip_env_cfg import Solo12BackflipEnvCfg as TaskCfg
else:
    from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg as TaskCfg


COLOR_OK = 0xFF63C763  # green
COLOR_SOFT = 0xFF3CC8F0  # amber
COLOR_HARD = 0xFF5050F0  # red


def joint_type_of(joint_name: str) -> str:
    """Return hip, thigh or calf for a Solo12 joint name."""
    for name in ("hip", "thigh", "calf"):
        if joint_name.endswith(f"_{name}_joint"):
            return name
    raise ValueError(f"Unknown Solo12 joint '{joint_name}'.")


def hard_limits_deg(cfg, joint_name: str, asymmetric: bool) -> tuple[float, float]:
    """Mirror Solo12Env._configure_joint_position_limits for one joint."""
    joint_type = joint_type_of(joint_name)
    if joint_type == "thigh" and asymmetric:
        side = "front" if joint_name.startswith(("FL_", "FR_")) else "rear"
        return tuple(getattr(cfg, f"joint_physical_limit_{side}_thigh"))
    return tuple(getattr(cfg, f"joint_physical_limit_{joint_type}"))


def soft_limits_deg(cfg, joint_name: str, hard: tuple[float, float]) -> tuple[float, float]:
    delta = getattr(cfg, f"joint_soft_limit_{joint_type_of(joint_name)}_delta")
    return (hard[0] + delta, hard[1] - delta)


class JointRow:
    """One slider plus the readout of where the standing pose sits inside the limits."""

    def __init__(self, cfg, joint_name: str, stand_deg: float):
        self.cfg = cfg
        self.name = joint_name
        self.stand_deg = stand_deg
        self.hard = (0.0, 0.0)
        self.soft = (0.0, 0.0)

        with ui.HStack(height=24, spacing=4):
            ui.Label(joint_name, width=125)
            self.model = ui.SimpleFloatModel(stand_deg)
            self.slider = ui.FloatSlider(model=self.model, min=-180.0, max=180.0)
            ui.FloatField(model=self.model, width=62)
            self.status = ui.Label("", width=42)
            self.info = ui.Label("", width=250)

    def apply_limits(self, asymmetric: bool):
        self.hard = hard_limits_deg(self.cfg, self.name, asymmetric)
        self.soft = soft_limits_deg(self.cfg, self.name, self.hard)
        self.slider.min = self.hard[0]
        self.slider.max = self.hard[1]
        self.info.text = (
            f"hard [{self.hard[0]:7.1f},{self.hard[1]:7.1f}]  soft [{self.soft[0]:7.1f},{self.soft[1]:7.1f}]"
            f"  stand {self.stand_deg:6.1f}  up {self.soft[1] - self.stand_deg:6.1f}"
            f"  dn {self.stand_deg - self.soft[0]:6.1f}"
        )
        self.clamp()

    def clamp(self):
        self.model.set_value(min(max(self.model.as_float, self.hard[0]), self.hard[1]))

    def refresh_status(self):
        value = self.model.as_float
        if value <= self.hard[0] + 1e-3 or value >= self.hard[1] - 1e-3:
            text, color = "HARD", COLOR_HARD
        elif value < self.soft[0] or value > self.soft[1]:
            text, color = "soft", COLOR_SOFT
        else:
            text, color = "ok", COLOR_OK
        if self.status.text != text:
            self.status.text = text
            self.status.set_style({"color": color})

    def set_deg(self, value: float):
        self.model.set_value(min(max(value, self.hard[0]), self.hard[1]))


class ViewerWindow:
    def __init__(self, cfg, joint_names: list[str], stand_deg: list[float], asymmetric: bool):
        self.cfg = cfg
        self.rows: list[JointRow] = []
        self.asymmetric = asymmetric
        self.sweep = False
        self._sweep_time = 0.0
        self.on_limits_changed = None

        self.window = ui.Window("Solo12 range of motion", width=980, height=520)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Sliders are joint angles in degrees. Slider ends are the hard limits.", height=18)
                    ui.Label("up / dn = degrees left from the standing pose to the SOFT limit.", height=18)

                    with ui.HStack(height=26, spacing=6):
                        ui.Label("Asymmetric thigh limits", width=170)
                        self.asym_model = ui.SimpleBoolModel(asymmetric)
                        ui.CheckBox(model=self.asym_model, width=24)
                        ui.Label("Sweep thighs", width=90)
                        self.sweep_model = ui.SimpleBoolModel(False)
                        ui.CheckBox(model=self.sweep_model, width=24)
                    self.asym_model.add_value_changed_fn(lambda m: self._set_asymmetric(m.as_bool))
                    self.sweep_model.add_value_changed_fn(lambda m: setattr(self, "sweep", m.as_bool))

                    with ui.HStack(height=26, spacing=6):
                        ui.Button("Standing pose", clicked_fn=self.to_standing)
                        ui.Button("Front thighs -> soft upper", clicked_fn=lambda: self._front_thighs("soft"))
                        ui.Button("Front thighs -> hard upper", clicked_fn=lambda: self._front_thighs("hard"))
                        ui.Button("All zero", clicked_fn=self.to_zero)

                    ui.Spacer(height=6)
                    for joint_name, stand in zip(joint_names, stand_deg, strict=True):
                        self.rows.append(JointRow(cfg, joint_name, stand))

        self.apply_limits()

    def _set_asymmetric(self, value: bool):
        self.asymmetric = value
        self.apply_limits()
        if self.on_limits_changed is not None:
            self.on_limits_changed()

    def apply_limits(self):
        for row in self.rows:
            row.apply_limits(self.asymmetric)

    def to_standing(self):
        self.sweep_model.set_value(False)
        for row in self.rows:
            row.set_deg(row.stand_deg)

    def to_zero(self):
        self.sweep_model.set_value(False)
        for row in self.rows:
            row.set_deg(0.0)

    def _front_thighs(self, which: str):
        """Push both front thighs to the top of their range, everything else standing."""
        self.sweep_model.set_value(False)
        for row in self.rows:
            if row.name.startswith(("FL_", "FR_")) and joint_type_of(row.name) == "thigh":
                row.set_deg(row.soft[1] if which == "soft" else row.hard[1])
            else:
                row.set_deg(row.stand_deg)

    def step_sweep(self, dt: float):
        """Drive every thigh across its full hard range so the travel is obvious."""
        if not self.sweep:
            return
        self._sweep_time += dt
        alpha = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.15 * self._sweep_time))
        for row in self.rows:
            if joint_type_of(row.name) == "thigh":
                row.set_deg(row.hard[0] + alpha * (row.hard[1] - row.hard[0]))

    def refresh(self):
        for row in self.rows:
            row.refresh_status()

    def targets_deg(self, device) -> torch.Tensor:
        values = [row.model.as_float for row in self.rows]
        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

    def hard_limits_tensor(self, device, num_instances: int) -> torch.Tensor:
        limits = torch.tensor([row.hard for row in self.rows], dtype=torch.float32, device=device)
        return torch.deg2rad(limits).unsqueeze(0).expand(num_instances, -1, -1).contiguous()


def print_summary(joint_names, stand_deg, cfg, asymmetric: bool):
    label = "asymmetric" if asymmetric else "symmetric"
    print(f"\n[INFO] Solo12 thigh range of motion, {label} limits, standing pose = {cfg.initial_position}")
    print(f"{'joint':<16}{'hard lo':>9}{'hard hi':>9}{'soft lo':>9}{'soft hi':>9}{'stand':>8}{'up(soft)':>10}{'up(hard)':>10}")
    for name, stand in zip(joint_names, stand_deg, strict=True):
        if joint_type_of(name) != "thigh":
            continue
        hard = hard_limits_deg(cfg, name, asymmetric)
        soft = soft_limits_deg(cfg, name, hard)
        print(
            f"{name:<16}{hard[0]:9.1f}{hard[1]:9.1f}{soft[0]:9.1f}{soft[1]:9.1f}"
            f"{stand:8.1f}{soft[1] - stand:10.1f}{hard[1] - stand:10.1f}"
        )
    print()


def main():
    cfg = TaskCfg()
    if args_cli.initial_position is not None:
        cfg.initial_position = args_cli.initial_position

    joint_names = list(cfg.joint_names)
    stand_pose = cfg.initial_joint_pos_by_name[cfg.initial_position.lower()]
    stand_deg = [math.degrees(stand_pose[name]) for name in joint_names]

    print_summary(joint_names, stand_deg, cfg, args_cli.asymmetric)

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=args_cli.dt, gravity=(0.0, 0.0, 0.0)))
    sim.set_camera_view([1.4, 1.2, 0.5], [0.0, 0.0, 0.2])

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.create_prim("/World/Origin", "Xform")

    robot_cfg = cfg.robot.copy()
    robot_cfg.prim_path = "/World/Origin/Robot"
    if robot_cfg.spawn.rigid_props is not None:
        robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)

    sim.reset()

    joint_ids, resolved = robot.find_joints(joint_names, preserve_order=True)
    joint_ids = list(joint_ids)
    if list(resolved) != joint_names:
        raise RuntimeError(f"Joint order mismatch: asked {joint_names}, got {list(resolved)}.")

    root_state = robot.data.default_root_state.clone()
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.reset()

    window = ViewerWindow(cfg, joint_names, stand_deg, args_cli.asymmetric)

    def push_limits():
        robot.write_joint_position_limit_to_sim(
            window.hard_limits_tensor(sim.device, robot.num_instances), joint_ids=joint_ids
        )
        print(f"[INFO] Applied {'asymmetric' if window.asymmetric else 'symmetric'} thigh limits.")

    window.on_limits_changed = push_limits
    push_limits()
    window.to_standing()

    fixed_pose, fixed_vel = root_state[:, :7].clone(), root_state[:, 7:].clone()
    zero_vel = torch.zeros((robot.num_instances, len(joint_ids)), device=sim.device)
    sim_dt = sim.get_physics_dt()

    print("[INFO] Ready. Move the sliders in the 'Solo12 range of motion' window.")

    while simulation_app.is_running():
        window.step_sweep(sim_dt)
        targets = torch.deg2rad(window.targets_deg(sim.device))

        robot.write_root_pose_to_sim(fixed_pose)
        robot.write_root_velocity_to_sim(fixed_vel)

        if args_cli.use_pd:
            robot.set_joint_position_target(targets, joint_ids=joint_ids)
        else:
            robot.write_joint_state_to_sim(targets, zero_vel, joint_ids=joint_ids)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        window.refresh()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
