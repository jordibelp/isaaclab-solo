# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Solo12 joint slider viewer for IsaacLab.

This script spawns only the Solo12 articulation, disables gravity, avoids creating a ground plane,
and exposes one GUI slider per controlled joint. Each slider acts like a raw policy action for the
manager-based joint-position action term:

    target_joint_pos = default_joint_pos + action_scale * raw_action

Usage example:

    ./isaaclab.sh -p /mnt/data/solo12_joint_slider_viewer.py --action_scale 0.25 --slider_limit 1.0

Notes:
- This script requires the Isaac Sim GUI. Do not run it headless.
- The slider order matches the reversed joint-name order currently used in your env config.
- By default, the base is frozen every step for easier visual inspection of joint mobility.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from isaaclab.app import AppLauncher


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Solo12 joint slider viewer without gravity or ground.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.25,
    help="Scale used exactly like the manager-based JointPositionAction: target = default + scale * action.",
)
parser.add_argument(
    "--slider_limit",
    type=float,
    default=1.0,
    help="Raw-action slider range is [-slider_limit, slider_limit].",
)
parser.add_argument("--dt", type=float, default=0.005, help="Physics dt.")
parser.add_argument(
    "--freeze_base",
    action="store_true",
    default=True,
    help="Keep the floating base visually frozen by rewriting its root pose/velocity every step.",
)
parser.add_argument(
    "--no_freeze_base",
    action="store_false",
    dest="freeze_base",
    help="Let the base move freely.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    raise ValueError("This script needs the Isaac Sim GUI because it creates joint sliders. Run without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------------------------------------------
# Imports after app launch
# --------------------------------------------------------------------------------------
import omni.ui as ui
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_tasks.manager_based.locomotion.velocity.config.solo12.solo_configs import SOLO12_FLAT_CFG


# Natural order for readability
JOINT_NAMES_NATURAL = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Match the current env config exactly: [...][::-1]
POLICY_JOINT_NAMES = JOINT_NAMES_NATURAL[::-1]


@dataclass
class SliderState:
    name: str
    model: ui.SimpleFloatModel


class JointSliderWindow:
    """Small omni.ui panel with one slider per joint."""

    def __init__(self, joint_names: list[str], action_scale: float, slider_limit: float):
        self._joint_names = joint_names
        self._action_scale = action_scale
        self._slider_limit = slider_limit
        self._states: list[SliderState] = []

        self.window = ui.Window("Solo12 joint action viewer", width=520, height=760)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Slider value = raw action", height=20)
                    ui.Label(
                        f"Target = default_joint_pos + ({self._action_scale:.3f}) * raw_action",
                        height=20,
                    )
                    ui.Label(
                        "Slider order matches your current env action order (reversed explicit joint list).",
                        height=20,
                    )
                    with ui.HStack(height=28):
                        ui.Button("Zero all sliders", clicked_fn=self.zero_all)

                    for i, joint_name in enumerate(self._joint_names):
                        model = ui.SimpleFloatModel(0.0)
                        self._states.append(SliderState(name=joint_name, model=model))
                        with ui.HStack(height=26):
                            ui.Label(f"[{i:02d}] {joint_name}", width=180)
                            ui.FloatSlider(model=model, min=-self._slider_limit, max=self._slider_limit)
                            ui.FloatField(model=model, width=80)

    def zero_all(self):
        for state in self._states:
            state.model.set_value(0.0)

    def get_raw_actions(self, device: torch.device) -> torch.Tensor:
        values = [state.model.as_float for state in self._states]
        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


# --------------------------------------------------------------------------------------
# Scene / simulation
# --------------------------------------------------------------------------------------
def design_scene() -> Articulation:
    # Light only. No ground plane.
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin", "Xform")

    robot_cfg = SOLO12_FLAT_CFG.copy()
    robot_cfg.prim_path = "/World/Origin/Robot"

    # Disable gravity on the robot rigid bodies for this viewer.
    if robot_cfg.spawn.rigid_props is not None:
        robot_cfg.spawn.rigid_props.disable_gravity = True

    # Keep the same initial pose from your config for visual consistency.
    robot = Articulation(cfg=robot_cfg)
    return robot


def initialize_robot(robot: Articulation, freeze_base: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Reset root/joints to defaults and return the fixed root pose/vel used when freeze_base=True."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    if freeze_base:
        print("[INFO] Base freeze enabled for easier joint inspection.")
    else:
        print("[INFO] Base freeze disabled. The floating base may drift while joints move.")

    return root_state[:, :7].clone(), root_state[:, 7:].clone()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=args_cli.dt, gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.8, 1.6, 0.9], [0.0, 0.0, 0.30])

    robot = design_scene()

    # Build the stage and initialize physics views / buffers.
    sim.reset()

    # Initialize robot state.
    fixed_root_pose, fixed_root_vel = initialize_robot(robot, freeze_base=args_cli.freeze_base)

    # Resolve joint ids in the same order as the policy/action term in your env.
    joint_ids, resolved_joint_names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    joint_ids = list(joint_ids)

    print("[INFO] Resolved policy joint order:")
    for i, (jid, name) in enumerate(zip(joint_ids, resolved_joint_names, strict=False)):
        print(f"  action[{i:02d}] -> joint_id={jid:02d}  {name}")

    if len(joint_ids) != len(POLICY_JOINT_NAMES):
        raise RuntimeError(
            f"Expected {len(POLICY_JOINT_NAMES)} resolved joints, but found {len(joint_ids)}."
        )

    default_joint_pos_subset = robot.data.default_joint_pos[:, joint_ids].clone()

    slider_window = JointSliderWindow(
        joint_names=list(resolved_joint_names),
        action_scale=args_cli.action_scale,
        slider_limit=args_cli.slider_limit,
    )

    sim_dt = sim.get_physics_dt()
    step_count = 0

    print("[INFO] Setup complete. Move the sliders in the UI window.")
    print("[INFO] Raw action formula: target = default_joint_pos + action_scale * slider_value")

    while simulation_app.is_running():
        raw_actions = slider_window.get_raw_actions(device=sim.device)
        joint_targets = default_joint_pos_subset + args_cli.action_scale * raw_actions

        if args_cli.freeze_base:
            robot.write_root_pose_to_sim(fixed_root_pose)
            robot.write_root_velocity_to_sim(fixed_root_vel)

        robot.set_joint_position_target(joint_targets, joint_ids=joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        # occasional terminal feedback
        if step_count % 200 == 0:
            actual = robot.data.joint_pos[:, joint_ids]
            max_abs_err = (joint_targets - actual).abs().max().item()
            print(f"[INFO] step={step_count:06d}  max|target-actual|={max_abs_err:.4f} rad")

        step_count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
