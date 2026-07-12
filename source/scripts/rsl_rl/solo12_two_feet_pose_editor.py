# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive reset-pose editor for the Solo12 two-feet task.

The editor references the canonical Solo12 USD without modifying it.  It lets the user edit the
floating-base pose and all twelve joints, align the rear feet to a requested clearance above the
floor, preview a randomized reset falling under gravity, and export the accepted nominal pose and
randomization ranges as JSON.

Typical use::

    ./isaaclab.sh -p source/scripts/rsl_rl/solo12_two_feet_pose_editor.py

The proposed starting pose has the base pitched to -85 degrees, so base +x points almost upward,
and uses the existing safe joint pose.  The editor initially places the rear feet 10 cm above the
floor.
The JSON is an authoring artifact; accepting a pose does not edit the training configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Author a Solo12 upright two-feet reset pose in Isaac Sim.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/solo12_two_feet_reset_pose.json"),
    help="JSON file written by the Accept button, relative to the current directory by default.",
)
parser.add_argument("--dt", type=float, default=0.005, help="Physics time step used by the drop preview.")
parser.add_argument(
    "--smoke_test",
    action="store_true",
    help="Headless test: align the proposal, export it, step briefly, then exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless and not args_cli.smoke_test:
    raise ValueError("The pose editor needs the Isaac Sim GUI. Remove --headless, or add --smoke_test.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_tasks.direct.solo12.solo12_env_cfg import JOINT_NAMES, SAFE_INITIAL_JOINT_POS, SOLO12_CFG

if not args_cli.smoke_test:
    import omni.ui as ui


PROPOSED_ROOT_RPY_DEG = (0.0, -85.0, 180.0)
PROPOSED_REAR_FOOT_CLEARANCE_M = 0.10
PROPOSED_JOINT_NOISE_RAD = 0.05
PROPOSED_ROOT_RPY_NOISE_DEG = (3.0, 5.0, 5.0)
FOOT_OFFSET_BY_CALF = {
    "FL_calf": (0.0, 0.009000003337860107, -0.1599999964237213),
    "FR_calf": (0.0, -0.009000003337860107, -0.1599999964237213),
    "RL_calf": (0.0, 0.009000003337860107, -0.1599999964237213),
    "RR_calf": (0.0, -0.009000003337860107, -0.1599999964237213),
}


def _natural_joint_values() -> list[float]:
    return [float(SAFE_INITIAL_JOINT_POS[name]) for name in JOINT_NAMES]


def _design_scene() -> Articulation:
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.create_prim("/World/Origin", "Xform")

    robot_cfg = SOLO12_CFG.copy()
    robot_cfg.prim_path = "/World/Origin/Robot"
    # Match Solo12TwoFeetEnvCfg so the physical drop preview uses the training controller gains.
    robot_cfg.actuators["legs"].stiffness = 15.0
    robot_cfg.actuators["legs"].damping = 0.5
    return Articulation(cfg=robot_cfg)


class PoseState:
    """Pose values shared between the UI and the simulation loop."""

    def __init__(self, joint_ids: list[int]):
        self.joint_ids = joint_ids
        self.edit_mode = True
        self.request_align_clearance = True
        self.request_randomized_preview = False
        self.request_accept = False
        self.request_return_to_edit = False
        self.randomized_preview_active = False
        self.nominal_joint_values = _natural_joint_values()

        if args_cli.smoke_test:
            self.root_values = [0.0, 0.0, 0.45, *PROPOSED_ROOT_RPY_DEG]
            self.joint_values = list(self.nominal_joint_values)
            self.clearance = PROPOSED_REAR_FOOT_CLEARANCE_M
            self.joint_noise = PROPOSED_JOINT_NOISE_RAD
            self.rpy_noise = list(PROPOSED_ROOT_RPY_NOISE_DEG)
            self.status = "Smoke test"
            return

        self.root_models = [
            ui.SimpleFloatModel(0.0),
            ui.SimpleFloatModel(0.0),
            ui.SimpleFloatModel(0.45),
            ui.SimpleFloatModel(PROPOSED_ROOT_RPY_DEG[0]),
            ui.SimpleFloatModel(PROPOSED_ROOT_RPY_DEG[1]),
            ui.SimpleFloatModel(PROPOSED_ROOT_RPY_DEG[2]),
        ]
        self.joint_models = [ui.SimpleFloatModel(value) for value in self.nominal_joint_values]
        self.clearance_model = ui.SimpleFloatModel(PROPOSED_REAR_FOOT_CLEARANCE_M)
        self.joint_noise_model = ui.SimpleFloatModel(PROPOSED_JOINT_NOISE_RAD)
        self.rpy_noise_models = [ui.SimpleFloatModel(value) for value in PROPOSED_ROOT_RPY_NOISE_DEG]
        self.status_model = ui.SimpleStringModel("Preparing proposed pose...")

    @property
    def root_values(self) -> list[float]:
        if args_cli.smoke_test:
            return self._root_values
        return [model.as_float for model in self.root_models]

    @root_values.setter
    def root_values(self, values: list[float]):
        self._root_values = values

    @property
    def joint_values(self) -> list[float]:
        if args_cli.smoke_test:
            return self._joint_values
        return [model.as_float for model in self.joint_models]

    @joint_values.setter
    def joint_values(self, values: list[float]):
        self._joint_values = values

    @property
    def clearance(self) -> float:
        if args_cli.smoke_test:
            return self._clearance
        return self.clearance_model.as_float

    @clearance.setter
    def clearance(self, value: float):
        self._clearance = value

    @property
    def joint_noise(self) -> float:
        if args_cli.smoke_test:
            return self._joint_noise
        return self.joint_noise_model.as_float

    @joint_noise.setter
    def joint_noise(self, value: float):
        self._joint_noise = value

    @property
    def rpy_noise(self) -> list[float]:
        if args_cli.smoke_test:
            return self._rpy_noise
        return [model.as_float for model in self.rpy_noise_models]

    @rpy_noise.setter
    def rpy_noise(self, values: list[float]):
        self._rpy_noise = values

    def set_status(self, message: str):
        if args_cli.smoke_test:
            self.status = message
        else:
            self.status_model.set_value(message)
        print(f"[POSE EDITOR] {message}")

    def set_root_z(self, value: float):
        if args_cli.smoke_test:
            self.root_values[2] = value
        else:
            self.root_models[2].set_value(value)

    def reset_proposal(self):
        values = [0.0, 0.0, 0.45, *PROPOSED_ROOT_RPY_DEG]
        for model, value in zip(self.root_models, values, strict=True):
            model.set_value(value)
        for model, value in zip(self.joint_models, self.nominal_joint_values, strict=True):
            model.set_value(value)
        self.edit_mode = True
        self.randomized_preview_active = False
        self.request_align_clearance = True
        self.set_status("Restored proposal; aligning rear feet to requested clearance")


class PoseEditorWindow:
    def __init__(self, state: PoseState, joint_limits: torch.Tensor):
        self.state = state
        self.window = ui.Window("Solo12 two-feet reset pose", width=610, height=900)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=7, height=0):
                    ui.Label("Edit mode freezes the exact pose. Drop preview enables gravity.", height=22)
                    ui.Label("Angles are degrees for the base and radians for joints.", height=22)
                    ui.Label("Canonical SoloFlat.usd is referenced read-only.", height=22)
                    ui.Separator(height=8)
                    ui.Label("Root pose", height=24)
                    root_specs = [
                        ("x [m]", -0.5, 0.5),
                        ("y [m]", -0.5, 0.5),
                        ("z [m]", 0.05, 1.0),
                        ("roll [deg]", -30.0, 30.0),
                        ("pitch [deg]", -110.0, -50.0),
                        ("yaw [deg]", -180.0, 180.0),
                    ]
                    for label, model, (low, high) in zip(
                        [spec[0] for spec in root_specs],
                        state.root_models,
                        [(spec[1], spec[2]) for spec in root_specs],
                        strict=True,
                    ):
                        self._float_control(label, model, low, high)

                    ui.Separator(height=8)
                    ui.Label("Joint pose", height=24)
                    for index, (name, model) in enumerate(zip(JOINT_NAMES, state.joint_models, strict=True)):
                        low = float(joint_limits[index, 0])
                        high = float(joint_limits[index, 1])
                        self._float_control(name, model, low, high)

                    ui.Separator(height=8)
                    ui.Label("Reset randomization (uniform ± range)", height=24)
                    self._float_control("joint [rad]", state.joint_noise_model, 0.0, 0.25)
                    for label, model in zip(
                        ("roll [deg]", "pitch [deg]", "yaw [deg]"), state.rpy_noise_models, strict=True
                    ):
                        self._float_control(label, model, 0.0, 20.0)
                    self._float_control("rear-foot clearance [m]", state.clearance_model, 0.0, 0.30)

                    with ui.HStack(height=34, spacing=6):
                        ui.Button("Place rear feet at clearance", clicked_fn=self._align)
                        ui.Button("Preview randomized drop", clicked_fn=self._preview)
                    with ui.HStack(height=34, spacing=6):
                        ui.Button("Return to edit", clicked_fn=self._return_to_edit)
                        ui.Button("Restore proposal", clicked_fn=state.reset_proposal)
                    ui.Button("Accept and export pose", height=38, clicked_fn=self._accept)
                    ui.Separator(height=8)
                    ui.StringField(state.status_model, read_only=True, height=50)

    @staticmethod
    def _float_control(label: str, model, low: float, high: float):
        with ui.HStack(height=27):
            ui.Label(label, width=185)
            ui.FloatSlider(model=model, min=low, max=high)
            ui.FloatField(model=model, width=92)

    def _align(self):
        self.state.request_align_clearance = True

    def _preview(self):
        self.state.request_randomized_preview = True

    def _return_to_edit(self):
        self.state.request_return_to_edit = True

    def _accept(self):
        self.state.request_accept = True


def _pose_tensors(state: PoseState, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = state.root_values
    rpy = torch.tensor([[math.radians(value) for value in root[3:6]]], device=device)
    quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    root_pose = torch.tensor([root[:3]], dtype=torch.float32, device=device)
    root_pose = torch.cat((root_pose, quat), dim=1)
    joint_pose = torch.tensor([state.joint_values], dtype=torch.float32, device=device)
    return root_pose, joint_pose


def _apply_exact_pose(robot: Articulation, state: PoseState, device: str):
    root_pose, joint_pose = _pose_tensors(state, device)
    zero_root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
    zero_joint_velocity = torch.zeros_like(joint_pose)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(zero_root_velocity)
    robot.write_joint_state_to_sim(joint_pose, zero_joint_velocity, joint_ids=state.joint_ids)
    robot.set_joint_position_target(joint_pose, joint_ids=state.joint_ids)


def _foot_positions(robot: Articulation, device: str) -> tuple[list[str], torch.Tensor]:
    body_ids, body_names = robot.find_bodies(".*_calf")
    offsets = torch.tensor(
        [FOOT_OFFSET_BY_CALF[str(name).split("/")[-1]] for name in body_names],
        dtype=torch.float32,
        device=device,
    )
    calf_pos = robot.data.body_pos_w[:, body_ids, :]
    calf_quat = robot.data.body_quat_w[:, body_ids, :]
    return list(body_names), calf_pos + math_utils.quat_apply(calf_quat, offsets.unsqueeze(0))


def _align_rear_feet(sim: SimulationContext, robot: Articulation, state: PoseState):
    _apply_exact_pose(robot, state, sim.device)
    sim.forward()
    robot.update(0.0)
    names, positions = _foot_positions(robot, sim.device)
    rear_indices = [index for index, name in enumerate(names) if str(name).split("/")[-1].startswith(("RL_", "RR_"))]
    if len(rear_indices) != 2:
        raise RuntimeError(f"Expected RL/RR calf bodies, found {names}")
    current_min_z = float(positions[0, rear_indices, 2].min().item())
    new_root_z = state.root_values[2] + state.clearance - current_min_z
    state.set_root_z(new_root_z)
    _apply_exact_pose(robot, state, sim.device)
    sim.forward()
    robot.update(0.0)
    _, positions = _foot_positions(robot, sim.device)
    heights = [float(positions[0, index, 2].item()) for index in rear_indices]
    state.set_status(
        f"Edit mode: rear feet at {heights[0]:.3f} m / {heights[1]:.3f} m; root z={new_root_z:.3f} m"
    )


def _start_randomized_drop(robot: Articulation, state: PoseState, device: str):
    root_pose, joint_pose = _pose_tensors(state, device)
    rpy_noise_deg = torch.tensor([state.rpy_noise], dtype=torch.float32, device=device)
    rpy_noise_deg *= torch.empty_like(rpy_noise_deg).uniform_(-1.0, 1.0)
    nominal_rpy = torch.tensor(
        [[math.radians(value) for value in state.root_values[3:6]]], dtype=torch.float32, device=device
    )
    noisy_rpy = nominal_rpy + torch.deg2rad(rpy_noise_deg)
    root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(noisy_rpy[:, 0], noisy_rpy[:, 1], noisy_rpy[:, 2])
    joint_pose += torch.empty_like(joint_pose).uniform_(-state.joint_noise, state.joint_noise)
    limits = robot.data.soft_joint_pos_limits[:, state.joint_ids, :]
    joint_pose = torch.clamp(joint_pose, limits[..., 0], limits[..., 1])

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))
    robot.write_joint_state_to_sim(joint_pose, torch.zeros_like(joint_pose), joint_ids=state.joint_ids)
    robot.set_joint_position_target(joint_pose, joint_ids=state.joint_ids)
    state.edit_mode = False
    state.randomized_preview_active = True
    state.set_status(
        "Randomized drop is running. Press Return to edit to restore the exact nominal pose."
    )


def _export_pose(robot: Articulation, state: PoseState, output_path: Path):
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = state.root_values
    payload = {
        "format_version": 1,
        "task": "solo12-two-feet",
        "root_position_m": dict(zip(("x", "y", "z"), root[:3], strict=True)),
        "root_rpy_deg": dict(zip(("roll", "pitch", "yaw"), root[3:6], strict=True)),
        "joint_position_rad": dict(zip(JOINT_NAMES, state.joint_values, strict=True)),
        "randomization": {
            "joint_position_uniform_plus_minus_rad": state.joint_noise,
            "root_rpy_uniform_plus_minus_deg": dict(zip(("roll", "pitch", "yaw"), state.rpy_noise, strict=True)),
            "root_linear_velocity_uniform_m_s": [0.0, 0.0],
            "root_angular_velocity_uniform_rad_s": [0.0, 0.0],
        },
        "authoring": {
            "requested_rear_foot_clearance_m": state.clearance,
            "canonical_usd_modified": False,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    state.set_status(f"Accepted pose exported to {output_path}")
    print(json.dumps(payload, indent=2))


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=args_cli.dt, gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, 1.4, 0.9], [0.0, 0.0, 0.35])
    robot = _design_scene()
    sim.reset()

    joint_ids, resolved_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    joint_ids = list(joint_ids)
    if list(resolved_names) != JOINT_NAMES:
        raise RuntimeError(f"Joint-order mismatch: expected {JOINT_NAMES}, resolved {resolved_names}")

    state = PoseState(joint_ids)
    joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids, :].detach().cpu()
    window = None if args_cli.smoke_test else PoseEditorWindow(state, joint_limits)

    _align_rear_feet(sim, robot, state)
    if args_cli.smoke_test:
        _export_pose(robot, state, args_cli.output)
        _start_randomized_drop(robot, state, sim.device)
        print("[POSE EDITOR] Stepping randomized drop preview", flush=True)
        for _ in range(2):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        if not torch.isfinite(robot.data.root_pos_w).all():
            raise RuntimeError("Non-finite robot state during drop-preview smoke test")
        print("[POSE EDITOR] Smoke test passed", flush=True)
        return

    print("[POSE EDITOR] Ready. Use the 'Solo12 two-feet reset pose' window.")
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if state.request_return_to_edit:
            state.request_return_to_edit = False
            state.edit_mode = True
            state.randomized_preview_active = False
            _apply_exact_pose(robot, state, sim.device)
            state.set_status("Returned to exact nominal edit pose")
        if state.request_align_clearance:
            state.request_align_clearance = False
            state.edit_mode = True
            state.randomized_preview_active = False
            _align_rear_feet(sim, robot, state)
        if state.request_randomized_preview:
            state.request_randomized_preview = False
            _start_randomized_drop(robot, state, sim.device)
        if state.request_accept:
            state.request_accept = False
            _export_pose(robot, state, args_cli.output)

        if state.edit_mode:
            _apply_exact_pose(robot, state, sim.device)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

    # Keep a strong reference to the window until Kit exits.
    _ = window


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
