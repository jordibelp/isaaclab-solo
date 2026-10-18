# SPDX-License-Identifier: BSD-3-Clause
"""Check actual done logic and USD-derived arm/paddle frames without Isaac Sim."""
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

FOLDER = Path(__file__).resolve().parents[1]
REPO = FOLDER.parents[4]
spec = importlib.util.spec_from_file_location("goalkeeper_reset_math", FOLDER / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


def config_literals():
    cls = next(n for n in ast.parse((FOLDER / "borinot_goalkeeper_env_cfg.py").read_text()).body
               if isinstance(n, ast.ClassDef))
    values = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def frame(position, quaternion_xyzw):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = position
    return result


def arm_frames(shoulder, elbow):
    model = json.loads((REPO / "source/scripts/tools/borinot_goalkeeper_viewer/robot_model.json").read_text())
    poses = {"base": np.eye(4)}
    axes = []
    for name, angle in zip(config_literals()["joint_names"], [shoulder, elbow]):
        joint = next(j for j in model["joints"] if j["name"] == name)
        anchor = poses[joint["parent"]] @ frame(joint["pos0"], joint["rot0"])
        axis = np.eye(3)["XYZ".index(joint["axis"])]
        axes.append(anchor[:3, :3] @ axis)
        turn = frame([0, 0, 0], Rotation.from_rotvec(angle * axis).as_quat())
        poses[joint["child"]] = anchor @ turn @ np.linalg.inv(frame(joint["pos1"], joint["rot1"]))
    return poses, axes


@pytest.mark.parametrize("shoulder,elbow", [(0, 0), (-1.1, -1.4), (1.1, 1.4), (.7, -1.2), (-.6, 1.3)])
def test_paddle_face_and_arm_sweep_are_parallel_to_goal(shoulder, elbow):
    cfg = config_literals()
    poses, axes = arm_frames(shoulder, elbow)
    root_quat = np.asarray(cfg.get("robot_home_quat", [1, 0, 0, 0]))
    root_rotation = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).as_matrix()
    paddle_quat = np.asarray(cfg["paddle_quat_ee"])
    normal = root_rotation @ poses["flying_arm_2__link2"][:3, :3] @ Rotation.from_quat(
        paddle_quat[[1, 2, 3, 0]]).apply([1, 0, 0])
    for axis in axes:
        world_axis = root_rotation @ axis
        assert abs(world_axis[0]) > 1 - 1e-8, "Arm sweep is not in the goal Y-Z plane"
        assert abs(np.dot(world_axis, normal)) > 1 - 1e-8, "Paddle face is perpendicular to arm sweep"
    assert normal[0] > 1 - 1e-8, "Paddle should face the incoming ball (+X)"


def test_neutral_shot_target_matches_aligned_paddle_at_reference_height():
    cfg = config_literals()
    poses, _ = arm_frames(0, 0)
    q = np.asarray(cfg["robot_home_quat"])
    root = frame([0, 0, 1.5], q[[1, 2, 3, 0]])
    position = (root @ poses["flying_arm_2__link2"] @ np.r_[cfg["paddle_offset_ee"], 1])[:3]
    np.testing.assert_allclose(position, [cfg["keeper_plane_x"], *cfg["target_center_yz"]], atol=1e-6)


@pytest.mark.parametrize("threshold_deg,roll_deg,pitch_deg,yaw_deg,expected", [
    (70.0, 0.0, 0.0, 150.0, False),  # yaw alone is not tilt
    (70.0, 69.0, 0.0, 0.0, False),
    (70.0, 71.0, 0.0, 0.0, True),
    (70.0, 0.0, 69.0, 150.0, False),
    (70.0, 0.0, 71.0, 150.0, True),
    (60.0, 65.0, 0.0, 0.0, True),
    (70.0, 65.0, 0.0, 0.0, False),  # changing degrees must change detection
    (20.0, 30.0, 0.0, 0.0, True),
    (100.0, 110.0, 0.0, 0.0, True),
])
def test_physics_tilt_detector_accepts_degrees(threshold_deg, roll_deg, pitch_deg, yaw_deg, expected):
    tree = ast.parse((FOLDER / "borinot_goalkeeper_env.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_sample_physics_events")
    detector = next(n for n in method.body if isinstance(n, ast.AugAssign)
                    and isinstance(n.target, ast.Subscript)
                    and isinstance(n.target.slice, ast.Constant) and n.target.slice.value == "tilt")
    rotation = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    gravity_b = torch.tensor(rotation.inv().apply([0.0, 0.0, -1.0])[None], dtype=torch.float64)
    env = SimpleNamespace(cfg=SimpleNamespace(robot_tilt_threshold_degs=threshold_deg),
                          _flags={"tilt": torch.zeros(1, dtype=torch.bool)})
    namespace = {"self": env, "math": math, "gravity_b": gravity_b}
    # Execute the actual detector, without importing or launching Isaac Sim.
    exec(compile(ast.Module(body=[detector], type_ignores=[]), "actual_tilt_detector", "exec"), namespace)
    assert env._flags["tilt"].item() == expected
    # Once detected, a violation stays latched for the rest of the action interval.
    env._flags["tilt"].fill_(True)
    gravity_b[0] = torch.tensor([0.0, 0.0, -1.0])
    exec(compile(ast.Module(body=[detector], type_ignores=[]), "actual_tilt_detector", "exec"), namespace)
    assert env._flags["tilt"].item()


@pytest.mark.parametrize("reset_contact,reset_tilt", [(False, False), (False, True), (True, False), (True, True)])
def test_reset_flags_are_independent_and_ball_outcomes_continue(reset_contact, reset_tilt):
    tree = ast.parse((FOLDER / "borinot_goalkeeper_env.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    namespace = {"torch": torch, "terminal_outcome": task_math.terminal_outcome}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_get_dones", "exec"), namespace)
    n = 6
    flags = {name: torch.zeros(n, dtype=torch.bool) for name in ("base", "arm", "crash", "tilt", "goal", "floor", "area", "miss")}
    flags["crash"][[0, 2, 4, 5]] = True
    flags["tilt"][[1, 2]] = True
    flags["goal"][3] = True
    flags["area"][4] = True
    env = SimpleNamespace(cfg=SimpleNamespace(require_recovery_for_save=False,
        reset_robot_on_robot_crash=reset_contact, self_collision_triggers_reset=True, reset_on_goal=False,
        reset_robot_on_tilt_threshold=reset_tilt,shot_timeout_s=3.0,is_finite_horizon=False),
        _flags=flags, _sample_physics_events=lambda: None, _clear=torch.zeros(n, dtype=torch.bool),
        _shot_steps=torch.tensor([1, 1, 1, 1, 1, 180]), _episode_steps=torch.ones(n,dtype=torch.long),
        _ball_active=torch.ones(n,dtype=torch.bool), step_dt=1/60, max_episode_length=1200)
    terminated, truncated = namespace["_get_dones"](env)
    assert terminated.tolist() == [reset_contact, reset_tilt, reset_contact or reset_tilt, False, True, reset_contact]
    assert env._outcome.tolist() == [8, 8, 8, 3, 8, 8], "Disabling reset must not remove the crash penalty"
    assert not truncated.any()
