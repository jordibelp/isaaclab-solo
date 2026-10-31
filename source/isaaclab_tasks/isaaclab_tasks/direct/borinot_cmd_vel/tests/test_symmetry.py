# SPDX-License-Identifier: BSD-3-Clause
"""Independent geometry, frame, action and TensorDict symmetry oracles."""
import ast
import importlib.util
import itertools
import inspect
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from test_control import config, method
from test_indi import indi
from test_wrench import wrench
from test_cbrt import cbrt

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("borinot_symmetry_test", FOLDER.parent / "borinot/symmetry.py")
symmetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry)
spec = importlib.util.spec_from_file_location("borinot_tracking_test", FOLDER / "task_math.py")
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)
MODES = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control", "PI_rotors_PD_joint_pos_control",
         "indi", "wrench", "cbrt", "cbrt_indi")
BITS = tuple(itertools.product((False, True), repeat=3))


def cfg(mode="indi", bits=(False, False, False), sac=False, acceleration=False):
    result = config()
    result.control_mode, result.sac_action_mapping = mode, sac
    result.predict_yaw_as_sincos, result.predict_thigh_joint_pos_as_sincos, result.predict_calf_joint_pos_as_sincos = bits
    result.indi, result.wrench, result.cbrt = indi.IndiControllerCfg(), wrench.WrenchControllerCfg(), cbrt.CbrtControllerCfg()
    result.obs_include_base_lin_acc = acceleration
    result.command_lin_vel_x_range = (-1., 1.)
    result.robot_home_quat = (1., 0., 0., 0.)
    if sac and mode == MODES[2]:
        raise ValueError("unsupported PI SAC")
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(result)
    return result


def observations(transform, n=19):
    generator = torch.Generator().manual_seed(721)
    obs = torch.randn(n, transform.observation_size, generator=generator, dtype=torch.float64)
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    q /= q.norm(dim=-1, keepdim=True)
    r = torch.from_numpy(Rotation.from_quat(q.numpy()).as_matrix())
    obs[:, transform.obs_slices["rotation"]] = r.flatten(1)
    if "gravity" in transform.obs_slices:
        obs[:, transform.obs_slices["gravity"]] = r[:, 2, :]*-1
    return obs, r


def test_rotor_chirality_not_just_position_permutation():
    audit = symmetry.rotor_symmetry_audit(cfg())
    assert audit["left_right"]["permutation"] == [1, 0, 4, 5, 2, 3]
    assert audit["left_right"]["valid"]
    for name in ("front_back", "rotate_180"):
        assert audit[name]["geometry_error_m"] == 0
        assert not audit[name]["valid"]
        assert audit[name]["wrench_error_per_n"] == pytest.approx(2*5.361e-8/3.847e-6)
    assert symmetry.BorinotSymmetry(cfg()).transforms == ("identity", "world_yaw_180")


@pytest.mark.parametrize("mode,sac", [(m, s) for m in MODES for s in (False, True) if not (s and m==MODES[2])])
@pytest.mark.parametrize("bits", BITS)
def test_all_control_modes_and_angle_encodings_are_involutions_without_input_mutation(mode, sac, bits):
    transform = symmetry.BorinotSymmetry(cfg(mode, bits, sac, acceleration=True))
    obs, _ = observations(transform)
    actions = torch.randn(19, transform.action_size, dtype=torch.float64)*2
    before_obs, before_actions = obs.clone(), actions.clone()
    for name in transform.transforms:
        torch.testing.assert_close(transform.observations(transform.observations(obs, name), name), obs, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(transform.actions(transform.actions(actions, name), name), actions, atol=1e-13, rtol=1e-13)
    assert torch.equal(before_obs, obs) and torch.equal(before_actions, actions)


@pytest.mark.parametrize("home_yaw", [0., .7, -math.pi/2])
def test_world_rotation_changes_only_world_frame_and_heading_references(home_yaw):
    task = cfg()
    task.robot_home_quat = (math.cos(home_yaw/2), 0., 0., math.sin(home_yaw/2))
    transform = symmetry.BorinotSymmetry(task)
    obs, rotations = observations(transform)
    result = transform.observations(obs, "world_yaw_180")
    world = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float64))
    expected = world@rotations
    torch.testing.assert_close(result[:, transform.obs_slices["rotation"]].reshape(-1, 3, 3), expected)
    torch.testing.assert_close(torch.linalg.det(expected), torch.ones(19, dtype=torch.float64))
    for key in ("velocity", "omega", "joint_position", "joint_velocity", "commands", "rotors", "joint_integral"):
        sl = transform.obs_slices[key]
        assert torch.equal(result[:, sl], obs[:, sl])


@pytest.mark.parametrize("mode", ["indi", "wrench"])
@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("pair", [False, True])
def test_world_acceleration_and_decoded_yaw_rotate_together(mode, sac, pair):
    task = cfg(mode, (pair, False, False), sac)
    task.robot_home_quat = (math.cos(.4), 0., 0., math.sin(.4))
    transform = symmetry.BorinotSymmetry(task)
    actions = torch.randn(21, task.action_space, dtype=torch.float64)*11
    slot = transform.action_slices["yaw"]
    if pair:
        actions[:2, slot] = 0
        actions[2:4, slot] = torch.tensor([[1., 0.], [-1., 0.]])
    else:
        actions[:5, slot] = torch.tensor([[-1.], [0.], [1.], [15.], [-14.]])
    output = transform.actions(actions, "world_yaw_180")
    def angle(a):
        if pair:
            vector = a[:, slot]
            result = torch.atan2(vector[:, 1], vector[:, 0])
            result[(vector==0).all(-1)] = 0.
        else:
            result = a[:, slot.start]
            if sac:
                result = result*math.pi
        if mode=="indi" or sac:
            result = result+transform.home_yaw
        return result
    delta = angle(output)-angle(actions)
    torch.testing.assert_close(delta.sin(), torch.zeros(21, dtype=torch.float64), atol=1e-13, rtol=0)
    torch.testing.assert_close(delta.cos(), -torch.ones(21, dtype=torch.float64), atol=1e-13, rtol=0)
    accel = transform.action_slices["acceleration"]
    torch.testing.assert_close(output[:, accel], actions[:, accel]*torch.tensor([-1., -1., 1.]))
    assert torch.equal(output[:, transform.action_slices["rates"]], actions[:, transform.action_slices["rates"]])


@pytest.mark.parametrize("pair", [False, True])
def test_sac_heading_bounds_and_limited_heading_rejection(pair):
    task = cfg("indi", (pair, False, False), True)
    transform = symmetry.BorinotSymmetry(task)
    actions = torch.rand(1000, task.action_space)*2-1
    actions[:2, transform.action_slices["yaw"]] = torch.tensor([[-1.], [1.]])
    output = transform.actions(actions, "world_yaw_180")
    assert output.abs().max() <= 1+1e-7
    task.indi.heading_range_degs = 90.
    if pair:
        symmetry.BorinotSymmetry(task)
    else:
        with pytest.raises(ValueError, match="full-circle"):
            symmetry.BorinotSymmetry(task)


def test_real_tracking_reward_matches_world_rotation():
    task = cfg()
    transform = symmetry.BorinotSymmetry(task)
    obs, rotation = observations(transform)
    for name in transform.transforms:
        augmented = transform.observations(obs, name)
        r = augmented[:, transform.obs_slices["rotation"]].reshape(-1, 3, 3)
        cmd = augmented[:, transform.obs_slices["commands"]]
        velocity = (r@augmented[:, transform.obs_slices["velocity"]][:, :, None]).squeeze(-1)
        omega = (r@augmented[:, transform.obs_slices["omega"]][:, :, None]).squeeze(-1)
        def quat(matrix):
            q = torch.from_numpy(Rotation.from_matrix(matrix.numpy()).as_quat())
            return torch.cat((q[:, 3:], q[:, :3]), -1)
        oracle = tracking.tracking_kernels(obs[:, transform.obs_slices["commands"]],
            (rotation@obs[:, transform.obs_slices["velocity"]][:, :, None]).squeeze(-1),
            (rotation@obs[:, transform.obs_slices["omega"]][:, :, None]).squeeze(-1), quat(rotation), .5, .5)
        result = tracking.tracking_kernels(cmd, velocity, omega, quat(r), .5, .5)
        for actual, expected in zip(result, oracle):
            torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)


def test_tensordict_alignment_action_only_observation_only_and_rotor_history():
    task = cfg("indi", (True, True, True), acceleration=True)
    transform = symmetry.BorinotSymmetry(task)
    env = SimpleNamespace(cfg=task)
    obs, _ = observations(transform)
    actions = torch.randn(19, task.action_space, dtype=torch.float64)
    td = TensorDict({"policy": obs, "critic": obs+0.3}, batch_size=[19])
    output, output_actions = symmetry.compute_symmetric_observations_actions(env, td, actions)
    obs_only, _ = symmetry.compute_symmetric_observations_actions(env, td, None)
    _, act_only = symmetry.compute_symmetric_observations_actions(env, None, actions)
    assert output.batch_size == (38,)
    assert torch.equal(output_actions, act_only)
    assert all(torch.equal(output[k], obs_only[k]) for k in td.keys())
    for index, name in enumerate(transform.transforms):
        torch.testing.assert_close(output["policy"][19*index:19*(index+1)], transform.observations(obs, name))
        if name != "identity":
            sl = transform.obs_slices["rotors"]
            assert torch.equal(output["policy"][19*index:19*(index+1), sl], obs[:, sl])


@pytest.mark.parametrize("position", [False, True])
def test_original_command_task_layout_and_pair_controls(position):
    tree = ast.parse((FOLDER.parent/"borinot/borinot_env_cfg.py").read_text())
    values = {}
    for node in next(n for n in tree.body if isinstance(n, ast.ClassDef)).body:
        try:
            if isinstance(node, ast.Assign):
                values[node.targets[0].id] = ast.literal_eval(node.value)
            elif isinstance(node, ast.AnnAssign):
                values[node.target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, AttributeError):
            pass
    values.update(des_joint_out=position, action_space=10 if position else 8,
                  torque_min=(-2.7, -2.7), torque_max=(2.7, 2.7))
    transform = symmetry.BorinotSymmetry(SimpleNamespace(**values))
    obs, _ = observations(transform)
    for name in transform.transforms:
        torch.testing.assert_close(transform.observations(transform.observations(obs, name), name), obs)


def test_fixed_goalkeeper_is_not_silently_augmented_as_command_tracking():
    with pytest.raises(ValueError, match="command-tracking"):
        symmetry.BorinotSymmetry(config())


def test_logged_source_is_the_transform_not_torch_no_grad_wrapper():
    repo = FOLDER.parents[4]
    tree = ast.parse((repo / "source/scripts/rsl_rl/train.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id=="symmetry_py" for t in n.targets))
    scope = {"inspect": inspect, "symmetry_enabled": True,
             "symmetry_fn": symmetry.compute_symmetric_observations_actions}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "source_receipt", "exec"), scope)
    assert Path(scope["symmetry_py"]).resolve() == (FOLDER.parent / "borinot/symmetry.py").resolve()


def test_actual_leg_joint_frames_are_not_body_reflection_symmetric():
    repo = FOLDER.parents[4]
    model = json.loads((repo / "source/scripts/tools/borinot_goalkeeper_viewer/robot_model.json").read_text())
    usd = repo / "source/isaaclab_assets/data/Robots/Borinot/borinotFlat.usd"
    assert hashlib.sha256(usd.read_bytes()).hexdigest() == model["usd_sha256"]
    joints = [next(j for j in model["joints"] if j["parent"]=="base"),
              next(j for j in model["joints"] if j["parent"]!="base")]
    def rotation(q):
        return Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    def tip(angles):
        p, r = np.zeros(3), np.eye(3)
        for joint, angle in zip(joints, angles):
            child = r@rotation(joint["rot0"])@Rotation.from_rotvec([angle, 0, 0]).as_matrix()@rotation(joint["rot1"]).T
            p = p+r@joint["pos0"]-child@joint["pos1"]
            r = child
        return p+r@np.array([0., -.136, -.00495])
    for signs in symmetry.SIGNS.values():
        s = np.array(signs)
        for angles in ([0., 0.], [.5, -.7], [-.6, 1.2]):
            # An exact mirror would give zero. The current CAD gives >12 cm.
            assert np.linalg.norm(tip(np.array(angles)*s[0])-s*tip(angles)) > .12
