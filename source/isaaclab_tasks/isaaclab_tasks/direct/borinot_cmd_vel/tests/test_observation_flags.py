# SPDX-License-Identifier: BSD-3-Clause
"""Check actual observation/config methods against explicit feature contracts."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scipy.spatial.transform import Rotation

from test_control import config as parent_config, method as parent_method
from test_indi import indi
from test_wrench import wrench
from test_cbrt import cbrt
from test_task import method
from test_symmetry import symmetry

FOLDER = Path(__file__).resolve().parents[1]
MODES = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control",
         "PI_rotors_PD_joint_pos_control", "indi", "wrench", "cbrt", "cbrt_indi")


def config(mode=MODES[0], **overrides):
    node = next(n for n in ast.parse((FOLDER / "borinot_cmd_vel_env_cfg.py").read_text()).body
                if isinstance(n, ast.ClassDef))
    values = vars(parent_config()).copy()
    for n in node.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    parent = type("Parent", (), {"configure_control": parent_method("borinot_goalkeeper_env_cfg.py", "configure_control")})
    # Compile the actual override with a real super() closure, without simulator imports.
    node.bases, node.decorator_list = [ast.Name(id="Parent", ctx=ast.Load())], []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "configure_control"]
    scope = {"Parent": parent, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "cmd_config", "exec"), scope)
    result = scope[node.name]()
    result.__dict__.update(values)
    result.robot.init_state = SimpleNamespace()
    result.scene = SimpleNamespace(env_spacing=64.)
    result.indi, result.wrench, result.cbrt = indi.IndiControllerCfg(), wrench.WrenchControllerCfg(), cbrt.CbrtControllerCfg()
    result.control_mode = mode
    for name, value in overrides.items():
        setattr(result, name, value)
    result.configure_control()
    return result


def matrix(q):
    return torch.tensor(Rotation.from_quat(q[:, [1, 2, 3, 0]].numpy()).as_matrix(), dtype=q.dtype)


def inverse(q, value):
    return (matrix(q).transpose(1, 2) @ value.unsqueeze(-1)).squeeze(-1)


def observation_env(cfg):
    q = torch.tensor([[1., 0, 0, 0], [math.cos(.3), math.sin(.3), 0, 0]], dtype=torch.float64)
    data = SimpleNamespace(root_quat_w=q, root_pos_w=torch.tensor([[5., 7., 1.25], [-2., 9., 4.5]], dtype=q.dtype),
        root_lin_vel_b=torch.arange(6, dtype=q.dtype).reshape(2, 3),
        root_ang_vel_b=torch.arange(6, 12, dtype=q.dtype).reshape(2, 3),
        joint_pos=torch.tensor([[.1, -.2], [.3, -.4]], dtype=q.dtype),
        joint_vel=torch.tensor([[.5, -.6], [.7, -.8]], dtype=q.dtype),
        body_lin_acc_w=torch.arange(6, dtype=q.dtype).reshape(2, 1, 3))
    controller_size = 8 if cfg.control_mode == "indi" else 6
    controller = SimpleNamespace(observation=lambda: torch.full((2, controller_size), .7, dtype=q.dtype))
    return SimpleNamespace(cfg=cfg, robot=SimpleNamespace(data=data), base_id=0, joint_ids=[0, 1],
        # Nonzero origin Z distinguishes absolute world height from clone-relative height.
        scene=SimpleNamespace(env_origins=torch.tensor([[100., 200., 10.], [300., 400., 20.]], dtype=q.dtype)),
        actions=torch.arange(2*cfg.action_space, dtype=q.dtype).reshape(2, -1),
        _gravity=torch.tensor([[0., 0, -1.]], dtype=q.dtype).expand(2, -1),
        _commands=torch.arange(8, dtype=q.dtype).reshape(2, 4), _command_steps_left=torch.ones(2),
        _episode_steps=torch.tensor([0, 1]), _indi=controller, _wrench=controller, _cbrt=controller,
        _rotor_thrust=torch.full((2, 6), 4., dtype=q.dtype), _hover_per_rotor=5.,
        _rate_integral=torch.ones(2, 3, dtype=q.dtype))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("height,gravity", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("acceleration", [False, True])
def test_feature_values_dimensions_and_symmetry(mode, height, gravity, acceleration):
    cfg = config(mode, include_base_height=height, projected_gravity_in_obs=gravity,
                 obs_include_base_lin_acc=acceleration)
    env = observation_env(cfg)
    result = method("_get_observations", quat_apply_inverse=inverse, matrix_from_quat=matrix)(env)
    obs = result["policy"]
    transform = symmetry.BorinotSymmetry(cfg)
    extra = {MODES[2]: 9, "indi": 8, "wrench": 6, "cbrt": 6, "cbrt_indi": 6}.get(mode, 0)
    expected_size = 23 + cfg.action_space + extra + int(height) + 3*int(gravity) + 3*int(acceleration)
    assert obs.shape == (2, expected_size)
    assert cfg.observation_space == cfg.state_space == transform.observation_size == expected_size
    assert result["policy"] is result["critic"]
    assert ("gravity" in transform.obs_slices) == gravity
    assert ("height" in transform.obs_slices) == height
    d = env.robot.data
    parts = [d.root_lin_vel_b, d.root_ang_vel_b]
    if acceleration:
        parts.append(inverse(d.root_quat_w, d.body_lin_acc_w[:, 0]) * torch.tensor([[0.], [1.]]))
    parts += [d.joint_pos, d.joint_vel, env.actions]
    if gravity:
        parts.append(inverse(d.root_quat_w, env._gravity))
    parts.append(matrix(d.root_quat_w).flatten(1))
    if height:
        parts.append(torch.tensor([[1.25], [4.5]], dtype=d.root_pos_w.dtype))
    parts.append(env._commands)
    if mode == MODES[2]:
        parts += [torch.full((2, 6), .8, dtype=obs.dtype), env._rate_integral]
    elif extra:
        parts.append(torch.full((2, extra), .7, dtype=obs.dtype))
    assert torch.equal(obs, torch.cat(parts, -1))
    augmented = transform.observations(obs, "world_yaw_180")
    if height:
        assert torch.equal(augmented[:, transform.obs_slices["height"]], d.root_pos_w[:, 2:3])
        before = obs.clone()
        d.root_pos_w[:, 2] += 2.
        after = method("_get_observations", quat_apply_inverse=inverse, matrix_from_quat=matrix)(env)["policy"]
        before[:, transform.obs_slices["height"]] += 2.
        assert torch.equal(after, before)
    assert torch.equal(transform.observations(augmented, "world_yaw_180"), obs)


def test_requested_defaults():
    cfg = config()
    assert cfg.include_base_height is True and cfg.projected_gravity_in_obs is False
    assert cfg.observation_space == cfg.state_space == 32


@pytest.mark.parametrize("field", ["include_base_height", "projected_gravity_in_obs"])
@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_nonboolean_flags_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        config(**{field: value})
