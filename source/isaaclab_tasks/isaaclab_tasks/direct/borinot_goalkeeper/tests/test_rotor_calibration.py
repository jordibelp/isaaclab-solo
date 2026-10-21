# SPDX-License-Identifier: BSD-3-Clause
"""Check the audited rotor map and its effective actuator limits on CPU."""
import ast
import math
from types import SimpleNamespace

import pytest
import torch

from test_control import FOLDER, MODES, action_env, config, method


EXPECTED_POSITIONS = ((0.0, -0.185, 0.0), (0.0, 0.185, 0.0),
                      (0.1602147, 0.0925, 0.0), (-0.1602147, -0.0925, 0.0),
                      (0.1602147, -0.0925, 0.0), (-0.1602147, 0.0925, 0.0))
EXPECTED_SIGNS = (1.0, -1.0, 1.0, -1.0, -1.0, 1.0)


def task_class(simple, suffix):
    task = "borinot_goalkeeper_simple" if simple else "borinot_goalkeeper"
    path = FOLDER.parent / task / f"{task}_env{suffix}.py"
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))


def task_config(simple):
    cfg = config()
    for node in task_class(simple, "_cfg").body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                setattr(cfg, node.targets[0].id, ast.literal_eval(node.value))
            except (ValueError, TypeError):
                pass
    cfg.km_over_kf = cfg.km / cfg.kf
    return cfg


def initialize_rotors(env, simple, mass):
    """Execute the actual constructor's mass, cap and allocation block."""
    env.robot.data.default_mass = torch.tensor([[mass]], dtype=torch.float64)
    env.robot.root_physx_view = SimpleNamespace(get_coms=lambda: torch.zeros(3, 1, 7))
    init = next(n for n in task_class(simple, "").body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "self.robot_mass_kg")
    end = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "self._rotor_thrust")
    exec(compile(ast.Module(body=init.body[start:end + 1], type_ignores=[]), "rotor_init", "exec"),
         {"self": env, "cfg": env.cfg, "torch": torch})
    # The existing CPU action fixture uses float64; real simulator tensors use float32.
    for name in ("_allocation", "_allocation_inverse", "_rotor_thrust"):
        setattr(env, name, getattr(env, name).to(env.actions.dtype))


@pytest.mark.parametrize("simple", [False, True])
def test_hardware_layout_and_coefficients(simple):
    cfg = task_config(simple)
    assert cfg.rotor_pos_b == EXPECTED_POSITIONS
    assert cfg.rotor_spin == EXPECTED_SIGNS
    assert all(math.isclose(math.hypot(x, y), 0.185, abs_tol=3e-8) for x, y, _ in cfg.rotor_pos_b)
    assert (cfg.kf, cfg.km, cfg.rotor_thrust_limit_n) == (3.847e-6, 5.361e-8, 10.0)
    assert cfg.km_over_kf == pytest.approx(0.013935534182479855)
    assert not cfg.remove_thrust_clipping
    assert cfg.motor_time_constant_s == 0.04


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("mass", [2.910538, 3.7])
@pytest.mark.parametrize("mode,sac", [(m, False) for m in MODES] + [(m, True) for m in MODES[:2]])
def test_ten_newton_cap_and_mapping_at_different_nominal_masses(simple, mass, mode, sac):
    env, _ = action_env(mode, motor_time_constant_s=0.0)
    env.cfg = task_config(simple)
    env.cfg.control_mode, env.cfg.sac_action_mapping = mode, sac
    env.cfg.motor_time_constant_s = 0.0
    env._sac_joint_offset = torch.zeros(2)
    initialize_rotors(env, simple, mass)
    assert env.cfg.thrust_to_weight * env._hover_per_rotor == pytest.approx(10.0)
    env.actions.fill_(100.0)
    if mode == MODES[2]:
        env.actions[:, 1:4] = 0.0
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(env._rotor_thrust, torch.full_like(env._rotor_thrust, 10.0))
    torch.testing.assert_close(env._actuator_commands[:, :6], torch.ones_like(env._actuator_commands[:, :6]))
    if sac:
        env.actions[:, :6] = torch.tensor([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0])
        method("borinot_goalkeeper_env.py", "_apply_action")(env)
        hover = mass * 9.81 / 6
        expected = torch.tensor([0., .5 * hover, hover, hover + .25 * (10 - hover),
                                 hover + .5 * (10 - hover), 10.], dtype=env._rotor_thrust.dtype)
        torch.testing.assert_close(env._rotor_thrust, expected.expand(3, -1))


@pytest.mark.parametrize("simple", [False, True])
def test_each_motor_wrench_matches_independent_cross_product(simple):
    env, _ = action_env(MODES[0], motor_time_constant_s=0.0)
    env.cfg = task_config(simple)
    initialize_rotors(env, simple, 2.910538)
    positions = torch.tensor(EXPECTED_POSITIONS)
    forces = torch.tensor([0., 0., 1.]).expand(6, -1)
    moments = torch.linalg.cross(positions, forces)
    moments[:, 2] = torch.tensor(EXPECTED_SIGNS) * (5.361e-8 / 3.847e-6)
    expected = torch.cat((torch.ones(6, 1), moments), dim=1)
    torch.testing.assert_close(env._allocation.T, expected.to(env._allocation.dtype))


@pytest.mark.parametrize("simple", [False, True])
def test_lower_twr_override_is_preserved(simple):
    env, _ = action_env(MODES[0], motor_time_constant_s=0.0)
    env.cfg = task_config(simple)
    env.cfg.thrust_to_weight = 1.5
    initialize_rotors(env, simple, 2.910538)
    assert env.cfg.thrust_to_weight == 1.5


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_physical_limit_rejected(simple, limit):
    cfg = task_config(simple)
    cfg.rotor_thrust_limit_n = limit
    resolver = next(n for n in task_class(simple, "_cfg").body
                    if isinstance(n, ast.FunctionDef) and n.name == "configure_control")
    scope = {"math": math}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), "cfg", "exec"), scope)
    with pytest.raises(ValueError, match="rotor_thrust_limit_n"):
        scope["configure_control"](cfg)
