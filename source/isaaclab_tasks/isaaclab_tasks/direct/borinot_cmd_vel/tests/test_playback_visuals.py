# SPDX-License-Identifier: BSD-3-Clause
"""Independent frame/geometry oracles; no simulator or USD import required."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scipy.spatial.transform import Rotation

FOLDER = Path(__file__).resolve().parents[1]
scope = {"torch": torch, "math": math}
for path, names in ((FOLDER / "task_math.py", {"heading_rotation"}),
                    (FOLDER.parents[1] / "controllers/borinot_indi.py", {"rotation_from_quat"})):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
tree = ast.parse((FOLDER / "playback_visuals.py").read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
exec(compile(ast.Module(body=nodes, type_ignores=[]), "playback_visuals.py", "exec"), scope)


def fixture_env(mode="cbrt_indi", sac=False):
    rotation = Rotation.from_euler("xyz", [.6, -.8, 1.2])
    q = torch.tensor(rotation.as_quat(scalar_first=True), dtype=torch.float64)[None]
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    cfg = SimpleNamespace(control_mode=mode, sac_action_mapping=sac, body_rate_scale=(2., 3., 4.),
                          thrust_to_weight=2., rotor_pos_b=((0., -.185, 0.), (0., .185, 0.),
                          (.16, .0925, 0.), (-.16, -.0925, 0.), (.16, -.0925, 0.), (-.16, .0925, 0.)))
    data = SimpleNamespace(root_pos_w=tensor([[10., -7., 3.]]), root_quat_w=q,
                           root_ang_vel_b=tensor([[-.3, .8, -.1]]), root_lin_vel_w=tensor([[.7, -.2, .4]]))
    env = SimpleNamespace(cfg=cfg, robot=SimpleNamespace(data=data), _hover_per_rotor=5.,
        _actuator_commands=tensor([[.1, .2, .3, .4, .5, .6, 0, 0]]),
        _rotor_thrust=tensor([[.5, 1., 1.5, 2., 2.5, 3.]]), actions=tensor([[.2, -.4, 1.5, 0, 0, 0]]),
        _commands=tensor([[.3, -.5, .7, 9.]]), _base_push_force_z_w=tensor([-12.]),
        _base_push_points_b=tensor([[.03, -.04, .05]]))
    return env, rotation.as_matrix()


@pytest.mark.parametrize("sac", [False, True])
def test_signal_sources_and_full_body_vs_heading_frames(sac):
    env, r = fixture_env(sac=sac)
    rotation = torch.tensor(r)
    heading = torch.tensor(Rotation.from_euler("z", 1.2).as_matrix())
    before = {k: v.clone() for k, v in vars(env).items() if isinstance(v, torch.Tensor)}
    result = scope["overlay_vectors"](env, scope["OVERLAYS"])
    for key in ("rotor_command", "rotor_actual"):
        origin, vector, values = result[key]
        expected = torch.arange(1., 7., dtype=torch.float64) * (1 if key == "rotor_command" else .5)
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(vector / .05, expected[:, None] * rotation[:, 2])
        hubs = torch.tensor(env.cfg.rotor_pos_b, dtype=torch.float64)
        hubs[:, 0] += -.012 if key == "rotor_command" else .012
        hubs[:, 2] += .025
        torch.testing.assert_close(origin, env.robot.data.root_pos_w + hubs @ rotation.T)
    expected_rate = torch.tensor([.4, -1.2, 4.] if sac else [.2, -.4, 1.5], dtype=torch.float64)
    torch.testing.assert_close(result["policy_rate"][2], expected_rate)
    torch.testing.assert_close(result["policy_rate"][1][0] / .25, rotation @ expected_rate)
    torch.testing.assert_close(result["actual_rate"][1][0] / .25, rotation @ env.robot.data.root_ang_vel_b[0])
    torch.testing.assert_close(result["command_velocity"][1][0], heading @ env._commands[0, :3])
    assert abs(float(result["command_velocity"][1][0, 2]) - .7) < 1e-12
    torch.testing.assert_close(result["actual_velocity"][1], env.robot.data.root_lin_vel_w)
    torch.testing.assert_close(result["disturbance"][1], torch.tensor([[0., 0., -.36]], dtype=torch.float64))
    torch.testing.assert_close(result["disturbance"][0][0], env.robot.data.root_pos_w[0] + rotation @ env._base_push_points_b[0])
    for key, value in before.items():
        assert torch.equal(getattr(env, key), value), key


@pytest.mark.parametrize("mode", ["wrench", "dsam_indi", "thrust_and_torque_control"])
def test_no_invented_policy_rate_for_other_modes(mode):
    env, _ = fixture_env(mode=mode)
    assert scope["overlay_vectors"](env, ["policy_rate"]) == {}


@pytest.mark.parametrize("size", [.25, 1., 3.])
def test_arrow_endpoints_magnitude_orientation_and_zero(size):
    generator = torch.Generator().manual_seed(81)
    vectors = torch.cat((torch.randn(100, 3, generator=generator, dtype=torch.float64),
                         torch.tensor([[0, 0, 0], [0, 0, -1], [0, 0, 1], [1e-5, 0, -1],
                                       [1e-9, -1e-9, 0]], dtype=torch.float64)))
    origins = torch.randn(vectors.shape, generator=generator, dtype=torch.float64)
    positions, quats, scales, indices = scope["arrow_instances"](origins, vectors, size)
    n = len(vectors)
    rotation = Rotation.from_quat(quats.numpy(), scalar_first=True).as_matrix()
    z = torch.from_numpy(rotation[:, :, 2])
    # Compute actual transformed primitive bases/tips, not the implementation formula.
    bases = positions[:n] - z[:n] * scales[:n, 2:3] / 2
    tips = positions[n:] + z[n:] * scales[n:, 2:3] / 2
    torch.testing.assert_close(bases, origins, atol=1e-10, rtol=0)
    torch.testing.assert_close(tips, origins + size*vectors, atol=1e-10, rtol=0)
    assert torch.isfinite(quats).all() and torch.isfinite(scales).all()
    assert not scales[100].any() and not scales[n+100].any()
    assert indices == [0]*n + [1]*n


def test_default_off_toggling_and_close_do_not_touch_simulation():
    visuals = scope["FlightVisuals"](None)
    assert not any(visuals.enabled.values())
    visuals.update()  # must not import Isaac/allocate markers/access env
    assert visuals.markers == {} and visuals.readout() == ""
    visibility = []
    visuals.markers["rotor_command"] = SimpleNamespace(set_visibility=visibility.append)
    visuals.set_enabled("rotor_command", True)
    visuals.values["rotor_command"] = torch.tensor([1., 2.])
    assert "1.00, +2.00 N" in visuals.readout()
    visuals.set_enabled("rotor_command", False)
    assert visibility == [False] and visuals.readout() == ""
    visuals.close()
    assert visibility == [False, False]
