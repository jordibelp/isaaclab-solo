# SPDX-License-Identifier: BSD-3-Clause
"""Control invariants needed when physical parameters differ between robots."""
import ast
from types import SimpleNamespace

import pytest
import torch

from test_control import FOLDER, action_env, config, method


def configure_randomization(cfg):
    node = next(n for n in ast.parse((FOLDER / "events.py").read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "configure_randomization")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "events.py", "exec"), scope)
    scope["configure_randomization"](cfg)


def test_requested_randomizations_enabled_by_default():
    assert config().event_randomization == {
        "joint_friction": True, "add_base_mass": True, "racket_mass": True, "base_com": True,
    }


@pytest.mark.parametrize("disabled", ["joint_friction", "add_base_mass", "racket_mass", "base_com"])
def test_single_ablation_preserves_other_terms_and_custom_parameters(disabled):
    cfg = config()
    terms = {tag: object() for tag in cfg.event_randomization}
    cfg.events = SimpleNamespace(**terms)
    cfg.event_randomization[disabled] = False  # same late update as Hydra
    configure_randomization(cfg)
    for tag, term in terms.items():
        assert getattr(cfg.events, tag) is (None if tag == disabled else term)


@pytest.mark.parametrize("global_off", [True, False])
def test_master_off_and_explicit_events_none(global_off):
    cfg = config()
    cfg.events = SimpleNamespace() if global_off else None
    cfg.include_events_randomization = not global_off
    configure_randomization(cfg)
    assert cfg.events is None


@pytest.mark.parametrize("change", ["typo", "missing", "not_bool"])
def test_invalid_toggle_fails_before_simulator_construction(change):
    cfg = config()
    cfg.events = SimpleNamespace()
    if change == "typo":
        cfg.event_randomization["racket_mas"] = False
    elif change == "missing":
        del cfg.event_randomization["racket_mass"]
    else:
        cfg.event_randomization["racket_mass"] = "false"
    with pytest.raises(ValueError, match="event_randomization"):
        configure_randomization(cfg)


def test_com_shift_changes_physical_moment_not_rotor_commands():
    env, calls = action_env("thrust_and_torque_control", motor_time_constant_s=0.0)
    env._base_com_offset = torch.tensor([[0.0, 0.0, 0.0], [0.01, -0.01, 0.0], [-0.01, 0.01, 0.01]])
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(env._rotor_thrust, torch.full_like(env._rotor_thrust, 5.0))
    # Six vertical 5 N forces: delta torque = -delta_com cross (0, 0, 30 N).
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.3, -0.3, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(calls["wrench"]["torques"][:, 0], expected)
