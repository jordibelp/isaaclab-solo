# SPDX-License-Identifier: BSD-3-Clause
"""Execute the evaluator's real pre-launch parser, without importing or starting Isaac Sim."""
import ast
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[5] / "scripts/rsl_rl/eval_borinot_goalkeeper.py"
TREE = ast.parse(SCRIPT.read_text())
STOP = next(i for i, node in enumerate(TREE.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "simulation_app" for t in node.targets))
PREFIX = ast.Module(body=[n for n in TREE.body[:STOP]
                         if not (isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")], type_ignores=[])
HYDRA = ["env.control_mode=thrust_and_torque_control", "env.shot_level=3",
         "env.critic_obs_includes_rotor_omega=True", "env.enabled_self_collisions=True",
         "env.ball_mass_range_kg=[0.05,0.2]", "env.rwd_scale_action_rate=-1.0",
         "env.rwd_scale_energy=-1.0e-6", "env.rwd_scale_thrust_limits=0.0",
         "env.rwd_scale_torque_limits=0.0", "env.rwd_scale_action_limits=0.0"]
DELAYS = ["--actuator_delays", "0.0", "0.01", "0.02", "0.04"]


def parse(monkeypatch, tokens):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *tokens])
    scope = {"AppLauncher": SimpleNamespace(add_app_launcher_args=lambda parser: None)}
    exec(compile(PREFIX, str(SCRIPT), "exec"), scope)
    return scope


@pytest.mark.parametrize("tokens", [DELAYS + HYDRA, HYDRA + DELAYS,
                                   HYDRA[:2] + DELAYS + HYDRA[2:]])
def test_delay_list_can_precede_follow_or_split_hydra_overrides(monkeypatch, tokens):
    scope = parse(monkeypatch, ["--checkpoint", "/weights.pt", *tokens])
    assert scope["args_cli"].actuator_delays == [0., .01, .02, .04]
    assert scope["hydra_args"] == HYDRA


def test_jordis_exact_command_preserves_every_override_and_original_provenance(monkeypatch):
    command = """--checkpoint ~/IsaacLab-dirty/logs/skrl/checkpoints/1026_1qrb60vv_best_model_curriculum_idx_5.pt --num_episodes 512 --seed 7 --actuator_delays 0.0 0.01 0.02 0.04 env.control_mode=thrust_and_torque_control env.shot_level=3 env.critic_obs_includes_rotor_omega=True env.enabled_self_collisions=True 'env.ball_mass_range_kg=[0.05,0.2]' env.rwd_scale_action_rate=-1.0 env.rwd_scale_energy=-1.0e-6 env.rwd_scale_thrust_limits=0.0 env.rwd_scale_torque_limits=0.0 env.rwd_scale_action_limits=0.0"""
    tokens = shlex.split(command)
    scope = parse(monkeypatch, tokens)
    assert scope["args_cli"].actuator_delays == [0., .01, .02, .04]
    assert scope["args_cli"].num_episodes == 512 and scope["args_cli"].seed == 7
    assert scope["hydra_args"] == HYDRA
    assert scope["command_line"] == [str(SCRIPT), *tokens]


def test_other_cli_options_and_repeated_hydra_keys_keep_their_order(monkeypatch):
    overrides = ["env.shot_level=1", "env.shot_level=3", "agent.seed=4", "+env.extra=True"]
    scope = parse(monkeypatch, ["--checkpoint", "/checkpoints/name=value.pt", overrides[0],
                  *DELAYS, overrides[1], "--num_envs", "64", overrides[2], "--label", "direct=baseline",
                  overrides[3], "--seed", "7"])
    assert scope["hydra_args"] == overrides
    assert scope["args_cli"].num_envs == 64 and scope["args_cli"].seed == 7
    assert scope["args_cli"].label == "direct=baseline"
    assert scope["args_cli"].checkpoint == "/checkpoints/name=value.pt"


def test_equals_option_form_and_scientific_notation(monkeypatch):
    scope = parse(monkeypatch, ["--checkpoint=/weights.pt", "--actuator_delays=0.0", "1e-2", *HYDRA])
    assert scope["args_cli"].actuator_delays == [0., .01] and scope["hydra_args"] == HYDRA


@pytest.mark.parametrize("values", [[], ["oops"], ["0.0", "0.02oops"]])
def test_empty_or_malformed_sweep_is_rejected_before_simulator_start(monkeypatch, values):
    with pytest.raises(SystemExit) as error:
        parse(monkeypatch, ["--checkpoint", "/weights.pt", "--actuator_delays", *values, *HYDRA])
    assert error.value.code == 2


@pytest.mark.parametrize("override", ["env.actuator_delay=0.02", "env.actuator_delay_range=[0.0,0.04]"])
def test_sweep_still_rejects_conflicting_environment_delay_overrides(monkeypatch, override):
    with pytest.raises(SystemExit) as error:
        parse(monkeypatch, ["--checkpoint", "/weights.pt", *DELAYS, override])
    assert error.value.code == 2


def test_environment_delay_without_sweep_is_unchanged(monkeypatch):
    scope = parse(monkeypatch, ["--checkpoint", "/weights.pt", "env.actuator_delay=0.02"])
    assert scope["args_cli"].actuator_delays is None
    assert scope["hydra_args"] == ["env.actuator_delay=0.02"]
