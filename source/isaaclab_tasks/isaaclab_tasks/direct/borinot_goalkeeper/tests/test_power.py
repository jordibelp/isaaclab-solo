# SPDX-License-Identifier: BSD-3-Clause
"""Check the real sampler/reward expressions without importing Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=["borinot", "borinot_goalkeeper"])
def task(request):
    name = request.param
    folder = Path(__file__).resolve().parents[2] / name
    tree = ast.parse((folder / f"{name}_env.py").read_text())
    methods = next(node for node in tree.body if isinstance(node, ast.ClassDef)).body
    sample = next(node for node in methods if isinstance(node, ast.FunctionDef) and node.name == "_sample_power")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[sample], type_ignores=[]), str(folder), "exec"), namespace)
    cfg = SimpleNamespace(kf=1.0, km=2.0, p_el_motor_efficiency=0.5, power_cost=0.0, decimation=4)
    thrust = torch.zeros(3, 6, dtype=torch.float64)
    torque = torch.zeros(3, 2, dtype=torch.float64)
    dq = torch.zeros_like(torque)
    env = SimpleNamespace(
        cfg=cfg, _kf=cfg.kf, _km=cfg.km, _thrusts=thrust, _rotor_thrust=thrust,
        _leg_torques=torque, _leg_dof_ids=[0, 1], joint_ids=[0, 1],
        robot=SimpleNamespace(data=SimpleNamespace(applied_torque=torque, joint_vel=dq)),
        _power_sum_w=torch.zeros(3, 2, dtype=torch.float64), _power_samples=0, _physics_samples=0,
    )
    return name, env, namespace["_sample_power"], tree, folder


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_completed_substeps_include_last_and_average_without_double_counting(task, decimation):
    _, env, sample, _, _ = task
    env.cfg.decimation = decimation
    expected = []
    for step in range(decimation):
        sample(env)  # before replacing the previous substep's outputs
        assert env._power_samples == step
        env._thrusts[:] = torch.tensor([[0]*6, [step+1]*6, [0, 1, 2, 3, 4, 5]], dtype=torch.float64)
        env._leg_torques[:] = torch.tensor([[0, 0], [2, -3], [-1, 4]], dtype=torch.float64)
        env.robot.data.joint_vel[:] = torch.tensor([[0, 0], [step+1, 2], [3, -2]], dtype=torch.float64)
        env._physics_samples += 1
        # Independent physical reference: drag torque times speed; absolute work per arm joint.
        drag_torque = env.cfg.km / env.cfg.kf * env._thrusts
        speed = (env._thrusts / env.cfg.kf).sqrt()
        rotor = (drag_torque * speed).sum(-1) / env.cfg.p_el_motor_efficiency
        arm = (env._leg_torques * env.robot.data.joint_vel).abs().sum(-1)
        expected.append(torch.stack((rotor, arm), dim=-1))
    assert env._power_samples == decimation - 1
    sample(env)  # final physics result, before rewards/resets
    assert env._power_samples == decimation
    torch.testing.assert_close(env._power_sum_w / decimation, torch.stack(expected).mean(0))
    previous = env._power_sum_w.clone()
    sample(env)
    sample(env)
    torch.testing.assert_close(env._power_sum_w, previous)
    assert env._power_samples == decimation
    assert env._power_sum_w[0].sum() == 0
    assert env._power_sum_w[2, 1] == 11 * decimation  # braking at one joint must not cancel the other


@pytest.mark.parametrize("weight", [0.0, 0.002])
def test_reward_penalizes_average_watts_not_joules(task, weight):
    name, env, _, tree, _ = task
    env.cfg.power_cost = weight
    env._power_sum_w[:] = torch.tensor([[8, 4], [16, 12], [20, 0]], dtype=torch.float64)
    reward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards")
    mean_expr = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "mean_power_w" for t in node.targets))
    if name == "borinot":
        penalty = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "power_pen" for t in node.targets))
        sign = 1
    else:
        terms = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "terms" for t in node.targets))
        penalty = next(value for key, value in zip(terms.keys, terms.values) if isinstance(key, ast.Constant) and key.value == "power")
        sign = -1
    namespace = {"self": env, "torch": torch}
    namespace["mean_power_w"] = eval(compile(ast.Expression(mean_expr), "mean", "eval"), namespace)
    actual = eval(compile(ast.Expression(penalty), "penalty", "eval"), namespace)
    torch.testing.assert_close(actual, sign * weight * torch.tensor([3, 7, 5], dtype=torch.float64))


def test_config_penalty_is_disabled_by_default(task):
    name, _, _, _, folder = task
    tree = ast.parse((folder / f"{name}_env_cfg.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "power_cost":
            assert ast.literal_eval(node.value) == 0.0
            return
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "power_cost" for t in node.targets):
            assert ast.literal_eval(node.value) == 0.0
            return
    pytest.fail("Missing power_cost configuration field")
