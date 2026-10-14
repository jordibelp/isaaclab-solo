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
    cfg = SimpleNamespace(kf=1.0, km=2.0, p_el_motor_efficiency=0.5,
                          energy_reward_scale=0.0, rwd_scale_energy=0.0, arm_copper_loss_coefficient=7.0, decimation=4)
    thrust = torch.zeros(3, 6, dtype=torch.float64)
    torque = torch.zeros(3, 2, dtype=torch.float64)
    dq = torch.zeros_like(torque)
    env = SimpleNamespace(
        cfg=cfg, _kf=cfg.kf, _km=cfg.km, _thrusts=thrust, _rotor_thrust=thrust,
        _leg_torques=torque, _leg_dof_ids=[0, 1], joint_ids=[0, 1],
        robot=SimpleNamespace(data=SimpleNamespace(applied_torque=torque, joint_vel=dq)),
        _power_sum_w=torch.zeros(3, 4, dtype=torch.float64), _power_samples=0, _physics_samples=0,
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
        # Independent reference: rotor drag torque times speed; current-based copper loss.
        drag_torque = env.cfg.km / env.cfg.kf * env._thrusts
        speed = (env._thrusts / env.cfg.kf).sqrt()
        rotor = (drag_torque * speed).sum(-1)
        rotor_loss = rotor / env.cfg.p_el_motor_efficiency - rotor
        arm = (env._leg_torques * env.robot.data.joint_vel).sum(-1)
        current = env._leg_torques / 0.225
        effective_resistance = env.cfg.arm_copper_loss_coefficient * 0.225**2
        copper = effective_resistance * current.square().sum(-1)
        expected.append(torch.stack((rotor, rotor_loss, arm, copper), dim=-1))
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
    assert env._power_sum_w[2, 2] == -11 * decimation  # preserve signed braking power
    assert env._power_sum_w[2, 3] == 119 * decimation  # copper loss stays positive


@pytest.mark.parametrize("weight", [0.0, -0.002, 0.002])
def test_reward_penalizes_average_watts_not_joules(task, weight):
    name, env, _, tree, _ = task
    setattr(env.cfg, "energy_reward_scale" if name == "borinot" else "rwd_scale_energy", weight)
    env._power_sum_w[:] = torch.tensor([[8, 4, 0, 0], [16, 12, 0, 0], [20, 0, 0, 0]], dtype=torch.float64)
    reward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards")
    mean_expr = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "mean_power_w" for t in node.targets))
    if name == "borinot":
        penalty = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "energy_reward" for t in node.targets))
    else:
        terms = next(node.value for node in ast.walk(reward) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "terms" for t in node.targets))
        penalty = next(value for key, value in zip(terms.keys, terms.values) if isinstance(key, ast.Constant) and key.value == "energy")
    namespace = {"self": env, "torch": torch}
    namespace["mean_power_w"] = eval(compile(ast.Expression(mean_expr), "mean", "eval"), namespace)
    actual = eval(compile(ast.Expression(penalty), "penalty", "eval"), namespace)
    torch.testing.assert_close(actual, weight * torch.tensor([3, 7, 5], dtype=torch.float64))


@pytest.mark.parametrize("torque,velocity,coefficient,mechanical,copper", [
    (2.0, 0.0, 7.0, 0.0, 28.0),    # holding costs power even without motion
    (2.0, 3.0, 7.0, 6.0, 28.0),
    (2.0, -3.0, 7.0, -6.0, 28.0),  # braking offsets some electrical input
    (1.0, -20.0, 7.0, -20.0, 7.0), # preserve net regenerative estimate, no clipping
    (2.0, 3.0, 0.0, 6.0, 0.0),
    (2.0, 0.0, 2.5, 0.0, 10.0),
])
def test_joint_mechanical_and_copper_power(task, torque, velocity, coefficient, mechanical, copper):
    _, env, sample, _, _ = task
    env.cfg.arm_copper_loss_coefficient = coefficient
    env._leg_torques[:, 0] = torque
    env.robot.data.joint_vel[:, 0] = velocity
    env._physics_samples = 1
    sample(env)
    expected = torch.tensor([0.0, 0.0, mechanical, copper], dtype=torch.float64)
    torch.testing.assert_close(env._power_sum_w, expected.expand_as(env._power_sum_w))


def test_config_energy_scale_is_nonpositive_and_copper_is_nonnegative(task):
    name, _, _, _, folder = task
    tree = ast.parse((folder / f"{name}_env_cfg.py").read_text())
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            values[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    scale_name = "energy_reward_scale" if name == "borinot" else "rwd_scale_energy"
    assert ast.literal_eval(values[scale_name]) <= 0.0
    assert ast.literal_eval(values["arm_copper_loss_coefficient"]) >= 0.0
