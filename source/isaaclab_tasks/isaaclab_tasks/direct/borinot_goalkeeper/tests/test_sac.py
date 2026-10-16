# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the real task action method and SAC constructor without Isaac Sim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl_sac.algorithms import SAC

from test_control import action_env, config, method


def algorithm_class():
    path = Path(__file__).resolve().parents[1] / "agents/rsl_rl_sac_cfg.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    scope = {"torch": torch, "SAC": SAC}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


def sac_env():
    env, calls = action_env("thrust_and_PD_joint_pos_control")
    env.cfg.sac_action_mapping = True
    env.cfg.sac_joint_pos_offset = (0.2, -0.3)
    env._sac_joint_offset = torch.tensor(env.cfg.sac_joint_pos_offset)
    env._sac_joint_soft_limits = torch.deg2rad(torch.tensor([[-75., 75.], [-200., 200.]]))
    return env, calls


def test_joint_target_endpoints_midpoint_and_nonzero_offset():
    env, calls = sac_env()
    upper, lower = algorithm_class()._compute_action_scaling(env, "cpu")
    u = torch.linspace(-1, 1, 3)[:, None]
    env.actions[:] = (upper-lower)/2 + (upper+lower)/2 * u
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    expected = torch.deg2rad(torch.tensor([[-75., -200.], [0., 0.], [75., 200.]], dtype=torch.float64))
    torch.testing.assert_close(calls["position"], expected, atol=4e-7, rtol=0)
    assert not calls["effort"].any() and not calls["velocity"].any()


def test_rotor_piecewise_mapping_and_lag_with_independent_rotor_commands():
    env, _ = sac_env()
    env.cfg.rotor_action_scale = 99.0  # legacy PPO scale must not affect SAC
    env.cfg.remove_thrust_clipping = True  # SAC bounds remain intrinsic
    env.actions[:, :6] = torch.tensor([-1., -.5, 0., .25, .5, 1.])
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    requested = torch.tensor([0., 2.5, 5., 8.125, 11.25, 17.5], dtype=torch.float64)
    torch.testing.assert_close(env._rotor_thrust, (5 + env._motor_alpha*(requested-5)).expand(3, -1))


@pytest.mark.parametrize("field,value", [
    ("control_mode", "thrust_and_torque_control"), ("control_mode", "PI_rotors_PD_joint_pos_control"),
    ("thrust_to_weight", 1.0), ("thrust_to_weight", float("nan")),
    ("joint_target_scale", (0.0, 1.0)), ("sac_joint_pos_offset", (4.0, 0.0)),
    ("joint_soft_limit_calf", (200.0, -200.0)), ("joint_physical_limit_thigh", (float("nan"), 90.0)),
])
def test_invalid_sac_contract_rejected_before_simulation(field, value):
    cfg = config()
    cfg.control_mode = "thrust_and_PD_joint_pos_control"
    cfg.sac_action_mapping = True
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)


@pytest.mark.parametrize("loss", ["two_hot", "mse"])
@pytest.mark.parametrize("initialize_offset", [False, True])
def test_constructor_dispatch_initialization_update_and_checkpoint(monkeypatch, loss, initialize_offset):
    env, _ = sac_env()
    env.num_actions, env.num_envs = 8, 3
    env.cfg.sac_q_offset_init_actions = initialize_offset
    obs = TensorDict({"policy": torch.randn(3, 5), "critic": torch.randn(3, 7)}, batch_size=[3])
    cls = algorithm_class()
    import rsl_rl_sac.algorithms.sac as module
    from rsl_rl_sac.utils import resolve_callable
    original = resolve_callable
    monkeypatch.setattr(module, "resolve_callable", lambda name: cls if name == "BorinotTest" else original(name))
    cfg = dict(num_steps_per_env=2, obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[16], init_noise_std=.15),
        critic=dict(class_name="SACCriticModel", hidden_dims=[16], layer_norm=True,
                    distributional_loss=loss, distributional_num_bins=101,
                    distributional_linear_limit=20.0 if loss == "two_hot" else 0.0),
        algorithm=dict(class_name="BorinotTest", replay_buffer_size=64, n_steps=5, gamma=.97,
                       num_mini_batches=1, mini_batch_size=3, q_reduction_method="mean_pi_q_none"))
    alg = SAC.construct_algorithm(obs, env, copy.deepcopy(cfg), "cpu")
    mean_head = [m for m in alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    initial = alg.actor._squash_and_scale(mean_head.bias[:8])
    if initialize_offset:
        torch.testing.assert_close(initial, torch.zeros(8), atol=1e-7, rtol=0)
    else:
        torch.testing.assert_close(initial[-2:]*env._joint_scale + env._sac_joint_offset,
                                   torch.zeros(2), atol=1e-7, rtol=0)
    actions = alg.actor(obs).detach()
    batch = (obs, actions, torch.tensor([[1.], [-1.], [.1]]), obs,
             torch.tensor([[1.], [1.], [0.]]), torch.tensor([[0.], [1.], [0.]]), torch.tensor([[1], [2], [5]]))
    alg.replay_buffer = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch]))
    alg.update()
    assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
    state = copy.deepcopy(alg.save())
    restored = SAC.construct_algorithm(obs, env, copy.deepcopy(cfg), "cpu")
    assert restored.load(state, load_cfg=None, strict=True)
    torch.testing.assert_close(restored.actor(obs), alg.actor(obs))
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]
    if loss == "two_hot":
        torch.testing.assert_close(alg.critic.value_support, torch.linspace(-20, 20, 101), atol=2e-6, rtol=0)
