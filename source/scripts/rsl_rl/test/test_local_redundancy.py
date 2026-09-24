# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical CPU/GPU checks without starting Isaac Sim.

Run with the command in LOCAL_REDUNDANCY.md from env_isaaclab.
"""

import ast
import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import local_redundancy as lr
import pytest
import torch
from rsl_rl.modules import ActorCritic
from rsl_rl_sac.models import SACActorModel, SACCriticModel
from tensordict import TensorDict
from torch import nn

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def cfg():
    # Read the real config defaults without importing Isaac Sim's environment
    # package. Real Hydra resolution is also exercised by the training smoke runs.
    tree = ast.parse((ROOT / "source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RslRlLocalRedundancyCfg")
    result = {node.target.id: ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.AnnAssign)}
    result.update(num_samples=19, batch_size=4)
    return result


@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"))]
)
def device(request):
    return request.param


def ppo(device="cpu", *, state_dependent_std=False):
    obs = TensorDict({"policy": torch.randn(23, 7, device=device)}, batch_size=[23])
    policy = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        3,
        actor_hidden_dims=[8, 5],
        critic_hidden_dims=[8, 5],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        state_dependent_std=state_dependent_std,
    ).to(device)
    policy.update_normalization(obs)
    policy.act(obs)  # non-leaf distribution cache, as in real training
    return SimpleNamespace(alg=SimpleNamespace(policy=policy), device=device), obs


def sac(device="cpu", *, state_dependent_std=True, distributional_loss="mse"):
    obs = TensorDict({"policy": torch.randn(23, 7, device=device)}, batch_size=[23])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(
        obs,
        groups,
        "actor",
        3,
        hidden_dims=[8, 5],
        obs_normalization=True,
        state_dependent_std=state_dependent_std,
        layer_norm=True,
    ).to(device)
    critic = SACCriticModel(
        obs,
        groups,
        "critic",
        1,
        hidden_dims=[8, 5],
        num_actions=3,
        distributional_loss=distributional_loss,
        obs_normalization=True,
        layer_norm=True,
    ).to(device)
    actor.update_normalization(obs)
    critic.update_normalization(obs)
    actor.sample_action_logp(obs)
    return SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic), device=device), obs


@pytest.mark.parametrize("distributional_loss", ["two_hot", "hl_gauss"])
def test_ce_critic_probe_still_measures_scalar_q(cfg, device, distributional_loss):
    runner, obs = sac(device, distributional_loss=distributional_loss)
    assert runner.alg.critic.critic1[-1].out_features > 1
    measured = lr.measure(runner, obs, cfg, 0)
    for name in ("critic1", "critic2"):
        assert measured[name]["local_redundancy_output_dim"] == 1
        assert math.isfinite(measured[name]["local_redundancy"])


def test_popart_critic_probe_measures_raw_unit_q(cfg, device):
    runner, obs = sac(device, distributional_loss="mse_target_norm_popart")
    base = lr.measure(runner, obs, cfg, 0)
    runner.alg.critic.popart_std.fill_(3.0)
    runner.alg.critic.popart_mean.fill_(-7.0)
    scaled = lr.measure(runner, obs, cfg, 0)
    for name in ("critic1", "critic2"):
        # Q = std * head + mean, so squared gradient norms grow by std^2 and ignore the mean.
        assert scaled[name]["local_redundancy"] == pytest.approx(9 * base[name]["local_redundancy"], rel=1e-4)


def test_linear_gradient_matches_closed_form_and_no_cancellation(device):
    model = nn.Linear(2, 1).to(device)
    x = torch.tensor([[3.0, 4.0], [3.0, 4.0]], device=device)
    noise = torch.tensor([[1.0], [-1.0]], device=device)
    norms, count = lr.regression_gradient_norms(model(x), noise, list(model.parameters()), target_std=2.0)
    # ||grad ell_i||^2 = z_i^2 / sigma^2 * (||x_i||^2 + 1).
    torch.testing.assert_close(norms, torch.full((2,), 26 / 4, dtype=torch.float64, device=device))
    assert count == 3
    assert all(parameter.grad is None for parameter in model.parameters())
    # A batch-mean gradient is zero here, while the correct mean squared norm is 6.5.
    assert norms.mean() > 0


def test_large_prediction_keeps_unit_noise(device):
    model = nn.Linear(2, 1).to(device)
    with torch.no_grad():
        model.bias.fill_(1.0e20)
    norms, _ = lr.regression_gradient_norms(
        model(torch.zeros(2, 2, device=device)),
        torch.ones(2, 1, device=device),
        list(model.parameters()),
        1.0,
    )
    torch.testing.assert_close(norms, torch.ones_like(norms))


@pytest.mark.parametrize("factory", [ppo, sac])
def test_microbatch_invariance_and_sampling_config(cfg, device, factory):
    runner, obs = factory(device)
    expected = lr.measure(runner, obs, cfg, 200)
    for size in (1, 7):
        actual = lr.measure(runner, obs, {**cfg, "batch_size": size}, 200)
        for name in expected:
            for key in expected[name]:
                assert actual[name][key] == pytest.approx(expected[name][key], rel=2.0e-6, abs=1.0e-8)
    real = lr.measure(runner, obs, {**cfg, "input_mode": "observations", "num_samples": 100}, 0)
    assert real["actor"]["local_redundancy_num_samples"] == 23
    resampled = lr.measure(runner, obs, {**cfg, "resample": True}, 0)
    assert resampled["actor"]["local_redundancy"] != expected["actor"]["local_redundancy"]
    fixed = {**cfg, "resample": False}
    assert lr.measure(runner, obs, fixed, 200) == lr.measure(runner, obs, fixed, 0)


@pytest.mark.parametrize("factory", [ppo, sac])
def test_training_state_rng_hooks_and_next_step_unchanged(cfg, device, factory):
    runner, obs = factory(device)
    model = getattr(runner.alg, "policy", None) or runner.alg.actor
    modules = [model] if hasattr(runner.alg, "policy") else [model, runner.alg.critic]
    params = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.Adam([p for p in params if p.requires_grad])
    for parameter in params:
        if parameter.requires_grad:
            parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    before = [copy.deepcopy(module.state_dict()) for module in modules]
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    distribution = model.distribution
    modes = [sub.training for module in modules for sub in module.modules()]
    baseline = lr.measure(runner, obs, cfg, 0)
    calls = []
    # These hooks emulate CBP forward capture and a loss-modifying regenerative hook.
    linear = next(sub for sub in model.modules() if isinstance(sub, nn.Linear))
    forward_handle = linear.register_forward_hook(lambda *_: calls.append("forward"))
    grad_handle = linear.weight.register_hook(lambda gradient: gradient + 1000)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    with torch.inference_mode():
        inference_obs = obs.clone()
        assert lr.measure(runner, inference_obs, cfg, 0) == baseline
    assert not calls
    assert model.distribution is distribution
    assert modes == [sub.training for module in modules for sub in module.modules()]
    for module, state in zip(modules, before):
        for key, value in module.state_dict().items():
            assert torch.equal(value, state[key])
    for parameter, gradient in zip(params, grads):
        assert (parameter.grad is None) if gradient is None else torch.equal(parameter.grad, gradient)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for key, value in optimizer_state["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, optimizer.state_dict()["state"][key][field])
    forward_handle.remove()
    grad_handle.remove()

    # Same subsequent stochastic input and Adam step with/without the probe.
    def step():
        optimizer.zero_grad()
        for parameter in params:
            if parameter.requires_grad:
                parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        return [p.detach().clone() for p in params]

    after_probe = step()
    for module, state in zip(modules, before):
        module.load_state_dict(state)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    without_probe = step()
    assert all(torch.equal(a, b) for a, b in zip(after_probe, without_probe))


def _race_class(filename, class_name):
    directory = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/agents"
    package = ModuleType("_local_redundancy_test_agents")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{filename}", directory / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


@pytest.mark.parametrize("shared", [False, True])
def test_race_encoders_included_and_shared_weights_counted_once(cfg, shared):
    cls = _race_class("env_params_conditioned_encoder_actor", "EnvParamsConditionedEncoderActor")
    obs = {"policy": torch.randn(23, 7)}
    model = cls(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        3,
        current_obs_dim=5,
        env_params_dim=2,
        env_params_encoder_hidden_dims=[4],
        env_params_latent_dim=2,
        actor_hidden_dims=[6],
        critic_hidden_dims=[6],
        actor_critic_share_latent_encoding=shared,
    )
    runner = SimpleNamespace(alg=SimpleNamespace(policy=model), device="cpu")
    results = lr.measure(runner, obs, cfg, 0)
    for name in ("actor", "critic"):
        expected = sum(p.numel() for p in getattr(model, name).parameters())
        expected += sum(p.numel() for p in getattr(model, f"{name}_env_params_encoder").parameters())
        assert results[name]["local_redundancy_parameter_count"] == expected
    for parameter in model.actor_env_params_encoder.parameters():
        parameter.requires_grad_(False)
    frozen = lr.measure(runner, obs, cfg, 0)
    assert frozen["actor"]["local_redundancy"] < results["actor"]["local_redundancy"]
    assert frozen["actor"]["local_redundancy_parameter_count"] < results["actor"]["local_redundancy_parameter_count"]


@pytest.mark.parametrize("factory", [ppo, sac])
def test_gaussian_boundary_and_mean_probe_ignore_running_stats_and_exploration(cfg, factory):
    runner, obs = factory(state_dependent_std=False)
    original = lr.measure(runner, obs, cfg, 0)
    model = getattr(runner.alg, "policy", None) or runner.alg.actor
    with torch.no_grad():
        getattr(model, "std", getattr(model, "log_std", None)).add_(5)
        for name, buffer in model.named_buffers():
            if "normalizer" in name and buffer.is_floating_point():
                buffer.add_(10)
    actual = lr.measure(runner, obs, cfg, 0)
    assert actual["actor"] == original["actor"]


def test_sac_both_online_critics_and_action_sources(cfg):
    runner, obs = sac()
    original = lr.measure(runner, obs, cfg, 0)
    with torch.no_grad():
        for name in ("critic1_target", "critic2_target"):
            for parameter in getattr(runner.alg.critic, name).parameters():
                parameter.add_(1000)
    assert lr.measure(runner, obs, cfg, 0) == original
    assert original["critic"]["local_redundancy"] == pytest.approx(
        (original["critic1"]["local_redundancy"] + original["critic2"]["local_redundancy"]) / 2
    )
    policy_actions = lr.measure(runner, obs, {**cfg, "sac_action_source": "policy_mean"}, 0)
    assert policy_actions["critic"]["local_redundancy"] != original["critic"]["local_redundancy"]
    assert policy_actions["actor"] == original["actor"]


def test_asymmetric_tcn_and_privileged_critic_paths(cfg, device):
    cls = _race_class("imu_tcn_actor_critic", "ActorCriticFootImuTcn")
    obs = {"policy": torch.randn(23, 29, device=device), "critic": torch.randn(23, 7, device=device)}
    model = cls(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        3,
        current_obs_dim=5,
        imu_history_len=8,
        imu_dim=3,
        tcn_channels=4,
        tcn_latent_dim=2,
        tcn_kernel_size=3,
        asymmetric_actor_critic=True,
        env_params_dim=2,
        env_params_encoder_hidden_dims=[4],
        env_params_latent_dim=2,
        actor_hidden_dims=[6],
        critic_hidden_dims=[6],
    ).to(device)
    runner = SimpleNamespace(alg=SimpleNamespace(policy=model), device=device)
    results = lr.measure(runner, obs, cfg, 0)
    for name, encoder in (("actor", model.actor_imu_encoder), ("critic", model.critic_env_params_encoder)):
        expected = sum(p.numel() for p in getattr(model, name).parameters())
        expected += sum(p.numel() for p in encoder.parameters())
        assert results[name]["local_redundancy_parameter_count"] == expected
        assert math.isfinite(results[name]["local_redundancy"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_samples", 1),
        ("batch_size", 0),
        ("interval", -1),
        ("interval", 1.5),
        ("actor_target_std", 0),
        ("critic_target_std", float("nan")),
        ("input_std", float("inf")),
        ("input_mode", "bad"),
        ("sac_action_source", "bad"),
        ("seed", -1),
    ],
)
def test_invalid_config_rejected(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        lr.validate_config(cfg)


def test_logging_cadence_disabled_and_failure_retry(cfg):
    runner, obs = ppo()
    scalars = []
    runner.writer = SimpleNamespace(add_scalar=lambda *args: scalars.append(args))
    lr.attach(runner, {**cfg, "enabled": False})
    lr.log(runner, {"it": 0, "obs": obs})
    assert not scalars
    lr.attach(runner, cfg)
    lr.log(runner, {"it": 1, "obs": obs})
    assert not scalars
    lr.log(runner, {"it": 0, "obs": obs})
    assert ("Plasticity/local_redundancy_valid", 1, 0) in scalars
    assert all(math.isfinite(value) for _, value, _ in scalars)
    with torch.no_grad():
        parameter = next(runner.alg.policy.actor.parameters())
        saved = parameter.clone()
        parameter.fill_(float("nan"))
    lr.log(runner, {"it": 100, "obs": obs})
    assert ("Plasticity/local_redundancy_valid", 0, 100) in scalars
    with torch.no_grad():
        parameter.copy_(saved)
    lr.log(runner, {"it": 200, "obs": obs})
    assert ("Plasticity/local_redundancy_valid", 1, 200) in scalars
