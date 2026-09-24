# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic regression-gradient plasticity probe (arXiv:2607.13432, Corollary 3.6).

Reports E[||grad_theta ell_i||^2], with ell_i the summed-output Gaussian NLL
and y_i = stop_gradient(f_theta(x_i)) + sigma * z_i, z_i ~ N(0, I).
This is the per-input, first-order lower-bound *proxy*, not redundancy in bits
or the norm of the mean gradient. No optimization is performed.
"""

from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict

import torch
from torch import nn


def validate_config(cfg: dict) -> None:
    """Fail before environment construction for malformed Hydra settings."""
    for key in ("interval", "num_samples", "batch_size"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"agent.local_redundancy.{key} must be a positive integer.")
    if cfg["num_samples"] < 2:
        raise ValueError("agent.local_redundancy.num_samples must be >= 2 for the standard error.")
    for key in ("input_std", "actor_target_std", "critic_target_std"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"agent.local_redundancy.{key} must be finite and positive.")
    if cfg["input_mode"] not in ("gaussian", "observations"):
        raise ValueError("agent.local_redundancy.input_mode must be gaussian or observations.")
    if cfg["sac_action_source"] not in ("uniform", "policy_mean"):
        raise ValueError("agent.local_redundancy.sac_action_source must be uniform or policy_mean.")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2**63:
        raise ValueError("agent.local_redundancy.seed must be an integer in [0, 2**63).")


def _probe_copy(module: nn.Module, *, gaussian_inputs: bool) -> nn.Module:
    # Distribution caches contain non-leaf tensors that PyTorch cannot deepcopy.
    # Do not copy training hooks: CBP hooks reference controllers and regenerative
    # gradient hooks would contaminate the probe. Parameter deepcopy drops .grad
    # and tensor hooks; preserve requires_grad and shared-parameter aliases.
    memo = {}
    for sub in module.modules():
        distribution = getattr(sub, "distribution", None)
        if distribution is not None:
            memo[id(distribution)] = None
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
            hooks = getattr(sub, name, None)
            if hooks is not None:
                memo[id(hooks)] = OrderedDict()
    result = copy.deepcopy(module, memo).eval().float()
    if gaussian_inputs:
        # x ~ N(0, s^2 I) at the normalized observation boundary, BEFORE learned
        # encoders, not raw sensor noise scaled by changing running statistics.
        for sub in result.modules():
            for name in ("actor_obs_normalizer", "critic_obs_normalizer", "obs_normalizer"):
                normalizer = getattr(sub, name, None)
                if normalizer is not None:
                    if any(True for _ in normalizer.parameters()):
                        raise ValueError("Gaussian input probing requires non-learned observation normalizers.")
                    setattr(sub, name, nn.Identity())
    return result


def regression_gradient_norms(
    prediction: torch.Tensor, noise: torch.Tensor, parameters: list[nn.Parameter], target_std: float
) -> tuple[torch.Tensor, int]:
    """Exact per-example gradient norms, vectorized over a small microbatch.

    The centered expression is algebraically f - (detach(f) + sigma*z), but
    avoids losing small noise when a critic's value is large. The 1/(2*sigma^2)
    Gaussian NLL convention fixes the otherwise arbitrary MSE scale.
    """
    residual = (prediction - prediction.detach()) / target_std - noise
    losses = 0.5 * residual.square().flatten(1).sum(1)
    count = prediction.shape[0]
    norms = torch.zeros(count, device=prediction.device, dtype=torch.float64)
    if not parameters or not losses.requires_grad:
        return norms, 0
    grads = torch.autograd.grad(
        losses,
        parameters,
        grad_outputs=torch.eye(count, device=losses.device, dtype=losses.dtype),
        is_grads_batched=True,
        allow_unused=True,
    )
    parameter_count = 0
    for parameter, grad in zip(parameters, grads):
        if grad is not None:
            norms += grad.detach().flatten(1).double().square().sum(1)
            parameter_count += parameter.numel()
    return norms, parameter_count


def _measure(forward, parameters, obs, *, cfg: dict, target_std: float, seed: int) -> dict[str, float]:
    device = next(iter(obs.values())).device
    count = next(iter(obs.values())).shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Generate all target noise together so batch_size affects only memory/runtime,
    # never the random-label dataset or metric normalization.
    with torch.no_grad():
        output_dim = forward({key: value[:1] for key, value in obs.items()}).flatten(1).shape[1]
    noise = torch.randn(count, output_dim, generator=generator).to(device)
    values = []
    parameter_count = 0
    for start in range(0, count, cfg["batch_size"]):
        batch = {key: value[start : start + cfg["batch_size"]] for key, value in obs.items()}
        prediction = forward(batch).flatten(1)
        norms, parameter_count = regression_gradient_norms(
            prediction, noise[start : start + prediction.shape[0]], parameters, target_std
        )
        values.append(norms)
    norms = torch.cat(values)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("Non-finite local-redundancy gradients.")
    mean = norms.mean().item()
    return {
        "local_redundancy": mean,
        "local_redundancy_per_output": mean / output_dim,
        "local_redundancy_stderr": norms.std(unbiased=True).item() / math.sqrt(count),
        "local_redundancy_num_samples": count,
        "local_redundancy_output_dim": output_dim,
        "local_redundancy_parameter_count": parameter_count,
    }


def measure(runner, obs, cfg: dict, iteration: int) -> dict[str, dict[str, float]]:
    """Probe PPO's full mean/value paths or SAC's latent mean and online Q1/Q2.

    All stochastic tensors use local CPU generators (stable across CPU/CUDA and
    microbatch sizes). No live module is forwarded, so even training hooks and
    cached action distributions remain untouched.
    """
    # Stride 3 keeps each iteration's input (+0) and noise (+1, +2) seeds disjoint across iterations.
    seed = (cfg["seed"] + (3 * iteration if cfg["resample"] else 0)) % (2**63)
    device = torch.device(runner.device)
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        gaussian = cfg["input_mode"] == "gaussian"
        available = next(iter(obs.values())).shape[0]
        count = cfg["num_samples"] if gaussian else min(cfg["num_samples"], available)
        if count < 2:
            raise ValueError("Local redundancy needs at least two observations; use gaussian inputs for one-env runs.")
        indices = None if gaussian else torch.randperm(available, generator=generator)[:count]
        inputs = {}
        for key, value in obs.items():
            if gaussian:
                inputs[key] = (torch.randn((count, *value.shape[1:]), generator=generator) * cfg["input_std"]).to(
                    device
                )
            else:
                inputs[key] = value[indices.to(value.device)].detach().to(device=device, dtype=torch.float32).clone()

        results = {}
        policy = getattr(runner.alg, "policy", None)
        if policy is not None:
            model = _probe_copy(policy, gaussian_inputs=gaussian)
            parameters = [p for p in model.parameters() if p.requires_grad]
            for name, forward in (("actor", model.act_inference), ("critic", model.evaluate)):
                if cfg[name]:
                    results[name] = _measure(
                        forward, parameters, inputs, cfg=cfg, target_std=cfg[f"{name}_target_std"], seed=seed + 1
                    )
        else:
            actor = _probe_copy(runner.alg.actor, gaussian_inputs=gaussian)

            def actor_mean(batch):
                output = actor.mlp(actor.get_latent(batch))
                return output[..., 0, :] if actor.state_dependent_std else output

            if cfg["actor"]:
                results["actor"] = _measure(
                    actor_mean,
                    [p for p in actor.parameters() if p.requires_grad],
                    inputs,
                    cfg=cfg,
                    target_std=cfg["actor_target_std"],
                    seed=seed + 1,
                )
            if cfg["critic"]:
                critic = _probe_copy(runner.alg.critic, gaussian_inputs=gaussian)
                with torch.no_grad():
                    if cfg["sac_action_source"] == "uniform":
                        unit = torch.rand(count, actor.output_dim, generator=generator).to(device) * 2 - 1
                        actions = actor.action_bias + actor.action_range * unit
                    else:
                        actions = actor(inputs)
                    # SAC's get_latent only concatenates/normalizes observations;
                    # learned Q layers (including LayerNorm) remain in each probe.
                    q_inputs = torch.cat((critic.get_latent(inputs), actions), dim=-1).detach()
                for name in ("critic1", "critic2"):
                    network = getattr(critic, name)
                    results[name] = _measure(
                        # Probe raw-unit Q in every mode: decode CE logits and PopArt's normalized head.
                        lambda batch, net=network: critic.q_from_output(net(batch["q"])),
                        [p for p in network.parameters() if p.requires_grad],
                        {"q": q_inputs},
                        cfg=cfg,
                        target_std=cfg["critic_target_std"],
                        seed=seed + 2,
                    )
                # Mean of the two comparable per-Q scores; not min(Q1,Q2), whose
                # gradient would hide one critic. Common labels make twins comparable.
                results["critic"] = {
                    key: (results["critic1"][key] + results["critic2"][key]) / 2
                    for key in ("local_redundancy", "local_redundancy_per_output")
                }
        return results


def attach(runner, cfg: dict) -> None:
    """Attach independently of the cheaper --plasticity-metrics diagnostics."""
    validate_config(cfg)
    if not cfg["enabled"] or not (cfg["actor"] or cfg["critic"]):
        return
    policy = getattr(runner.alg, "policy", None)
    models = [policy] if policy is not None else [getattr(runner.alg, name, None) for name in ("actor", "critic")]
    if any(model is None or getattr(model, "is_recurrent", False) for model in models):
        raise ValueError(
            "Local redundancy supports feed-forward PPO/SAC (including TCN encoders), not recurrent agents."
        )
    runner._local_redundancy_cfg = dict(cfg)
    print(
        f"[INFO]: Local redundancy enabled: {cfg['input_mode']} inputs, {cfg['num_samples']} samples, "
        f"microbatch {cfg['batch_size']}, every {cfg['interval']} iterations; actor={cfg['actor']}, "
        f"critic={cfg['critic']}. Diagnostic copies only, no optimizer updates.",
        flush=True,
    )


def log(runner, locs: dict) -> None:
    cfg = getattr(runner, "_local_redundancy_cfg", None)
    writer = getattr(runner, "writer", None)
    if writer is None:
        writer = getattr(getattr(runner, "logger", None), "writer", None)
    if cfg is None or writer is None or getattr(runner, "disable_logs", False):
        return
    iteration = int(locs["it"])
    if iteration % cfg["interval"]:
        return
    start = time.perf_counter()
    try:
        metrics = measure(runner, locs["obs"], cfg, iteration)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        # Do not kill a long training run for diagnostics, nor permanently hide
        # transient failures. Mark the missing point and retry next interval.
        writer.add_scalar("Plasticity/local_redundancy_valid", 0, iteration)
        print(f"[WARN]: Local redundancy failed at iteration {iteration}: {exc}", flush=True)
        return
    for name, scalars in metrics.items():
        for key, value in scalars.items():
            writer.add_scalar(f"Plasticity/summary/{name}/{key}", value, iteration)
    writer.add_scalar("Plasticity/local_redundancy_valid", 1, iteration)
    writer.add_scalar("Plasticity/local_redundancy_seconds", time.perf_counter() - start, iteration)
