# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plasticity diagnostics logged during training.

Implements the four metrics of Appendix B in "The Impact of On-Policy
Parallelized Data Collection on Deep Reinforcement Learning Networks"
(arXiv:2506.03404):

  (a) feature rank      - approximate rank (Yang et al. 2019): smallest k such
                          that the top-k squared singular values of the feature
                          matrix retain >= 99% of the total squared-singular-value
                          energy. ``feature_rank`` keeps the legacy last-hidden-
                          activation value; ``feature_rank_i`` reports layer i.
  (b) % dormant units   - percentage of hidden units whose mean |activation| over
                          a batch is below eps=1e-5 (the paper's reading of
                          Sokar et al. 2023). The Sokar-normalized variant
                          (s_i = E|h_i| / mean_j E|h_j| <= tau) is reported
                          alongside as ``dormant_tau_pct`` for comparability with
                          solo_race_plasticity_eval.py.
  (c) weight norm       - global L2 norm of the network parameters.
  (d) gradient kurtosis - kurtosis E[(L - mu)^4] / var(L)^2 of the log-transformed
                          absolute gradients L = log(|g| + eps), pooled over all
                          gradient entries of the network (Garg et al. 2021).
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_DORMANT_EPS = 1.0e-5
DEFAULT_DORMANT_TAU = 0.025
DEFAULT_RANK_THRESHOLD = 0.99
DEFAULT_KURTOSIS_EPS = 1.0e-8
DEFAULT_SAMPLE_CAP = 4096

_ACTIVATION_MODULES = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Tanh,
    nn.Sigmoid,
    nn.Softplus,
    nn.Hardswish,
)


def _subsample_rows(x: torch.Tensor, sample_cap: int) -> torch.Tensor:
    if x.shape[0] <= sample_cap:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:sample_cap]
    return x[idx]


def feature_rank(features: torch.Tensor, threshold: float = DEFAULT_RANK_THRESHOLD) -> float:
    """Smallest k whose top-k squared singular values keep >= threshold of the total energy."""
    f = features.detach().float()
    if f.ndim != 2 or f.shape[0] < 2 or f.numel() == 0 or not torch.isfinite(f).all():
        return float("nan")
    try:
        sigma = torch.linalg.svdvals(f)
    except Exception:
        return float("nan")
    energy = sigma.square()
    total = energy.sum()
    if total <= 0:
        return float("nan")
    cumulative = torch.cumsum(energy, dim=0) / total
    k = int((cumulative < threshold).sum().item()) + 1
    return float(min(k, sigma.numel()))


def dormant_metrics(
    activations: list[torch.Tensor],
    eps: float = DEFAULT_DORMANT_EPS,
    tau: float = DEFAULT_DORMANT_TAU,
) -> dict[str, float]:
    """Network-wide dormant-unit percentages pooled over the given hidden activations."""
    total_units = 0
    dormant_eps_units = 0
    dormant_tau_units = 0
    for act in activations:
        a = act.detach().float()
        a = a.reshape(-1, a.shape[-1])
        score = a.abs().mean(dim=0)
        total_units += score.numel()
        dormant_eps_units += int((score < eps).sum().item())
        layer_mean = float(score.mean().item())
        if layer_mean > 0.0:
            dormant_tau_units += int((score / layer_mean <= tau).sum().item())
        else:
            dormant_tau_units += score.numel()
    if total_units == 0:
        return {}
    return {
        "dormant_pct": 100.0 * dormant_eps_units / total_units,
        "dormant_tau_pct": 100.0 * dormant_tau_units / total_units,
    }


def weight_norm(params) -> float:
    """Global L2 norm over the given parameters."""
    squares = [p.detach().float().square().sum() for p in params]
    if not squares:
        return float("nan")
    return float(torch.stack(squares).sum().sqrt().item())


def gradient_kurtosis(params, eps: float = DEFAULT_KURTOSIS_EPS) -> float:
    """Kurtosis of log(|grad| + eps) pooled over all gradient entries of the given parameters."""
    grads = [p.grad.detach().flatten().float() for p in params if p.grad is not None]
    if not grads:
        return float("nan")
    logs = (torch.cat(grads).abs() + eps).log()
    centered = logs - logs.mean()
    variance = centered.square().mean()
    if not torch.isfinite(variance) or variance <= 0:
        return float("nan")
    return float((centered.pow(4).mean() / variance.square()).item())


def mlp_hidden_activations(
    net: nn.Module, x: torch.Tensor, sample_cap: int = DEFAULT_SAMPLE_CAP
) -> list[torch.Tensor]:
    """Forward x through an MLP-like nn.Sequential, returning each post-activation hidden feature."""
    if not isinstance(net, nn.Sequential):
        return []
    x = _subsample_rows(x, sample_cap)
    activations: list[torch.Tensor] = []
    for layer in net:
        x = layer(x)
        if isinstance(layer, _ACTIVATION_MODULES):
            activations.append(x)
    return activations


def collect_hidden_activations(module: nn.Module, forward_fn, sample_cap: int = DEFAULT_SAMPLE_CAP) -> list[torch.Tensor]:
    """Capture post-activation hidden features produced inside ``module`` while running ``forward_fn``.

    Uses transient forward hooks so it works with any input plumbing (normalizers,
    history encoders, shared networks). A shared activation-module instance reused at
    several depths still yields one record per call, in execution order.
    """
    records: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs, output) -> None:
        if torch.is_tensor(output):
            t = output.detach()
            t = t.reshape(-1, t.shape[-1])
            records.append(_subsample_rows(t, sample_cap).float())

    handles = [sub.register_forward_hook(_hook) for sub in module.modules() if isinstance(sub, _ACTIVATION_MODULES)]
    if not handles:
        return []
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for handle in handles:
            handle.remove()
    return records


def activation_plasticity_metrics(
    activations: list[torch.Tensor],
    *,
    rank_threshold: float = DEFAULT_RANK_THRESHOLD,
    dormant_eps: float = DEFAULT_DORMANT_EPS,
    dormant_tau: float = DEFAULT_DORMANT_TAU,
) -> dict[str, float]:
    """Feature rank and dormant-unit percentages over collected hidden activations."""
    if not activations:
        return {}
    metrics = dormant_metrics(activations, eps=dormant_eps, tau=dormant_tau)
    layer_ranks = [feature_rank(act, threshold=rank_threshold) for act in activations]
    metrics["feature_rank"] = layer_ranks[-1]
    metrics.update({f"feature_rank_{i}": rank for i, rank in enumerate(layer_ranks)})
    return metrics


class GradKurtosisCapture:
    """Captures per-group gradient kurtosis at optimizer-step time.

    Wraps ``optimizer.step`` so gradients are read while they are still present
    (and, under AMP, after ``GradScaler`` has unscaled them). The computation only
    runs on the first step after :meth:`arm`, keeping steady-state overhead at zero.
    """

    def __init__(self, param_groups: dict[str, list[nn.Parameter]], eps: float = DEFAULT_KURTOSIS_EPS) -> None:
        self.param_groups = {name: list(params) for name, params in param_groups.items()}
        self.eps = float(eps)
        self._armed = True
        self.last: dict[str, float] = {}

    def arm(self) -> None:
        self._armed = True

    def capture_if_armed(self) -> None:
        if not self._armed:
            return
        captured_any = False
        for name, params in self.param_groups.items():
            kurtosis = gradient_kurtosis(params, eps=self.eps)
            if kurtosis == kurtosis:  # skip NaN (no grads yet / degenerate variance)
                self.last[name] = kurtosis
                captured_any = True
        # Stay armed on a fully failed capture (e.g. non-finite grads while the AMP
        # loss scale settles) and retry on the next optimizer step.
        self._armed = not captured_any

    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        original_step = optimizer.step

        def _step_with_grad_kurtosis(*args, **kwargs):
            self.capture_if_armed()
            return original_step(*args, **kwargs)

        optimizer.step = _step_with_grad_kurtosis
