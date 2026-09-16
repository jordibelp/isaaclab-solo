# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plasticity diagnostics logged during training.

Implements the four metrics of Appendix B in "The Impact of On-Policy
Parallelized Data Collection on Deep Reinforcement Learning Networks"
(arXiv:2506.03404):

  (a) feature rank      - approximate rank (Yang et al. 2019): smallest k such
                          that the top-k squared singular values of the feature
                          matrix retain >= 99% of the total squared-singular-
                          value energy. ``feature_rank`` is the raw rank (in
                          units/neurons), ``feature_rank_frac`` divides it by the
                          layer's feature count, and ``feature_num`` is the layer
                          width.
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

:func:`activation_plasticity_metrics` returns (a) and (b) twice: once per hidden
layer, and once as a network-level summary (dormant percentages pooled over all
units, feature-rank statistics as the median across layers).
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


def _feature_count(features: torch.Tensor) -> int:
    if features.ndim == 0:
        return 0
    return int(features.shape[-1])


def dormant_counts(
    activations: torch.Tensor,
    eps: float = DEFAULT_DORMANT_EPS,
    tau: float = DEFAULT_DORMANT_TAU,
) -> tuple[int, int, int]:
    """Dormant-unit counts ``(units, dormant_eps, dormant_tau)`` for one hidden layer."""
    a = activations.detach().float()
    a = a.reshape(-1, a.shape[-1])
    score = a.abs().mean(dim=0)
    layer_mean = float(score.mean().item())
    if layer_mean > 0.0:
        dormant_tau = int((score / layer_mean <= tau).sum().item())
    else:
        dormant_tau = score.numel()
    return score.numel(), int((score < eps).sum().item()), dormant_tau


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


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if value == value)  # drop NaN
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    return finite[mid] if len(finite) % 2 else 0.5 * (finite[mid - 1] + finite[mid])


def activation_plasticity_metrics(
    activations: list[torch.Tensor],
    *,
    rank_threshold: float = DEFAULT_RANK_THRESHOLD,
    dormant_eps: float = DEFAULT_DORMANT_EPS,
    dormant_tau: float = DEFAULT_DORMANT_TAU,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Feature-rank and dormant-unit metrics over collected hidden activations.

    Returns ``(summary, per_layer)``. ``per_layer`` maps each metric name to one
    value per hidden layer, in execution order; ``summary`` holds the same metric
    names reduced to one scalar for the whole network (dormant percentages pooled
    over every unit, feature ranks as the median across layers).
    """
    if not activations:
        return {}, {}

    per_layer: dict[str, list[float]] = {
        "dormant_pct": [],
        "dormant_tau_pct": [],
        "feature_rank": [],
        "feature_rank_frac": [],
        "feature_num": [],
    }
    total_units = 0
    total_dormant_eps = 0
    total_dormant_tau = 0
    for act in activations:
        units, dormant_eps_units, dormant_tau_units = dormant_counts(act, eps=dormant_eps, tau=dormant_tau)
        total_units += units
        total_dormant_eps += dormant_eps_units
        total_dormant_tau += dormant_tau_units
        num_features = _feature_count(act)
        rank = feature_rank(act, threshold=rank_threshold)
        per_layer["dormant_pct"].append(100.0 * dormant_eps_units / units if units else float("nan"))
        per_layer["dormant_tau_pct"].append(100.0 * dormant_tau_units / units if units else float("nan"))
        per_layer["feature_rank"].append(rank)
        per_layer["feature_rank_frac"].append(rank / float(num_features) if num_features > 0 else float("nan"))
        per_layer["feature_num"].append(float(num_features))

    summary = {
        "dormant_pct": 100.0 * total_dormant_eps / total_units if total_units else float("nan"),
        "dormant_tau_pct": 100.0 * total_dormant_tau / total_units if total_units else float("nan"),
        "feature_rank": _median(per_layer["feature_rank"]),
        "feature_rank_frac": _median(per_layer["feature_rank_frac"]),
        "feature_num": per_layer["feature_num"][-1],
    }
    return summary, per_layer


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
