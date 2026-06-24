from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable

import torch
from torch import nn


VALID_CBP_UTIL_TYPES = {
    "contribution",
    "zero_contribution",
    "adaptable_contribution",
    "weight",
    "adaptation",
    "feature_by_input",
    "random",
}
VALID_CBP_INITS = {"default", "xavier", "lecun", "kaiming"}


@dataclass(frozen=True)
class CBPSpec:
    """Describe one hidden feature group eligible for continual backprop."""

    name: str
    input_layer: nn.Module
    output_layer: nn.Module
    capture_layer: nn.Module
    feature_transform: tuple[nn.Module, ...] = ()
    norm_layer: nn.Module | None = None


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _infer_hidden_size(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, nn.Conv1d):
        return int(module.out_channels)
    raise TypeError(f"Unsupported continual-backprop input layer: {module.__class__.__name__}")


def _infer_output_kind(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Conv1d):
        return "conv1d"
    if isinstance(module, nn.LSTM):
        return "lstm"
    raise TypeError(f"Unsupported continual-backprop output layer: {module.__class__.__name__}")


def _init_bound(layer: nn.Module, init: str, activation: str = "relu") -> float:
    activation = activation.lower()
    if activation in {"gelu", "elu", "swish"}:
        activation = "relu"
    if isinstance(layer, nn.Linear):
        in_features = layer.in_features
        out_features = layer.out_features
    elif isinstance(layer, nn.Conv1d):
        in_features = layer.in_channels * layer.kernel_size[0]
        out_features = layer.out_channels
    else:
        raise TypeError(f"Unsupported continual-backprop layer for init bound: {layer.__class__.__name__}")

    if init == "default":
        return sqrt(1 / in_features)
    if init == "xavier":
        return nn.init.calculate_gain(activation) * sqrt(6 / (in_features + out_features))
    if init == "lecun":
        return sqrt(3 / in_features)
    return nn.init.calculate_gain(activation) * sqrt(3 / in_features)


def _zero_optimizer_state_slice(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    indexer: tuple[Any, ...],
) -> None:
    state = optimizer.state.get(parameter, None)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        tensor = state.get(key, None)
        if isinstance(tensor, torch.Tensor):
            tensor[indexer] = 0


class _CBPGroup:
    def __init__(
        self,
        spec: CBPSpec,
        replacement_rate: float,
        maturity_threshold: int,
        decay_rate: float,
        util_type: str,
        init: str,
        accumulate: bool,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.input_layer = spec.input_layer
        self.output_layer = spec.output_layer
        self.capture_layer = spec.capture_layer
        self.feature_transform = tuple(spec.feature_transform)
        self.norm_layer = spec.norm_layer
        self.replacement_rate = float(replacement_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.decay_rate = float(decay_rate)
        self.util_type = str(util_type)
        self.init = str(init)
        self.accumulate = bool(accumulate)
        self.output_kind = _infer_output_kind(self.output_layer)
        self.hidden_size = _infer_hidden_size(self.input_layer)
        self.device = _module_device(self.input_layer)
        self.util = torch.zeros(self.hidden_size, device=self.device)
        self.mean_feature_act = torch.zeros(self.hidden_size, device=self.device)
        self.ages = torch.zeros(self.hidden_size, device=self.device)
        self.accumulated_replacements = 0.0
        self.total_replacements = 0
        self._last_features: torch.Tensor | None = None
        self._capture_handle = self.capture_layer.register_forward_hook(self._capture_features)
        self._weight_bound = _init_bound(self.input_layer, self.init)

    def _capture_features(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            features = output
            if self.feature_transform:
                with torch.no_grad():
                    for module in self.feature_transform:
                        features = module(features)
            self._last_features = features.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "util": self.util.detach().cpu(),
            "mean_feature_act": self.mean_feature_act.detach().cpu(),
            "ages": self.ages.detach().cpu(),
            "accumulated_replacements": float(self.accumulated_replacements),
            "total_replacements": int(self.total_replacements),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        util = state_dict.get("util", None)
        if isinstance(util, torch.Tensor) and util.shape == self.util.shape:
            self.util.copy_(util.to(self.device))
        mean_feature_act = state_dict.get("mean_feature_act", None)
        if isinstance(mean_feature_act, torch.Tensor) and mean_feature_act.shape == self.mean_feature_act.shape:
            self.mean_feature_act.copy_(mean_feature_act.to(self.device))
        ages = state_dict.get("ages", None)
        if isinstance(ages, torch.Tensor) and ages.shape == self.ages.shape:
            self.ages.copy_(ages.to(self.device))
        self.accumulated_replacements = float(state_dict.get("accumulated_replacements", self.accumulated_replacements))
        self.total_replacements = int(state_dict.get("total_replacements", self.total_replacements))

    def _flatten_features(self) -> torch.Tensor | None:
        if self._last_features is None:
            return None
        features = self._last_features
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features.reshape(-1, features.shape[-1]).to(self.device)

    def _input_weight_magnitude(self) -> torch.Tensor:
        if isinstance(self.input_layer, nn.Linear):
            return self.input_layer.weight.detach().abs().mean(dim=1)
        if isinstance(self.input_layer, nn.Conv1d):
            return self.input_layer.weight.detach().abs().mean(dim=(1, 2))
        raise TypeError(f"Unsupported continual-backprop input layer: {self.input_layer.__class__.__name__}")

    def _output_weight_magnitude(self) -> torch.Tensor:
        if isinstance(self.output_layer, nn.Linear):
            return self.output_layer.weight.detach().abs().mean(dim=0)
        if isinstance(self.output_layer, nn.Conv1d):
            return self.output_layer.weight.detach().abs().mean(dim=(0, 2))
        if isinstance(self.output_layer, nn.LSTM):
            return self.output_layer.weight_ih_l0.detach().abs().mean(dim=0)
        raise TypeError(f"Unsupported continual-backprop output layer: {self.output_layer.__class__.__name__}")

    def _reset_input_layer(self, indices: torch.Tensor) -> None:
        with torch.no_grad():
            if isinstance(self.input_layer, nn.Linear):
                self.input_layer.weight.data[indices, :] = torch.empty(
                    indices.numel(),
                    self.input_layer.in_features,
                    device=self.device,
                ).uniform_(-self._weight_bound, self._weight_bound)
                if self.input_layer.bias is not None:
                    self.input_layer.bias.data[indices] = 0
            elif isinstance(self.input_layer, nn.Conv1d):
                self.input_layer.weight.data[indices, :, :] = torch.empty(
                    indices.numel(),
                    self.input_layer.in_channels,
                    self.input_layer.kernel_size[0],
                    device=self.device,
                ).uniform_(-self._weight_bound, self._weight_bound)
                if self.input_layer.bias is not None:
                    self.input_layer.bias.data[indices] = 0
            else:
                raise TypeError(f"Unsupported continual-backprop input layer: {self.input_layer.__class__.__name__}")

    def _zero_output_connections(self, indices: torch.Tensor, bias_correction: torch.Tensor) -> None:
        with torch.no_grad():
            if isinstance(self.output_layer, nn.Linear):
                if self.output_layer.bias is not None:
                    correction = (
                        self.output_layer.weight.data[:, indices] * bias_correction[indices].unsqueeze(0)
                    ).sum(dim=1)
                    self.output_layer.bias.data += correction
                self.output_layer.weight.data[:, indices] = 0
            elif isinstance(self.output_layer, nn.Conv1d):
                self.output_layer.weight.data[:, indices, :] = 0
            elif isinstance(self.output_layer, nn.LSTM):
                self.output_layer.weight_ih_l0.data[:, indices] = 0
            else:
                raise TypeError(f"Unsupported continual-backprop output layer: {self.output_layer.__class__.__name__}")

    def _reset_norm_layer(self, indices: torch.Tensor) -> None:
        if self.norm_layer is None:
            return
        with torch.no_grad():
            if hasattr(self.norm_layer, "weight") and self.norm_layer.weight is not None:
                self.norm_layer.weight.data[indices] = 1
            if hasattr(self.norm_layer, "bias") and self.norm_layer.bias is not None:
                self.norm_layer.bias.data[indices] = 0

    def _zero_optimizer_state(self, optimizer: torch.optim.Optimizer, indices: torch.Tensor) -> None:
        if isinstance(self.input_layer, nn.Linear):
            _zero_optimizer_state_slice(optimizer, self.input_layer.weight, (indices, slice(None)))
            if self.input_layer.bias is not None:
                _zero_optimizer_state_slice(optimizer, self.input_layer.bias, (indices,))
        elif isinstance(self.input_layer, nn.Conv1d):
            _zero_optimizer_state_slice(optimizer, self.input_layer.weight, (indices, slice(None), slice(None)))
            if self.input_layer.bias is not None:
                _zero_optimizer_state_slice(optimizer, self.input_layer.bias, (indices,))

        if isinstance(self.output_layer, nn.Linear):
            _zero_optimizer_state_slice(optimizer, self.output_layer.weight, (slice(None), indices))
        elif isinstance(self.output_layer, nn.Conv1d):
            _zero_optimizer_state_slice(optimizer, self.output_layer.weight, (slice(None), indices, slice(None)))
        elif isinstance(self.output_layer, nn.LSTM):
            _zero_optimizer_state_slice(optimizer, self.output_layer.weight_ih_l0, (slice(None), indices))

        if self.norm_layer is not None:
            if hasattr(self.norm_layer, "weight") and self.norm_layer.weight is not None:
                _zero_optimizer_state_slice(optimizer, self.norm_layer.weight, (indices,))
            if hasattr(self.norm_layer, "bias") and self.norm_layer.bias is not None:
                _zero_optimizer_state_slice(optimizer, self.norm_layer.bias, (indices,))

    def step(self, optimizer: torch.optim.Optimizer) -> int:
        features = self._flatten_features()
        self.ages += 1
        if self.replacement_rate <= 0 or features is None:
            return 0

        if features.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"continual backprop feature capture for group '{self.name}' produced "
                f"{features.shape[-1]} features, but the hidden layer has {self.hidden_size}. "
                "This would disable replacements for the group."
            )

        bias_correction = 1.0 - self.decay_rate ** self.ages.clamp_min(1)
        self.mean_feature_act.mul_(self.decay_rate)
        self.mean_feature_act.add_((1.0 - self.decay_rate) * features.mean(dim=0))
        bias_corrected_act = self.mean_feature_act / bias_correction.clamp_min(1e-12)

        output_weight_mag = self._output_weight_magnitude()
        input_weight_mag = self._input_weight_magnitude().clamp_min(1e-12)

        if self.util_type == "weight":
            new_util = output_weight_mag
        elif self.util_type == "contribution":
            new_util = output_weight_mag * features.abs().mean(dim=0)
        elif self.util_type == "adaptation":
            new_util = 1 / input_weight_mag
        elif self.util_type == "zero_contribution":
            new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)
        elif self.util_type == "adaptable_contribution":
            new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
        elif self.util_type == "feature_by_input":
            new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
        elif self.util_type == "random":
            new_util = torch.rand_like(self.util)
        else:
            new_util = torch.zeros_like(self.util)

        self.util.mul_(self.decay_rate)
        self.util.add_((1.0 - self.decay_rate) * new_util)
        bias_corrected_util = self.util / bias_correction.clamp_min(1e-12)

        eligible = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible.numel() == 0:
            return 0

        expected = self.replacement_rate * float(eligible.numel())
        if self.accumulate:
            self.accumulated_replacements += expected
            replace_count = int(self.accumulated_replacements)
            self.accumulated_replacements -= replace_count
        else:
            replace_count = int(expected)
            if replace_count < 1 and expected > 0 and torch.rand((), device=self.device).item() <= expected:
                replace_count = 1
        replace_count = min(replace_count, int(eligible.numel()))
        if replace_count <= 0:
            return 0

        to_replace = eligible[torch.topk(-bias_corrected_util[eligible], replace_count).indices]
        self._reset_input_layer(to_replace)
        self._zero_output_connections(to_replace, bias_correction)
        self._reset_norm_layer(to_replace)
        self._zero_optimizer_state(optimizer, to_replace)
        self.util[to_replace] = 0
        self.mean_feature_act[to_replace] = 0
        self.ages[to_replace] = 0
        self.total_replacements += replace_count
        return replace_count


class ContinualBackpropManager:
    """Manage continual-backprop feature replacement across multiple groups."""

    def __init__(
        self,
        specs: Iterable[CBPSpec],
        *,
        replacement_rate: float,
        maturity_threshold: int,
        decay_rate: float,
        util_type: str,
        init: str,
        accumulate: bool,
    ) -> None:
        self.groups = [
            _CBPGroup(
                spec=spec,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                decay_rate=decay_rate,
                util_type=util_type,
                init=init,
                accumulate=accumulate,
            )
            for spec in specs
        ]
        self.replacement_rate = float(replacement_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.decay_rate = float(decay_rate)
        self.util_type = str(util_type)
        self.init = str(init)
        self.accumulate = bool(accumulate)
        self.last_replacements: dict[str, int] = {}
        self.total_replacements_by_group: dict[str, int] = {group.name: 0 for group in self.groups}
        self.optimizer_steps = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "replacement_rate": self.replacement_rate,
            "maturity_threshold": self.maturity_threshold,
            "decay_rate": self.decay_rate,
            "util_type": self.util_type,
            "init": self.init,
            "accumulate": self.accumulate,
            "last_replacements": dict(self.last_replacements),
            "total_replacements_by_group": dict(self.total_replacements_by_group),
            "optimizer_steps": int(self.optimizer_steps),
            "groups": {group.name: group.state_dict() for group in self.groups},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        groups = state_dict.get("groups", {})
        if not isinstance(groups, dict):
            return
        for group in self.groups:
            group_state = groups.get(group.name, None)
            if isinstance(group_state, dict):
                group.load_state_dict(group_state)
                self.total_replacements_by_group[group.name] = int(group.total_replacements)
        last_replacements = state_dict.get("last_replacements", {})
        if isinstance(last_replacements, dict):
            self.last_replacements = {str(key): int(value) for key, value in last_replacements.items()}
        totals = state_dict.get("total_replacements_by_group", {})
        if isinstance(totals, dict):
            for key, value in totals.items():
                if key in self.total_replacements_by_group:
                    self.total_replacements_by_group[str(key)] = int(value)
        self.optimizer_steps = int(state_dict.get("optimizer_steps", self.optimizer_steps))

    def after_optimizer_step(self, optimizer: torch.optim.Optimizer) -> dict[str, int]:
        self.optimizer_steps += 1
        replacements: dict[str, int] = {}
        for group in self.groups:
            count = group.step(optimizer)
            replacements[group.name] = count
            self.total_replacements_by_group[group.name] = int(group.total_replacements)
        self.last_replacements = replacements
        return replacements

    def summary(self) -> dict[str, Any]:
        return {
            "replacement_rate": self.replacement_rate,
            "maturity_threshold": self.maturity_threshold,
            "decay_rate": self.decay_rate,
            "util_type": self.util_type,
            "init": self.init,
            "accumulate": self.accumulate,
            "groups": [group.name for group in self.groups],
            "optimizer_steps": int(self.optimizer_steps),
            "total_replacements": int(sum(self.total_replacements_by_group.values())),
            "total_replacements_by_group": dict(self.total_replacements_by_group),
        }


def build_continual_backprop_manager(
    specs: Iterable[CBPSpec],
    *,
    replacement_rate: float,
    maturity_threshold: int,
    decay_rate: float,
    util_type: str,
    init: str,
    accumulate: bool,
) -> ContinualBackpropManager:
    spec_list = list(specs)
    if not spec_list:
        raise ValueError("continual backprop was enabled, but no eligible feature groups were found")
    if util_type not in VALID_CBP_UTIL_TYPES:
        raise ValueError(f"unsupported cbp util type: {util_type}")
    if init not in VALID_CBP_INITS:
        raise ValueError(f"unsupported cbp init: {init}")
    if replacement_rate < 0:
        raise ValueError("cbp replacement rate must be non-negative")
    if maturity_threshold < 0:
        raise ValueError("cbp maturity threshold must be non-negative")
    if not 0 <= decay_rate < 1:
        raise ValueError("cbp decay rate must be in [0, 1)")
    return ContinualBackpropManager(
        spec_list,
        replacement_rate=replacement_rate,
        maturity_threshold=maturity_threshold,
        decay_rate=decay_rate,
        util_type=util_type,
        init=init,
        accumulate=accumulate,
    )


def _ordered_leaf_modules(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Module]]:
    """Return leaf modules in execution order, preserving repeated module instances."""

    children = list(module._modules.items())
    if not children:
        return [(prefix, module)]

    leaves: list[tuple[str, nn.Module]] = []
    for child_name, child in children:
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        leaves.extend(_ordered_leaf_modules(child, child_prefix))
    return leaves


def cbp_specs_for_sequential_mlp(name: str, net: nn.Module) -> list[CBPSpec]:
    """Create CBP specs for hidden features in an MLP-like Sequential module."""

    if not isinstance(net, nn.Sequential):
        return []

    leaves = _ordered_leaf_modules(net)
    linear_positions = [idx for idx, (_module_name, layer) in enumerate(leaves) if isinstance(layer, nn.Linear)]
    specs: list[CBPSpec] = []
    for layer_idx, input_pos in enumerate(linear_positions[:-1]):
        output_pos = linear_positions[layer_idx + 1]
        input_name, input_layer = leaves[input_pos]
        _output_name, output_layer = leaves[output_pos]
        feature_transform = tuple(module for _module_name, module in leaves[input_pos + 1 : output_pos])

        norm_layer = None
        for _module_name, module in leaves[input_pos + 1 : output_pos]:
            if isinstance(module, nn.LayerNorm):
                norm_layer = module
                break

        specs.append(
            CBPSpec(
                name=f"{name}.{input_name}",
                input_layer=input_layer,
                output_layer=output_layer,
                capture_layer=input_layer,
                feature_transform=feature_transform,
                norm_layer=norm_layer,
            )
        )
    return specs


def collect_actor_critic_cbp_specs(policy: nn.Module) -> list[CBPSpec]:
    """Collect hidden-feature CBP specs from an RSL-RL actor-critic module."""

    if hasattr(policy, "cbp_specs") and callable(policy.cbp_specs):
        return list(policy.cbp_specs())

    specs: list[CBPSpec] = []
    for name in ("actor", "critic"):
        module = getattr(policy, name, None)
        if module is not None:
            specs.extend(cbp_specs_for_sequential_mlp(name, module))

    input_layer_ids: set[int] = set()
    for spec in specs:
        input_layer_id = id(spec.input_layer)
        if input_layer_id in input_layer_ids:
            raise ValueError(
                "continual backprop found shared hidden layers between actor and critic. "
                "That topology needs an explicit policy.cbp_specs() implementation."
            )
        input_layer_ids.add(input_layer_id)
    return specs
