# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl_sac.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-layer perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        layer_norm: bool = False,
        last_activation: str | None = None,
        batch_norm: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            layer_norm: Whether to apply layer normalization after hidden linear layers.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            batch_norm: Whether to apply batch normalization after hidden linear layers. It replaces layer
                normalization when both are requested.
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        for in_dim, out_dim in zip([input_dim, *hidden_dims_processed[:-1]], hidden_dims_processed):
            # BatchNorm subtracts the batch mean, so a bias in front of it only drifts on Adam's rounding noise.
            layers.append(nn.Linear(in_dim, out_dim, bias=not batch_norm))
            if batch_norm:
                # XQC (arXiv:2509.25174): running statistics decay by 0.99 per update, eps 1e-3.
                layers.append(nn.BatchNorm1d(out_dim, momentum=0.01, eps=1e-3))
            elif layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x
