# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
import random
from typing import Any

import numpy as np
import torch
import warp as wp


@dataclass
class RngState:
    """Snapshot of Python, NumPy, and Torch process-global RNG state."""

    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor]
    pythonhashseed: str | None
    cublas_workspace_config: str | None
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool


def _set_optional_env_var(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def capture_rng_state() -> RngState:
    """Capture Python, NumPy, Torch CPU, and initialized CUDA RNG state."""

    cuda_states = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.random.get_rng_state().clone(),
        cuda_states=cuda_states,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def restore_rng_state(state: RngState) -> None:
    """Restore a state captured with :func:`capture_rng_state`."""

    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.random.set_rng_state(state.torch_state)

    if state.cuda_states and torch.cuda.is_available():
        for device_index, cuda_state in enumerate(state.cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state, device=device_index)

    _set_optional_env_var("PYTHONHASHSEED", state.pythonhashseed)
    _set_optional_env_var("CUBLAS_WORKSPACE_CONFIG", state.cublas_workspace_config)
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.use_deterministic_algorithms(state.deterministic_algorithms)


class RngStream:
    """A saved process-global RNG stream that can be used without perturbing the caller."""

    def __init__(self, state: RngState, seed: int | None = None):
        self.state = state
        self.seed = seed

    @classmethod
    def from_seed(cls, seed: int | None, torch_deterministic: bool = False) -> RngStream:
        """Create a stream initialized from ``seed`` while preserving the caller's RNG state."""

        outer_state = capture_rng_state()
        # configure_seed also initializes Warp's global seed. Warp does not expose
        # a public state snapshot API, so RngStream can only switch Python/NumPy/Torch states.
        resolved_seed = configure_seed(seed, torch_deterministic=torch_deterministic)
        stream_state = capture_rng_state()
        restore_rng_state(outer_state)
        return cls(stream_state, resolved_seed)

    @contextmanager
    def use(self) -> Iterator[None]:
        """Temporarily switch to this stream and save its advanced state on exit."""

        outer_state = capture_rng_state()
        restore_rng_state(self.state)
        try:
            yield
        finally:
            try:
                self.state = capture_rng_state()
            finally:
                restore_rng_state(outer_state)


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp).

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: If True, enables deterministic mode for torch operations.

    Returns:
        The seed value that was set.
    """
    if seed is None or seed == -1:
        seed = 42 if torch_deterministic else random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wp.rand_init(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
