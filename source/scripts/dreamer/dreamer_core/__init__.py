# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""High-performance DreamerV3 core.

A faithful, efficient PyTorch DreamerV3 implementation adapted for Isaac Lab's
GPU-resident vectorized environments.  The design mirrors the efficient
reproduction in the R2-Dreamer codebase (block-diagonal GRU, RMSNorm, symexp
two-hot heads, LaProp + adaptive gradient clipping, a single fused backward with
frozen network copies for imagination, AMP, torch.compile, and replay-context
latent caching), while dropping the decoder-free / augmentation variants so that
only the DreamerV3 objective remains.
"""

from .agent import DreamerAgent, DreamerConfig  # noqa: F401
from .buffer import SequenceReplayBuffer  # noqa: F401
