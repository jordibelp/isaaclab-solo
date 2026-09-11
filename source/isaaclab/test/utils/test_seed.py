# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first.

This is only needed because importing isaaclab.utils initializes USD-backed helper modules.
"""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import random

import numpy as np
import torch

from isaaclab.utils.seed import RngStream, configure_seed


def _draw_signature() -> tuple[float, list[float], torch.Tensor]:
    return (
        random.random(),
        np.random.random(3).tolist(),
        torch.rand(3),
    )


def _assert_signature_equal(
    actual: tuple[float, list[float], torch.Tensor],
    expected: tuple[float, list[float], torch.Tensor],
):
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_rng_stream_preserves_outer_state_and_advances_inner_state():
    configure_seed(123)
    stream = RngStream.from_seed(999)

    with stream.use():
        inner_first = _draw_signature()
    outer_first = _draw_signature()
    with stream.use():
        inner_second = _draw_signature()

    configure_seed(123)
    _assert_signature_equal(outer_first, _draw_signature())

    configure_seed(999)
    _assert_signature_equal(inner_first, _draw_signature())
    _assert_signature_equal(inner_second, _draw_signature())


def test_rng_stream_sequence_is_independent_from_agent_random_consumption():
    def run_with_agent_burn(torch_draw_count: int):
        configure_seed(11)
        env_stream = RngStream.from_seed(42)
        torch.rand(torch_draw_count)
        random.random()
        np.random.random(torch_draw_count % 7 + 1)
        with env_stream.use():
            first_env_draw = _draw_signature()
        torch.rand(torch_draw_count + 31)
        random.random()
        np.random.random(torch_draw_count % 5 + 2)
        with env_stream.use():
            second_env_draw = _draw_signature()
        return first_env_draw, second_env_draw

    small_agent = run_with_agent_burn(1)
    large_agent = run_with_agent_burn(10_000)

    for actual, expected in zip(large_agent, small_agent, strict=True):
        _assert_signature_equal(actual, expected)
