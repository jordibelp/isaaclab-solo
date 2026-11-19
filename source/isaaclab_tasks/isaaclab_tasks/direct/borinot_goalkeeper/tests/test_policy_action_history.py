# SPDX-License-Identifier: BSD-3-Clause
from types import SimpleNamespace

import pytest
import torch

from test_control import action_history


@pytest.mark.parametrize("steps, extra", [(0, 0), (1, 0), (2, 8), (7, 48)])
def test_history_dimension(steps, extra):
    cfg = SimpleNamespace(include_prev_action_steps=steps, action_space=8)
    assert action_history.extra_action_history_dim(cfg) == extra


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "3"])
def test_invalid_lengths(invalid):
    with pytest.raises(ValueError, match="non-negative integer"):
        action_history.extra_action_history_dim(SimpleNamespace(include_prev_action_steps=invalid, action_space=8))


@pytest.mark.parametrize("steps", [2, 4, 7])
def test_history_wraparound_partial_reset_and_rereads(steps):
    latest = torch.zeros(3, 8)
    history = action_history.OlderPolicyActions(latest, steps)
    expected = torch.zeros(3, steps - 1, 8)
    critic = latest.new_ones(3, 21)
    for tick in range(2 * steps + 1):
        expected = torch.cat((latest[:, None], expected[:, :-1]), dim=1)
        history.push(latest)
        latest.copy_(torch.arange(24).reshape(3, 8) + tick * 31 + 1)
        if tick == steps:
            ids = torch.tensor([0, 2])
            latest[ids] = 0
            expected[ids] = 0
            history.reset(ids)
        obs = {"policy": latest, "critic": critic}
        for _ in range(2):
            actual = history.append(obs)
            torch.testing.assert_close(actual["policy"], torch.cat([latest, expected.flatten(1)], -1))
            assert actual["critic"] is critic
            shared = history.append({"policy": latest, "critic": latest}, critic=True)
            assert shared["critic"] is shared["policy"]
            torch.testing.assert_close(shared["critic"], actual["policy"])
