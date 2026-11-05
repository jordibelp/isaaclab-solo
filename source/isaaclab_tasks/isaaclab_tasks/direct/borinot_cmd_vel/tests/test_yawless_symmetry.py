# SPDX-License-Identifier: BSD-3-Clause
"""Yawless policy layout is compatible with the existing world-yaw augmentation."""
import pytest
import torch

from test_symmetry import cfg, observations, symmetry


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("pairs", [(False, False), (True, False), (False, True), (True, True)])
def test_yawless_symmetry_has_no_heading_slot_and_keeps_all_controller_memory(sac, pairs):
    transform = symmetry.BorinotSymmetry(cfg("model_wrench_yawless", (False, *pairs), sac))
    assert "yaw" not in transform.action_slices
    assert transform.action_size == 8+sum(pairs)
    assert transform.obs_slices["rotors"].stop-transform.obs_slices["rotors"].start == 6
    obs, _ = observations(transform)
    actions = torch.randn(19, transform.action_size, dtype=torch.float64)
    other = transform.actions(actions, "world_yaw_180")
    torch.testing.assert_close(other[:, :2], -actions[:, :2])
    torch.testing.assert_close(other[:, 2:], actions[:, 2:])
    torch.testing.assert_close(transform.actions(other, "world_yaw_180"), actions)
    torch.testing.assert_close(transform.observations(transform.observations(obs, "world_yaw_180"), "world_yaw_180"), obs)
