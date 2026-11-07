# SPDX-License-Identifier: BSD-3-Clause
"""Hybrid action/observation symmetry matches its nominal model-wrench mode."""
import importlib

import pytest
import torch

from test_symmetry import cfg, observations, symmetry
from test_control import method
from test_wrench import wrench


@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("pairs", [(False, False), (True, False), (False, True), (True, True)])
def test_hybrid_world_heading_transform_keeps_body_observer_contract(yawless, sac, pairs):
    nominal = "model_wrench_yawless" if yawless else "model_wrench"
    base = cfg(nominal, (False, *pairs), sac, acceleration=True)
    expected = symmetry.BorinotSymmetry(base)
    base.control_mode += "_indi"
    hybrid = importlib.import_module(wrench.__package__+".borinot_model_wrench_indi")
    base.model_wrench_indi = hybrid.ModelWrenchIndiCfg()
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(base)
    actual = symmetry.BorinotSymmetry(base)
    assert actual.action_slices == expected.action_slices
    assert actual.obs_slices == expected.obs_slices
    obs, _ = observations(expected)
    actions = torch.randn(19, actual.action_size, dtype=torch.float64)
    for name in actual.transforms:
        torch.testing.assert_close(actual.actions(actions, name), expected.actions(actions, name), atol=0, rtol=0)
        torch.testing.assert_close(actual.observations(obs, name), expected.observations(obs, name), atol=0, rtol=0)
