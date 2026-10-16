# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check sparse device reads, measured angles, joint order and task-limit semantics without Kit."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from joint_position_plots import JointPositionPlots


@pytest.fixture
def panel(monkeypatch):
    def build(self):
        self.window = SimpleNamespace(visible=True, destroy=Mock())
        self.time_label = SimpleNamespace(text="")
        self.values = [SimpleNamespace(text="", style={}) for _ in self.names]
        self.plots = [Mock() for _ in self.names]

    monkeypatch.setattr(JointPositionPlots, "_build_window", build)

    class Data:
        reads = 0
        joint_pos_limits = torch.deg2rad(torch.tensor([[[-179., 179.], [-60., 60.], [-145., 55.]]]))

        @property
        def joint_pos(self):
            self.reads += 1
            return torch.deg2rad(torch.tensor([[-160., 10., -150.], [0., 0., 0.]]))

    data = Data()
    env = SimpleNamespace(
        _robot=SimpleNamespace(joint_names=["FR_calf_joint", "FL_hip_joint", "FL_thigh_joint"], data=data),
        _joint_ids=[1, 2, 0],
        _joint_soft_pos_limits=torch.deg2rad(torch.tensor([[[-40., 40.], [-125., 35.], [-155., 155.]]])),
    )
    return JointPositionPlots(env)


def test_sparse_sampling_does_not_read_device_between_ticks(panel):
    for step in range(201):
        panel.update(step * 0.005)
    assert panel.env._robot.data.reads == 3  # 0.0, 0.5, 1.0, not 201 physics ticks
    assert [t for t, _ in panel.history] == [0.0, 0.5, 1.0]


def test_actual_joint_order_angles_limits_and_warnings(panel):
    assert panel.names == ["FL_hip_joint", "FL_thigh_joint", "FR_calf_joint"]
    assert panel.physical_deg[1] == pytest.approx((-145., 55.), abs=1e-5)
    assert panel.soft_deg[1] == pytest.approx((-125., 35.), abs=1e-5)
    panel.update(0.)
    assert panel.history[-1][1] == pytest.approx([10., -150., -160.], abs=1e-5)
    assert "inside soft" in panel.values[0].tooltip
    assert "PHYSICAL LIMIT" in panel.values[1].tooltip
    assert "outside soft" in panel.values[2].tooltip
    # Measured angles remain visible, not clamped into configured bounds.
    assert "-150.0 deg" in panel.values[1].text


def test_closed_panel_does_not_read_device(panel):
    panel.window.visible = False
    assert panel.update(1.) is False
    assert panel.env._robot.data.reads == 0


def test_reopen_does_not_connect_across_unsampled_gap(panel):
    panel.update(0.)
    panel.window.visible = False
    panel.update(1.)
    panel.window.visible = True
    panel.update(3.)
    assert [t for t, _ in panel.history] == [3.]


def test_history_is_bounded_and_reset_does_not_join_episodes(panel):
    for i in range(150):
        panel.update(i * .5)
    assert len(panel.history) == 121
    assert panel.history[0][0] == 14.5
    panel.reset_history()
    assert not panel.history
    # Reset does not bypass the sample-rate gate.
    assert panel.update(74.6) is False
    panel.update(75.)
    assert len(panel.history) == 1
    assert "75.0 to 75.0" in panel.time_label.text


@pytest.mark.parametrize("interval", [0., -1., float("nan"), float("inf")])
def test_reject_invalid_interval(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        JointPositionPlots(None, interval)
