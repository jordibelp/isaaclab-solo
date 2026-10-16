# SPDX-License-Identifier: BSD-3-Clause
"""Play-only score/camera semantics; no simulator or UI imports required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("keeper_play_ui", Path(__file__).parents[1] / "playback_ui.py")
play = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play)


def test_score_counts_resolved_outcomes_not_contacts_or_idle_steps():
    score = play.MatchScore()
    for code in range(10):
        score.record(code, False)
    assert score.shots == 0
    for code in (3, 7, 1, 2, 4, 5, 6, 8, 9):
        score.record(code, True)
        for _ in range(100):
            score.record(code, False)
    assert score.shots == 9
    assert score.counts[3] == score.counts[7] == 1
    assert score.other == 7
    score.clear()
    assert score.shots == score.other == 0
    assert score.last == "Waiting for the first shot"


@pytest.mark.parametrize("mode", play.CAMERA_MODES)
def test_camera_finite_noncoincident_and_free_preserves_manual_view(mode):
    cfg = SimpleNamespace(goal_center=(-.45, 0., 1.2), launch_x_range=(1.9, 2.1), robot_home=(0., 0., 1.5))
    pose = play.camera_pose(cfg, mode, robot=(.1, .2, 1.7), time_s=10.)
    if mode == "free":
        assert pose is None
    else:
        assert np.isfinite(pose).all()
        assert np.linalg.norm(np.subtract(*pose)) > 1.


def test_follow_moves_with_robot_but_orbit_uses_simulated_time():
    cfg = SimpleNamespace(goal_center=(-.45, 0., 1.2), launch_x_range=(1.9, 2.1), robot_home=(0., 0., 1.5))
    a = np.array(play.camera_pose(cfg, "follow", robot=(0., 0., 1.5)))
    b = np.array(play.camera_pose(cfg, "follow", robot=(.1, .2, 1.8)))
    np.testing.assert_allclose(b - a, [[.1, .2, .3]] * 2)
    assert play.camera_pose(cfg, "orbit", time_s=0) != play.camera_pose(cfg, "orbit", time_s=10)


def test_static_camera_does_not_overwrite_mouse_and_free_stops_tracking():
    writes = []
    class Root:
        def cpu(self):
            return self

        def tolist(self):
            return [0., 0., 1.5]

        def __sub__(self, _):
            return self

    root = Root()
    env = SimpleNamespace(cfg=SimpleNamespace(viewer=SimpleNamespace(eye=(4., -4., 3.), lookat=(0., 0., 1.)),
            goal_center=(-.45, 0., 1.2), launch_x_range=(1.9, 2.1)),
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=[root])), scene=SimpleNamespace(env_origins=[root]),
        viewport_camera_controller=SimpleNamespace(update_view_location=lambda **kw: writes.append(kw)))
    controls = play.PlayControls(env, show_ui=False)
    controls.update(0)
    controls.update(1)
    assert len(writes) == 1
    controls.set_camera("follow")
    controls.update(2)
    controls.update(3)
    assert len(writes) == 3
    controls.set_camera("free")
    controls.update(4)
    assert len(writes) == 3
    controls.request_reset()
    controls.toggle_pause()
    assert controls.reset_requested and controls.paused
