import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "source/scripts/rsl_rl/play_direct_0325.py"


def load_helpers(*names):
    tree = ast.parse(SCRIPT.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Path": Path,
        "TRACKING_COMMAND_DURATION_S": 5.0,
        "Any": object,
        "copy": copy,
        "math": math,
        "np": np,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def test_rsl_summary_includes_signed_and_directional_errors():
    summarize, = load_helpers("_summarize_command_tracking")
    summary = summarize(
        [0.02, 0.04],
        [(0.5, 0.0, 0.0)],
        [(0.4, 0.1), (0.6, -0.1)],
        [0.2, -0.2],
        0,
    )

    assert summary["vx_error_signed_mean_mps"] == pytest.approx(0.0)
    assert summary["vy_error_signed_mean_mps"] == pytest.approx(0.0)
    assert summary["wz_error_signed_mean_radps"] == pytest.approx(0.0)
    assert summary["vxy_error_along_command_mean_mps"] == pytest.approx(0.0)
    assert summary["vxy_error_along_command_std_mps"] == pytest.approx(0.1)


def test_rsl_tracking_sequence_controls_evaluation_duration():
    duration, = load_helpers("_evaluation_duration_s")
    assert duration(2000.0, [(0.5, 0.0, 0.0)] * 8) == 40.0
    assert duration(2000.0, []) == 2000.0


def test_rsl_wandb_history_matches_mujoco_metric_names(tmp_path):
    log_csv, = load_helpers("_log_command_tracking_csv_to_wandb")
    csv_path = tmp_path / "command_tracking.csv"
    csv_path.write_text(
        "time_s,cmd_vx,cmd_vy,cmd_wz,velocity_vx,velocity_vy,yaw_rate_wz,"
        "vxy_error_norm,wz_error_abs,vx_error_signed,vy_error_signed,wz_error_signed,"
        "vxy_error_along_command\n"
        "0.02,0.5,0,0,0.4,0.1,-0.2,0.141421,0.2,-0.1,0.1,-0.2,-0.1\n"
    )

    class Run:
        def __init__(self):
            self.metrics = []
            self.rows = []

        def define_metric(self, name, **kwargs):
            self.metrics.append((name, kwargs))

        def log(self, row):
            self.rows.append(row)

    run = Run()
    assert log_csv(run, csv_path) == 1
    assert run.rows[0]["tracking/error_vx_signed_mps"] == pytest.approx(-0.1)
    assert run.rows[0]["tracking/error_vxy_along_command_mps"] == pytest.approx(-0.1)
    assert ("tracking/error_vxy_along_command_mps", {"step_metric": "tracking/time_s"}) in run.metrics


def test_rsl_joint_plots_include_effective_soft_and_physical_limits(tmp_path, monkeypatch):
    save_plots, = load_helpers("_save_command_tracking_plots")

    import matplotlib.axes

    horizontal_lines = []
    joint_traces = []
    ylabels = []
    original_axhline = matplotlib.axes.Axes.axhline
    original_plot = matplotlib.axes.Axes.plot
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def record_axhline(self, y=0, *args, **kwargs):
        horizontal_lines.append((float(y), kwargs.get("label")))
        return original_axhline(self, y, *args, **kwargs)

    def record_plot(self, *args, **kwargs):
        if kwargs.get("label") in {r"$q$", r"$q_{des}$"}:
            joint_traces.append((np.asarray(args[1]), kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    def record_set_ylabel(self, ylabel, *args, **kwargs):
        ylabels.append(ylabel)
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", record_axhline)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", record_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", record_set_ylabel)
    paths = save_plots(
        tmp_path,
        [0.02, 0.04],
        [(0.5, 0.0, 0.0)],
        [(0.1, 0.0), (0.2, 0.0)],
        [0.0, 0.0],
        ["FL_hip_joint", "FR_hip_joint"],
        [[0.0, 0.0], [0.1, -0.1]],
        [[0.05, -0.05], [0.1, -0.1]],
        np.asarray([[-1.2, 1.2], [-1.3, 1.3]]),
        np.asarray([[-0.8, 0.8], [-0.9, 0.9]]),
    )

    assert (math.degrees(-0.8), "soft lower") in horizontal_lines
    assert (math.degrees(0.9), "soft upper") in horizontal_lines
    assert (math.degrees(-1.2), "hard lower") in horizontal_lines
    assert (math.degrees(1.3), "hard upper") in horizontal_lines
    np.testing.assert_allclose(joint_traces[0][0], np.rad2deg([0.0, 0.1]))
    np.testing.assert_allclose(joint_traces[1][0], np.rad2deg([0.05, 0.1]))
    assert ylabels.count("deg") == 2
    assert tmp_path / "joint_position_vs_limits_left.png" in paths
    assert tmp_path / "joint_position_vs_limits_right.png" in paths


def test_rsl_sac_inference_cfg_does_not_allocate_training_replay_buffer():
    cfg_to_dict, inference_cfg = load_helpers("_cfg_to_dict", "_off_policy_inference_runner_cfg")
    training_cfg = {"algorithm": {"replay_buffer_size": 5_000_000}, "num_steps_per_env": 24}

    play_cfg = inference_cfg(training_cfg, num_envs=3)

    assert play_cfg["algorithm"]["replay_buffer_size"] == 3
    assert training_cfg["algorithm"]["replay_buffer_size"] == 5_000_000
