import ast
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
