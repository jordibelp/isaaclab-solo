# SPDX-License-Identifier: BSD-3-Clause
"""Simulator-independent tests of scenarios, perturbation isolation, and task metrics."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from borinot_eval.metrics import OUTCOMES, paired_goal_changes, summarize
from borinot_eval.report import render_report
from borinot_eval.spec import apply_condition, plan_scenarios, validate


def sample_recipe(tmp):
    policies = []
    for rate in (50, 20, 50):
        ckpt = tmp / f"checkpoint-{len(policies)}.pt"
        ckpt.touch()
        policies.append({"checkpoint": ckpt.name, "policy_hz": rate})
    return {
        "task": "Borinot-goto-minTime", "episodes": 8, "num_envs": 3, "seed": 12, "policies": policies,
        "scenario": {"command": [3, 0, 1.5], "deadline_s": 2.37,
                     "sample_box": {"half_extents": [7, 1, 0.5], "deadline_s": [2.2, 2.6]}},
    }


def test_plan_pairs_targets_and_physical_deadlines_at_different_policy_rates(tmp_path):
    spec = validate(sample_recipe(tmp_path), tmp_path)
    rows = plan_scenarios(spec)
    assert rows[0] == {"scenario_id": 0, "target_xyz_m": [3, 0, 1.5], "deadline_s": 2.4}
    assert len({tuple(row["target_xyz_m"]) for row in rows}) == 8
    assert [row["scenario_id"] for row in rows] == list(range(8))
    for r in rows:
        assert 2.2 <= r["deadline_s"] <= 2.6
        for p in spec["policies"]:
            ticks = p["policy_hz"] * r["deadline_s"]
            assert ticks == pytest.approx(round(ticks))
    assert plan_scenarios(spec | {"policies": list(reversed(spec["policies"]))}) == rows
    assert [c["delay_ms"] for c in spec["conditions"]] == [0., 20.]


def test_invalid_range_and_unknown_randomization_rejected(tmp_path):
    src = sample_recipe(tmp_path)
    with pytest.raises(ValueError, match="reversed"):
        validate(src | {"conditions": [{"delay_range_ms": [20, 0]}]}, tmp_path)
    with pytest.raises(ValueError, match="Unknown randomization"):
        validate(src | {"conditions": [{"randomizations": ["sensor noise"]}]}, tmp_path)
    with pytest.raises(ValueError, match="distinct labels"):
        validate(src | {"conditions": [{"delay_ms": 20}, {"delay_ms": 20}]}, tmp_path)
    src["scenario"]["sample_box"]["deadline_s"] = [2.21, 2.28]
    with pytest.raises(ValueError, match="shared by all policy rates"):
        plan_scenarios(validate(src, tmp_path))


def test_repeat_velocity_commands_and_keeper_only_stage_inputs(tmp_path):
    src = sample_recipe(tmp_path)
    del src["scenario"]
    for task, command in (("Borinot-cmd-vel", [1, 2, 3, 0.5]), ("Borinot-Direct-v0", [1, 2, .5])):
        policy = {"checkpoint": src["policies"][0]["checkpoint"]}
        spec = validate(src | {"task": task, "policies": [policy], "scenario": {"command": command}}, tmp_path)
        assert all(r["velocity_command"] == command for r in plan_scenarios(spec))
    gk = validate(src | {"task": "Borinot-Goalkeeper-Direct-v0"}, tmp_path)
    assert plan_scenarios(gk) == [{"scenario_id": i} for i in range(8)]


def test_delay_only_is_nominal_without_destroying_force_stage_or_motor_model():
    flight = NS(motor_time_constant_range_s=(.035, .045), thrust_shared_gain_delta=.1, thrust_rotor_gain_delta=.03,
                km_gain_delta=.1, wind_enabled=True, wind_gust_max_m_s=2., wind_steady_max_m_s=3., external_moment_max_nm=.04)
    cfg = NS(flight_randomization=flight, include_events_randomization=True,
             event_randomization={"base_com": True}, reset_xy_noise_m=.015, reset_rpy_noise_degs=(2., 1., 3.),
             forces_applied_to_base_curriculum=(0., 4.), motor_time_constant_s=.04)
    apply_condition(cfg, {"delay_ms": 20, "randomizations": ["gust", "base_com"]})
    assert cfg.enable_domain_randomization and cfg.actuator_delay == .02 and cfg.actuator_delay_range is None
    assert cfg.motor_time_constant_s == .04 and cfg.forces_applied_to_base_curriculum == (0., 4.)
    assert cfg.reset_xy_noise_m == 0. and cfg.reset_rpy_noise_degs == (0., 0., 0.)
    assert not flight.km_gain_delta and not flight.thrust_shared_gain_delta and not flight.thrust_rotor_gain_delta
    assert cfg.include_events_randomization and cfg.event_randomization["base_com"]
    assert flight.wind_enabled and flight.wind_gust_max_m_s == 2. and flight.wind_steady_max_m_s == 0.


def base_row(i):
    return {"scenario_id": i, "return": -1., "energy_j": 50., "duration_s": 2.,
            "robot_failure": False, "actuator_delay_ms": 20.}


def goal(i, success, arrival, settled, duration=2.):
    return base_row(i) | {"success": success, "first_arrival_s": arrival, "first_settled_s": settled,
                          "terminal_distance_m": .1, "minimum_distance_m": .01,
                          "target_xyz_m": [3., 0., 1.5], "deadline_s": 2.4, "duration_s": duration}


def test_goal_statistics_keep_spatial_entries_separate_from_settled_success():
    rows = [goal(0, False, .5, None), goal(1, True, 1., 2.), goal(2, False, None, None)]
    summary = summarize("goal", rows)
    assert summary["success_rate"] == 1/3 and summary["spatial_arrival_rate"] == 2/3
    assert summary["mean_first_arrival_s"] == .75 and summary["mean_success_duration_s"] == 2.
    assert summarize("goal", [rows[0]])["mean_success_duration_s"] is None
    with pytest.raises(ValueError, match="one complete episode"):
        summarize("goal", [rows[0], rows[0]])


def test_paired_only_counts_common_successes_for_time_and_rejects_unmatched_commands():
    a = [goal(0, True, 1., 1.7, 1.7), goal(1, True, 1., 1.4, 1.4), goal(2, False, 1., None)]
    b = [goal(0, True, 1., 1.9, 1.9), goal(1, False, 1., None), goal(2, False, None, None)]
    delta = paired_goal_changes(a, list(reversed(b)))
    assert delta["success_delta_pp"] == pytest.approx(-100/3)
    assert delta["successes_lost"] == 1 and delta["succeeded_in_both"] == 1
    assert delta["mean_paired_success_duration_delta_s"] == pytest.approx(.2)
    b[0]["deadline_s"] += .2
    with pytest.raises(ValueError, match="identical scenario"):
        paired_goal_changes(a, b)


def test_goalkeeper_denominator_counts_shots_not_episodes_and_handles_reset_drops():
    rows = []
    for i, (saves, misses, failure, drop) in enumerate(((2, 3, False, False), (0, 0, True, True), (0, 0, False, True))):
        outcomes = {f"outcome_{name}": 0 for name in OUTCOMES}
        rows.append(base_row(i) | outcomes | {
            "outcome_save": saves, "outcome_miss": misses, "resolved_shots": saves+misses,
            "saves": saves, "racket_contacts": saves+misses, "robot_failure": failure, "unresolved_shot": drop})
    result = summarize("keeper", rows)
    assert result["save_rate"] == 2/5 and result["save_rate_including_failure_drops"] == 2/6
    assert result["unresolved_at_failure"] == result["unresolved_at_timeout"] == 1
    assert summarize("keeper", rows[1:])["save_rate"] is None


def test_report_escapes_checkpoint_labels_as_data(tmp_path):
    spec = validate(sample_recipe(tmp_path), tmp_path)
    row = goal(0, False, None, None)
    data = {"rotors": [], "leg_links": 0, "episodes": [row],
            "traces": [{"scenario_id": 0, "dt": .02, "frames": [[0, 0, 1.5, 1, 0, 0, 0]]}]}
    record = {"policy": "</script><script>unsafe</script>", "condition": "20 ms", "policy_hz": 50, "episodes": 1}
    render_report(tmp_path, spec, [{"record": record, "data": data}], [])
    html = (tmp_path / "report.html").read_text()
    assert "</script><script>unsafe</script>" not in html
    assert "\\u003c/script>" in html and "&lt;/script&gt;" in html
