# SPDX-License-Identifier: BSD-3-Clause
"""Task-native denominators and matched-scenario comparisons."""
import math
from statistics import mean

# Matches both Goalkeeper tasks' outcome enums.
OUTCOMES = ("ongoing", "base_hit", "arm_hit", "goal", "ball_floor", "leave_goal_area", "deadline",
            "save", "robot_crash", "miss")
SAVE = OUTCOMES.index("save")


def average(values):
    valid = [x for x in values if x is not None]
    return mean(valid) if valid else None


def ratio(n, d):
    return n / d if d else None


def wilson(successes, n):
    if not n:
        return [None, None]
    p, z = successes / n, 1.959963984540054
    middle = (p + z * z / (2 * n)) / (1 + z * z / n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0., middle - width), min(1., middle + width)]


def summarize(kind, rows):
    if not rows or len({r["scenario_id"] for r in rows}) != len(rows):
        raise ValueError("Each planned trial must have one complete episode.")
    n = len(rows)
    failures = sum(r["robot_failure"] for r in rows)
    result = {"episodes": n, "robot_failure_count": failures, "robot_failure_rate": failures / n}
    for key in ("duration_s", "return", "actuator_delay_ms", "energy_j"):
        result[f"mean_{key}"] = average(r[key] for r in rows)
    if kind == "goal":
        success = sum(r["success"] for r in rows)
        result.update(successes=success, success_rate=success / n,
                      success_ci95_low=wilson(success, n)[0], success_ci95_high=wilson(success, n)[1],
                      spatial_arrival_rate=sum(r["first_arrival_s"] is not None for r in rows) / n,
                      settled_at_least_once_rate=sum(r["first_settled_s"] is not None for r in rows) / n)
        for key in ("terminal_distance_m", "minimum_distance_m", "first_arrival_s", "first_settled_s"):
            result[f"mean_{key}"] = average(r[key] for r in rows)
        result["mean_success_duration_s"] = average(r["duration_s"] for r in rows if r["success"])
        result["mean_first_settled_s_on_success"] = average(r["first_settled_s"] for r in rows if r["success"])
    elif kind == "keeper":
        shots, saves, contacts = [sum(r[k] for r in rows) for k in ("resolved_shots", "saves", "racket_contacts")]
        failure_drops = sum(r["unresolved_shot"] and r["robot_failure"] for r in rows)
        time_limit_drops = sum(r["unresolved_shot"] and not r["robot_failure"] for r in rows)
        result.update(resolved_shots=shots, saves=saves, save_rate=ratio(saves, shots),
                      racket_contact_rate=ratio(contacts, shots),
                      save_rate_including_failure_drops=ratio(saves, shots + failure_drops),
                      unresolved_at_failure=failure_drops, unresolved_at_timeout=time_limit_drops,
                      outcome_counts={name: sum(r[f"outcome_{name}"] for r in rows) for name in OUTCOMES if name != "ongoing"})
    else:
        for key in ("velocity_error_m_s", "velocity_rmse_m_s", "yaw_error_rad_s", "full_horizon_tracking_score"):
            result[f"mean_{key}"] = average(r.get(key) for r in rows)
    return result


def paired_goal_changes(baseline, delayed):
    a, b = [{r["scenario_id"]: r for r in rows} for rows in (baseline, delayed)]
    if not a or a.keys() != b.keys() or any(
        a[i]["target_xyz_m"] != b[i]["target_xyz_m"] or a[i]["deadline_s"] != b[i]["deadline_s"] for i in a
    ):
        raise ValueError("Paired comparison requires identical scenario IDs, targets and deadlines.")
    improved = sum(not a[i]["success"] and b[i]["success"] for i in a)
    regressed = sum(a[i]["success"] and not b[i]["success"] for i in a)
    common = [i for i in a if a[i]["success"] and b[i]["success"]]
    return {"paired_trials": len(a), "success_delta_pp": (improved - regressed) * 100 / len(a),
            "successes_gained": improved, "successes_lost": regressed, "succeeded_in_both": len(common),
            "mean_paired_success_duration_delta_s": average(b[i]["duration_s"] - a[i]["duration_s"] for i in common)}
