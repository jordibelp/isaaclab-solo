# SPDX-License-Identifier: BSD-3-Clause
"""Build a standalone metrics report and synchronized schematic pose replay."""
import json
from html import escape
from pathlib import Path


def pretty(x, rate=False):
    if x is None:
        return "—"
    if rate:
        return f"{100 * x:.1f}%"
    if isinstance(x, float):
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return escape(str(x))


def table(rows, columns):
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{pretty(r.get(k), k.endswith('_rate'))}</td>" for k, _ in columns) + "</tr>" for r in rows)
    return f"<div class=scroll><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def render_report(out, spec, runs, comparisons):
    task, kind = spec["task"], spec["kind"]
    columns = [("policy", "Policy"), ("condition", "Condition"), ("policy_hz", "Policy Hz"), ("episodes", "Trials")]
    columns += {
        "goal": [("successes", "Settled successes"), ("success_rate", "Success"), ("spatial_arrival_rate", "Reached"),
                 ("mean_success_duration_s", "Success time · s"), ("mean_terminal_distance_m", "Final error · m")],
        "keeper": [("resolved_shots", "Resolved shots"), ("saves", "Saves"), ("save_rate", "Saves / resolved"),
                   ("save_rate_including_failure_drops", "With failure drops"), ("racket_contact_rate", "Racket contact")],
        "velocity": [("mean_velocity_error_m_s", "XYZ error · m/s"), ("mean_yaw_error_rad_s", "Yaw error · rad/s"),
                     ("mean_full_horizon_tracking_score", "Training tracking score")],
        "legacy": [("mean_velocity_error_m_s", "XY error · m/s"), ("mean_yaw_error_rad_s", "Yaw error · rad/s")],
    }[kind]
    columns += [("robot_failure_rate", "Robot failure"), ("mean_return", "Mean return"),
                ("mean_actuator_delay_ms", "Realized delay · ms")]
    paired = ""
    if comparisons:
        paired = "<h2>Matched goal outcomes</h2>" + table(comparisons, [
            ("policy", "Policy"), ("baseline", "Reference"), ("condition", "Condition"),
            ("success_delta_pp", "Success change · pp"), ("successes_lost", "Successes lost"),
            ("successes_gained", "Successes gained"), ("succeeded_in_both", "Succeeded in both"),
            ("mean_paired_success_duration_delta_s", "Time change if both succeed · s")])
    blurb = {
        "goal": "Success uses the task’s actual stop rule: the robot must be close enough and slow enough, "
                "without a safety failure. Reached only measures position. Success time only includes successful trials.",
        "keeper": "Save rate uses resolved shots, not contacts. We also show the rate when unresolved shots lost to "
                  "robot failures count against the policy. Launch sequences can diverge after trajectories diverge.",
        "velocity": "Each robot receives the same fixed command. The training score averages the XY tracking "
                    "kernel over the full horizon, so an early failure leaves the remaining time at zero.",
        "legacy": "Each robot receives the same fixed command. XY and yaw errors use the robot's body frame.",
    }[kind]
    serial = {"kind": kind, "runs": [{
        "label": r["record"]["policy"], "condition": r["record"]["condition"],
        "hz": r["record"]["policy_hz"], "rotors": r["data"]["rotors"], "legLinks": r["data"]["leg_links"],
        "trials": [trace | {"episode": r["data"]["episodes"][trace["scenario_id"]]} for trace in r["data"]["traces"]],
    } for r in runs if r["data"]["traces"]]}
    payload = json.dumps(serial, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report.html").read_text()
    (out / "report.html").write_text(template.replace("@@TITLE@@", escape(task))
        .replace("@@SUBTITLE@@", f"Fixed stage {spec['stage']} · {spec['episodes']} trials per condition · seed {spec['seed']}")
        .replace("@@SUMMARY@@", table([r["record"] for r in runs], columns)).replace("@@PAIRED@@", paired)
        .replace("@@BLURB@@", escape(blurb)).replace("@@DATA@@", payload))
