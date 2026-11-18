# SPDX-License-Identifier: BSD-3-Clause
"""Typed, simulator-free input and scenario plan for Borinot evaluation."""
from __future__ import annotations

import json
import math
import random
import re
import shlex
from pathlib import Path

TASKS = {
    "Borinot-goto": "goal",
    "Borinot-goto-minTime": "goal",
    "Borinot-cmd-vel": "velocity",
    "Borinot-Goalkeeper-Direct-v0": "keeper",
    "Borinot-Goalkeeper-Simple-Direct-v0": "keeper",
    "Borinot-Direct-v0": "legacy",
}
AGENTS = ("rsl_rl_cfg_entry_point", "rsl_rl_sac_cfg_entry_point", "rsl_rl_sac_mse_cfg_entry_point")
RANDOMIZATIONS = {
    "joint_friction": "Joint friction", "add_base_mass": "Base mass + inertia", "base_com": "Base centre of mass",
    "racket_mass": "Racket mass", "motor_lag": "Motor response", "thrust_gain": "Thrust gain", "yaw_gain": "Yaw gain",
    "reset_jitter": "Robot reset jitter", "vertical_push": "Vertical pushes",
    "steady_wind": "Steady wind", "gust": "Wind gusts", "moment": "External moment",
}
DEFAULT_COMMANDS = {"goal": (3., 0., 1.5), "velocity": (1., 0., 0., 0.), "legacy": (1., 0., 0.)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fields(obj, allowed, context):
    if not isinstance(obj, dict) or (unknown := obj.keys() - set(allowed)):
        raise ValueError(f"Unknown or invalid fields in {context}: {obj if not isinstance(obj, dict) else sorted(unknown)}")


def number(value, name, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be a finite number" + (f" >= {minimum}." if minimum is not None else "."))
    return float(value)


def vector(values, n, name, minimum=None):
    if not isinstance(values, (tuple, list)) or len(values) != n:
        raise ValueError(f"{name} needs {n} numbers.")
    return [number(x, name, minimum) for x in values]


def integer(value, name, low=0):
    if type(value) is not int or value < low:
        raise ValueError(f"{name} must be an integer >= {low}.")
    return value


def split_env_overrides(text):
    if not isinstance(text, str):
        raise ValueError("Hydra overrides must be a text line.")
    tokens = shlex.split(text)
    for token in tokens:
        if not re.fullmatch(r"env(?:\.[A-Za-z_][A-Za-z0-9_]*)+", token.partition("=")[0]) or "=" not in token:
            raise ValueError(f"Expected an env.path=value override, got {token!r}.")
    return tokens


def parse_env_overrides(text):
    """Parse Hydra values in order, without launching Hydra or a shell."""
    tokens = split_env_overrides(text)
    if not tokens:
        return []
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import HydraException

    result = []
    try:
        overrides = OverridesParser.create().parse_overrides(tokens)
        for override in overrides:
            if override.is_sweep_override():
                raise ValueError("Hydra sweeps are not supported here. Use the GUI conditions for delay sweeps.")
            value = override.value()
            for key in reversed(override.key_or_group.split(".")[1:]):
                value = {key: value}
            result.append(value)
    except HydraException as exc:
        raise ValueError(str(exc)) from exc
    return result


def validate(recipe, base, *, require_weights=True):
    fields(recipe, ("task", "seed", "stage", "episodes", "num_envs", "replay_count", "scenario",
                    "env", "hydra_overrides", "policies", "conditions"), "recipe")
    hydra_overrides = recipe.get("hydra_overrides", "")
    split_env_overrides(hydra_overrides)
    task = recipe.get("task", "Borinot-goto-minTime")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {', '.join(TASKS)}.")
    kind = TASKS[task]
    shared = recipe.get("env", {})
    if not isinstance(shared, dict):
        raise ValueError("env must be a JSON object.")
    episodes = integer(recipe.get("episodes", 64), "episodes", 1)
    replay = integer(recipe.get("replay_count", 4), "replay_count")
    if not 0 <= replay <= 16:
        raise ValueError("replay_count must be between 0 and 16.")
    scenario = recipe.get("scenario", {})
    fields(scenario, ("command", "deadline_s", "sample_box") if kind == "goal" else
           (() if kind == "keeper" else ("command",)), "scenario")
    if kind != "keeper":
        default = DEFAULT_COMMANDS[kind]
        scenario = scenario | {"command": vector(scenario.get("command", default), len(default), "scenario.command")}
    if kind == "goal":
        scenario["deadline_s"] = number(scenario.get("deadline_s", 2.4), "deadline_s", .001)
        if "sample_box" in scenario:
            box = scenario["sample_box"]
            fields(box, ("center", "half_extents", "deadline_s"), "sample_box")
            low, high = vector(box.get("deadline_s", [2.2, 2.6]), 2, "sample_box.deadline_s", .001)
            if high < low:
                raise ValueError("sample_box deadline range is reversed.")
            scenario["sample_box"] = {
                "center": vector(box.get("center", [0, 0, 1.5]), 3, "sample_box.center"),
                "half_extents": vector(box.get("half_extents"), 3, "sample_box.half_extents", 0),
                "deadline_s": [low, high],
            }

    policies = []
    for p in recipe.get("policies", []):
        fields(p, ("label", "checkpoint", "control_mode", "policy_hz", "agent", "env", "runner", "training_run"), "policy")
        if not isinstance(p.get("checkpoint"), str) or not p["checkpoint"].strip():
            raise ValueError("Each policy needs a local checkpoint path.")
        checkpoint = Path(p["checkpoint"]).expanduser()
        checkpoint = (base / checkpoint).resolve()
        if require_weights and not checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint}")
        p = p | {"checkpoint": str(checkpoint), "label": p.get("label") or checkpoint.stem,
                 "agent": p.get("agent", AGENTS[0])}
        if p["agent"] not in AGENTS or not isinstance(p["label"], str):
            raise ValueError("Invalid policy label or agent registry key.")
        for key in ("env", "runner"):
            if not isinstance(p.get(key, {}), dict):
                raise ValueError(f"policy.{key} must be a JSON object.")
        if kind != "legacy":
            p["policy_hz"] = integer(p.get("policy_hz", p.get("env", {}).get("policy_hz", shared.get("policy_hz", 100))),
                                     "policy_hz", 1)
        elif "control_mode" in p or "policy_hz" in p:
            raise ValueError("For legacy Borinot, omit control_mode and policy_hz; use env.sim.dt and env.decimation.")
        policies.append(p)
    if not policies or len({p["label"] for p in policies}) != len(policies):
        raise ValueError("At least one policy is required, with distinct labels.")

    conditions = []
    for c in recipe.get("conditions", [{"delay_ms": 0}, {"delay_ms": 20}]):
        fields(c, ("label", "delay_ms", "delay_range_ms", "randomizations"), "condition")
        if "delay_ms" in c and "delay_range_ms" in c:
            raise ValueError("Use fixed delay_ms or delay_range_ms, not both.")
        if "delay_range_ms" in c:
            bounds = vector(c["delay_range_ms"], 2, "delay_range_ms", 0)
            if bounds[0] > bounds[1]:
                raise ValueError("delay_range_ms is reversed.")
            c = c | {"delay_range_ms": bounds}
            label = f"{bounds[0]:g}–{bounds[1]:g} ms"
        else:
            delay = number(c.get("delay_ms", 0), "delay_ms", 0)
            c = c | {"delay_ms": delay}
            label = f"{delay:g} ms"
        names = c.get("randomizations", [])
        if not isinstance(names, list) or any(not isinstance(x, str) or x not in RANDOMIZATIONS for x in names):
            raise ValueError(f"Unknown randomization in {names!r}.")
        supported = set(RANDOMIZATIONS)
        if kind == "legacy":
            supported -= {"joint_friction", "add_base_mass", "base_com", "racket_mass", "vertical_push"}
        elif kind == "keeper":
            supported.discard("vertical_push")
        if set(names) - supported:
            raise ValueError(f"Randomizations not available on {task}: {sorted(set(names) - supported)}")
        conditions.append(c | {"label": c.get("label") or label, "randomizations": names})
    if not conditions or any(not isinstance(c["label"], str) for c in conditions) or len(
        {c["label"] for c in conditions}
    ) != len(conditions):
        raise ValueError("At least one condition is required, with distinct labels.")

    return {"task": task, "kind": kind, "seed": integer(recipe.get("seed", 42), "seed"),
            "stage": integer(recipe.get("stage", 0), "stage"), "episodes": episodes,
            "num_envs": min(episodes, integer(recipe.get("num_envs", 64), "num_envs", 1)),
            "replay_count": min(episodes, replay), "scenario": scenario, "env": shared,
            "hydra_overrides": hydra_overrides.strip(), "policies": policies, "conditions": conditions}


def plan_scenarios(spec):
    """Make one explicit, indexed scenario list shared by every condition and policy."""
    kind, scenario = spec["kind"], spec["scenario"]
    rng = random.Random(spec["seed"])
    rows = []
    if kind == "goal":
        common_hz = math.gcd(*(p["policy_hz"] for p in spec["policies"]))
        box = scenario.get("sample_box")
        if box:
            lo, hi = box["deadline_s"]
            lo, hi = math.ceil(lo * common_hz - 1e-9), math.floor(hi * common_hz + 1e-9)
            if lo > hi:
                raise ValueError("Sample deadline range has no tick shared by all policy rates; widen the range.")
    for idx in range(spec["episodes"]):
        row = {"scenario_id": idx}
        if kind == "goal":
            target = scenario["command"]
            deadline_ticks = math.ceil(scenario["deadline_s"] * common_hz - 1e-9)
            if idx and box:
                target = [c + rng.uniform(-r, r) for c, r in zip(box["center"], box["half_extents"])]
                deadline_ticks = rng.randint(lo, hi)
            row.update(target_xyz_m=target, deadline_s=deadline_ticks / common_hz)
        elif kind in ("velocity", "legacy"):
            row["velocity_command"] = scenario["command"]
        rows.append(row)
    return rows


def apply_condition(cfg, condition):
    """Select perturbations after plant/agent overrides. Delay is independent of the other switches."""
    selected = set(condition["randomizations"])
    flight = cfg.flight_randomization
    cfg.enable_domain_randomization = True
    cfg.actuator_delay = condition.get("delay_ms", 0.0) / 1000
    cfg.actuator_delay_range = (tuple(x / 1000 for x in condition["delay_range_ms"])
                                if "delay_range_ms" in condition else None)
    if hasattr(cfg, "include_events_randomization"):
        cfg.event_randomization = {key: key in selected for key in cfg.event_randomization}
        cfg.include_events_randomization = any(cfg.event_randomization.values())
    if "motor_lag" not in selected:
        flight.motor_time_constant_range_s = None
    if "thrust_gain" not in selected:
        flight.thrust_shared_gain_delta = flight.thrust_rotor_gain_delta = 0.0
    if "yaw_gain" not in selected:
        flight.km_gain_delta = 0.0
    flight.wind_enabled = bool(selected & {"steady_wind", "gust", "moment"})
    for name, field in (("steady_wind", "wind_steady_max_m_s"), ("gust", "wind_gust_max_m_s"),
                        ("moment", "external_moment_max_nm")):
        if name not in selected:
            setattr(flight, field, 0.0)
    if "reset_jitter" not in selected:
        for key in ("reset_xy_noise", "reset_z_noise", "reset_joint_noise", "reset_xy_noise_m", "reset_z_noise_m"):
            if hasattr(cfg, key):
                setattr(cfg, key, 0.0)
        if hasattr(cfg, "reset_rpy_noise_degs"):
            cfg.reset_rpy_noise_degs = (0., 0., 0.)
    # Evaluation evaluates the requested mode, not learned action/observation corruption.
    cfg.action_noise_model = cfg.observation_noise_model = None
