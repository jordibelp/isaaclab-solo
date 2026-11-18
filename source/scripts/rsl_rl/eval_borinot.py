# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate Borinot checkpoints from one JSON recipe, without a training run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from borinot_eval.metrics import paired_goal_changes
from borinot_eval.spec import parse_env_overrides, plan_scenarios, read_json, validate, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIT = "--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false"


def run_worker(job_path, device):
    log = job_path.parent / "sim.log"
    cmd = [sys.executable, str(HERE / "borinot_eval_worker.py"), "--job", str(job_path),
           "--headless", "--device", device, f"--kit_args={KIT}"]
    write_json(job_path.parent / "argv.json", cmd)
    with log.open("w") as out:
        process = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=900)
    result = job_path.parent / "result.json"
    if process.returncode or not result.is_file():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-65:])
        raise RuntimeError(f"Simulator worker failed (exit {process.returncode}): {log}\n{tail}")
    return read_json(result)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()


def run(spec, output, device):
    plan = plan_scenarios(spec)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "recipe.json", {k: v for k, v in spec.items() if k != "kind"})
    write_json(output / "scenarios.json", plan)
    write_json(output / "provenance.json", {
        "argv": sys.argv, "commit": git("rev-parse", "HEAD"), "tracked_status": git("status", "--porcelain", "-uno"),
        "checkpoints": [{"label": p["label"], "path": p["checkpoint"], "sha256": file_hash(p["checkpoint"]),
                         "training_run": p.get("training_run")} for p in spec["policies"]],
    })
    scripts = output / "evaluator-source"
    scripts.mkdir()
    for f in (HERE / "eval_borinot.py", HERE / "borinot_eval_worker.py", *sorted((HERE / "borinot_eval").glob("*.py")),
              HERE / "borinot_eval/report.html"):
        shutil.copyfile(f, scripts / f.name)
    runs, per_episode = [], []
    total = len(spec["conditions"]) * len(spec["policies"])
    for c, condition in enumerate(spec["conditions"]):
        for p, policy in enumerate(spec["policies"]):
            folder = output / f"condition-{c + 1:02d}" / f"policy-{p + 1:02d}"
            folder.mkdir(parents=True)
            job = {"spec": spec, "policy": policy, "condition": condition, "scenarios": plan}
            write_json(folder / "job.json", job)
            print(f"[{len(runs) + 1}/{total}] {policy['label']} · {condition['label']}", flush=True)
            data = run_worker(folder / "job.json", device)
            if [row.get("scenario_id") for row in data["episodes"]] != list(range(len(plan))):
                raise RuntimeError(f"Incomplete evaluation in {folder}.")
            record = {"policy": policy["label"], "checkpoint_name": Path(policy["checkpoint"]).name,
                      "training_run": policy.get("training_run"), "condition": condition["label"],
                      "control_mode": data["control_mode"], "policy_hz": data["policy_hz"],
                      "physics_hz": data["physics_hz"], "stage": spec["stage"], "seed": spec["seed"],
                      **data["summary"]}
            per_episode.extend({"policy": record["policy"], "condition": record["condition"], **r}
                               for r in data["episodes"])
            runs.append({"record": record, "data": data})
            write_csv(output / "summary.csv", [r["record"] for r in runs])
            write_csv(output / "episodes.csv", per_episode)
    comparisons = []
    if spec["kind"] == "goal":
        baseline = spec["conditions"][0]
        for condition in spec["conditions"][1:]:
            # Only attribute a difference to latency if the remaining perturbations match.
            if set(condition["randomizations"]) != set(baseline["randomizations"]):
                continue
            for p in spec["policies"]:
                a, b = (next(r["data"]["episodes"] for r in runs if r["record"]["policy"] == p["label"]
                             and r["record"]["condition"] == condition_label)
                        for condition_label in (baseline["label"], condition["label"]))
                comparisons.append({"policy": p["label"], "baseline": baseline["label"], "condition": condition["label"],
                                    **paired_goal_changes(a, b)})
    write_json(output / "summary.json", {"task": spec["task"], "conditions": [r["record"] for r in runs],
                                         "paired_goal_changes": comparisons})
    if comparisons:
        write_csv(output / "paired-goal-changes.csv", comparisons)
    return runs, comparisons


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path, help="JSON defining checkpoints, task, scenario and perturbations.")
    source.add_argument("--from-results", type=Path, help="Rebuild or upload an existing result, without running the simulator.")
    parser.add_argument("--output", type=Path, help="New output directory. Default: logs/borinot_eval/<timestamp>.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()
    recipe_path = args.recipe or args.from_results / "recipe.json"
    try:
        spec = validate(read_json(recipe_path), recipe_path.resolve().parent, require_weights=not args.from_results)
        plan_scenarios(spec)
        if not args.from_results:
            parse_env_overrides(spec["hydra_overrides"])
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.from_results:
        if args.output:
            parser.error("--output cannot be used with --from-results.")
        output = args.from_results.resolve()
        summary = read_json(output / "summary.json")
        comparisons = summary["paired_goal_changes"]
        runs = []
        for c, _ in enumerate(spec["conditions"], 1):
            for p, _ in enumerate(spec["policies"], 1):
                data = read_json(output / f"condition-{c:02d}" / f"policy-{p:02d}" / "result.json")
                runs.append({"record": summary["conditions"][len(runs)], "data": data})
    else:
        output = (args.output or ROOT / "logs/borinot_eval" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
        print(f"Output: {output}", flush=True)
        runs, comparisons = run(spec, output, args.device)

    from borinot_eval.report import render_report
    render_report(output, spec, runs, comparisons)
    print(f"Report: {output / 'report.html'}", flush=True)

    if args.wandb != "disabled":
        from borinot_eval.wandb_log import publish
        publish(output, spec, runs, args.wandb, args.wandb_entity)


if __name__ == "__main__":
    main()
