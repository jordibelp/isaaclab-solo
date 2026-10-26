# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate Borinot goalkeeper PPO or SAC weights: save fraction over N complete robot episodes.

A save is the task's own shot outcome (code 7), as in Curriculum/stage_pure_save_fraction, counted over
every shot of N complete episodes at one fixed shot level. Each environment scores only its first
episodes, so short failed episodes are not over-represented. --actuator_delays sweeps fixed
policy-command delays in one launch; every delay restarts from the same seed. Results:
<out_dir>/<label>_<time>.json and one <out_dir>/summary.csv row per delay.

./isaaclab.sh -p source/scripts/rsl_rl/eval_borinot_goalkeeper.py --checkpoint <model.pt> --num_episodes 512 --seed 0 --actuator_delays 0 0.01 0.02 0.04 env.control_mode=indi
"""

import argparse
import csv
import hashlib
import inspect
import json
import re
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

TASKS = ("Borinot-Goalkeeper-Direct-v0", "Borinot-Goalkeeper-Simple-Direct-v0")
OUTCOMES = ("ongoing", "base_hit", "arm_hit", "goal", "ball_floor", "leave_goal_area", "deadline", "save",
            "robot_crash", "miss")  # the simplified task uses only "goal" and "save" (no goal)
SAVE = OUTCOMES.index("save")
CSV_FIELDS = ("time", "label", "task", "algorithm", "control_mode", "shot_level", "seed", "episodes",
              "actuator_delay_s", "shots", "save_fraction", "save_ci95_low", "save_ci95_high",
              "save_fraction_counting_reset_drops", "racket_contact_fraction", "robot_failure_fraction",
              "robot_failures_per_minute", "mean_episode_duration_s", "mean_episode_return",
              "checkpoint_name", "checkpoint_path", "result_json")
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default=TASKS[0], choices=TASKS)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="rsl_rl_sac_cfg_entry_point for SAC weights.")
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint; optimizer state is not restored.")
parser.add_argument("--num_episodes", type=int, default=512, help="Complete robot episodes per evaluated delay.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments, at most num_episodes.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--actuator_delays", type=float, nargs="+", default=None, metavar="SECONDS",
                    help="Fixed policy-command delays to sweep. Omitted: env.actuator_delay or env.actuator_delay_range.")
parser.add_argument("--label", default=None, help="Policy name in summary.csv. Default: the checkpoint file stem.")
parser.add_argument("--out_dir", type=Path, default=None, help="Default: <checkpoint folder>/eval.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_episodes < 1 or args_cli.num_envs < 1:
    parser.error("num_episodes and num_envs must be positive.")
command_line = list(sys.argv)
# Evaluation stays at one shot level and uses eager forwards unless explicitly overridden.
defaults = {"env.curriculum": "False", "agent.algorithm.torch_compile": "False",
            "agent.local_redundancy.enabled": "False"}
override_keys = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
if args_cli.actuator_delays is not None and override_keys & {"env.actuator_delay", "env.actuator_delay_range"}:
    parser.error("Use either --actuator_delays or env.actuator_delay / env.actuator_delay_range, not both.")
sys.argv = [sys.argv[0]] + [f"{key}={value}" for key, value in defaults.items()
                          if key not in override_keys] + hydra_args
simulation_app = AppLauncher(args_cli).app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401 -- registers the tasks
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_goalkeeper.actuator_delay import ActuatorDelay, delay_steps
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env import BorinotGoalkeeperEnv
from isaaclab_tasks.direct.borinot_goalkeeper_simple.borinot_goalkeeper_simple_env import BorinotGoalkeeperSimpleEnv


class ResetRecorder:
    """Keep what each robot reset discards: an unresolved shot and the episode's delay."""

    def _reset_idx(self, env_ids):
        # _get_rewards() has already parked every resolved ball; a still-active ball was abandoned.
        self.unresolved_at_reset = self._ball_active.clone()
        self.delay_before_reset = self._actuator_delay.seconds.clone()
        super()._reset_idx(env_ids)


class GoalkeeperEvalEnv(ResetRecorder, BorinotGoalkeeperEnv):
    pass


class SimpleEvalEnv(ResetRecorder, BorinotGoalkeeperSimpleEnv):
    pass


def evaluate(env, policy, policy_model, quota, simple):
    """Roll out until every environment completed its quota; return one row per counted episode."""
    raw, n = env.unwrapped, env.num_envs
    completed = torch.zeros(n, dtype=torch.long, device=env.device)
    episode = {name: torch.zeros(n, device=env.device) for name in ("steps", "return", "shots", "saves", "contacts")}
    shot_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
    outcomes = torch.zeros(len(OUTCOMES), dtype=torch.long, device=env.device)
    rows = {name: [] for name in (*episode, "robot_failure", "unresolved_shot", "actuator_delay_s",
                                 "env_index", "episode_index")}
    max_steps = (int(quota.max()) + 1) * raw.max_episode_length
    obs, steps = env.get_observations(), 0
    while (completed < quota).any():
        steps += 1
        if steps > max_steps:
            raise RuntimeError("Evaluation outlasted the episode time limit; check env.episode_length_s.")
        # A shot belongs to the episode that was running when this step started.
        counting = completed < quota
        actions = policy(obs)
        if not torch.isfinite(actions).all():
            raise RuntimeError("The checkpoint produced non-finite actions.")
        obs, rewards, dones, info = env.step(actions)
        if not torch.isfinite(rewards).all() or not torch.isfinite(obs["policy"]).all():
            raise RuntimeError("The environment produced non-finite rewards or observations.")
        policy_model.reset(dones)
        ended = dones.bool()
        finished = info["goalkeeper_shot_finished"] & counting
        codes = info["goalkeeper_outcome"]
        if simple:  # the shot's first robot contact was the racket
            shot_contact |= info["goalkeeper_events"]["hit_racket"]
            contact = shot_contact.clone()
            shot_contact &= ~(info["goalkeeper_shot_finished"] | ended)
        else:  # the paddle touched the ball during the shot
            contact = info["goalkeeper_had_paddle_hit"]
        episode["steps"] += counting
        episode["return"] += rewards * counting
        episode["shots"] += finished
        episode["saves"] += finished & (codes == SAVE)
        episode["contacts"] += finished & contact
        outcomes += torch.bincount(codes[finished], minlength=len(OUTCOMES))
        record = (ended & counting).nonzero().flatten()
        if len(record):
            for name, values in episode.items():
                rows[name] += values[record].tolist()
            rows["robot_failure"] += raw.reset_terminated[record].tolist()
            rows["unresolved_shot"] += raw.unresolved_at_reset[record].tolist()
            rows["actuator_delay_s"] += raw.delay_before_reset[record].tolist()
            rows["env_index"] += record.tolist()
            rows["episode_index"] += completed[record].tolist()
        for values in episode.values():
            values[ended] = 0
        completed += ended
    return rows, outcomes.tolist()


def interval(numerator, denominator, seed, samples=2000):
    """95% percentile bootstrap of sum(numerator) / sum(denominator), resampling whole episodes."""
    if not denominator.sum():
        return [None, None]
    rng = np.random.default_rng(seed)
    ratios = []
    for start in range(0, samples, 250):
        index = rng.integers(0, len(numerator), (min(250, samples - start), len(numerator)))
        totals = denominator[index].sum(1)
        valid = totals > 0
        ratios.append(numerator[index].sum(1)[valid] / totals[valid])
    return [float(v) for v in np.percentile(np.concatenate(ratios), [2.5, 97.5])]


def summarize(rows, outcomes, step_dt, seed):
    data = {name: np.asarray(values, dtype=float) for name, values in rows.items()}
    shots, saves = data["shots"], data["saves"]
    failed, unresolved = data["robot_failure"] > 0, data["unresolved_shot"] > 0
    dropped = int((failed & unresolved).sum())  # shot abandoned by a robot-failure reset
    flight_min = data["steps"].sum() * step_dt / 60
    return {
        "episodes": len(shots), "shots": int(shots.sum()), "saves": int(saves.sum()),
        "save_fraction": float(saves.sum() / shots.sum()) if shots.sum() else None,
        "save_fraction_ci95": interval(saves, shots, seed),
        "save_fraction_counting_reset_drops": (float(saves.sum() / (shots.sum() + dropped))
                                               if shots.sum() + dropped else None),
        "racket_contact_fraction": float(data["contacts"].sum() / shots.sum()) if shots.sum() else None,
        "robot_failure_episodes": int(failed.sum()), "robot_failure_fraction": float(failed.mean()),
        "robot_failure_fraction_ci95": interval(failed.astype(float), np.ones_like(shots), seed),
        "robot_failures_per_minute": float(failed.sum() / flight_min),
        "mean_episode_duration_s": float(data["steps"].mean() * step_dt),
        "mean_episode_return": float(data["return"].mean()),
        "mean_shots_per_episode": float(shots.mean()),
        "unresolved_shots_at_robot_reset": dropped,
        "unresolved_shots_at_time_limit": int((~failed & unresolved).sum()),
        "outcomes": dict(zip(OUTCOMES, outcomes)),
        "realized_actuator_delay_s": {"min": float(data["actuator_delay_s"].min()),
                                      "mean": float(data["actuator_delay_s"].mean()),
                                      "max": float(data["actuator_delay_s"].max())},
        "per_episode": rows,
    }


def append_summary(path, rows):
    exists = path.exists()
    if exists and path.read_text().splitlines()[:1] != [",".join(CSV_FIELDS)]:
        raise RuntimeError(f"{path} has different columns; choose another --out_dir.")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if env_cfg.curriculum:
        parser.error("Evaluation requires a fixed shot level: use env.curriculum=False and env.shot_level=N.")
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    infos = torch.load(checkpoint, map_location="cpu", weights_only=False).get("infos") or {}
    saved_level = infos.get("best_model_curriculum_idx")
    if saved_level is None:
        saved_level = infos.get("curriculum_global_idx")
    if "env.shot_level" in override_keys:
        level_source = "command line"
    elif saved_level is not None:
        env_cfg.shot_level, level_source = int(saved_level), "checkpoint"
    else:
        level_source = "config default"
    env_cfg.scene.num_envs = min(args_cli.num_envs, args_cli.num_episodes)
    env_cfg.seed = agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.device = env_cfg.sim.device
    configure_environment = getattr(agent_cfg, "configure_environment", None)
    if configure_environment is not None:
        configure_environment(env_cfg)
    env_cfg.configure_control()  # derived action/observation sizes and rates after Hydra
    env_cfg.robot.init_state.pos = tuple(env_cfg.robot_home)
    delays = args_cli.actuator_delays or [None]
    for delay in delays:
        if delay is not None:
            delay_steps(delay, env_cfg.physics_hz, "--actuator_delays")  # fail before the simulator scene
    simple = args_cli.task == TASKS[1]
    env = (SimpleEvalEnv if simple else GoalkeeperEvalEnv)(cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OffPolicyRunner":
            # Evaluation never collects replay; avoid allocating the training-sized buffer.
            agent_cfg.algorithm.replay_buffer_size = env.num_envs
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
            policy_model = runner.alg.actor
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_optimizer=False)
            policy_model = runner.alg.policy
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        raw, n = env.unwrapped, env.num_envs
        quota = torch.full((n,), args_cli.num_episodes // n, dtype=torch.long, device=env.device)
        quota[: args_cli.num_episodes % n] += 1
        algorithm = "SAC" if agent_cfg.class_name == "OffPolicyRunner" else "PPO"
        print(f"[INFO] {args_cli.task}: {algorithm}, {env_cfg.control_mode}, {env_cfg.physics_hz}/{env_cfg.policy_hz} Hz, "
              f"shot_level={env_cfg.shot_level} ({level_source}), {args_cli.num_episodes} episodes on {n} envs, "
              f"seed={args_cli.seed}, actuator delays [s]={delays if delays != [None] else 'from config'}", flush=True)
        results = []
        with torch.inference_mode():
            for delay in delays:
                if delay is not None:
                    raw.cfg.actuator_delay = delay
                    raw._actuator_delay = ActuatorDelay(raw.cfg, n, raw.cfg.action_space, raw.device)
                # The same command repeats every number. GPU contact rounding still depends on what ran
                # earlier in this process, so a delay's place in the sweep acts like another seed.
                raw.seed(args_cli.seed)
                env.reset()
                policy_model.reset(torch.ones(n, dtype=torch.bool, device=env.device))
                start = time.perf_counter()
                rows, outcomes = evaluate(env, policy, policy_model, quota, simple)
                summary = summarize(rows, outcomes, raw.step_dt, args_cli.seed)
                summary["actuator_delay_s"] = (raw.cfg.actuator_delay if raw.cfg.actuator_delay_range is None
                                               else list(raw.cfg.actuator_delay_range))
                summary["wall_time_s"] = time.perf_counter() - start
                results.append(summary)
                low, high = summary["save_fraction_ci95"]
                score = (f"{summary['save_fraction']:.3f} [{low:.3f}, {high:.3f}]"
                         if summary["save_fraction"] is not None else "unavailable (no resolved shots)")
                print(f"[EVAL] delay {summary['actuator_delay_s']} s: save {score} "
                      f"over {summary['shots']} shots in {summary['episodes']} episodes; "
                      f"robot failures {summary['robot_failure_fraction']:.1%} of episodes, "
                      f"mean episode {summary['mean_episode_duration_s']:.1f} s ({summary['wall_time_s']:.0f} s wall)",
                      flush=True)
        label = args_cli.label or Path(checkpoint).stem
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        out_dir = args_cli.out_dir or Path(checkpoint).parent / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', label)}_{stamp}.json"
        report = {
            "task": args_cli.task, "algorithm": algorithm, "label": label,
            "checkpoint_name": Path(checkpoint).name, "checkpoint_path": str(checkpoint),
            "command": command_line, "hydra_overrides": hydra_args, "seed": args_cli.seed,
            "num_envs": n, "episodes_per_delay": args_cli.num_episodes, "control_mode": env_cfg.control_mode,
            "sac_action_mapping": env_cfg.sac_action_mapping, "physics_hz": env_cfg.physics_hz,
            "policy_hz": env_cfg.policy_hz, "episode_length_s": env_cfg.episode_length_s,
            "shot_level": env_cfg.shot_level, "shot_level_source": level_source,
            "checkpoint_saved_level": saved_level, "shot_level_end": raw.shot_level,
            "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
            "results": results,
        }
        # Keep effective settings and source bytes: a git revision alone cannot reproduce a dirty worktree.
        repo = Path(__file__).resolve().parents[3]
        sources = {__file__, *raw.get_code_snapshot_files()}
        for obj in (raw, env_cfg, agent_cfg):
            sources.update(inspect.getsourcefile(cls) for cls in type(obj).__mro__ if cls is not object)
        report["environment_config"] = json.loads(json.dumps(env_cfg.to_dict(), default=str))
        report["agent_config"] = json.loads(json.dumps(agent_cfg.to_dict(), default=str))
        report["checkpoint_sha256"] = file_hash(checkpoint)
        report["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        report["source_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).splitlines()
        report["source_snapshot"] = {}
        for source in sorted(path for path in sources if path):
            path = Path(source).resolve()
            report["source_snapshot"][str(path)] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": path.read_text()}
        report["metric_definition"] = "Task saves / resolved shots in exactly N complete robot episodes at a fixed stage."
        report["repeatability"] = ("Keep seed, num_envs, device, settings and delay order fixed. PhysX rounding can "
                                   "depend on earlier rollouts; separate one-delay launches avoid sweep-order effects. "
                                   "Equal seeds do not guarantee identical shots across different policies.")
        result_path.write_text(json.dumps(report, indent=1))
        append_summary(out_dir / "summary.csv", [{
            "time": stamp, "label": label, "task": args_cli.task, "algorithm": algorithm,
            "control_mode": env_cfg.control_mode, "shot_level": env_cfg.shot_level, "seed": args_cli.seed,
            "episodes": result["episodes"], "actuator_delay_s": result["actuator_delay_s"], "shots": result["shots"],
            "save_fraction": result["save_fraction"], "save_ci95_low": result["save_fraction_ci95"][0],
            "save_ci95_high": result["save_fraction_ci95"][1],
            "save_fraction_counting_reset_drops": result["save_fraction_counting_reset_drops"],
            "racket_contact_fraction": result["racket_contact_fraction"],
            "robot_failure_fraction": result["robot_failure_fraction"],
            "robot_failures_per_minute": result["robot_failures_per_minute"],
            "mean_episode_duration_s": result["mean_episode_duration_s"],
            "mean_episode_return": result["mean_episode_return"], "checkpoint_name": Path(checkpoint).name,
            "checkpoint_path": str(checkpoint), "result_json": str(result_path)} for result in results])
        compact = {**{k: v for k, v in report.items() if k not in (
            "environment_config", "agent_config", "source_snapshot")},
                   "results": [{k: v for k, v in r.items() if k != "per_episode"} for r in results],
                   "result_json": str(result_path)}
        print("GOALKEEPER_EVAL=" + json.dumps(compact), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
