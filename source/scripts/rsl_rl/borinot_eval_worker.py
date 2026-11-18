# SPDX-License-Identifier: BSD-3-Clause
"""Internal Isaac Sim worker: one checkpoint/condition per fresh process."""
import argparse
import inspect
import math
import shutil
import sys
import traceback
from pathlib import Path

from borinot_eval.metrics import OUTCOMES, SAVE, summarize
from borinot_eval.spec import apply_condition, read_json, write_json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
job = read_json(args.job)
out = args.job.parent
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.borinot.borinot_env import BorinotEnv
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env import BorinotCmdVelEnv
from isaaclab_tasks.direct.borinot_goalkeeper.actuator_delay import actuator_delay_bounds
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env import BorinotGoalkeeperEnv
from isaaclab_tasks.direct.borinot_goalkeeper_simple.borinot_goalkeeper_simple_env import BorinotGoalkeeperSimpleEnv
from isaaclab_tasks.direct.borinot_goto.borinot_goto_env import BorinotGoToEnv
from isaaclab_tasks.direct.borinot_goto.task_math import goal_success_parts
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class RecordTransition:
    """Read native outcomes and state at the last policy tick, before DirectRLEnv auto-reset."""

    def _get_rewards(self):
        reward = super()._get_rewards()
        if not getattr(self, "recording", False):
            return reward
        data, info = self.robot.data, self.extras
        fields = {
            "return": reward, "energy_j": self._power_sum_w.sum(-1) * self.physics_dt,
            "actuator_delay_ms": self._actuator_delay.seconds * 1000,
            "robot_failure": self.reset_terminated,
            "done": self.reset_terminated | self.reset_time_outs,
        }
        if self.kind == "goal":
            goal = info["goal"]
            fields.update(terminal_distance_m=goal["distance_m"], minimum_distance_m=self._minimum_goal_distance,
                          first_arrival_s=self._first_arrival_s, success=goal["success"],
                          deadline_reached=goal["deadline_reached"], robot_failure=goal["failed"])
            inside, slow, still = goal_success_parts(self.cfg, goal["distance_m"],
                                                     data.root_lin_vel_w, data.root_ang_vel_w)
            fields["settled"] = inside & slow & still & ~goal["failed"]
        elif self.kind == "keeper":
            finished = info["goalkeeper_shot_finished"]
            codes = info["goalkeeper_outcome"]
            if self.simple:
                self.shot_touched |= info["goalkeeper_events"]["hit_racket"]
                contact = self.shot_touched
            else:
                contact = info["goalkeeper_had_paddle_hit"]
            fields.update(resolved_shots=finished, saves=finished & (codes == SAVE), racket_contacts=finished & contact,
                          unresolved_shot=self._ball_active)
            fields.update({f"outcome_{name}": finished & (codes == i) for i, name in enumerate(OUTCOMES)})
            self.shot_touched &= ~(finished | fields["done"])
        elif self.kind == "velocity":
            yaw_error = data.root_ang_vel_w[:, 2] - self._commands[:, 3]
            fields.update(velocity_error_m_s=info["tracking_error_m_s"]["v_xyz"],
                          yaw_error_rad_s=yaw_error.abs(), tracking_xy=info["tracking"]["xy"])
        else:
            base = self._base_body_id[0]
            quat = data.body_quat_w[:, base]
            vxy = quat_apply_inverse(quat, data.body_lin_vel_w[:, base])[:, :2]
            wz = quat_apply_inverse(quat, data.body_ang_vel_w[:, base])[:, 2]
            fields.update(velocity_error_m_s=(vxy - self._cmd_step[:, :2]).norm(dim=-1),
                          yaw_error_rad_s=(wz - self._cmd_step[:, 2]).abs())
        self.packet = {key: value.clone() for key, value in fields.items()}
        if self.trace_count:
            self.trace = self.capture_frame()
        return reward

    def capture_frame(self):
        n, data = self.trace_count, self.robot.data
        origins = self.scene.env_origins[:n]
        points = (data.body_link_pos_w[:n, self.leg_ids] - origins[:, None]).flatten(1)
        pose = [data.root_pos_w[:n] - origins, data.root_quat_w[:n], points]
        if self.kind == "keeper":
            pose += [self.ball.data.root_pos_w[:n] - origins, self._ball_active[:n, None].float()]
        return torch.cat(pose, dim=1)


class MatchGoal:
    def _sample_training_goals(self, start_position_w, env_ids):
        return self.eval_targets[env_ids] + self.scene.env_origins[env_ids], self.eval_deadline_steps[env_ids]


class MatchVelocity:
    def _resample_commands(self, env_ids):
        self._commands[env_ids] = self.eval_command
        self._command_steps_left[env_ids] = self.max_episode_length


class MatchPushes:
    def _update_pushes(self):
        if self.eval_pushes:
            super()._update_pushes()


def make_env(spec, cfg, condition, rows):
    kind, task = spec["kind"], spec["task"]
    simple = task == "Borinot-Goalkeeper-Simple-Direct-v0"
    parents = {"goal": (MatchGoal, MatchPushes, BorinotGoToEnv),
               "velocity": (MatchVelocity, MatchPushes, BorinotCmdVelEnv),
               "keeper": (BorinotGoalkeeperSimpleEnv if simple else BorinotGoalkeeperEnv,),
               "legacy": (BorinotEnv,)}[kind]
    # All reward/termination logic is the native task. Only command sampling and instrumentation differ.
    env = type("EvaluationEnv", (RecordTransition, *parents), {})(cfg)
    env.kind, env.simple, env.trace_count = kind, simple, 0
    env.eval_pushes = "vertical_push" in condition["randomizations"]
    if not env.eval_pushes and hasattr(env, "_force_limits"):
        env._force_limits.zero_()
    if kind == "goal":
        install_goals(env, rows[:env.num_envs])
    elif kind == "velocity":
        env.eval_command = torch.tensor(spec["scenario"]["command"], device=env.device)
    elif kind == "keeper":
        env.shot_touched = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.leg_ids = [i for i, name in enumerate(env.robot.body_names) if name.endswith(("link1", "link2"))]
    return env


def install_goals(env, rows):
    padded = rows + [rows[-1]] * (env.num_envs - len(rows))
    env.eval_targets = torch.tensor([r["target_xyz_m"] for r in padded], device=env.device)
    env.eval_deadline_steps = torch.tensor([round(r["deadline_s"] / env.step_dt) for r in padded], device=env.device)
    if any(abs(r["deadline_s"] / env.step_dt - round(r["deadline_s"] / env.step_dt)) > 1e-5 for r in rows):
        raise ValueError("A scenario deadline falls off this policy's tick grid.")


def config_overrides(cfg, requested):
    """Match the configclass numeric types (JSON 0 and 0.0 denote the same float setting)."""
    def coerce(current, value):
        if isinstance(value, dict):
            return {k: coerce(current.get(k) if isinstance(current, dict) else getattr(current, k, None), v)
                    for k, v in value.items()}
        return float(value) if type(current) is float and type(value) is int else value

    cfg.from_dict(coerce(cfg, requested))


def make_cfg(spec, policy, condition):
    task = spec["task"]
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, policy["agent"])
    config_overrides(cfg, spec["env"])
    config_overrides(cfg, policy.get("env", {}))
    config_overrides(agent, policy.get("runner", {}))
    if hasattr(cfg, "configure_control"):
        if "control_mode" in policy:
            cfg.control_mode = policy["control_mode"]
        cfg.policy_hz = policy["policy_hz"]
        cfg.curriculum = cfg.skip_curriculum = False
    if spec["kind"] == "goal":
        cfg.goal_curriculum_start_level, cfg.goal_debug_vis = spec["stage"], False
    elif spec["kind"] == "velocity":
        cfg.curriculum_start_level = spec["stage"]
    elif spec["kind"] == "keeper":
        cfg.shot_level = spec["stage"]
    else:
        cfg.is_fixed_cmd, cfg.train_fixed_cmd, cfg.eval_cmds = True, tuple(spec["scenario"]["command"]), ()
    cfg.scene.num_envs = spec["num_envs"]
    cfg.seed = agent.seed = spec["seed"]
    if args.device:
        cfg.sim.device = args.device
    agent.device = cfg.sim.device
    configure_agent = getattr(agent, "configure_environment", None)
    if configure_agent is not None:
        configure_agent(cfg)
    if hasattr(agent.algorithm, "torch_compile"):
        agent.algorithm.torch_compile = False
    if hasattr(agent, "local_redundancy"):
        agent.local_redundancy.enabled = False
    apply_condition(cfg, condition)
    actuator_delay_bounds(cfg)
    return cfg, agent


def policy_from_checkpoint(wrapper, agent, policy):
    checkpoint = policy["checkpoint"]
    try:
        if agent.class_name == "OffPolicyRunner":
            agent.algorithm.replay_buffer_size = wrapper.num_envs
            runner = OffPolicyRunner(wrapper, agent.to_dict(), log_dir=None, device=agent.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": False, "optimizer": False,
                                              "iteration": False}, map_location=agent.device)
            model = runner.alg.actor
        elif agent.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=None, device=agent.device)
            runner.load(checkpoint, load_optimizer=False, map_location=agent.device)
            model = runner.alg.policy
        else:
            raise ValueError(f"Unsupported runner {agent.class_name!r}.")
    except (RuntimeError, KeyError, ValueError) as exc:
        raise ValueError(f"Checkpoint {Path(checkpoint).name} is incompatible with this task/agent config. "
                         f"Effective observation/action sizes: {wrapper.unwrapped.cfg.observation_space}/"
                         f"{wrapper.num_actions}. Check training mode, policy architecture, and observation flags.\n{exc}") from exc
    return runner.get_inference_policy(device=wrapper.device), model


def roll_batch(raw, wrapper, infer, model, scenarios, trace_count):
    n, device, dt = len(scenarios), raw.device, raw.step_dt
    if raw.kind == "goal":
        install_goals(raw, scenarios)
    raw.trace_count = trace_count
    raw.recording = True
    if raw.kind == "keeper":
        raw.shot_touched.zero_()
    obs, _ = wrapper.reset()
    if raw.kind == "goal" and not (
        torch.equal(raw._goal_deadline_steps, raw.eval_deadline_steps)
        and torch.equal(raw._goal_position_w, raw.eval_targets + raw.scene.env_origins)
    ):
        raise RuntimeError("The reset did not install the planned goal and deadline.")
    model.reset(torch.ones(raw.num_envs, device=device, dtype=torch.bool))
    active = torch.ones(n, device=device, dtype=torch.bool)
    first_settled = torch.full((n,), -1., device=device)
    frames = [raw.capture_frame()] if trace_count else []
    delivered = [None] * n
    sums, horizon = {}, raw.max_episode_length + 1
    if raw.kind == "goal":
        horizon = max(round(r["deadline_s"] / dt) for r in scenarios) + 1
    for tick in range(1, horizon + 1):
        action = infer(obs)
        if not torch.isfinite(action).all():
            raise RuntimeError("Nonfinite policy actions.")
        obs, reward, done, _ = wrapper.step(action)
        if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
            raise RuntimeError("Nonfinite simulation reward or policy observation.")
        model.reset(done.bool())
        packet = {k: v[:n] for k, v in raw.packet.items()}
        sample = {"return": packet["return"], "energy_j": packet["energy_j"]}
        if raw.kind == "goal":
            new_settled = active & packet["settled"] & (first_settled < 0)
            first_settled[new_settled] = tick * dt
        elif raw.kind == "keeper":
            sample.update({k: v for k, v in packet.items() if k.startswith("outcome_") or
                           k in ("resolved_shots", "saves", "racket_contacts")})
        else:
            v = packet["velocity_error_m_s"]
            sample.update(velocity_error_m_s=v, squared_velocity_error=v.square(), yaw_error_rad_s=packet["yaw_error_rad_s"])
            if raw.kind == "velocity":
                sample["tracking_xy"] = packet["tracking_xy"] * dt
        for key, value in sample.items():
            if value.dtype == torch.bool:
                value = value.to(torch.int32)  # Count every resolved shot, not merely whether one occurred.
            if key not in sums:
                sums[key] = value * active
            else:
                sums[key] += value * active
        if trace_count:
            frames.append(raw.trace)
        ended = packet["done"] & active
        indices = ended.nonzero().flatten().tolist()
        if indices:
            keep = {"actuator_delay_ms", "robot_failure"}
            if raw.kind == "goal":
                keep |= {"terminal_distance_m", "minimum_distance_m", "first_arrival_s", "success", "deadline_reached"}
                packet["first_settled_s"] = first_settled
                keep.add("first_settled_s")
            elif raw.kind == "keeper":
                keep.add("unresolved_shot")
            final = {k: packet[k][indices].tolist() for k in sorted(keep)}
            totals = {k: value[indices].tolist() for k, value in sums.items()}
            for j, i in enumerate(indices):
                row = scenarios[i] | {k: v[j] for k, v in final.items()} | {k: v[j] for k, v in totals.items()}
                row.update(ticks=tick, duration_s=tick * dt)
                if raw.kind == "goal":
                    for key in ("first_arrival_s", "first_settled_s"):
                        if row[key] < 0:
                            row[key] = None
                if raw.kind in ("velocity", "legacy"):
                    row["velocity_error_m_s"] /= tick
                    row["yaw_error_rad_s"] /= tick
                    row["velocity_rmse_m_s"] = math.sqrt(row.pop("squared_velocity_error") / tick)
                    if raw.kind == "velocity":
                        row["full_horizon_tracking_score"] = row.pop("tracking_xy") / raw.cfg.episode_length_s
                delivered[i] = row
        active &= ~ended
        if not active.any():
            raw.recording = False
            traces = []
            if trace_count:
                positions = torch.stack(frames).transpose(0, 1).cpu()
                traces = [{"scenario_id": delivered[i]["scenario_id"], "dt": dt,
                           "frames": positions[i, :delivered[i]["ticks"] + 1].tolist()} for i in range(trace_count)]
            return delivered, traces
    pending = active.nonzero().flatten().tolist()
    raise RuntimeError(f"Trials {pending} exceeded {horizon} ticks of {dt:g}s. "
                       f"Last done mask: {packet['done'].tolist()}; "
                       f"counters: {raw.episode_length_buf[:n].tolist()}; "
                       f"deadlines: {getattr(raw, '_goal_deadline_steps', None)}")


def main():
    spec, policy, condition, plan = (job[k] for k in ("spec", "policy", "condition", "scenarios"))
    cfg, agent = make_cfg(spec, policy, condition)
    raw = make_env(spec, cfg, condition, plan)
    try:
        wrapper = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        infer, model = policy_from_checkpoint(wrapper, agent, policy)
        rows, traces = [], []
        with torch.inference_mode():
            for start in range(0, len(plan), raw.num_envs):
                raw.seed(spec["seed"] + start)
                batch, replay = roll_batch(raw, wrapper, infer, model, plan[start:start + raw.num_envs],
                                           min(raw.num_envs, max(0, spec["replay_count"] - start)))
                rows += batch
                traces += replay
                print(f"Completed {len(rows)}/{len(plan)} trials", flush=True)
        write_json(out / "effective-env.json", cfg.to_dict())
        write_json(out / "effective-agent.json", agent.to_dict())
        sources = set(raw.get_code_snapshot_files() if hasattr(raw, "get_code_snapshot_files") else ())
        sources.update(inspect.getsourcefile(cls) for cls in type(cfg).__mro__ if cls is not object)
        sources.update(inspect.getsourcefile(cls) for cls in type(raw).__mro__ if cls is not object)
        repo = Path(__file__).resolve().parents[3]
        for name in sorted(s for s in sources if s):
            path = Path(name).resolve()
            if path.is_relative_to(repo):
                target = out / "source" / path.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        result = {"summary": summarize(raw.kind, rows), "episodes": rows, "traces": traces,
                  "physics_hz": 1 / raw.physics_dt, "policy_hz": 1 / raw.step_dt,
                  "rotors": list(cfg.rotor_pos_b), "leg_links": len(raw.leg_ids),
                  "control_mode": getattr(cfg, "control_mode", "legacy")}
    finally:
        raw.close()
    write_json(out / "result.json", result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()  # Kit shutdown can mask an uncaught Python exception.
        raise
    finally:
        app.close()
