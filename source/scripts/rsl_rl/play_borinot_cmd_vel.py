# SPDX-License-Identifier: BSD-3-Clause
"""Play Borinot PPO/SAC weights with live heading vx/vy, world vz and world-Z yaw-rate commands."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel", choices=("Borinot-cmd-vel", "Borinot-Cmd-Vel-Direct-v0"))
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--cmd_init", "--cmd", type=float, nargs=4, default=(0.0, 0.0, 0.0, 0.0),
                    metavar=("VX", "VY", "VZ", "WZ"), help="Initial command: m/s, m/s, m/s, rad/s.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Play-only timeout; omitted keeps training default.")
parser.add_argument("--camera", "--camera_mode", choices=("follow", "chase", "side", "top", "free"), default="follow")
parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--random-commands", action="store_true", help="Use training command sampling instead of live controls.")
parser.add_argument("--curriculum-level", type=int, default=None,
                    help="Fixed force stage. Default: saved checkpoint stage, then environment default.")
for axis in ("vx", "vy", "vz", "wz"):
    parser.add_argument(f"--{axis}_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or not math.isfinite(args.duration_s) or args.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive and finite.")
if not all(map(math.isfinite, args.cmd_init)):
    parser.error("cmd_init must contain four finite values: vx vy vz wz.")
if args.episode_length_s is not None and (not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0):
    parser.error("episode_length_s must be positive and finite.")
for axis in ("vx", "vy", "vz", "wz"):
    bounds = getattr(args, f"{axis}_range")
    if bounds is not None and (not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]):
        parser.error(f"{axis}_range must contain two finite values with MIN < MAX.")
defaults = {"env.curriculum": "False", "agent.algorithm.torch_compile": "False",
            "agent.local_redundancy.enabled": "False"}
override_keys = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
sys.argv = [sys.argv[0]] + [f"{key}={value}" for key, value in defaults.items() if key not in override_keys] + hydra_args
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_cmd_vel.playback import BorinotCmdVelPlayEnv, checkpoint_force_level
from isaaclab_tasks.direct.borinot_cmd_vel.playback_ui import FlightControls, RANGE_FIELDS, camera_pose


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    checkpoint = retrieve_file_path(str(Path(args.checkpoint).expanduser())
                                    if args.checkpoint.startswith("~") else args.checkpoint)
    # Inspect saved stage before constructing the env; optimizer state is never restored.
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = metadata.get("infos") or {}
    env_cfg.curriculum_start_level = (args.curriculum_level if args.curriculum_level is not None else
                                     checkpoint_force_level(infos, env_cfg.curriculum_start_level, override_keys))
    del metadata
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg.device = env_cfg.sim.device
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(env_cfg)
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.configure_control()
    ranges = []
    for index, (axis, field) in enumerate(zip(("vx", "vy", "vz", "wz"), RANGE_FIELDS)):
        bounds = getattr(args, f"{axis}_range")
        if bounds is None:
            low, high = getattr(env_cfg, field)
            bounds = (min(low, args.cmd_init[index], 0.0), max(high, args.cmd_init[index], 0.0))
            if bounds[0] == bounds[1]:
                bounds = (bounds[0] - 1.0, bounds[1] + 1.0)
        if not bounds[0] <= args.cmd_init[index] <= bounds[1] or not bounds[0] <= 0 <= bounds[1]:
            raise ValueError(f"{axis}_range must include cmd_init and zero (for Zero commands).")
        ranges.append(bounds)
    env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
    pose = camera_pose(args.camera, env_cfg.robot_home)
    if pose is not None:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = pose
    env = RslRlVecEnvWrapper(BorinotCmdVelPlayEnv(
        cfg=env_cfg, command=None if args.random_commands else args.cmd_init), clip_actions=agent_cfg.clip_actions)
    controls = None
    try:
        if agent_cfg.class_name == "OffPolicyRunner":
            agent_cfg.algorithm.replay_buffer_size = env.num_envs
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
            policy_model = runner.alg.actor
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_optimizer=False)
            policy_model = runner.alg.policy
        policy = runner.get_inference_policy(device=env.device)
        controls = FlightControls(env.unwrapped, args.cmd_init, ranges, args.camera,
                                  show_ui=args.ui and not args.random_commands and env.unwrapped.sim.has_gui())
        observation = env.get_observations()
        total_steps = math.ceil(args.duration_s / env.unwrapped.step_dt)
        completed = episodes = 0
        tracking = torch.zeros(3, device=env.device)
        applied_command = tuple(args.cmd_init)
        print(f"[INFO] Borinot: {env_cfg.control_mode}, {env_cfg.physics_hz}/{env_cfg.policy_hz} Hz, "
              f"command={'random' if args.random_commands else applied_command} (vx vy vz wz), "
              f"force stage={env.unwrapped.force_level}, episode={env_cfg.episode_length_s:g} s, "
              f"checkpoint={Path(checkpoint).name}", flush=True)
        with torch.inference_mode():
            while completed < total_steps and simulation_app.is_running():
                start = time.perf_counter()
                if controls.reset_requested:
                    observation, _ = env.reset()
                    policy_model.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                    controls.reset_requested = False
                    controls.manual_resets += 1
                if not args.random_commands and tuple(controls.command) != applied_command:
                    applied_command = tuple(controls.command)
                    env.unwrapped.set_play_command(applied_command)
                    observation = env.get_observations()
                controls.update(completed * env.unwrapped.step_dt)
                if controls.paused:
                    env.unwrapped.sim.render()
                    time.sleep(0.02)
                    continue
                action = policy(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite policy action.")
                observation, reward, dones, info = env.step(action)
                if not (torch.isfinite(reward).all()
                        and all(torch.isfinite(value).all() for value in observation.values())):
                    raise RuntimeError("Non-finite policy rollout.")
                policy_model.reset(dones)
                tracking += torch.stack([info["tracking"][key].mean() for key in ("xyz", "yaw", "xy")])
                episodes += int(dones.sum())
                completed += 1
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        print("BORINOT_CMD_VEL_PLAY " + json.dumps({
            "task": args.task, "control_mode": env_cfg.control_mode, "steps": completed,
            "checkpoint_name": Path(checkpoint).name, "checkpoint_path": checkpoint,
            "physics_hz": env_cfg.physics_hz, "policy_hz": env_cfg.policy_hz,
            "episode_length_s": env_cfg.episode_length_s,
            "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
            "simulated_s": completed * env.unwrapped.step_dt, "completed_episodes": episodes,
            "force_level": env.unwrapped.force_level,
            "force_limit_n": float(env.unwrapped._force_limits[env.unwrapped.force_level]),
            "command_vx_vy_vz_wz": None if args.random_commands else list(applied_command),
            "final_env_0_command": env.unwrapped._commands[0].cpu().tolist(),
            "manual_resets": controls.manual_resets, "camera": controls.camera,
            "mean_tracking_xyz_yaw_xy": (tracking / max(1, completed)).cpu().tolist(),
            "finite": True,
        }), flush=True)
    finally:
        if controls is not None:
            controls.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
