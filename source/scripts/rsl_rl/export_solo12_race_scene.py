# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a USD snapshot of the Solo12 race scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export a USD snapshot of the Solo12 race scene.")
parser.add_argument("--task", type=str, default="Isaac-Solo12-Race-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=str, default="/home/jordibelp/IsaacLab/scene_exports/solo12_race_scene.usda")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# keep Hydra-like args out of the application launcher
import sys

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

# register the task IDs after the simulation app is up
import isaaclab.sim as sim_utils
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks.direct.solo12_race  # noqa: F401
import gymnasium as gym


def main():
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=False)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    sim_utils.update_stage()
    ok = sim_utils.save_stage(str(output_path), save_and_reload_in_place=False)
    print(f"Saved: {ok} -> {output_path}")
    env.close()


if __name__ == "__main__":
    main()
