# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Velocity_Flat_Solo12_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_configs:Solo12FlatEnvCfg",
        # Reuse the existing Go2 SKRL flat PPO config, as requested.
        "skrl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.solo12.agents:"
            "skrl_ppo_cfg.yaml"
        ),
    },
)

gym.register(
    id="Velocity_Flat_Solo12_Play_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_configs:Solo12FlatEnvCfg_PLAY",
        "skrl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.solo12.agents:"
            "skrl_ppo_cfg.yaml"
        ),
    },
)


gym.register(
    id="Velocity_Rough_Solo12_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_configs:Solo12RoughEnvCfg",
        "skrl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.solo12.agents:"
            "skrl_ppo_cfg.yaml"
        ),
    },
)

gym.register(
    id="Velocity_Rough_Solo12_Play_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_configs:Solo12RoughEnvCfg_PLAY",
        "skrl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.solo12.agents:"
            "skrl_ppo_cfg.yaml"
        ),
    },
)
