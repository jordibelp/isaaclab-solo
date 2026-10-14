# SPDX-License-Identifier: BSD-3-Clause
"""Borinot goalkeeper: one airborne shot, paddle-only interception, and recovery."""

import gymnasium as gym

gym.register(
    id="Borinot-Goalkeeper-Direct-v0",
    entry_point=f"{__name__}.borinot_goalkeeper_env:BorinotGoalkeeperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.borinot_goalkeeper_env_cfg:BorinotGoalkeeperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BorinotGoalkeeperPPORunnerCfg",
    },
)
