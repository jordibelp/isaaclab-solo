# SPDX-License-Identifier: BSD-3-Clause
"""Borinot goalkeeper with a simplified reward: task + integrity + desirability."""

import gymnasium as gym

gym.register(
    id="Borinot-Goalkeeper-Simple-Direct-v0",
    entry_point=f"{__name__}.borinot_goalkeeper_simple_env:BorinotGoalkeeperSimpleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.borinot_goalkeeper_simple_env_cfg:BorinotGoalkeeperSimpleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BorinotGoalkeeperSimplePPORunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoalkeeperSimpleSACRunnerCfg",
        "rsl_rl_sac_mse_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoalkeeperSimpleSACMSERunnerCfg",
    },
)
