# SPDX-License-Identifier: BSD-3-Clause
"""Borinot position commands with an explicit, finite task deadline."""
import gymnasium as gym

for task_id in ("Borinot-goto", "Borinot-GoTo-Direct-v0"):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.borinot_goto_env:BorinotGoToEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.borinot_goto_env_cfg:BorinotGoToEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BorinotGoToPPORunnerCfg",
            "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoToSACRunnerCfg",
            "rsl_rl_sac_mse_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoToSACMSERunnerCfg",
        },
    )
