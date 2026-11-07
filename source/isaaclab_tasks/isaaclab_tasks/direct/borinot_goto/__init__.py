# SPDX-License-Identifier: BSD-3-Clause
"""Borinot position commands with an explicit, finite task deadline."""
import gymnasium as gym

for task_ids, variant in (
    (("Borinot-goto", "Borinot-GoTo-Direct-v0"), ""),
    (("Borinot-goto-minTime", "Borinot-GoTo-MinTime-Direct-v0"), "MinTime"),
):
    for task_id in task_ids:
        gym.register(
            id=task_id,
            entry_point=f"{__name__}.borinot_goto_env:BorinotGoToEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.borinot_goto_env_cfg:BorinotGoTo{variant}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BorinotGoTo{variant}PPORunnerCfg",
                "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoTo{variant}SACRunnerCfg",
                "rsl_rl_sac_mse_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotGoTo{variant}SACMSERunnerCfg",
            },
        )
