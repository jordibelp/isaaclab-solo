# SPDX-License-Identifier: BSD-3-Clause
"""Borinot command-conditioned linear velocity and yaw-rate tracking."""
import gymnasium as gym

for task_id in ("Borinot-cmd-vel", "Borinot-Cmd-Vel-Direct-v0"):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.borinot_cmd_vel_env:BorinotCmdVelEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.borinot_cmd_vel_env_cfg:BorinotCmdVelEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BorinotCmdVelPPORunnerCfg",
            "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotCmdVelSACRunnerCfg",
            "rsl_rl_sac_mse_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:BorinotCmdVelSACMSERunnerCfg",
        },
    )
