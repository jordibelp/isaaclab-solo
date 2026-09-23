# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from ..solo12 import agents


gym.register(
    id="solo12-backflip",
    entry_point=f"{__name__}.solo12_backflip_env:Solo12BackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo12_backflip_env_cfg:Solo12BackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Solo12PPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Solo12PPORunnerWithSymmetryCfg"
        ),
        "rsl_rl_sac_cfg_entry_point": f"{agents.__name__}.rsl_rl_sac_cfg:Solo12SACRunnerCfg",
    },
)
