# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="solo12-backflip",
    entry_point=f"{__name__}.solo12_backflip_env:Solo12BackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo12_backflip_env_cfg:Solo12BackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.backflip_runner_cfg:Solo12BackflipPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{__name__}.backflip_runner_cfg:Solo12BackflipPPORunnerWithSymmetryCfg"
        ),
        "rsl_rl_sac_cfg_entry_point": f"{__name__}.backflip_sac_runner_cfg:Solo12BackflipSACRunnerCfg",
    },
)
