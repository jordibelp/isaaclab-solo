# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


_ENV_KWARGS = {
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
}


gym.register(
    id="solo12_cat_laas",
    entry_point="cat_envs.tasks.utils.cat.cat_env:CaTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg",
        **_ENV_KWARGS,
    },
)

gym.register(
    id="solo12_cat_laas_play",
    entry_point="cat_envs.tasks.utils.cat.cat_env:CaTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg_PLAY",
        **_ENV_KWARGS,
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-Laas-v0",
    entry_point="cat_envs.tasks.utils.cat.cat_env:CaTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg",
        **_ENV_KWARGS,
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-Laas-Play-v0",
    entry_point="cat_envs.tasks.utils.cat.cat_env:CaTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg_PLAY",
        **_ENV_KWARGS,
    },
)
