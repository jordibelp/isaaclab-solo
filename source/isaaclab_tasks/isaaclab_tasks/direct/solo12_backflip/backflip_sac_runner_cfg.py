# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..solo12.agents.rsl_rl_sac_cfg import Solo12SACRunnerCfg


@configclass
class Solo12BackflipSACRunnerCfg(Solo12SACRunnerCfg):
    wandb_project = "solo-backflip"
    wandb_entity = "jordibelp"
