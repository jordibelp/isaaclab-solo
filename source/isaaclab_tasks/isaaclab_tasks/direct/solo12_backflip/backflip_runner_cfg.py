# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..solo12.agents.rsl_rl_ppo_cfg import Solo12PPORunnerCfg, Solo12PPORunnerWithSymmetryCfg


@configclass
class Solo12BackflipPPORunnerCfg(Solo12PPORunnerCfg):
    wandb_project = "solo-backflip"
    wandb_entity = "jordibelp"


@configclass
class Solo12BackflipPPORunnerWithSymmetryCfg(Solo12PPORunnerWithSymmetryCfg):
    wandb_project = "solo-backflip"
    wandb_entity = "jordibelp"
