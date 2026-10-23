# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_sac_cfg import (
    BorinotGoalkeeperSACMSERunnerCfg, BorinotGoalkeeperSACRunnerCfg,
)


@configclass
class BorinotCmdVelSACRunnerCfg(BorinotGoalkeeperSACRunnerCfg):
    experiment_name = "borinot_cmd_vel_sac"
    run_name = "Borinot cmd vel SAC CE"


@configclass
class BorinotCmdVelSACMSERunnerCfg(BorinotGoalkeeperSACMSERunnerCfg):
    experiment_name = "borinot_cmd_vel_sac"
    run_name = "Borinot cmd vel SAC MSE"
