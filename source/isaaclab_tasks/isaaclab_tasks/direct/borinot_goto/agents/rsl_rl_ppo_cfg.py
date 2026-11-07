# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_cmd_vel.agents.rsl_rl_ppo_cfg import BorinotCmdVelPPORunnerCfg


@configclass
class BorinotGoToPPORunnerCfg(BorinotCmdVelPPORunnerCfg):
    experiment_name = "borinot_goto"
    run_name = "Borinot go-to PPO"
    algorithm = BorinotCmdVelPPORunnerCfg().algorithm.replace(gamma=1.0)


@configclass
class BorinotGoToMinTimePPORunnerCfg(BorinotGoToPPORunnerCfg):
    experiment_name = "borinot_goto_min_time"
    run_name = "Borinot minimum-time go-to PPO"
