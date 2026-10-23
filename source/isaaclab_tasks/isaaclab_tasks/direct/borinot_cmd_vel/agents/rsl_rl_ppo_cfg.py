# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_ppo_cfg import BorinotGoalkeeperPPORunnerCfg


@configclass
class BorinotCmdVelPPORunnerCfg(BorinotGoalkeeperPPORunnerCfg):
    experiment_name = "borinot_cmd_vel"
    run_name = "Borinot cmd vel PPO"
