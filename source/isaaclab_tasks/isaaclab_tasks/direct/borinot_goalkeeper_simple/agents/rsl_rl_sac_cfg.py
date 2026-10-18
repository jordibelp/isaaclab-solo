# SPDX-License-Identifier: BSD-3-Clause
"""The original goalkeeper SAC recipes, logged under their own experiment name."""
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_sac_cfg import (
    BorinotGoalkeeperSACMSERunnerCfg, BorinotGoalkeeperSACRunnerCfg,
)


@configclass
class BorinotGoalkeeperSimpleSACRunnerCfg(BorinotGoalkeeperSACRunnerCfg):
    experiment_name = "borinot_goalkeeper_simple_sac"
    run_name = "Borinot goalkeeper simple SAC CE"


@configclass
class BorinotGoalkeeperSimpleSACMSERunnerCfg(BorinotGoalkeeperSACMSERunnerCfg):
    experiment_name = "borinot_goalkeeper_simple_sac"
    run_name = "Borinot goalkeeper simple SAC MSE"
