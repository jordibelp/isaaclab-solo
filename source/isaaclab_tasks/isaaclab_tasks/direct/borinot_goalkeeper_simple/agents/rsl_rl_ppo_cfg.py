# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_ppo_cfg import BorinotGoalkeeperPPORunnerCfg


@configclass
class BorinotGoalkeeperSimplePPORunnerCfg(BorinotGoalkeeperPPORunnerCfg):
    """The original goalkeeper PPO recipe, logged under its own experiment name."""

    experiment_name = "borinot_goalkeeper_simple"
    run_name = "Borinot goalkeeper simple PPO"
