# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_cmd_vel.agents.rsl_rl_sac_cfg import BorinotCmdVelSACRunnerCfg, BorinotCmdVelSACMSERunnerCfg


@configclass
class BorinotGoToSACRunnerCfg(BorinotCmdVelSACRunnerCfg):
    experiment_name = "borinot_goto_sac"
    run_name = "Borinot go-to SAC CE"
    algorithm = BorinotCmdVelSACRunnerCfg().algorithm.replace(gamma=1.0)
    # The inherited +/-20 support is too small for one-off -10*d terminal costs.
    # Wider support includes the default -100 safety failure. Not a strict return bound.
    critic = BorinotCmdVelSACRunnerCfg().critic.replace(distributional_num_bins=401, distributional_linear_limit=200.0)


@configclass
class BorinotGoToSACMSERunnerCfg(BorinotCmdVelSACMSERunnerCfg):
    experiment_name = "borinot_goto_sac"
    run_name = "Borinot go-to SAC MSE"
    algorithm = BorinotCmdVelSACMSERunnerCfg().algorithm.replace(gamma=1.0)
