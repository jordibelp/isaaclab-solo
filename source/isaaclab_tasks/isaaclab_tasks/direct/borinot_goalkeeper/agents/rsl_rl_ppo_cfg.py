# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot.agents.rsl_rl_ppo_cfg import BorinotPPORunnerCfg


@configclass
class BorinotGoalkeeperPPORunnerCfg(BorinotPPORunnerCfg):
    """Shared Borinot PPO recipe with the goalkeeper's observation groups."""

    experiment_name = "borinot_goalkeeper"
    run_name = "Borinot goalkeeper PPO"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
