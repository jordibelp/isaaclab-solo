# SPDX-License-Identifier: BSD-3-Clause
"""Goalkeeper SAC, using Jordi's Solo12 recipe with task-specific action bounds."""
import torch

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlLocalRedundancyCfg
from isaaclab_tasks.direct.solo12.agents.rsl_rl_sac_cfg import Solo12SACRunnerCfg
from rsl_rl_sac.algorithms import SAC


class BorinotGoalkeeperSAC(SAC):
    @staticmethod
    def _compute_action_scaling(env, device):
        task = getattr(env, "unwrapped", env)
        if not task.cfg.sac_action_mapping:
            raise ValueError("Borinot SAC requires sac_action_mapping=True; use source/scripts/rsl_rl/train.py.")
        # Rotor actions stay in [-1, 1]; the environment maps them around hover.
        # Joint actions retain the existing target-scale units, like Solo12 SAC.
        limits = task._sac_joint_soft_limits.to(device)
        offset = task._sac_joint_offset.to(device)
        scale = task._joint_scale.to(device)
        upper = torch.cat((torch.ones(6, device=device), (limits[:, 1] - offset) / scale))
        lower = torch.cat((torch.ones(6, device=device), (offset - limits[:, 0]) / scale))
        return upper, lower


@configclass
class BorinotGoalkeeperSACRunnerCfg(Solo12SACRunnerCfg):
    """Linear two-hot CE by default; remaining defaults match the supplied Solo command."""
    experiment_name = "borinot_goalkeeper_sac"
    run_name = "Borinot goalkeeper SAC CE"
    max_iterations = 4500
    optimizer = "adamW"
    adam_beta2 = 0.95
    weight_decay = 0.001
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None
    local_redundancy = RslRlLocalRedundancyCfg(enabled=True, interval=50, batch_size=64)
    save_replay_buffer = False
    save_replay_buffer_every = 250

    actor = Solo12SACRunnerCfg().actor.replace(hidden_dims=[1024, 512, 256])
    critic = Solo12SACRunnerCfg().critic.replace(
        hidden_dims=[1024, 512, 256], layer_norm=True, distributional_loss="two_hot",
        distributional_num_bins=101, distributional_linear_limit=20.0,
    )
    algorithm = Solo12SACRunnerCfg().algorithm.replace(
        class_name=f"{__name__}:BorinotGoalkeeperSAC", target_entropy_scale=0.166,
        num_mini_batches=400, q_reduction_method="mean_pi_q_none", torch_compile=True,
    )

    def configure_environment(self, env_cfg):
        """Select the action contract before constructing the simulator and spaces."""
        if env_cfg.control_mode != "thrust_and_PD_joint_pos_control":
            raise ValueError(
                "Borinot SAC position targets require env.control_mode=thrust_and_PD_joint_pos_control "
                "(the PPO default is torque control)."
            )
        env_cfg.sac_action_mapping = True
        env_cfg.configure_control()


@configclass
class BorinotGoalkeeperSACMSERunnerCfg(BorinotGoalkeeperSACRunnerCfg):
    run_name = "Borinot goalkeeper SAC MSE"
    critic = BorinotGoalkeeperSACRunnerCfg().critic.replace(distributional_loss="mse", distributional_linear_limit=0.0)
