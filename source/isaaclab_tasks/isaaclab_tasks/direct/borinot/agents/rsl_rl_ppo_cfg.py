# SPDX-License-Identifier: BSD-3-Clause
"""Borinot PPO recipe matched to Solo12 run 8958tyzt, without Solo symmetry."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlLocalRedundancyCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

# This PPO wrapper is task-independent: compiled forwards and gradient diagnostics.
# Import its config module as well to register Solo12PPO with the RSL-RL runner.
from isaaclab_tasks.direct.solo12.agents.rsl_rl_ppo_cfg import Solo12PpoAlgorithmCfg


@configclass
class BorinotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    optimizer: str = "adamW"
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    weight_decay = 0.001
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 469
    experiment_name = "borinot_rsl_rl_ppo"
    run_name = "Borinot PPO"
    logger = "wandb"
    wandb_project = "borinotIsaacLab"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    local_redundancy = RslRlLocalRedundancyCfg(enabled=True, interval=50, batch_size=64)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = Solo12PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        torch_compile=True,
    )
