# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

_SOLO12_AGENTS_DIR = Path(__file__).resolve().parent
_ISAACLAB_ROOT = _SOLO12_AGENTS_DIR.parents[5]
_BORINOT_SKRL_SCRIPTS_DIR = _ISAACLAB_ROOT / "source" / "scripts" / "skrl"
if str(_BORINOT_SKRL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BORINOT_SKRL_SCRIPTS_DIR))

import solo12_symmetry


@configclass
class Solo12LaasPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 45573
    save_interval = 469
    experiment_name = "borinot_ppo_experiments_fast"
    run_name = "[ClusterIRI]-LAAS USD/actuator DR remove_root_lin_vel_b_from_obs=True"
    logger = "wandb"
    wandb_project = "borinotIsaacLab"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[x for x in [256, 128, 64]],
        critic_hidden_dims=[x for x in [256, 128, 64]],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Solo12LaasPPORunnerWithSymmetryCfg(Solo12LaasPPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=solo12_symmetry.compute_symmetric_observations_actions,
        ),
    )
