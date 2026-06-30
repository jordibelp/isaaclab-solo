# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class Solo12DreamerV3RunnerCfg:
    """Small DreamerV3-style runner config for the direct Solo12 task."""

    seed = 42
    device = "cuda:0"
    num_envs = 1024
    max_iterations = 10000
    steps_per_env = 24 # 4
    num_batches_trained_per_iteration = 16
    prefill_steps = 8192
    replay_size = 2_000_000
    batch_size = 2048 # 16
    batch_length = 24 # 32

    hidden_vector_deter_dims = 128
    stoch_dim = 32 #16
    num_bins_encoding = 32 #16
    encoder_hidden_dims = [128, 128]
    model_hidden_dim = 256
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]

    # Upstream DreamerV3 name: imag_length. This local config keeps the older
    # local name while imag_last below controls the K replay states used as starts.
    imag_horizon = 15
    imag_last = 0
    filter_done_imagination_starts = True
    # discount = 0.997 == 6.67 seconds horizon
    discount = 0.99 # 2 seconds effective horizon
    # Discount used for imagination. I think for locomotion 2s is more than sufficient.
    lambda_ = 0.95
    free_nats = 1.0
    kl_dyn_scale = 0.5
    kl_rep_scale = 0.1
    reward_loss_scale = 1.0
    continue_loss_scale = 1.0
    obs_loss_scale = 1.0
    reward_value_num_bins = 255
    reward_value_symlog_range = 20.0
    actor_entropy_scale = 3.0e-4
    normalize_actor_returns = True
    return_norm_rate = 0.01
    return_norm_limit = 1.0
    return_norm_percentile_low = 5.0
    return_norm_percentile_high = 95.0

    model_lr = 1.0e-4
    actor_lr = 3.0e-5
    critic_lr = 1.0e-4
    grad_clip = 100.0
    slow_critic_tau = 0.01

    use_cbp = False
    cbp_replacement_rate = 1.0e-4
    cbp_maturity_threshold = 10_000
    cbp_decay_rate = 0.99
    cbp_util_type = "contribution"
    cbp_init = "kaiming"
    cbp_accumulate = True

    log_interval = 10
    save_interval = 100
    experiment_name = "solo12_dreamer_v3"
    run_name = "[Local]-Solo12 simple DreamerV3"
    logger = "wandb"
    wandb_project = "solo12-dreamer"
    wandb_entity = None
