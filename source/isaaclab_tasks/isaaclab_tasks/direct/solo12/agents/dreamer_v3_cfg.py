# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class Solo12DreamerV3RunnerCfg:
    """High-performance DreamerV3 runner config for the direct Solo12 task.

    The learning core lives in ``source/scripts/dreamer/dreamer_core`` (block-GRU
    RSSM, symexp two-hot heads, LaProp + adaptive gradient clipping, fused
    single-backward with frozen-network imagination, AMP + torch.compile, and a
    GPU-resident replay buffer with latent caching).
    """

    seed = 42
    device = "cuda:0"
    num_envs = 1024
    max_iterations = 10000
    steps_per_env = 24
    num_batches_trained_per_iteration = 16
    prefill_steps = 8192
    replay_size = 2_000_000
    batch_size = 2048
    batch_length = 24

    # --- world-model / RSSM sizes ---
    hidden_vector_deter_dims = 128  # deterministic GRU state (must be divisible by `blocks`)
    stoch_dim = 32                  # number of discrete latent variables
    num_bins_encoding = 32          # classes per discrete latent variable
    encoder_hidden_dims = [128, 128]
    model_hidden_dim = 256
    head_hidden_dims = [256, 256]   # reward / continue / decoder head trunks
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]

    # --- block-GRU / RSSM structure (DreamerV3) ---
    blocks = 8            # block-diagonal groups in the recurrent core
    obs_layers = 1        # posterior MLP depth
    img_layers = 2        # prior MLP depth
    dyn_layers = 1        # block-GRU hidden layers
    unimix = 0.01         # uniform mixture on discrete latents

    command_outside_observation = False
    # Mix a fraction of most-recent sequences into each batch (replaces the old
    # online replay queue). 0 -> pure uniform sampling.
    use_uniform_replay_buffer_with_online_queue = True
    replay_recent_fraction = 0.1
    replay_storage_device = "cuda:0"  # keep replay on GPU for zero-copy collection

    imag_horizon = 15
    # discount = 0.997 == 6.67 seconds horizon
    discount = 0.99  # 2 seconds effective horizon; sufficient for locomotion
    lambda_ = 0.95
    free_nats = 1.0
    kl_dyn_scale = 0.5
    kl_rep_scale = 0.1
    reward_loss_scale = 1.0
    continue_loss_scale = 1.0
    obs_loss_scale = 1.0
    policy_loss_scale = 1.0
    value_loss_scale = 1.0
    repval_loss_scale = 0.3   # DreamerV3 replay-based critic loss (0 disables)
    slowreg = 1.0             # slow-critic EMA regularizer strength
    reward_value_num_bins = 255
    reward_value_symlog_range = 20.0
    actor_entropy_scale = 3.0e-4
    actor_min_std = 0.1
    actor_max_std = 1.0
    normalize_actor_returns = True
    return_norm_rate = 0.01
    return_norm_limit = 1.0
    return_norm_percentile_low = 5.0
    return_norm_percentile_high = 95.0

    # --- optimization (LaProp + adaptive gradient clipping, DreamerV3) ---
    model_lr = 1.0e-4
    actor_lr = 3.0e-5
    critic_lr = 1.0e-4
    laprop_beta1 = 0.9
    laprop_beta2 = 0.999
    laprop_eps = 1.0e-20
    agc_clip = 0.3            # adaptive gradient clip ratio
    agc_pmin = 1.0e-3
    warmup_steps = 1000
    slow_critic_tau = 0.01
    grad_clip = 100.0         # kept for reference; AGC is used instead

    # --- runtime efficiency ---
    use_amp = True            # fp16 autocast + GradScaler
    # Keep compile opt-in for cluster runs. On IRICluster, Torch Inductor can
    # stall while querying nvidia-smi during the first compiled update.
    use_compile = False

    use_cbp = False
    cbp_replacement_rate = 1.0e-4
    cbp_maturity_threshold = 10_000
    cbp_decay_rate = 0.99
    cbp_util_type = "contribution"
    cbp_init = "kaiming"
    cbp_accumulate = True

    log_interval = 10
    save_interval = 25
    save_best_checkpoint = True
    experiment_name = "solo12_dreamer_v3"
    run_name = "[Local]-Solo12 high-performance DreamerV3"
    logger = "wandb"
    wandb_project = "solo12-dreamer"
    wandb_entity = None
