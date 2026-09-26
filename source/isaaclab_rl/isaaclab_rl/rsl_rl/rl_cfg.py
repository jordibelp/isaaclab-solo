# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlSacActorModelCfg:
    """Configuration for the tanh-squashed SAC actor from RSL-RL-SAC."""

    class_name: str = "SACActorModel"
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = MISSING
    init_noise_std: float = MISSING
    state_dependent_std: bool = True
    """Whether the actor network outputs log standard deviation as a function of state."""
    layer_norm: bool = False
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@configclass
class RslRlSacCriticModelCfg:
    """Configuration for the twin-Q SAC critic."""

    class_name: str = "SACCriticModel"
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = MISSING
    layer_norm: bool = False
    distributional_loss: str = "mse"
    """Critic loss: "mse", scalar-target "two_hot"/"hl_gauss", distributional Bellman "c51",
    or "mse_target_norm_popart" for MSE on PopArt-normalized targets."""
    distributional_num_bins: int = 255
    """Odd number of symmetrically spaced symexp atoms (two_hot and hl_gauss only)."""
    distributional_symlog_limit: float = 8.0
    """Two-hot/HL-Gauss support is symexp(linspace(-limit, limit, num_bins)), in reward units."""
    hl_gauss_sigma_ratio: float = 0.75
    """HL-Gauss label width as a fraction of the atom spacing; 0.75 is the published default."""
    c51_num_atoms: int = 101
    """Number of equally spaced C51 atoms; FastSAC uses 101."""
    c51_v_min: float = -20.0
    c51_v_max: float = 20.0
    """C51 support bounds in reward units; FastSAC uses [-20, 20]."""
    popart_beta: float = 3e-4
    """Step size of PopArt's running target mean/std (mse_target_norm_popart only); 3e-4 as in Hessel et al. 2018."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""


@configclass
class RslRlSacAlgorithmCfg:
    """Configuration for SAC as released with arXiv:2605.24975."""

    class_name: str = "SAC"
    replay_buffer_size: int = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    mini_batch_size: int = MISSING
    actor_learning_rate: float = MISSING
    critic_learning_rate: float = MISSING
    alpha_learning_rate: float = MISSING
    gamma: float = MISSING
    tau: float = MISSING
    alpha: float = MISSING
    auto_alpha: bool = MISSING
    target_entropy_scale: float = 1.0
    max_grad_norm: float = MISSING
    policy_frequency: int = MISSING
    n_steps: int = MISSING
    q_reduction_method: Literal["min", "mean", "mean_pi_q_none"] = "min"
    """How the twin critics are combined in the Bellman target and the actor loss.

    "min" is clipped double Q-learning. "mean" uses their average in both places (see FastSAC, arXiv:2512.01996).
    "mean_pi_q_none" follows the FastSAC reference code: the actor loss uses the average, and each critic
    bootstraps from its own target network, with no reduction in the target.
    """
    actor_optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    critic_optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    alpha_optimizer: Literal["adam", "adamw"] = "adam"
    torch_compile: bool = False
    """Compile the per-mini-batch Bellman target, critic loss and actor objective with ``torch.compile``.

    The math is unchanged; the compiler fuses it into fewer GPU kernels. The first update is slower while
    the kernels compile.
    """
    rnd_cfg: RslRlRndCfg | None = None
    symmetry_cfg: RslRlSymmetryCfg | None = None
    symmetry_log_interval: int = 100
    """Iterations between two measurements of the mirror loss when it is only logged.

    With ``--symmetry-mode=augmentation`` the mirror loss is a diagnostic, so it is computed on one mini-batch
    every this many iterations. The ``loss`` and ``both`` modes still compute it on every actor update, because
    there it is part of the actor loss.
    """


#########################
# Runner configurations #
#########################


@configclass
class RslRlLocalRedundancyCfg:
    """Read-only regression-gradient probe; see source/scripts/rsl_rl/LOCAL_REDUNDANCY.md."""

    enabled: bool = True
    actor: bool = True
    critic: bool = True
    interval: int = 100
    num_samples: int = 4096
    batch_size: int = 16
    input_mode: Literal["gaussian", "observations"] = "gaussian"
    input_std: float = 1.0
    actor_target_std: float = 1.0
    critic_target_std: float = 1.0
    sac_action_source: Literal["uniform", "policy_mean"] = "uniform"
    seed: int = 1729
    resample: bool = True


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    local_redundancy: RslRlLocalRedundancyCfg = RslRlLocalRedundancyCfg(enabled=False)
    """Hydra-configurable actor/critic local-redundancy diagnostics for PPO and SAC."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    weight_decay: float = 0.0
    """PPO policy or SAC actor/critic weight decay. SAC alpha uses a separate setting."""

    adam_beta1: float = 0.9
    """PPO policy or SAC actor/critic first Adam beta coefficient. Default is 0.9."""

    adam_beta2: float = 0.999
    """PPO policy or SAC actor/critic second Adam beta coefficient. Default is 0.999."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    local_redundancy: RslRlLocalRedundancyCfg = RslRlLocalRedundancyCfg()

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the RSL-RL-SAC off-policy runner."""

    class_name: str = "OffPolicyRunner"
    sac_alpha_optimizer: str = "adamW"
    """SAC entropy-temperature optimizer: ``adam`` or ``adamW`` (case-insensitive)."""
    sac_alpha_adam_beta1: float = 0.9
    sac_alpha_adam_beta2: float = 0.999
    """SAC entropy-temperature Adam beta coefficients; independent of actor and critic."""
    sac_alpha_adam_weight_decay: float = 0.0
    """SAC entropy-temperature optimizer weight decay; independent of actor and critic."""
    distributional_critic_ce: bool = False
    """Deprecated alias for critic.distributional_loss="two_hot"; set that instead."""
    local_redundancy: RslRlLocalRedundancyCfg = RslRlLocalRedundancyCfg()
    actor: RslRlSacActorModelCfg = MISSING
    critic: RslRlSacCriticModelCfg = MISSING
    algorithm: RslRlSacAlgorithmCfg = MISSING
    log_interval: int = 1
    start_training: int = 1
    """Iterations of pure data collection before the first gradient update."""

    save_replay_buffer: bool = False
    """Whether to snapshot the replay buffer to ``<log_dir>/replay_buffer.pt`` during training.

    Enable this when the run is meant to seed a later sim-to-online fine-tuning run, which
    reuses the pretraining transitions as retained replay (arXiv:2602.20220).
    """

    save_replay_buffer_every: int = 500
    """Iterations between replay-buffer snapshots.

    The same file is overwritten each time, so a run that is stopped before ``max_iterations``
    still leaves usable replay data behind without keeping one copy per snapshot.
    """
