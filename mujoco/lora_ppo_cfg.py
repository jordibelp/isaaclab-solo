"""Default PPO/LoRA agent configuration for MJX fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class LoraPolicyCfg:
    trainable_layers: str = "all"
    rank: int = 1
    lora_alpha: float = 1.0
    log_std_range: tuple[float, float] = (-4.0, 0.0)


@dataclass
class LoraPpoAlgorithmCfg:
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    learning_rate_range: tuple[float, float] = (1.0e-5, 1.0e-2)
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.002
    max_grad_norm: float = 0.5


@dataclass
class LoraPpoAgentCfg:
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 50
    policy: LoraPolicyCfg = field(default_factory=LoraPolicyCfg)
    algorithm: LoraPpoAlgorithmCfg = field(default_factory=LoraPpoAlgorithmCfg)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_AGENT_CFG = LoraPpoAgentCfg()
