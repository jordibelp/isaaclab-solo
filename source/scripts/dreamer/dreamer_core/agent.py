# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""High-performance DreamerV3 agent.

Efficiency techniques (the "5x" bundle):
  * one fused backward over the combined world-model + actor + critic loss;
  * imagination rolled out through *frozen* network views that share storage with
    the live nets (up-to-date weights, no autograd graph through the rollout);
  * AMP fp16 autocast + GradScaler, and ``torch.compile(reduce-overhead)`` with
    CUDA graphs over the whole gradient computation;
  * a single LaProp optimizer (per-group LRs) with adaptive gradient clipping;
  * block-diagonal GRU + RMSNorm RSSM and a GPU-resident buffer.

Fidelity to DreamerV3: symexp two-hot reward/value, unimix discrete latents with
free-bits KL, percentile return normalization, a slow-critic EMA regularizer
(``slowreg``), replay-based critic learning (``repval``), and the bounded-normal
continuous actor.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from . import networks
from .distributions import symlog
from .optim import LaProp, clip_grad_agc_
from .rssm import RSSM, RSSMState


@dataclass
class DreamerConfig:
    obs_dim: int
    command_dim: int
    action_dim: int
    device: str = "cuda:0"

    # RSSM / model
    deter: int = 512
    stoch: int = 32
    discrete: int = 32
    model_hidden: int = 256
    blocks: int = 8
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1
    unimix: float = 0.01
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    head_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    num_bins: int = 255
    symlog_range: float = 20.0
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0

    # loss scales
    kl_free: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    decoder_scale: float = 1.0
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    policy_scale: float = 1.0
    value_scale: float = 1.0
    repval_scale: float = 0.3
    act_entropy_scale: float = 3e-4
    slowreg: float = 1.0

    # actor-critic
    imag_horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95
    normalize_actor_returns: bool = True
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_pct_low: float = 5.0
    return_norm_pct_high: float = 95.0
    slow_critic_tau: float = 0.02

    # optim / runtime
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    laprop_beta1: float = 0.9
    laprop_beta2: float = 0.999
    laprop_eps: float = 1e-20
    agc_clip: float = 0.3
    agc_pmin: float = 1e-3
    warmup_steps: int = 1000
    use_amp: bool = True
    use_compile: bool = True


class DreamerAgent(nn.Module):
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.command_dim = int(cfg.command_dim)

        # --- world model -------------------------------------------------
        self.encoder = networks.make_mlp(cfg.obs_dim, list(cfg.encoder_hidden_dims) + [cfg.model_hidden], None)
        embed_dim = cfg.model_hidden
        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=cfg.action_dim,
            deter=cfg.deter,
            stoch=cfg.stoch,
            discrete=cfg.discrete,
            hidden=cfg.model_hidden,
            blocks=cfg.blocks,
            obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
            dyn_layers=cfg.dyn_layers,
            unimix_ratio=cfg.unimix,
        )
        self.feat_dim = self.rssm.feat_size + self.command_dim
        self.decoder = networks.MLPDecoderHead(self.feat_dim, list(cfg.head_hidden_dims), cfg.obs_dim)
        self.reward = networks.TwoHotHead(self.feat_dim, list(cfg.head_hidden_dims), cfg.num_bins, cfg.symlog_range, outscale=0.0)
        self.cont = networks.ContinueHead(self.feat_dim, list(cfg.head_hidden_dims))

        # --- actor-critic ------------------------------------------------
        self.actor = networks.Actor(self.feat_dim, cfg.action_dim, list(cfg.actor_hidden_dims), cfg.actor_min_std, cfg.actor_max_std)
        self.value = networks.TwoHotHead(self.feat_dim, list(cfg.critic_hidden_dims), cfg.num_bins, cfg.symlog_range, outscale=0.0)
        self._slow_value = networks.TwoHotHead(self.feat_dim, list(cfg.critic_hidden_dims), cfg.num_bins, cfg.symlog_range, outscale=0.0)
        self._slow_value.load_state_dict(self.value.state_dict())
        for p in self._slow_value.parameters():
            p.requires_grad_(False)
        self.return_ema = networks.ReturnEMA(cfg.return_norm_rate, cfg.return_norm_limit, cfg.return_norm_pct_low, cfg.return_norm_pct_high)

        self.to(self.device)

        # --- optimizer (single, per-group LRs) ---------------------------
        world_params = list(self.encoder.parameters()) + list(self.rssm.parameters()) \
            + list(self.decoder.parameters()) + list(self.reward.parameters()) + list(self.cont.parameters())
        self._optimizer = LaProp(
            [
                {"params": world_params, "lr": cfg.model_lr},
                {"params": list(self.actor.parameters()), "lr": cfg.actor_lr},
                {"params": list(self.value.parameters()), "lr": cfg.critic_lr},
            ],
            betas=(cfg.laprop_beta1, cfg.laprop_beta2),
            eps=cfg.laprop_eps,
        )
        self._base_lrs = [cfg.model_lr, cfg.actor_lr, cfg.critic_lr]
        self._trainable = [p for p in self.parameters() if p.requires_grad]
        self._scaler = GradScaler(enabled=cfg.use_amp)
        self.num_optimization_steps = 0

        # Frozen storage-sharing views of every net used in imagination.
        self._make_frozen_views()

        self.cbp_managers: dict[str, object] = {}

        self._cal_grad_fn = self._cal_grad
        if cfg.use_compile:
            self._cal_grad_fn = torch.compile(self._cal_grad, mode="reduce-overhead")

    # ------------------------------------------------------------- frozen
    def _make_frozen_views(self) -> None:
        """Create frozen module copies whose params share storage with the live nets.

        Because ``.data`` aliases the live parameter tensors (updated in place by
        the optimizer), these always reflect current weights yet never build an
        autograd graph — so imagination rollouts stay cheap.
        """
        def frozen_like(module: nn.Module) -> nn.Module:
            clone = copy.deepcopy(module)
            for (_, src), (_, dst) in zip(module.named_parameters(), clone.named_parameters()):
                dst.data = src.data
                dst.requires_grad_(False)
            for (_, src), (_, dst) in zip(module.named_buffers(), clone.named_buffers()):
                dst.data = src.data
            clone.eval()
            return clone

        self._frozen_encoder = frozen_like(self.encoder)
        self._frozen_rssm = frozen_like(self.rssm)
        self._frozen_actor = frozen_like(self.actor)
        self._frozen_reward = frozen_like(self.reward)
        self._frozen_cont = frozen_like(self.cont)
        self._frozen_value = frozen_like(self.value)
        self._frozen_slow_value = frozen_like(self._slow_value)

    @torch.no_grad()
    def _update_slow_value(self) -> None:
        tau = self.cfg.slow_critic_tau
        for s, v in zip(self._slow_value.parameters(), self.value.parameters()):
            s.data.mul_(1.0 - tau).add_(v.data, alpha=tau)

    # -------------------------------------------------------------- helpers
    def _feature(self, state: RSSMState, command: torch.Tensor) -> torch.Tensor:
        feat = self.rssm.get_feat(state)
        if self.command_dim:
            feat = torch.cat([feat, command], dim=-1)
        return feat

    def _frozen_feature(self, state: RSSMState, command: torch.Tensor) -> torch.Tensor:
        feat = self._frozen_rssm.get_feat(state)
        if self.command_dim:
            feat = torch.cat([feat, command], dim=-1)
        return feat

    @torch.no_grad()
    def initial_state(self, batch_size: int) -> RSSMState:
        return self.rssm.initial(batch_size, self.device)

    @torch.no_grad()
    def act(self, state: RSSMState, prev_action: torch.Tensor, command: torch.Tensor, is_first: torch.Tensor,
            obs: torch.Tensor, *, eval: bool = False) -> tuple[torch.Tensor, RSSMState]:
        """One environment-facing policy step (posterior update + action)."""
        embed = self._frozen_encoder(symlog(obs))
        state, _ = self._frozen_rssm.obs_step(state, prev_action, embed, is_first)
        feat = self._frozen_feature(state, command)
        dist = self._frozen_actor(feat)
        action = dist.mode if eval else dist.rsample()
        return action, state

    # -------------------------------------------------------------- update
    def update(self, batch: dict[str, torch.Tensor], writeback, buffer) -> dict[str, torch.Tensor]:
        if self.cfg.use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        self._update_slow_value()
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.use_amp):
            (post_stoch, post_deter), metrics = self._cal_grad_fn(batch)
        self._scaler.unscale_(self._optimizer)
        clip_grad_agc_(self._trainable, self.cfg.agc_clip, self.cfg.agc_pmin)
        self._apply_warmup_lr()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad(set_to_none=True)
        self.num_optimization_steps += 1
        for name, manager in self.cbp_managers.items():
            manager.after_optimizer_step(self._optimizer)
        buffer.update_latents(writeback, post_stoch.detach(), post_deter.detach())
        # Clone out of the (possibly CUDA-graph) static output buffers so callers can
        # read these later without a sync and without them being overwritten.
        return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in metrics.items()}

    def _apply_warmup_lr(self) -> None:
        warmup = self.cfg.warmup_steps
        scale = 1.0 if not warmup else min(1.0, (self.num_optimization_steps + 1) / warmup)
        for group, base in zip(self._optimizer.param_groups, self._base_lrs):
            group["lr"] = base * scale

    # ------------------------------------------------------------ gradient
    def _cal_grad(self, batch: dict[str, torch.Tensor]):
        cfg = self.cfg
        obs = batch["obs"]
        command = batch["command"]
        prev_action = batch["prev_action"]
        reward = batch["reward"]
        is_first = batch["is_first"]
        is_terminal = batch["is_terminal"].float()
        is_last = batch["is_last"].float()
        B, T = obs.shape[0], obs.shape[1]
        initial = RSSMState(stoch=batch["initial_stoch"], deter=batch["initial_deter"])

        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        # === world model ===
        embed = self.encoder(symlog(obs))
        post, post_logit = self.rssm.observe(embed, prev_action, initial, is_first)
        prior_logit = self.rssm.prior_logit(post.deter)
        dyn, rep = self.rssm.kl_loss(post_logit, prior_logit, cfg.kl_free)
        losses["dyn"] = dyn.mean()
        losses["rep"] = rep.mean()

        feat = self._feature(post, command)
        losses["decoder"] = -self.decoder(feat).log_prob(obs).mean()
        losses["reward"] = -self.reward(feat).log_prob(reward).mean()
        cont_target = 1.0 - is_terminal
        losses["cont"] = nn.functional.binary_cross_entropy_with_logits(self.cont.logits(feat), cont_target)

        metrics["prior_entropy"] = self.rssm.get_dist(prior_logit).entropy().mean()
        metrics["post_entropy"] = self.rssm.get_dist(post_logit).entropy().mean()

        # === imagination (frozen rollout, no graph through the sequence) ===
        start = RSSMState(
            stoch=post.stoch.reshape(B * T, *post.stoch.shape[2:]).detach(),
            deter=post.deter.reshape(B * T, post.deter.shape[-1]).detach(),
        )
        start_command = command.reshape(B * T, self.command_dim).detach() if self.command_dim else command.reshape(B * T, 0)
        imag_feat, imag_action = self._imagine(start, start_command, cfg.imag_horizon + 1)
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()

        imag_reward = self._frozen_reward(imag_feat).mean().unsqueeze(-1)
        imag_cont = torch.sigmoid(self._frozen_cont.logits(imag_feat)).unsqueeze(-1)
        imag_value = self._frozen_value(imag_feat).mean().unsqueeze(-1)
        imag_slow = self._frozen_slow_value(imag_feat).mean().unsqueeze(-1)

        disc = cfg.discount
        weight = torch.cumprod(imag_cont * disc, dim=1)
        term = 1.0 - imag_cont
        last = torch.zeros_like(imag_cont)
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, cfg.lambda_)
        ret_offset, ret_scale = self.return_ema(ret, update=True) if cfg.normalize_actor_returns else (ret.new_zeros(()), ret.new_ones(()))
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + cfg.act_entropy_scale * entropy))

        value_dist = self.value(imag_feat)
        tar = torch.cat([ret, torch.zeros_like(ret[:, -1:])], dim=1).detach()
        # Regress the (grad-carrying) value onto detached targets: return + slow-critic EMA.
        value_lp = (-value_dist.log_prob(tar.squeeze(-1)) - cfg.slowreg * value_dist.log_prob(imag_slow.squeeze(-1).detach()))
        losses["value"] = torch.mean(weight[:, :-1].squeeze(-1).detach() * value_lp[:, :-1])

        metrics["imag_reward"] = imag_reward.mean()
        metrics["imag_value"] = imag_value.mean()
        metrics["imag_return"] = ret.mean()
        metrics["imag_cont"] = imag_cont.mean()
        metrics["return_scale"] = ret_scale
        metrics["action_entropy"] = entropy.mean()

        # === replay critic (repval): grad flows back into the world model ===
        if cfg.repval_scale > 0.0:
            rep_reward = reward.unsqueeze(-1)
            # Bootstrap targets are detached; the world-model gradient flows only
            # through self.value(feat) below.
            rep_value = self._frozen_value(feat.detach()).mean().unsqueeze(-1)
            rep_slow = self._frozen_slow_value(feat.detach()).mean().unsqueeze(-1)
            boot = ret[:, 0].reshape(B, T, 1)
            rep_ret = self._lambda_return(
                is_last.unsqueeze(-1), is_terminal.unsqueeze(-1), rep_reward, rep_value, boot, disc, cfg.lambda_
            )
            rep_tar = torch.cat([rep_ret, torch.zeros_like(rep_ret[:, -1:])], dim=1).detach()
            rep_weight = 1.0 - is_last
            rep_value_dist = self.value(feat)
            rep_lp = (-rep_value_dist.log_prob(rep_tar.squeeze(-1)) - cfg.slowreg * rep_value_dist.log_prob(rep_slow.squeeze(-1).detach()))
            losses["repval"] = torch.mean(rep_weight[:, :-1] * rep_lp[:, :-1])

        scales = {
            "dyn": cfg.kl_dyn_scale, "rep": cfg.kl_rep_scale, "decoder": cfg.decoder_scale,
            "reward": cfg.reward_scale, "cont": cfg.cont_scale, "policy": cfg.policy_scale,
            "value": cfg.value_scale, "repval": cfg.repval_scale,
        }
        total = sum(scales[k] * v for k, v in losses.items())
        self._scaler.scale(total).backward()

        metrics["total_loss"] = total.detach()
        for k, v in losses.items():
            metrics[f"loss/{k}"] = v.detach()
        return (post.stoch.detach(), post.deter.detach()), metrics

    @torch.no_grad()
    def _imagine(self, start: RSSMState, command: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the (frozen) policy forward in latent space; no autograd graph."""
        state = start
        feats, actions = [], []
        for _ in range(horizon):
            feat = self._frozen_feature(state, command)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            state, _ = self._frozen_rssm.img_step(state, action)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @staticmethod
    @torch.no_grad()
    def _lambda_return(last, term, reward, value, boot, disc, lamb):
        """DreamerV3 continuation-weighted lambda return (bootstrap target; returns length T-1)."""
        live = (1.0 - term)[:, 1:] * disc
        cont = (1.0 - last)[:, 1:] * lamb
        interm = reward[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], dim=1)

    # ------------------------------------------------------------- CBP / IO
    def cbp_target_mlps(self) -> dict[str, list[nn.Sequential]]:
        """Grouped ``nn.Sequential`` MLPs eligible for Continual-Backprop."""
        return {
            "world": [self.encoder, self.rssm._obs_net, self.rssm._img_net,
                      self.decoder.net, self.reward.net, self.cont.net],
            "actor": [self.actor.net],
            "critic": [self.value.net],
        }

    def attach_cbp(self, managers: dict[str, object]) -> None:
        self.cbp_managers = {k: v for k, v in managers.items() if v is not None}

    @staticmethod
    def metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        return {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}

    def save(self, path, extra: dict | None = None) -> None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "num_optimization_steps": self.num_optimization_steps,
        }
        if self.cbp_managers:
            payload["cbp"] = {name: m.state_dict() for name, m in self.cbp_managers.items()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self._scaler.load_state_dict(ckpt["scaler"])
        self.num_optimization_steps = int(ckpt.get("num_optimization_steps", 0))
        cbp_state = ckpt.get("cbp", {})
        for name, manager in self.cbp_managers.items():
            if name in cbp_state:
                manager.load_state_dict(cbp_state[name])
        # Re-point frozen views at the (possibly reloaded) live parameter storage.
        self._make_frozen_views()
        return ckpt
