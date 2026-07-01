# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmark: legacy DreamerV3 training step vs. the high-performance core.

Runs entirely on synthetic Solo12-shaped tensors (no Isaac Sim), so it isolates
the *learning update* cost — exactly where the efficiency work lands — and fits
on a small GPU.

``LegacyCore`` faithfully mirrors the hot path of the previous
``source/scripts/dreamer/train.py`` (plain ``GRUCell`` RSSM, LayerNorm MLPs,
per-timestep Python loss loop, three separate Adam optimizers / three backward
passes, fp32, autograd through the imagination rollout).  It is then compared
against the new core with efficiency features toggled on progressively.

Usage:
    python benchmark_update.py                 # default B=512, T=24
    python benchmark_update.py --batch 1024 --iters 30
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer_core.agent import DreamerAgent, DreamerConfig

DEV = torch.device("cuda:0")

# Solo12 defaults (command folded into obs).
OBS_DIM, CMD_DIM, ACT_DIM = 36, 0, 12
DETER, STOCH, DISCRETE, HIDDEN = 128, 32, 32, 256
NUM_BINS, SYMLOG_RANGE = 255, 20.0
IMAG_HORIZON, DISCOUNT, LAMBDA = 15, 0.99, 0.95


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ---------------------------------------------------------------- legacy core
def legacy_mlp(inp, hidden, out):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class LegacyTwoHot(nn.Module):
    def __init__(self, inp, hidden, num_bins=NUM_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.net = legacy_mlp(inp, hidden, num_bins)
        half = symexp(torch.linspace(-SYMLOG_RANGE, 0, (num_bins - 1) // 2 + 1))
        self.register_buffer("bins", torch.cat((half, -half[:-1].flip(0))))

    def logits(self, x):
        return self.net(x)

    def pred(self, x):
        p = torch.softmax(self.net(x), -1)
        return (p * self.bins).sum(-1)

    def loss(self, x, target):
        logits = self.net(x)
        below = (torch.searchsorted(self.bins, target.detach(), right=True) - 1).clamp(0, self.num_bins - 1)
        above = (below + 1).clamp(0, self.num_bins - 1)
        eq = below == above
        db = torch.where(eq, torch.ones_like(target), (self.bins[below] - target).abs())
        da = torch.where(eq, torch.ones_like(target), (self.bins[above] - target).abs())
        tot = db + da
        twohot = F.one_hot(below, self.num_bins) * (da / tot).unsqueeze(-1) + F.one_hot(above, self.num_bins) * (db / tot).unsqueeze(-1)
        return -(twohot * F.log_softmax(logits, -1)).sum(-1)


class LegacyCore(nn.Module):
    """Mirrors the previous train.py update (GRUCell, per-step loop, 3 optimizers)."""

    def __init__(self):
        super().__init__()
        self.flat = STOCH * DISCRETE
        feat = DETER + self.flat + CMD_DIM
        self.encoder = legacy_mlp(OBS_DIM, [128, 128], HIDDEN)
        self.gru = nn.GRUCell(self.flat + ACT_DIM, DETER)
        self.prior = legacy_mlp(DETER, [HIDDEN], self.flat)
        self.posterior = legacy_mlp(DETER + HIDDEN, [HIDDEN], self.flat)
        self.decoder = legacy_mlp(feat, [HIDDEN, HIDDEN], OBS_DIM)
        self.reward = LegacyTwoHot(feat, [HIDDEN, HIDDEN])
        self.cont = legacy_mlp(feat, [HIDDEN, HIDDEN], 1)
        self.actor = legacy_mlp(feat, [256, 128, 64], 2 * ACT_DIM)
        self.critic = LegacyTwoHot(feat, [256, 128, 64])
        self.target = LegacyTwoHot(feat, [256, 128, 64])
        self.target.load_state_dict(self.critic.state_dict())
        self.to(DEV)
        self.mopt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.gru.parameters())
                                     + list(self.prior.parameters()) + list(self.posterior.parameters())
                                     + list(self.decoder.parameters()) + list(self.reward.parameters())
                                     + list(self.cont.parameters()), lr=1e-4)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def _st(self, logits):
        logits = logits.reshape(-1, STOCH, DISCRETE)
        p = torch.softmax(logits, -1)
        idx = torch.multinomial(p.reshape(-1, DISCRETE), 1).reshape(-1, STOCH)
        oh = F.one_hot(idx, DISCRETE).float()
        return oh + p - p.detach()

    def feature(self, deter, stoch):
        return torch.cat([deter, stoch.reshape(stoch.shape[0], self.flat)], -1)

    def update(self, batch):
        obs, act, rew = batch["obs"], batch["prev_action"], batch["reward"]
        term = batch["is_terminal"].float()
        T = obs.shape[1]
        B = obs.shape[0]
        deter = torch.zeros(B, DETER, device=DEV)
        stoch = torch.zeros(B, STOCH, DISCRETE, device=DEV)
        stoch[..., 0] = 1.0
        # --- world model: per-timestep Python loop (legacy) ---
        model_loss = torch.zeros((), device=DEV)
        deters, stochs = [], []
        for t in range(T):
            deter = self.gru(torch.cat([stoch.reshape(B, self.flat), act[:, t]], -1), deter)
            prior_logits = self.prior(deter)
            embed = self.encoder(symlog(obs[:, t]))
            post_logits = self.posterior(torch.cat([deter, embed], -1))
            stoch = self._st(post_logits)
            feat = self.feature(deter, stoch)
            obs_loss = F.mse_loss(self.decoder(feat), symlog(obs[:, t]), reduction="none").sum(-1)
            rew_loss = self.reward.loss(feat, rew[:, t])
            cont_loss = F.binary_cross_entropy_with_logits(self.cont(feat).squeeze(-1), 1 - term[:, t])
            pl = post_logits.reshape(-1, STOCH, DISCRETE)
            ql = prior_logits.reshape(-1, STOCH, DISCRETE)
            kl_dyn = (torch.softmax(pl.detach(), -1) * (torch.log_softmax(pl.detach(), -1) - torch.log_softmax(ql, -1))).sum((-1, -2)).clamp(min=1.0)
            kl_rep = (torch.softmax(pl, -1) * (torch.log_softmax(pl, -1) - torch.log_softmax(ql.detach(), -1))).sum((-1, -2)).clamp(min=1.0)
            model_loss = model_loss + obs_loss.mean() + rew_loss.mean() + cont_loss.mean() + 0.5 * kl_dyn.mean() + 0.1 * kl_rep.mean()
            deters.append(deter)
            stochs.append(stoch)
        model_loss = model_loss / T
        self.mopt.zero_grad(set_to_none=True)
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.mopt.param_groups[0]["params"], 100.0)
        self.mopt.step()

        # --- actor-critic: imagination with autograd through the rollout ---
        deter = torch.stack(deters, 1).reshape(-1, DETER).detach()
        stoch = torch.stack(stochs, 1).reshape(-1, STOCH, DISCRETE).detach()
        feats, logps, ents, rews, discs = [], [], [], [], []
        for _ in range(IMAG_HORIZON):
            feat = self.feature(deter, stoch).detach()
            mean, log_std = self.actor(feat).chunk(2, -1)
            std = torch.exp(log_std.clamp(-5, 2))
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            action = torch.tanh(raw)
            logp = (dist.log_prob(raw) - torch.log1p(-action.pow(2) + 1e-6)).sum(-1)
            ent = dist.entropy().sum(-1)
            with torch.no_grad():
                deter = self.gru(torch.cat([stoch.reshape(stoch.shape[0], self.flat), action], -1), deter)
                stoch = self._st(self.prior(deter))
                nf = self.feature(deter, stoch)
                r = self.reward.pred(nf)
                c = torch.sigmoid(self.cont(nf).squeeze(-1))
            feats.append(feat); logps.append(logp); ents.append(ent); rews.append(r); discs.append(DISCOUNT * c)
        feats = torch.stack(feats); logps = torch.stack(logps); ents = torch.stack(ents)
        rews = torch.stack(rews); discs = torch.stack(discs)
        with torch.no_grad():
            vals = self.target.pred(feats.reshape(-1, feats.shape[-1])).reshape(IMAG_HORIZON, -1)
            boot = self.target.pred(self.feature(deter, stoch))
            nv = boot; rets = []
            for t in reversed(range(IMAG_HORIZON)):
                nvv = boot if t == IMAG_HORIZON - 1 else vals[t + 1]
                nv = rews[t] + discs[t] * ((1 - LAMBDA) * nvv + LAMBDA * nv)
                rets.append(nv)
            rets = torch.stack(list(reversed(rets)))
            weights = torch.cumprod(torch.cat([torch.ones_like(discs[:1]), discs[:-1]]), 0)
        adv = (rets - self.critic.pred(feats.reshape(-1, feats.shape[-1])).reshape(IMAG_HORIZON, -1).detach())
        actor_loss = -(weights * (logps * adv.detach() + 3e-4 * ents)).mean()
        self.aopt.zero_grad(set_to_none=True); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0); self.aopt.step()
        closs = (weights * self.critic.loss(feats.detach().reshape(-1, feats.shape[-1]), rets.detach().reshape(-1)).reshape(IMAG_HORIZON, -1)).mean()
        self.copt.zero_grad(set_to_none=True); closs.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0); self.copt.step()
        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.critic.parameters()):
                tp.mul_(0.99).add_(p, alpha=0.01)
        return model_loss


# ------------------------------------------------------------------- harness
def make_batch(B, T):
    return {
        "obs": torch.randn(B, T, OBS_DIM, device=DEV),
        "command": torch.zeros(B, T, 0, device=DEV),
        "prev_action": torch.tanh(torch.randn(B, T, ACT_DIM, device=DEV)),
        "reward": torch.randn(B, T, device=DEV),
        "is_first": torch.rand(B, T, device=DEV) < 0.05,
        "is_terminal": torch.rand(B, T, device=DEV) < 0.02,
        "is_last": torch.rand(B, T, device=DEV) < 0.03,
        "initial_stoch": torch.zeros(B, STOCH, DISCRETE, device=DEV),
        "initial_deter": torch.zeros(B, DETER, device=DEV),
    }


class MockBuf:
    def update_latents(self, *_):
        pass


def time_fn(step_fn, B, T, iters, warmup):
    for _ in range(warmup):
        step_fn(make_batch(B, T))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        step_fn(make_batch(B, T))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000.0
    peak = torch.cuda.max_memory_allocated() / 1e6
    return dt, peak


def new_core(use_amp, use_compile):
    cfg = DreamerConfig(
        obs_dim=OBS_DIM, command_dim=CMD_DIM, action_dim=ACT_DIM, device="cuda:0",
        deter=DETER, stoch=STOCH, discrete=DISCRETE, model_hidden=HIDDEN, blocks=8,
        encoder_hidden_dims=[128, 128], actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
        imag_horizon=IMAG_HORIZON, discount=DISCOUNT, lambda_=LAMBDA,
        use_amp=use_amp, use_compile=use_compile,
    )
    agent = DreamerAgent(cfg)
    buf = MockBuf()
    wb = (torch.zeros(1, 1, dtype=torch.long, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV))

    def step(batch):
        B = batch["obs"].shape[0]
        wbb = (torch.zeros(B, batch["obs"].shape[1], dtype=torch.long, device=DEV), torch.arange(B, device=DEV))
        return agent.update(batch, wbb, buf)

    return step


MODES = {
    "legacy": ("legacy (GRUCell, 3x Adam, fp32, no compile)", lambda: LegacyCore().update),
    "new_fp32": ("new core (fp32, no compile)", lambda: new_core(False, False)),
    "new_amp": ("new core (+AMP fp16)", lambda: new_core(True, False)),
    "new_compile": ("new core (+AMP +torch.compile)", lambda: new_core(True, True)),
}


def run_single(mode, B, T, iters, warmup):
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    name, build = MODES[mode]
    warm = max(warmup, 8) if "compile" in name else warmup
    step = build()
    dt, peak = time_fn(step, B, T, iters, warm)
    reserved = torch.cuda.max_memory_reserved() / 1e6
    # Machine-readable line consumed by the --driver aggregator.
    print(f"RESULT\t{mode}\t{name}\t{dt:.3f}\t{peak:.1f}\t{reserved:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mode", choices=list(MODES), default=None, help="Run a single variant (used by --driver).")
    ap.add_argument("--driver", action="store_true", help="Spawn one clean subprocess per variant and aggregate.")
    args = ap.parse_args()

    if args.mode is not None:
        run_single(args.mode, args.batch, args.seq, args.iters, args.warmup)
        return

    if not args.driver:
        args.driver = True  # default behaviour: drive all variants

    import subprocess
    print(f"Device: {torch.cuda.get_device_name(0)}  |  batch={args.batch} seq={args.seq} iters={args.iters}\n")
    rows = {}
    for mode in MODES:
        cmd = [sys.executable, __file__, "--mode", mode, "--batch", str(args.batch),
               "--seq", str(args.seq), "--iters", str(args.iters), "--warmup", str(args.warmup)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT\t")), None)
        if line is None:
            err = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"{MODES[mode][0]:48s}  FAILED: {err[:90]}")
            continue
        _, _m, name, dt, peak, reserved = line.split("\t")
        rows[mode] = (name, float(dt), float(peak), float(reserved))
        print(f"{name:48s}  {float(dt):8.2f} ms/update   peak(alloc) {float(peak):7.0f} MB   reserved {float(reserved):7.0f} MB")

    if "legacy" in rows and "new_compile" in rows:
        base, best = rows["legacy"][1], rows["new_compile"][1]
        print(f"\n>>> Speedup (legacy -> full new core): {base / best:.2f}x  ({base:.1f} -> {best:.1f} ms/update)")
        if "new_fp32" in rows:
            print(f"    - arch/fused/vectorized alone (fp32):  {base / rows['new_fp32'][1]:.2f}x")
        if "new_amp" in rows:
            print(f"    - + AMP fp16:                          {base / rows['new_amp'][1]:.2f}x")


if __name__ == "__main__":
    main()
