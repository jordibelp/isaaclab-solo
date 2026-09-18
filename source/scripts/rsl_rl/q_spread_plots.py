# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot how much of a categorical SAC critic's support is used, and how wrong it is.

Input is one or more ``.npz`` files written by ``play_direct_0325.py --q_value_log``,
from a checkpoint trained with ``agent.distributional_critic_ce=True``.

The spread panels are reported in **symlog units**: the ``x`` in ``Q = sign(x)(exp|x| - 1)``,
so the axes are directly comparable to ``agent.critic.distributional_symlog_limit``. Runs
with different limits can therefore be overlaid, and the dotted verticals mark each run's
own limit. The estimation-error panels are in raw reward units instead.

    # One run:
    ./isaaclab.sh -p source/scripts/rsl_rl/q_spread_plots.py q_log.npz

    # Compare checkpoints, write a PNG and skip the interactive window:
    ./isaaclab.sh -p source/scripts/rsl_rl/q_spread_plots.py a.npz b.npz \
        --labels "limit 5.0" "limit 4.0" --out spread.png --no-show

The twin critics are mixed as ``(p1 + p2) / 2`` before any spread statistic is computed,
matching the ``CriticDist/*`` scalars logged during training. Critic disagreement therefore
widens the reported spread, exactly as it widens the range of values the pair can encode.

The last two panels compare ``min(Q1, Q2)`` — the estimate SAC actually acts on — against
the return it predicts, using the training ``gamma`` stored in the log. Three things make
that comparison easy to get wrong, and all three are handled explicitly:

* **Entropy.** SAC's Q predicts the *soft* return, which adds ``alpha * -log_pi`` for every
  step after the evaluated action. The logged per-step ``log_prob`` supplies it; without it
  the error carries a systematic bias of roughly ``alpha * H / (1 - gamma)``.
* **Truncation.** A timed-out episode is missing a ``gamma ** (remaining + 1)``-weighted
  tail, so late steps barely observe their own return. Every step reports the fraction of
  its return that was observed, and ``--min-observed`` keeps the statistics to steps where
  that fraction is high. Terminated episodes are complete and always count.
* **Determinism.** Q is defined under the stochastic policy, so a deterministic rollout
  adds a mismatch of its own. Record with ``--q_value_log_stochastic`` to remove it.

Caveat: these critics are trained on *scalar* Bellman targets with a two-hot cross-entropy
loss, not with a distributional Bellman projection. The width of a distribution is fit
error plus target spread; it is not a calibrated estimate of return uncertainty.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))

from rsl_rl_sac.models import symlog_distribution_stats  # noqa: E402

ACTIVE_PROB = 0.1
SUMMARY_KEYS = ("mean", "std", "q05", "q50", "q95", "q05_q95_width", "active_atoms", "effective_atoms", "edge_mass")


def discounted_returns(
    reward: np.ndarray,
    done: np.ndarray,
    time_out: np.ndarray,
    gamma: float,
    entropy_bonus: np.ndarray | None,
    bootstrap: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Realized return behind each step, walking backwards through the log.

    All arrays are ``(steps, envs)``. A *terminated* episode has a true value of zero after
    its last step. A *truncated* one — a timeout, or an episode still running when the log
    ends — does not: cutting it at zero would drag the return down toward the episode
    boundary. Passing ``bootstrap`` closes it with ``V(s')`` the way training does, keeping
    the return flat across a timeout.

    ``entropy_bonus`` is ``alpha * -log_pi`` per step. SAC's Q predicts the *soft* return,
    which adds that bonus for every step after the one being evaluated, so passing it gives
    the quantity Q was actually trained to match. ``bootstrap`` supplies the matching
    ``plain`` and ``soft`` continuation values.

    ``observed`` reports how much of each return came from real rewards rather than from the
    critic's own estimate, which is what makes a bootstrapped comparison partly circular.
    """
    steps, envs = reward.shape
    out = {key: np.zeros_like(reward) for key in ("plain", "soft")}
    remaining = np.zeros((steps, envs), dtype=np.int64)
    terminal = np.zeros((steps, envs), dtype=bool)
    ends = done > 0
    terminates = ends & (time_out <= 0)
    zero = np.zeros(envs)

    # Carried state describes step t+1. It starts empty: nothing is observed past the log.
    carry = {key: np.zeros(envs) for key in ("plain", "soft", "bonus")}
    carry_remaining = np.zeros(envs, dtype=np.int64)
    carry_terminal = np.zeros(envs, dtype=bool)
    carry_valid = np.zeros(envs, dtype=bool)

    for t in range(steps - 1, -1, -1):
        # The tail continues only when this step does not end the episode and t+1 was logged.
        continues = ~ends[t] & carry_valid
        # Everything else is either a true terminal (value zero) or a cut that can bootstrap.
        cut_short = ~continues & ~(ends[t] & terminates[t])
        for key in ("plain", "soft"):
            tail = carry[key] + (carry["bonus"] if key == "soft" else 0.0)
            closing = zero if bootstrap is None else np.where(cut_short, bootstrap[key][t], 0.0)
            out[key][t] = reward[t] + gamma * np.where(continues, tail, closing)
        remaining[t] = np.where(continues, carry_remaining + 1, 0)
        terminal[t] = np.where(ends[t], terminates[t], continues & carry_terminal)

        carry["plain"], carry["soft"] = out["plain"][t], out["soft"][t]
        carry["bonus"] = entropy_bonus[t] if entropy_bonus is not None else zero
        carry_remaining, carry_terminal = remaining[t], terminal[t]
        carry_valid = np.ones(envs, dtype=bool)

    out["bootstrap_weight"] = np.where(terminal, 0.0, gamma ** (remaining + 1))
    out["observed"] = 1.0 - out["bootstrap_weight"]
    out["remaining"] = remaining.astype(np.float64)
    return out


class Run:
    """One ``--q_value_log`` file, reduced to per-step statistics in symlog units."""

    def __init__(self, path: Path, label: str | None, drop_first: int, use_bootstrap: bool = True) -> None:
        data = np.load(path)
        if data["value_support"].size == 0:
            raise ValueError(f"{path} has no categorical support; it was logged from a scalar critic.")
        self.path = path
        self.label = label or path.stem
        self.support = torch.from_numpy(data["value_support"]).float()
        self.atoms = self.support.sign() * self.support.abs().log1p()
        self.symlog_limit = float(self.atoms[-1])
        # (steps, envs, 2, atoms) -> mix the twin critics, then flatten the env axis.
        probs = torch.from_numpy(data["probs"]).float()[drop_first:]
        self.probs = probs.mean(dim=2)
        self.q = torch.from_numpy(data["q"]).float()[drop_first:]
        self.done = torch.from_numpy(data["done"]).float()[drop_first:]
        self.stats = symlog_distribution_stats(self.probs, self.atoms)
        self.episodes = int(self.done.sum())
        self.samples = self.probs.shape[0] * self.probs.shape[1]

        # SAC acts on min(Q1, Q2), so that is the estimate whose error matters.
        self.q_min = self.q.min(dim=-1).values.numpy().astype(np.float64)
        self.gamma = float(data["gamma"]) if "gamma" in data else None
        self.alpha = float(data["alpha"]) if "alpha" in data else None
        self.deterministic = bool(data["deterministic"]) if "deterministic" in data else None
        self.returns = None
        if self.gamma is None:
            return
        reward = data["reward"][drop_first:].astype(np.float64)
        bonus = None
        if "log_prob" in data and self.alpha is not None:
            bonus = self.alpha * -data["log_prob"][drop_first:].astype(np.float64)
        self.entropy_bonus = bonus

        # V(s') for closing a truncated episode, as training does, instead of assuming zero.
        self.bootstrap = None
        self.bootstrap_available = "bootstrap_q" in data
        if use_bootstrap and self.bootstrap_available:
            q_next = data["bootstrap_q"][drop_first:].astype(np.float64)
            self.bootstrap = {"plain": q_next, "soft": q_next}
            if "bootstrap_log_prob" in data and self.alpha is not None:
                # The soft value subtracts the entropy of the action taken at s'.
                self.bootstrap["soft"] = q_next - self.alpha * data["bootstrap_log_prob"][drop_first:]

        self.returns = discounted_returns(
            reward,
            data["done"][drop_first:].astype(np.float64),
            data["time_out"][drop_first:].astype(np.float64),
            self.gamma,
            bonus,
            self.bootstrap,
        )

    def target_return(self) -> tuple[np.ndarray, str]:
        """The return Q should match, preferring the entropy-corrected one when available."""
        if self.entropy_bonus is not None:
            return self.returns["soft"], "soft"
        return self.returns["plain"], "plain"

    def error(self) -> np.ndarray:
        """Q - G, the estimation error at every logged step."""
        return self.q_min - self.target_return()[0]

    def flat(self, key: str) -> np.ndarray:
        return self.stats[key].reshape(-1).numpy()

    def per_step(self, key: str) -> np.ndarray:
        """Average over envs, keeping the time axis."""
        return self.stats[key].mean(dim=1).numpy()

    def occupancy(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean probability per atom, and the fraction of samples where that atom is used."""
        flat = self.probs.reshape(-1, self.probs.shape[-1])
        return flat.mean(dim=0).numpy(), (flat > ACTIVE_PROB).float().mean(dim=0).numpy()

    def valid(self, min_observed: float) -> np.ndarray:
        """Steps whose realized return is complete enough to compare against."""
        return self.returns["observed"] >= min_observed

    def summary(self, min_observed: float) -> dict[str, float]:
        out = {key: float(self.stats[key].mean()) for key in SUMMARY_KEYS}
        out["symlog_limit"] = self.symlog_limit
        out["symlog_std_across_states"] = float(self.stats["mean"].std(unbiased=False))
        out["q_raw_mean"] = float(self.q.mean())
        out["q_raw_min"] = float(self.q.min())
        out["q_raw_max"] = float(self.q.max())
        out["critic_disagreement"] = float((self.q[..., 0] - self.q[..., 1]).abs().mean())
        if self.returns is None:
            return out

        out["gamma"] = self.gamma
        keep = self.valid(min_observed)
        out["valid_fraction"] = float(keep.mean())
        if not keep.any():
            return out
        error = self.error()[keep]
        out["error_mean"] = float(error.mean())
        out["error_abs_mean"] = float(np.abs(error).mean())
        out["error_std"] = float(error.std())
        out["return_mean"] = float(self.target_return()[0][keep].mean())
        out["q_on_valid"] = float(self.q_min[keep].mean())
        if self.entropy_bonus is not None:
            out["alpha"] = self.alpha
            out["entropy_bonus_per_step"] = float(self.entropy_bonus.mean())
            # How much of the error the entropy term explains.
            out["error_mean_no_entropy"] = float((self.q_min - self.returns["plain"])[keep].mean())
        return out


SPREAD_FIELDS = [
    ("symlog_limit", "support limit +/-"),
    ("mean", "mean"),
    ("std", "within-state std"),
    ("symlog_std_across_states", "across-state std"),
    ("q05", "q05"),
    ("q95", "q95"),
    ("q05_q95_width", "q05-q95 width"),
    ("active_atoms", f"atoms above {ACTIVE_PROB:g}"),
    ("effective_atoms", "effective atoms"),
    ("edge_mass", "edge mass"),
    ("q_raw_mean", "Q raw mean"),
    ("q_raw_min", "Q raw min"),
    ("q_raw_max", "Q raw max"),
    ("critic_disagreement", "mean |Q1-Q2| raw"),
]

ERROR_FIELDS = [
    ("gamma", "gamma"),
    ("alpha", "alpha"),
    ("entropy_bonus_per_step", "entropy bonus/step"),
    ("q_on_valid", "mean min(Q1,Q2)"),
    ("return_mean", "mean realized G"),
    ("error_mean", "mean error Q - G"),
    ("error_mean_no_entropy", "  ... ignoring entropy"),
    ("error_abs_mean", "mean |Q - G|"),
    ("error_std", "error std"),
    ("valid_fraction", "fraction of steps used"),
]


def print_summary(runs: list[Run], min_observed: float) -> None:
    rows = [(run.label, run.summary(min_observed), run) for run in runs]
    width = max(len(label) for label, _, _ in rows) + 2

    def table(title, fields):
        present = [(key, pretty) for key, pretty in fields if any(key in s for _, s, _ in rows)]
        if not present:
            return
        print(f"\n{title}")
        print(f"{'metric':>24}" + "".join(f"{label:>{width}}" for label, _, _ in rows))
        for key, pretty in present:
            cells = "".join(
                f"{summary[key]:>{width}.4g}" if key in summary else f"{'-':>{width}}"
                for _, summary, _ in rows
            )
            print(f"{pretty:>24}{cells}")

    table("Categorical critic spread (symlog units unless stated otherwise)", SPREAD_FIELDS)
    print(f"{'episodes / samples':>24}" + "".join(f"{f'{r.episodes} / {r.samples}':>{width}}" for _, _, r in rows))
    table(f"Estimation error in reward units (steps with >= {min_observed:.0%} of G observed)", ERROR_FIELDS)

    for _, summary, run in rows:
        if summary["edge_mass"] > 1e-3:
            print(
                f"[WARN] {run.label}: {summary['edge_mass']:.3g} mean mass on the outermost atoms. "
                "The support is likely too narrow for this policy."
            )
        if run.returns is None:
            print(f"[WARN] {run.label}: no gamma in the log, so no estimation error. Re-record it.")
        elif run.bootstrap is None and not run.bootstrap_available:
            print(
                f"[WARN] {run.label}: the log predates bootstrap_q, so truncated episodes close at "
                "zero and the return sags toward every timeout. Re-record it."
            )
        elif run.bootstrap is None:
            print(f"[NOTE] {run.label}: --no-bootstrap, so truncated episodes close at zero.")
        elif summary.get("valid_fraction", 0.0) == 0.0:
            print(
                f"[WARN] {run.label}: every step draws more than {1 - min_observed:.0%} of its "
                "return from the bootstrapped value. Episodes are short relative to the "
                "1/(1-gamma) horizon; lower --min-observed or record longer episodes."
            )
        elif run.entropy_bonus is None:
            print(
                f"[WARN] {run.label}: no log_prob in the log, so the error ignores SAC's entropy "
                "bonus and will read biased. Re-record with the current play script."
            )
        elif run.deterministic:
            print(
                f"[NOTE] {run.label}: recorded deterministically. Q is defined under the stochastic "
                "policy, so some error is expected; --q_value_log_stochastic removes that mismatch."
            )


def env_mean(values: np.ndarray) -> np.ndarray:
    """Average over envs, ignoring masked-out steps instead of poisoning the whole step."""
    keep = ~np.isnan(values)
    counts = keep.sum(axis=1)
    return np.where(counts > 0, np.nansum(np.nan_to_num(values), axis=1) / np.maximum(counts, 1), np.nan)


def add_error_panels(axes, runs, colors, min_observed: float) -> None:
    """Q against the return it predicts, and the error between them, over play steps."""
    value_axis, error_axis = axes
    observed_axis = error_axis.twinx()

    for run, color in zip(runs, colors):
        if run.returns is None:
            continue
        target, kind = run.target_return()
        steps = np.arange(target.shape[0])
        keep = run.valid(min_observed)
        # Averaging over envs matches the other time-series panels.
        value_axis.plot(steps, run.q_min.mean(axis=1), color=color, label=f"{run.label}: min(Q1,Q2)")
        value_axis.plot(steps, target.mean(axis=1), color=color, linestyle="--", alpha=0.8,
                        label=f"{run.label}: realized G ({kind})")

        error = run.error()
        used = env_mean(np.where(keep, error, np.nan))
        error_axis.plot(steps, used, color=color, label=run.label)
        # Excluded steps are drawn faintly so the truncated tail stays visible but unmistakable.
        error_axis.plot(steps, env_mean(np.where(keep, np.nan, error)), color=color, alpha=0.18)
        observed_axis.plot(steps, run.returns["observed"].mean(axis=1), color=color, linestyle=":", alpha=0.5)
        if keep.any():
            error_axis.axhline(error[keep].mean(), color=color, linestyle="-.", alpha=0.5)

    value_axis.set_title("Critic estimate against the return it predicts (dashed)")
    value_axis.set_xlabel("play step")
    value_axis.set_ylabel("reward units")
    value_axis.legend(loc="upper left", fontsize=8)

    error_axis.set_title(
        "Estimation error Q - G — faint: mostly bootstrapped, so closer to a TD residual; "
        "dash-dot: mean over used steps"
    )
    error_axis.set_xlabel("play step")
    error_axis.set_ylabel("Q - G  (reward units)")
    error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    observed_axis.set_ylabel("fraction of G from real rewards (dotted)")
    observed_axis.set_ylim(0, 1.05)
    observed_axis.axhline(min_observed, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    error_axis.legend(loc="upper left", fontsize=8)


def build_figure(runs: list[Run], min_observed: float):
    import matplotlib.pyplot as plt

    with_error = any(run.returns is not None for run in runs)
    rows = 5 if with_error else 3
    figure, axes = plt.subplots(rows, 1, figsize=(11, 4 * rows), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    occupancy_axis = axes[0].twinx()
    if with_error:
        add_error_panels(axes[3:5], runs, colors, min_observed)

    for run, color in zip(runs, colors):
        mean_prob, used_fraction = run.occupancy()
        atoms = run.atoms.numpy()
        axes[0].plot(atoms, mean_prob, color=color, label=f"{run.label}: mean p")
        occupancy_axis.plot(atoms, used_fraction, color=color, linestyle="--", alpha=0.7)
        for edge in (-run.symlog_limit, run.symlog_limit):
            axes[0].axvline(edge, color=color, linestyle=":", alpha=0.6)

        steps = np.arange(run.probs.shape[0])
        center = run.per_step("mean")
        width = run.per_step("std")
        axes[1].plot(steps, center, color=color, label=run.label)
        axes[1].fill_between(steps, center - width, center + width, color=color, alpha=0.2)
        axes[1].fill_between(steps, run.per_step("q05"), run.per_step("q95"), color=color, alpha=0.08)
        for step in np.flatnonzero(run.done.sum(dim=1).numpy() > 0):
            axes[1].axvline(step, color=color, linestyle=":", alpha=0.35)

        axes[2].hist(run.flat("std"), bins=60, color=color, alpha=0.5, label=run.label, density=True)

    axes[0].set_title(f"Support occupancy — solid: mean probability; dashed: fraction of steps with p > {ACTIVE_PROB:g}")
    axes[0].set_xlabel(
        "atom value in symlog units,  Q = sign(x) * (exp(|x|) - 1)   (dotted verticals: each run's symlog limit)"
    )
    axes[0].set_ylabel("mean probability")
    occupancy_axis.set_ylabel(f"fraction of steps with p > {ACTIVE_PROB:g}")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].set_title("Distribution center per step — dark band: +/- 1 std, light band: q05-q95, dotted: episode end")
    axes[1].set_xlabel("play step")
    axes[1].set_ylabel("symlog units")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].set_title("Width of a single state-action's distribution, over all logged steps")
    axes[2].set_xlabel("within-state std in symlog units")
    axes[2].set_ylabel("density")
    axes[2].legend(loc="upper right", fontsize=8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+", help="One or more --q_value_log .npz files.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Plot label per log; defaults to filename.")
    parser.add_argument("--out", type=Path, default=None, help="PNG path. Defaults to <first log>_spread.png.")
    parser.add_argument("--no-show", action="store_true", help="Only write the PNG, do not open a window.")
    parser.add_argument(
        "--drop-first",
        type=int,
        default=0,
        help="Skip this many initial steps, e.g. to exclude a settling transient after reset.",
    )
    parser.add_argument(
        "--min-observed",
        type=float,
        default=0.9,
        help=(
            "Estimation-error statistics only use steps where at least this fraction of the "
            "return comes from real rewards rather than the bootstrapped critic value."
        ),
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Close truncated episodes at zero instead of V(s'), to see the truncation bias.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        parser.error(f"--labels has {len(args.labels)} entries but {len(args.logs)} logs were given.")
    labels = args.labels or [None] * len(args.logs)

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Run(path, label, args.drop_first, not args.no_bootstrap) for path, label in zip(args.logs, labels)]
    print_summary(runs, args.min_observed)

    figure = build_figure(runs, args.min_observed)
    out = args.out or args.logs[0].with_name(f"{args.logs[0].stem}_spread.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f"\n[RESULT] Spread figure: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
