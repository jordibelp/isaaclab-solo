# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot how much of a categorical SAC critic's support is actually used.

Input is one or more ``.npz`` files written by ``play_direct_0325.py --q_value_log``,
from a checkpoint trained with ``agent.distributional_critic_ce=True``.

Everything is reported in **symlog units**: the ``x`` in ``Q = sign(x) * (exp(|x|) - 1)``,
so the axes are directly comparable to ``agent.critic.distributional_symlog_limit``. Runs
with different limits can therefore be overlaid, and the dotted verticals mark each run's
own limit.

    # One run:
    ./isaaclab.sh -p source/scripts/rsl_rl/q_spread_plots.py q_log.npz

    # Compare checkpoints, write a PNG and skip the interactive window:
    ./isaaclab.sh -p source/scripts/rsl_rl/q_spread_plots.py a.npz b.npz \
        --labels "limit 5.0" "limit 4.0" --out spread.png --no-show

The twin critics are mixed as ``(p1 + p2) / 2`` before any statistic is computed, matching
the ``CriticDist/*`` scalars logged during training. Critic disagreement therefore widens
the reported spread, exactly as it widens the range of values the pair can encode.

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


class Run:
    """One ``--q_value_log`` file, reduced to per-step statistics in symlog units."""

    def __init__(self, path: Path, label: str | None, drop_first: int) -> None:
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

    def flat(self, key: str) -> np.ndarray:
        return self.stats[key].reshape(-1).numpy()

    def per_step(self, key: str) -> np.ndarray:
        """Average over envs, keeping the time axis."""
        return self.stats[key].mean(dim=1).numpy()

    def occupancy(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean probability per atom, and the fraction of samples where that atom is used."""
        flat = self.probs.reshape(-1, self.probs.shape[-1])
        return flat.mean(dim=0).numpy(), (flat > ACTIVE_PROB).float().mean(dim=0).numpy()

    def summary(self) -> dict[str, float]:
        out = {key: float(self.stats[key].mean()) for key in SUMMARY_KEYS}
        out["symlog_limit"] = self.symlog_limit
        out["symlog_std_across_states"] = float(self.stats["mean"].std(unbiased=False))
        out["q_raw_mean"] = float(self.q.mean())
        out["q_raw_min"] = float(self.q.min())
        out["q_raw_max"] = float(self.q.max())
        out["critic_disagreement"] = float((self.q[..., 0] - self.q[..., 1]).abs().mean())
        return out


def print_summary(runs: list[Run]) -> None:
    rows = [(run.label, run.summary(), run) for run in runs]
    width = max(len(label) for label, _, _ in rows) + 2
    fields = [
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
    print("\nCategorical critic spread (symlog units unless stated otherwise)")
    print(f"{'metric':>22}" + "".join(f"{label:>{width}}" for label, _, _ in rows))
    for key, pretty in fields:
        print(f"{pretty:>22}" + "".join(f"{summary[key]:>{width}.4g}" for _, summary, _ in rows))
    print(f"{'episodes / samples':>22}" + "".join(f"{f'{r.episodes} / {r.samples}':>{width}}" for _, _, r in rows))
    for _, summary, run in rows:
        if summary["edge_mass"] > 1e-3:
            print(
                f"[WARN] {run.label}: {summary['edge_mass']:.3g} mean mass on the outermost atoms. "
                "The support is likely too narrow for this policy."
            )


def build_figure(runs: list[Run]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    occupancy_axis = axes[0].twinx()

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
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        parser.error(f"--labels has {len(args.labels)} entries but {len(args.logs)} logs were given.")
    labels = args.labels or [None] * len(args.logs)

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Run(path, label, args.drop_first) for path, label in zip(args.logs, labels)]
    print_summary(runs)

    figure = build_figure(runs)
    out = args.out or args.logs[0].with_name(f"{args.logs[0].stem}_spread.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f"\n[RESULT] Spread figure: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
