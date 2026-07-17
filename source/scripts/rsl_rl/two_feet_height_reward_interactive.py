"""Interactive notebook view of the Solo12 two-feet height reward.

From a notebook opened at the IsaacLab root, run:

    %run source/scripts/rsl_rl/two_feet_height_reward_interactive.py
"""

from __future__ import annotations

import numpy as np


def two_feet_height_reward(
    average_foot_height_m: np.ndarray, alpha: float, threshold_m: float
) -> np.ndarray:
    """Evaluate the normalized height kernel used by ``Solo12Env``."""
    height = np.asarray(average_foot_height_m, dtype=float)
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")
    if threshold_m < 0.0:
        raise ValueError(f"threshold_m must be non-negative, got {threshold_m}.")
    below_threshold = np.exp(-alpha * np.square(threshold_m - height))
    return np.where(height >= threshold_m, 1.0, below_threshold)


def launch_interactive_plot(alpha: float = 8.0, threshold_m: float = 0.6) -> None:
    """Display live alpha/threshold controls in a Jupyter notebook."""
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display
    except ImportError as exc:
        raise RuntimeError(
            "This interactive plot needs matplotlib, IPython, and ipywidgets in the notebook kernel."
        ) from exc

    alpha_slider = widgets.FloatSlider(
        value=float(alpha),
        min=0.0,
        max=30.0,
        step=0.1,
        description="alpha",
        continuous_update=True,
        readout_format=".1f",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )
    threshold_slider = widgets.FloatSlider(
        value=float(threshold_m),
        min=0.0,
        max=1.0,
        step=0.01,
        description="threshold [m]",
        continuous_update=True,
        readout_format=".2f",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )

    def draw(alpha: float, threshold_m: float) -> None:
        heights = np.linspace(0.0, 1.2, 1201)
        reward = two_feet_height_reward(heights, alpha, threshold_m)

        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        ax.plot(heights, reward, color="tab:blue", linewidth=2.5, label="normalized height reward")
        ax.axvline(
            threshold_m,
            color="tab:orange",
            linestyle="--",
            linewidth=1.6,
            label=rf"$h_{{des}}={threshold_m:.2f}$ m",
        )
        ax.fill_between(heights, 0.0, reward, where=heights < threshold_m, color="tab:blue", alpha=0.10)
        ax.set_xlim(0.0, 1.2)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xlabel(r"average airborne-foot height $h_a$ [m]")
        ax.set_ylabel("normalized reward")
        ax.set_title(
            r"$r(h_a)=\exp[-\alpha(h_{des}-h_a)^2]$ for $h_a<h_{des}$; otherwise $r=1$"
            f"\nalpha={alpha:.1f}, h_des={threshold_m:.2f} m"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show()

    output = widgets.interactive_output(
        draw,
        {"alpha": alpha_slider, "threshold_m": threshold_slider},
    )
    display(widgets.VBox((alpha_slider, threshold_slider, output)))


if __name__ == "__main__":
    launch_interactive_plot()
