import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import slip_plots


class TestPolarSlipPlots(unittest.TestCase):
    def _dataframe(self):
        rows = []
        directions = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        for foot, (x, y) in zip(slip_plots.FEET, directions):
            for sample, (contact, angle) in enumerate(((1, 20.0), (1, 25.0), (0, 80.0))):
                rows.append(
                    {
                        "sim_time_s": sample * 0.0025,
                        "foot": foot,
                        "contact": contact,
                        "contact_id": 0 if contact else -1,
                        "friction_direction_footprint_x": x,
                        "friction_direction_footprint_y": y,
                        "angle_deg": angle,
                        "angle_dyn_deg": 15.0,
                        "angle_static_deg": 35.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_contact_filter_and_footprint_azimuth(self):
        theta, radius, dynamic, static = slip_plots._polar_contact_data(self._dataframe(), slip_plots.FEET)

        self.assertEqual(theta.size, 8)
        np.testing.assert_allclose(np.unique(np.round(theta, 8)), [-np.pi / 2, 0.0, np.pi / 2, np.pi])
        np.testing.assert_allclose(np.rad2deg(radius), [20.0, 25.0] * 4)
        np.testing.assert_allclose(np.rad2deg(dynamic), 15.0)
        np.testing.assert_allclose(np.rad2deg(static), 35.0)

    def test_contact_timing_stats(self):
        stats = slip_plots._contact_timing_stats(self._dataframe(), slip_plots.FEET)

        self.assertAlmostEqual(stats["FL"]["contact_time_pct_of_race"], 100.0 * 2.0 / 3.0)
        self.assertAlmostEqual(stats["FL"]["mean_contact_duration_s"], 0.005)
        self.assertEqual(stats["FL"]["contact_episode_count"], 1)
        self.assertAlmostEqual(stats["all_feet"]["contact_time_pct_of_race"], 100.0 * 2.0 / 3.0)
        self.assertAlmostEqual(stats["all_feet"]["mean_contact_duration_s"], 0.005)
        self.assertEqual(stats["all_feet"]["contact_episode_count"], 4)

    def test_tracks_are_split_by_foot_and_contact_episode(self):
        tracks = slip_plots._polar_contact_tracks(self._dataframe(), slip_plots.FEET)

        self.assertEqual(len(tracks), 4)
        self.assertTrue(all(theta.size == radius.size == 2 for theta, radius in tracks))

    def test_forward_is_at_top_and_samples_are_above_friction_rings(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        df = self._dataframe()
        theta, radius, dynamic, static = slip_plots._polar_contact_data(df, slip_plots.FEET)
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="polar")
        scatter = slip_plots._draw_polar_density(
            ax,
            theta,
            radius,
            dynamic,
            static,
            tracks=slip_plots._polar_contact_tracks(df, slip_plots.FEET),
            title="test",
        )

        self.assertEqual(ax.get_theta_offset(), np.pi / 2.0)
        self.assertGreater(scatter.get_zorder(), max(line.get_zorder() for line in ax.lines))

    def test_requested_colormap_is_used(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        df = self._dataframe()
        theta, radius, dynamic, static = slip_plots._polar_contact_data(df, slip_plots.FEET)
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="polar")
        scatter = slip_plots._draw_polar_density(
            ax, theta, radius, dynamic, static, title="test", cmap="cividis"
        )

        self.assertEqual(scatter.cmap.name, "cividis")

    def test_tracks_use_one_faint_color_independent_of_density(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure

        df = self._dataframe()
        theta, radius, dynamic, static = slip_plots._polar_contact_data(df, slip_plots.FEET)
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="polar")
        slip_plots._draw_polar_density(
            ax,
            theta,
            radius,
            dynamic,
            static,
            tracks=slip_plots._polar_contact_tracks(df, slip_plots.FEET),
            title="test",
        )

        track_collection = next(item for item in ax.collections if isinstance(item, LineCollection))
        self.assertEqual(len(track_collection.get_colors()), 1)
        self.assertEqual(track_collection.get_alpha(), slip_plots.POLAR_TRACK_ALPHA)

    def test_saves_one_plot_per_foot_and_one_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "slip.csv"
            self._dataframe().to_csv(csv_path, index=False)

            paths = slip_plots.save_polar_slip_figures(str(csv_path), str(directory / "slip"), dpi=30)

            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths))
            self.assertEqual(Path(paths[-1]).name, "slip_polar_all_feet.png")

    def test_straight_velocity_plot_labels_race_finish_time(self):
        from unittest.mock import patch

        from matplotlib.figure import Figure

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "slip.csv"
            pd.DataFrame(
                {
                    "sim_time_s": (0.0, 0.5, 1.0),
                    "base_velocity_straight_mps": (0.0, 1.0, 0.5),
                }
            ).to_csv(csv_path, index=False)
            slip_plots.save_plot_metadata(
                str(csv_path),
                task="race",
                title_extra=None,
                finish_time_s=1.0,
            )
            captured = {}

            def capture_figure(figure, *_args, **_kwargs):
                captured["figure"] = figure

            with patch.object(Figure, "savefig", autospec=True, side_effect=capture_figure):
                slip_plots.save_straight_velocity_figure(
                    str(csv_path),
                    str(directory / "velocity.png"),
                )

            ax = captured["figure"].axes[0]
            self.assertIn("Race finish: 1.000 s", [item.get_text() for item in ax.texts])
            self.assertTrue(
                any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), (1.0, 1.0)) for line in ax.lines)
            )


if __name__ == "__main__":
    unittest.main()
