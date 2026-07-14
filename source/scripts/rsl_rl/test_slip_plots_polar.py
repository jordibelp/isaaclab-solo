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
            for contact, angle in ((1, 20.0), (1, 25.0), (0, 80.0)):
                rows.append(
                    {
                        "foot": foot,
                        "contact": contact,
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

    def test_saves_one_plot_per_foot_and_one_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "slip.csv"
            self._dataframe().to_csv(csv_path, index=False)

            paths = slip_plots.save_polar_slip_figures(str(csv_path), str(directory / "slip"), dpi=30)

            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths))
            self.assertEqual(Path(paths[-1]).name, "slip_polar_all_feet.png")


if __name__ == "__main__":
    unittest.main()
