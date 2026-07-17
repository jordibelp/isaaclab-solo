import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from two_feet_height_reward_interactive import two_feet_height_reward


def test_two_feet_height_reward_matches_environment_kernel():
    heights = np.array((0.4, 0.5, 0.6, 0.8))

    actual = two_feet_height_reward(heights, alpha=8.0, threshold_m=0.6)

    expected = np.array((np.exp(-8.0 * 0.2**2), np.exp(-8.0 * 0.1**2), 1.0, 1.0))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(("alpha", "threshold"), ((-1.0, 0.6), (8.0, -0.1)))
def test_two_feet_height_reward_rejects_negative_parameters(alpha, threshold):
    with pytest.raises(ValueError):
        two_feet_height_reward(np.array((0.5,)), alpha=alpha, threshold_m=threshold)
