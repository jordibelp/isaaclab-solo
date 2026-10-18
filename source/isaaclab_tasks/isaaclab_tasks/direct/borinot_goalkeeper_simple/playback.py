# SPDX-License-Identifier: BSD-3-Clause
"""Play-only scenery. The registered training environment is never modified."""

from isaaclab_tasks.direct.borinot_goalkeeper.playback import GoalkeeperPlaybackMixin

from .borinot_goalkeeper_simple_env import BorinotGoalkeeperSimpleEnv


class BorinotGoalkeeperSimplePlayEnv(GoalkeeperPlaybackMixin, BorinotGoalkeeperSimpleEnv):
    """Simplified-reward task with the original recent-ball playback."""
