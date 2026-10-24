# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility imports; the shared implementation lives outside this task."""
from isaaclab_tasks.controllers.borinot_indi import (  # noqa: F401
    INDI_TILT_METHODS,
    Butterworth,
    IndiController,
    IndiControllerCfg,
    RotorAllocator,
    attitude_acceleration,
    attitude_error_vectors,
    box_qp,
    quaternion_exp,
    quaternion_from_rotation,
    quaternion_log,
    quaternion_multiply,
    rotation_from_quat,
    thrust_attitude,
)
