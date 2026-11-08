# SPDX-License-Identifier: BSD-3-Clause
"""Portable inertial settings transcribed from borinot_description's inference URDF.

Values refer to the short arm at description revision 6b3e09ce.
The 127 g lower link already includes its 104 g nut. The versioned USD base
already includes all motors and propellers; never add the legacy motor points.
"""
from isaaclab.utils import configclass


@configclass
class BorinotInertialPropertiesCfg:
    include_motor_masses: bool = False
    use_inference_leg_inertias: bool = True

    # Historical opt-in for the OLD borinotFlat.usd only. Rejected on the new
    # inclusive-base asset, including when an older saved config enables it.
    motor_mass_kg: float = 0.005
    motor_mass_positions_b: tuple = (
        (0.0, -0.185, 0.05), (0.0, 0.185, 0.05),
        (0.1602146997, 0.0925, 0.05), (-0.1602146997, -0.0925, 0.05),
        (0.1602146997, -0.0925, 0.05), (-0.1602146997, 0.0925, 0.05),
    )

    # Upper link then calf. CoM positions and full own-CoM tensors use each
    # link's local axes, not its principal axes or the base frame.
    leg_masses_kg: tuple = (0.14853845, 0.127)
    leg_com_positions_l: tuple = ((0.00001377, -0.078707, -0.01935853), (0.0, -0.111706, -0.015478))
    leg_inertias_l_kg_m2: tuple = (
        ((0.00041107, 0.0, 0.0), (0.0, 0.00003024, -0.00004671), (0.0, -0.00004671, 0.00041193)),
        ((0.00018274, 0.0, 0.0), (0.0, 0.000019135, -0.000018119), (0.0, -0.000018119, 0.00019053)),
    )
