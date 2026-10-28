# SPDX-License-Identifier: BSD-3-Clause
"""Portable inertial settings transcribed from borinot_description's inference URDF.

Values refer to borinot_flying_arm_2.urdf at description revision 76708fb.
The inference calf already includes its load: 115 g is not a bare 15 g shaft.
"""
from isaaclab.utils import configclass


@configclass
class BorinotInertialPropertiesCfg:
    include_motor_masses: bool = True
    use_inference_leg_inertias: bool = True

    # Ideal point masses, NOT the URDF rotor links' intrinsic inertia tensors.
    motor_mass_kg: float = 0.005
    motor_mass_positions_b: tuple = (
        (0.0, -0.185, 0.05), (0.0, 0.185, 0.05),
        (0.1602146997, 0.0925, 0.05), (-0.1602146997, -0.0925, 0.05),
        (0.1602146997, -0.0925, 0.05), (-0.1602146997, 0.0925, 0.05),
    )

    # Upper link then calf. CoM positions and full own-CoM tensors use each
    # link's local axes, not its principal axes or the base frame.
    leg_masses_kg: tuple = (0.14853845, 0.115)
    leg_com_positions_l: tuple = ((0.00001377, -0.078707, -0.01935853), (0.0, -0.1149, -0.021))
    leg_inertias_l_kg_m2: tuple = (
        ((0.00041107, 0.0, 0.0), (0.0, 0.00003024, -0.00004671), (0.0, -0.00004671, 0.00041193)),
        ((0.00017554, 0.0, 0.0), (0.0, 0.00001972, 0.0), (0.0, 0.0, 0.00017938)),
    )
