# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass


@configclass
class BorinotFlightRandomizationCfg:
    """Physical plant uncertainty shared by the Borinot environments."""

    motor_time_constant_range_s = (0.035, 0.045)  # per rotor, once at startup; None uses nominal lag
    thrust_shared_gain_delta = 0.10  # uniform signed additive deviation from 1, per vehicle
    thrust_rotor_gain_delta = 0.03  # additional independent signed deviation, per rotor
    km_gain_delta = 0.10  # uniform signed relative deviation of km, shared per vehicle

    wind_enabled = False  # independent of the vertical-push curriculum and startup event randomization
    wind_steady_max_m_s = 2.0  # horizontal world-frame speed, held for an episode
    wind_gust_max_m_s = 2.0  # extra horizontal wind velocity during a pulse
    wind_drag_n_per_m_s_squared = 0.08  # isotropic drag: coefficient * |air-relative velocity| * vector
    external_moment_max_nm = 0.04  # norm bound for a world-frame moment about the base CoM
    gust_interval_range_s = (0.8, 1.6)  # start-to-start; first pulse starts at a random phase
    gust_duration_range_s = (0.2, 0.5)
