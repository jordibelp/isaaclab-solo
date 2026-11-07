#!/usr/bin/env python3
"""Play Borinot go-to (live XYZ goal and deadline) weights in MuJoCo (sim-to-sim).

Runs source/scripts/rsl_rl/play_borinot_goto.py unchanged on the MuJoCo backend:
same flags, Hydra overrides, control panel and reports. See mujoco/BORINOT_SIM2SIM.md.
"""

from borinot_sim2sim.app import run_play_script

if __name__ == "__main__":
    run_play_script("play_borinot_goto.py", default_substeps=1)
