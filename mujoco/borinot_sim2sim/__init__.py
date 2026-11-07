"""Run the Borinot Isaac Lab tasks on a MuJoCo physics backend.

The real task code is reused: environment configs, observations, rewards, resets,
controllers, play-mode goal/command logic and the omni.ui control panels. Only the
physics engine, the renderer and a thin layer of Isaac Lab runtime APIs are replaced.

Call :func:`borinot_sim2sim.bootstrap.bootstrap` before importing any ``isaaclab`` or
``isaaclab_tasks`` module. See ``mujoco/BORINOT_SIM2SIM.md``.
"""
