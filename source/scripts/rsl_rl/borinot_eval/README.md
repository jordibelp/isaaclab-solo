# Borinot Eval

Compare frozen checkpoints without training. The evaluator runs one checkpoint
and condition at a time in a fresh headless Isaac Sim process. Each trial counts
once, even if one robot fails much earlier than another.

## Desktop launcher

From an activated IsaacLab environment, run the GTK launcher with the Ubuntu
system Python (with GTK 3/PyGObject installed):

```bash
/usr/bin/python3 source/scripts/rsl_rl/borinot_eval_gui.py
```

1. Choose the task and its fixed stage. Set a goal and deadline, a velocity
   command, or a goalkeeper shot stage.
2. Add the local checkpoints. Match each checkpoint's **training** controller,
   policy rate, agent and observation settings. Use the training config rather
   than the filename. If you have them, load a saved recipe instead. The config
   buttons let you set additional shared environment and per-policy settings.
   For quick changes, paste space-separated settings into **Hydra env overrides**,
   for example `env.reset_robot_on_tilt_threshold=False`. The line is saved
   as `hydra_overrides` in the recipe and applies to every policy and condition.
   It takes priority over the environment JSON. Dedicated controls for the
   scenario, policy rate, stage, delay, and randomizations still take priority.
3. Select each condition to set its delay and perturbations. Fixed delay is
   written in milliseconds (for example `20`). A range is written `min:max`
   (for example `0:20`). With no boxes checked, you get nominal physics with
   only the selected delay. Use `0` and `20` for the baseline and stress test.
   **+ Condition** copies the selected delay and checkboxes into a new, separately
   named condition. Change the new copy without affecting the original.
4. Run. **Open report** shows the metrics and side-by-side pose replay. The
   panels share a clock, camera, scenario, play/pause and scrubber. They freeze
   when their respective episodes end. It is a schematic rendered from measured
   simulator poses, not live Isaac Sim windows or a camera video.

## CLI and saved recipes

```bash
./isaaclab.sh -p source/scripts/rsl_rl/eval_borinot.py --recipe comparison.json
```

A minimal GoTo MinTime recipe (edit the checkpoint path and training settings):

```json
{
  "task": "Borinot-goto-minTime",
  "stage": 5,
  "seed": 42,
  "episodes": 128,
  "num_envs": 64,
  "replay_count": 4,
  "scenario": {
    "command": [3.0, 0.0, 1.5],
    "deadline_s": 2.4,
    "sample_box": {
      "center": [0.0, 0.0, 1.5],
      "half_extents": [7.0, 1.0, 0.5],
      "deadline_s": [2.2, 2.6]
    }
  },
  "env": {
    "physics_hz": 200,
    "obs_include_base_lin_acc": true,
    "obs_include_base_lin_acc_steps": 2
  },
  "policies": [
    {"label": "Direct thrust", "checkpoint": "/path/to/trusted.pt",
     "control_mode": "thrust_and_torque_control", "policy_hz": 50}
  ],
  "conditions": [{"delay_ms": 0}, {"delay_ms": 20}]
}
```

- The GUI and CLI read the same format. Use trusted local checkpoint files only.
- **GoTo / GoTo MinTime:** first trial uses the exact specified environment-local
  XYZ. An optional box supplies the other targets; without it, the target and
  deadline repeat. Requested fixed deadlines round up to a tick shared by all
  selected policy rates. Sampled deadlines are chosen from shared ticks inside
  the range. These are explicit evaluation scenarios; the task's native
  deadline sampler is not used. Set success thresholds to the training values
  if they differ from the current task defaults.
- **CmdVel:** `"scenario":{"command":[vx,vy,vz,yaw_rate]}` holds the command
  for every trial. Horizontal velocity is in heading axes; vertical and yaw
  rates are in world axes. **Legacy** `Borinot-Direct-v0` uses
  `[vx,vy,yaw_rate]` in body axes and derives its policy rate from physics
  dt and decimation. Omit `policy_hz` and `control_mode` for legacy.
- **Goalkeeper / Goalkeeper Simple:** omit `scenario`. Both use native
  shots, resets and outcomes at the fixed shot stage. They share the evaluation
  seed, not a scripted shot timeline. Once policies diverge, later shots can
  differ, so shot results are not reported as paired trials.
- **Independent settings:** `env` sets common task settings. Each policy can
  override them with its own `env`, select the agent registry key with `agent`
  (`rsl_rl_cfg_entry_point`, `rsl_rl_sac_cfg_entry_point`, or
  `rsl_rl_sac_mse_cfg_entry_point` where registered), and set `runner` options
  for its network. Checkpoint loading is strict and skips optimizer state.
- **Perturbations:** `randomizations` is a list per condition. The selectable
  keys are in [spec.py](spec.py). Set numeric ranges in the environment config,
  for example `env.flight_randomization.wind_steady_max_m_s` in the shared JSON
  object. When all choices are off, the evaluator overrides existing defaults
  for physical events, motor variation, reset jitter and disturbances. Nominal
  motor lag stays in place. Vertical pushes use the selected stage's configured
  force, which can be zero. The launch seed and scenario plan are saved.
  With randomization on, matching seeds do not guarantee matching later gusts,
  pushes or resets when episode timing differs.
- **Delay:** 20 ms at 200 Hz is four physics ticks. It is a full policy period at
  50 Hz, but only 0.4 periods at 20 Hz. This is a delay of the policy command
  entering the low-level controller; it is not measurement delay. An explicit
  sweep value or range endpoint must be on the physics grid.

## What the numbers mean

- GoTo success is the task's actual settled-success test. Plain GoTo scores it
  at the deadline; MinTime can end sooner when settled. First entry inside the
  tolerance is reported separately. Time-to-success statistics only count
  successful trials. The paired table also shows time differences among trials
  that succeed under both conditions; compare success rates first.
- Velocity tasks report per-episode mean and RMS tracking error and yaw-rate
  error. CmdVel also reports the training XY tracking kernel averaged over the
  full episode horizon, counting time lost to failures as zero score.
- Keeper metrics separate saves, racket contact, and robot failures. Native
  save rate divides saves by resolved shots; a second rate includes attempts
  abandoned by a robot failure. A trial with no resolved shots does not create a
  zero-denominator percentage.
- Reports include simulator power estimates integrated by time, not measured
  battery energy. Reward can depend on the policy rate and training config; use
  the task metrics rather than treating raw return as a universal score.
- If only one checkpoint was trained per controller, these results compare those
  checkpoint recipes, not independently replicated controller populations.
  Repeat a fixed scenario for debugging; vary it for meaningful robustness data.

## Outputs and W&B

By default results stay local under `logs/borinot_eval/`. `--output` selects a
new directory. Keep `recipe.json`, `scenarios.json`, `summary.csv`,
`episodes.csv`, and `report.html` together. Each worker folder includes its
effective configs, log, recorded poses, and task source. The top-level
provenance file includes the exact checkpoint SHA-256 and repo revision.

Use `--wandb online` (or the GUI checkbox) to publish summaries and all trial
rows. The evaluator requires an existing **private** `borinot-eval` project in
the selected entity. There is one W&B run per checkpoint and condition. Local
results are written first; do not re-run the simulation to retry an upload:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/eval_borinot.py --from-results <output-dir> --wandb online
```

Use `--wandb offline` for a local W&B log. Complete results and the schematic
replay are available without an account or internet connection.
