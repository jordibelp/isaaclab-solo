# Borinot command delays and goalkeeper evaluation

## What is available

All four Borinot tasks support `env.actuator_delay` and `env.actuator_delay_range`:

| Task | Delay support | Hit/save evaluation |
|---|---|---|
| `Borinot-Goalkeeper-Direct-v0` | All control modes; PPO and SAC | New goalkeeper evaluator |
| `Borinot-Goalkeeper-Simple-Direct-v0` | All control modes; PPO and SAC | Same evaluator, with `--task` |
| `Borinot-cmd-vel` | All control modes; PPO and SAC | Use its existing tracking playback, not a goalkeeper score |
| `Borinot-Direct-v0` | Legacy rotor/torque and rotor/joint-angle interfaces | No goalkeeper shots in this task |

Defaults are `actuator_delay = 0.0` and `actuator_delay_range = None`.
Existing action and observation sizes do not change. The delay is not added to
the actor or critic inputs. Existing checkpoint requirements still apply:
match its algorithm, control mode, action units, observations, and physical settings.
Do not switch an INDI checkpoint into direct thrust/torque mode.

## Where the delay acts

We delay the policy command **before** INDI, PI, joint PD/PID, and the motor
model. Each controller still runs every physics step and reads current state.
Thus an INDI reference can be old while its stabilizing feedback remains current.
Both rotor and arm commands receive the same delay within one environment.

This represents policy computation/communication latency. It does **not** model
dead time between an onboard controller and a physical motor, delayed sensors,
or a delayed feedback loop. Those would be different experiments.
The existing `motor_time_constant_s` remains a separate first-order motor lag;
the new command delay does not replace or disable it.

Previous-action observations and raw policy-output limit costs retain the
newest policy output. `Desired_Thrust/*` and `Desired_Torque/*` also describe
the newest direct policy request, not its later arrival. Physical command-rate
and power terms use the commands actually reaching the control/plant path.

The [quadrotor action-space paper](https://arxiv.org/html/2202.10796v1) motivates
a communication-delay comparison. Our INDI versus direct-control comparison
is a new experiment, not a reproduction of its CTBR versus SRT benchmark.
Whether INDI wins remains a hypothesis to test.

We queue commands every physics step. Delay values must be exact multiples of
the physics timestep. At 200 Hz, valid values include 0, 5, 10, 15, and 20 ms.
At 300 Hz, 10 ms is three steps. Off-grid values fail with an error; we do not
silently round. For the legacy task, the grid comes from `env.sim.dt` (120 Hz
by default), not a separate frequency setting.

- A fixed delay stays the same across robot episodes.
- A range samples an integer delay uniformly from all included physics-step
  values, independently per environment, on every physical robot reset.
- A sampled value stays fixed during that episode. It is not per-step jitter.
- Ball-only relaunches keep both the sampled delay and queued commands.
- A physical reset clears only that environment's queue and fills it with zero
  actions. In the modern tasks, these are hover/home/zero-effort references.
  The legacy task retains its own zero-action mapping, including its thrust
  mode and configured joint-torque bias.
- Setting both a nonzero fixed delay and a range is an error.
- The modern tasks log `Episode/actuator_delay_s`: the mean realized delay of
  the completed robot episodes in each reset batch, before resampling.

IsaacLab checks config types. Use `0.0`, not `0`, for a float override.
Quote range overrides as shown below. These ranges are **seconds**, unlike
Solo12's `actuation_delay_range`, whose values are step counts.

## Local environment setup

Run from `/home/jordibelp/IsaacLab` after:

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

The following commands are templates. Replace checkpoint paths and retain all
settings needed to match the corresponding training run. In particular, the
evaluator restores weights and normalizers, **not** the saved environment YAML.

## A. Playback with a fixed 20 ms command delay

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --checkpoint /absolute/path/indi_model.pt env.control_mode=indi env.actuator_delay=0.02
```

For direct control, use its own checkpoint and
`env.control_mode=thrust_and_torque_control`. For the simple task, add
`--task Borinot-Goalkeeper-Simple-Direct-v0`. For SAC weights, add
`--agent rsl_rl_sac_cfg_entry_point`.

## B. Evaluate a checkpoint over N complete robot episodes

```bash
./isaaclab.sh -p source/scripts/rsl_rl/eval_borinot_goalkeeper.py --checkpoint /absolute/path/indi_model.pt --num_episodes 512 --num_envs 256 --seed 7 --actuator_delays 0.0 0.01 0.02 0.04 --label indi-no-delay-training --out_dir /absolute/path/borinot-delay-results env.control_mode=indi env.shot_level=3
```

The evaluator is headless by default. It does not train or upload to W&B.
For SAC, add `--agent rsl_rl_sac_cfg_entry_point`. For the simple task, add its
`--task` as above. Omit `--actuator_delays` to use the environment's fixed or
random-range delay setting instead. Mixing the sweep flag with environment
delay overrides is rejected.

Difficulty is fixed: `env.curriculum=True` is rejected. Set the **same explicit
`env.shot_level` for every compared checkpoint**. Without that override, the
evaluator selects `best_model_curriculum_idx`, then `curriculum_global_idx`
from checkpoint metadata, or falls back to the current config default. That
convenience is not a guarantee of matched difficulty between two checkpoints.

N means complete **robot episodes**, not N shots. Each environment contributes
a fixed quota of its first episodes. The quotas sum to exactly N, including
when N is not divisible by the number of environments. Short failed episodes
cannot keep filling the sample while slower environments are still running.
Every terminal step is included. Recurrent policies reset their state on done.

### Scores and output files

Each invocation writes one `<label>_<timestamp>.json` and appends one row per
delay to `summary.csv` in the output folder. The JSON includes:

- `save_fraction`: total task saves divided by total resolved shots across the
  counted episodes. This uses the same save outcome as the curriculum's
  `Curriculum/stage_pure_save_fraction`; it is not merely racket contact.
  The original task uses its configured save/recovery rules. The simple task
  uses its own no-goal outcome. Do not compare scores across these two tasks.
- `racket_contact_fraction`: resolved shots with a paddle touch in the original
  task; shots whose first robot contact was the racket in the simple task.
- Robot failure fraction, failures per simulated minute, episode duration,
  return, shot counts, and each outcome count. Read these alongside saves:
  a policy that saves a shot and then crashes is not a robust keeper.
- `save_fraction_counting_reset_drops`: also counts unresolved shots abandoned
  by robot-failure resets as unsuccessful attempts. Unresolved shots at a
  time limit are reported separately. The main score intentionally retains
  the task's resolved-shot denominator for comparison with the curriculum.
- A 95% percentile bootstrap interval, resampling whole robot episodes rather
  than pretending shots within one episode are independent. With no resolved
  shots, save/contact scores and their intervals are unavailable (`null`).
- Every counted episode's environment index, episode index, steps, return,
  shots, saves, contact count, termination status, abandoned-shot status, and
  realized delay in seconds.
- Checkpoint filename, full path and SHA-256; command and seed; effective
  environment/agent settings; source revision/status; and source text with
  hashes for the evaluator, task/config inheritance, and shared controllers.

No optimizer is restored, and SAC evaluation does not allocate the training-
sized replay buffer. Model forwards are deterministic, but GPU physics can
depend on tiny contact-rounding differences and earlier rollouts. Keep the
seed, environment count, device, settings, and sweep order fixed. For the
strongest repeatability, use **one fixed delay per fresh process** and repeat
the comparison across seeds. A common seed does not guarantee identical shots
across different policies: resets consume random numbers at different times.
Bootstrap intervals describe episode variation for one policy/seed, not the
variation across independently trained policies.

## C. Train with delay randomization

Add these overrides to the usual training command for either modern interface:

```bash
env.actuator_delay=0.0 'env.actuator_delay_range=[0.0,0.04]'
```

For example, from the activated local environment:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Direct-v0 --headless env.control_mode=indi env.actuator_delay=0.0 'env.actuator_delay_range=[0.0,0.04]'
```

That draws 0, 5, ..., 40 ms at 200 Hz. Use the same range for the direct-control
condition. A fixed training delay uses `env.actuator_delay=0.02` with no range.
To test a randomized-trained policy at one fixed delay, explicitly clear the
range (`env.actuator_delay_range=null`) if it is present in your command/config.
For the legacy 120 Hz task, choose compatible endpoints, such as `[0.0,0.05]`.

## Suggested comparison

Train four conditions: INDI and direct thrust/torque, each with no added delay
and with the same randomized delay range. Evaluate all four on the same fixed
test-delay grid and shot level, across several training and evaluation seeds.
Match motor lag, physics/policy clocks, plant randomization, observations,
episode/reset rules, training budget, and checkpoint selection. Keep the
existing mode-specific action units and architectures explicit. Test both
inside and beyond the training delay range. Do not infer superiority from
the short implementation-smoke checkpoints.

## Verification

CPU coverage includes queue timing, range sampling, selected resets, controller
feedback timing, all shared interfaces, legacy mappings, exact evaluation
quotas, terminal steps, zero-shot results, bootstrap statistics, and CSV output.
Live probes check arrival time, continued INDI feedback, random-range resets,
and finite rollouts. Training/public-playback and checkpoint-evaluation smoke
results are integration checks, not evidence of learned robustness.
