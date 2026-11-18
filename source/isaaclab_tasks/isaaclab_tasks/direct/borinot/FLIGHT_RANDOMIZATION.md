# Borinot flight randomization

Goalkeeper, Goalkeeper Simple, CmdVel, GoTo (including MinTime), and legacy Borinot
use the same flight randomization settings under `env.flight_randomization`.

## Enabled by default

- **Motor response:** each rotor samples a time constant uniformly between **35 and
  45 ms**. This is a first-order lag on rotor speed, except for legacy PI control,
  which keeps its first-order lag on thrust.
- **Thrust calibration:** each vehicle draws a shared uniform offset of **±0.10**.
  Each rotor adds its own uniform offset of **±0.03**. The two offsets add to 1
  and multiply nominal `kf`, giving a total gain from **0.87 to 1.13**.
- **Yaw calibration:** actual `km` is nominal `km` times a **0.90–1.10** gain
  shared across the vehicle. It is sampled independently of `kf`. The actual
  reaction torque per unit thrust therefore depends on both coefficients.
- **Policy delay:** each reset draws a new delay of **0–20 ms** on the physics grid.
  At 200 Hz, that is 0, 5, 10, 15 or 20 ms. At legacy 120 Hz, the largest step
  below 20 ms is 16.67 ms. This delays the command entering control, not the
  measurements INDI/PI/PD uses or the motor command leaving the controller.

Rotor coefficients and time constants are sampled once per clone and persist
across episode resets. The controller's model and command limits remain nominal.
Actual thrust can differ from the nominal command ceiling because the plant gain
also affects thrust at maximum commanded speed. No observation dimensions change.

For old-checkpoint comparisons, disable the new defaults explicitly:

```text
env.flight_randomization.motor_time_constant_range_s=null
env.flight_randomization.thrust_shared_gain_delta=0.0
env.flight_randomization.thrust_rotor_gain_delta=0.0
env.flight_randomization.km_gain_delta=0.0
env.actuator_delay_range=null
```

This restores the previous fixed motor lag (instant response on legacy Borinot)
and, with `env.actuator_delay=0.0`, disables the delay. The existing startup
mass, CoM, inertia and friction events still use their own settings.

## Optional wind, gusts, and external moments

Enable with `env.flight_randomization.wind_enabled=True`. Each clone samples
horizontal wind up to **2 m/s** at reset and holds it for the episode. It adds
gusts up to another **2 m/s** and independently draws a world-frame moment up
to **0.04 N·m** in magnitude for each pulse. Pulses rise and fall together over
**0.2–0.5 s**, with starts **0.8–1.6 s** apart after a random initial phase.
Zero the respective speed or moment limits to isolate individual effects.

Wind produces quadratic drag based on air velocity relative to the base, with
a coefficient of **0.08 N/(m/s)²**. The wind force acts at the base CoM and the
external moment acts about the CoM. These effects do not replace the separate
vertical push or its off-centre moment. The vertical-push curriculum still
controls only the vertical pushes; wind and moments can act at any stage.
These are starting ranges for simulation tests, not measured hardware values.
