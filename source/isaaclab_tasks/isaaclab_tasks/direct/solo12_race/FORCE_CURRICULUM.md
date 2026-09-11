# Race backward-force curriculum

The single-process RSL-RL PPO path in `source/scripts/rsl_rl/train.py` evaluates this curriculum once after each complete rollout. The initial force is `env.backward_force`; `env.backward_force_curriculum` lists subsequent forces. An empty list disables force changes.

## Defaults

| Setting | Default | Purpose |
| --- | ---: | --- |
| `env.min_iterations_with_curriculum_stage` | `64` | Minimum complete rollouts at each force, including the initial force. |
| `env.backward_force_curriculum_window_iterations` | `10` | Recent rollout window used to pool outcomes. |
| `env.backward_force_curriculum_min_episodes` | `500` | Minimum eligible completed episodes required for a decision. |
| `env.backward_force_curriculum_sr_threshold` | `0.7` | Promote only when the pooled success rate is **strictly greater** than 70%. |

These are conservative starting defaults for roughly 10,000 parallel environments, not calibrated convergence or confidence guarantees. At 32 steps per rollout and a control timestep of 0.01 s, 64 rollouts give **20.48 simulated seconds per environment**, and ten rollouts give 3.2 seconds. The dwell default allows approximately a full 20-second episode horizon at a force before considering promotion. Change it if rollout length, timestep, or episode horizon changes substantially.

The dwell counter measures time spent at a force, not consecutive successful iterations. It increases even for empty rollouts. Episode counts measure the amount of evidence. Both gates must pass, and at most one force change is allowed per rollout.

## Which episodes count?

- Completed episodes include successes, failed terminations, and timeouts. Running episodes are not outcomes yet. A timeout and termination on the same step count as one ending, not two.
- An episode is eligible for the **curriculum** only if all its simulated steps used the current force stage. An ongoing episode that spans a force change is excluded when it ends; the next episode in that environment is eligible.
- An episode reset on the final rollout step has not yet taken any steps under the old force. It is tagged with the new stage if promotion occurs at that boundary.
- Startup/reset calls with no completed simulated episode contribute no outcome.
- Exclusion applies only to curriculum statistics. Mixed-stage episodes remain in PPO training and in the general episode success metric. No forced resets, reward changes, or PPO data filtering are added.

## Window behavior

Pool counts, not percentages:

`success rate = sum(successful eligible episodes) / sum(completed eligible episodes)`

Keep all outcomes from the last ten rollouts. If those contain fewer than 500 eligible endings, retain the smallest number of additional older rollout batches from **this stage** needed to reach 500. If the stage has not yet produced 500 endings in total, wait. Whole rollout batches are retained, so the count can slightly exceed the minimum. Empty rollouts occupy time in the window but add no outcomes.

For example:

- **500 eligible endings per rollout:** use the most recent ten rollouts, containing 5,000 episodes.
- **20 eligible endings per rollout:** ten rollouts contain only 200 episodes, so use 25 rollouts to reach 500. On the next rollout the oldest batch falls out: the window does not grow forever.
- If the rate later increases, older batches are discarded as soon as enough recent evidence is available; the window contracts back to ten rollouts.

This is a streaming retention rule, not a re-query of old runs or a cumulative lifetime average. It adapts to small environment counts without repeatedly failing a fixed-window sample requirement. Only nonempty batches are stored. The tradeoff is that sparse runs can use older-policy outcomes from the same stage; the actual window length is logged so this lag is visible. There is no additional maximum age cap.

All history is cleared on promotion. Only a rollout with **new eligible endings** can trigger a promotion. Waiting out the dwell period or aging old failures out of the recent window cannot, on its own, promote from stale success evidence. A rollout with no endings is no data, not 0% success; a rollout with endings that all fail is genuinely 0%.

## Logging

| Metric | Meaning |
| --- | --- |
| `Episode/successRate`, `Episode/finishRatio` | Successful episodes / all completed episodes in this rollout, pooled by episode count, **including mixed-stage episodes**. Absent if no episodes ended. |
| `Episode/success_count`, `Episode/completed_count` | Total successful/completed episodes in the rollout. |
| `Curriculum/backward_force_eligible_success_count`, `Curriculum/backward_force_eligible_completed_count` | Current rollout's stage-pure successful/completed episodes; zero for empty or entirely mixed-stage rollouts. |
| `Curriculum/backward_force_success_rate` | Pooled stage-pure window rate used for decisions. Absent when the window has no outcomes; retained window evidence can be logged even when there are no new eligible endings. |
| `Curriculum/backward_force_window_episodes`, `Curriculum/backward_force_window_successes` | Denominator/numerator of the decision statistic, before any promotion clears history. |
| `Curriculum/backward_force_window_iterations` | Actual temporal width of the window, including any extension and empty rollouts; limited by current stage age at startup. |
| `Curriculum/backward_force_rollout_N`, `Curriculum/backward_force_rollout_stage`, `Curriculum/backward_force_rollout_stage_iterations` | Force, stage, and completed stage rollouts for the data/decision on this row. |
| `Curriculum/backward_force_N`, `Curriculum/backward_force_stage`, `Curriculum/backward_force_stage_iterations` | State **after** the decision; selected stage age is zero on a transition row. |
| `Curriculum/backward_force_promoted` | One on a promotion row, otherwise zero. |

The general `Episode/successRate` previously averaged the success fractions of control steps equally. For example, 1/1 plus 60/100 yielded 80%; it now correctly yields 61/101 = 60.4%. Comparisons against old logged runs must account for this definition change. Other episode metrics retain their previous aggregation.

## Restart and scope

New environments, including checkpoint restarts, start at the configured `env.backward_force` with empty evidence and a zero dwell counter. The race force curriculum is **not restored from policy checkpoints**, preserving the previous restart behavior. To restart at a later force, specify that initial force and the remaining stage list explicitly.

This integration is for single-process RSL-RL PPO. `train.py` rejects an enabled race force curriculum with distributed PPO or the SAC runner, whose per-rollout callbacks are not supported here. A constant backward force with an empty curriculum remains allowed.

To use these settings explicitly (they are also the defaults):

```bash
env.min_iterations_with_curriculum_stage=64 env.backward_force_curriculum_window_iterations=10 env.backward_force_curriculum_min_episodes=500 env.backward_force_curriculum_sr_threshold=0.7
```

An existing explicit override such as `env.min_iterations_with_curriculum_stage=5` still wins over the new default. A shorter dwell no longer permits mixed-stage outcomes into the statistic, but it still gives the policy less time at each force and can emphasize faster-ending episodes soon after a transition.
