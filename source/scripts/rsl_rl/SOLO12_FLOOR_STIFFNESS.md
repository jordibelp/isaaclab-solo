# Solo12 floor-stiffness randomization

## Enable it

Append this setting to the existing two-feet IMU teacher command:

```bash
'env.delta_x_floor_stiffness_range_mm=[0.2,5.0]'
```

Keep `env.compliant_contact_damping=60.0` if you want that constant damping. You can also keep `env.compliant_contact_stiffness=3000.0`: an enabled indentation range overrides that stiffness on the complete floor.

| Range | Behavior |
| --- | --- |
| Omitted, or `[0.0,0.0]` | Preserve the existing fixed `compliant_contact_stiffness` and damping. No geometry or spawn-pool change. |
| `[0.2,5.0]` | Sample each floor tile's reference indentation uniformly between 0.2 and 5.0 mm, then convert it to stiffness. |
| `[2.0,2.0]` | Set every floor tile to the stiffness corresponding to a 2.0 mm reference indentation. |

Enabled endpoints must be finite and satisfy `0 < min <= max`. A mixed interval such as `[0.0,5.0]` is invalid: zero indentation would require infinite stiffness. Use the existing fixed-stiffness setting for rigid ground. The feature currently requires generated terrain, which the supplied two-feet curriculum creates automatically. It rejects a plain infinite plane instead of silently sampling one shared value.

## What is a patch, and when does it change?

A patch is an existing generated terrain tile. In this task there are 120 tiles, each 13 by 13 metres. Each gets one independent sample at scene creation. The outer border gets another sample. Samples stay fixed during resets and curriculum changes. Robots on the same physical tile share its material. A new run seed creates a new material layout.

We do not resample a shared tile when one robot resets. Many parallel robots use each tile; changing its material would also affect robots whose episodes are still running.

The early flat phases normally visit only rows 4 and 5 of flat column 2. When floor randomization is enabled, they visit all ten rows of that same flat column. The geometry is flat in all those rows. This exposes ten fixed floor samples instead of two, without changing the flat terrain type or the rough-stage spawn pool. Later rough phases keep their existing rows and weighted columns. There is a finite library of physical floors, not an independent new continuous sample for every episode.

## Calibration and precedence

Stiffness is the nominal robot weight divided by two and by the sampled indentation in metres. The reference is two feet sharing a static load equally. The nominal mass comes from all robot rigid bodies after the configured base mass and optional front-foot payloads are applied. It is measured before mass randomization. The gravity magnitude comes from the simulator configuration.

The supplied robot configuration has a nominal mass of about 3.02857 kg. The requested 0.2–5.0 mm interval therefore corresponds to about 2,971–74,276 N/m. Damping stays constant. Friction and its existing randomization are unchanged.

The floor does not change its stiffness when the number of supporting feet changes. Actual indentation also depends on load sharing, motion, contact geometry, and solver settings. For example, a sphere resting exactly on a triangle seam can engage two contact springs. The range is a reference calibration, not a clamp on measured foot penetration.

## Implementation and records

The enabled path splits the existing collision triangles into static tile colliders. It preserves every triangle, the original visible terrain, and the original height-query mesh. The original mesh's collision is disabled to avoid duplicate contacts. It does not add one floor per environment. Materials are authored before PhysX starts; there are no material writes or sampling operations in the training step or reset loop.

A first attempt using material subsets on the single three-million-triangle mesh stalled during local PhysX loading. Separate static tile colliders started successfully. This is why the implementation uses separate collision meshes rather than per-face material bindings.

Each training run saves `params/floor_stiffness.json` with the exact sampled indentation and stiffness grid, border values, reference mass, gravity, support-foot count, seed, and damping. The environment YAML saves the requested interval. The training source snapshot includes `floor_stiffness.py` and the exact `floor_stiffness.json` draws. Snapshotting the helper and the seed makes the sampler reviewable; the JSON preserves the actual draws.

## Full supplied training command

This adds only the new range to Jordi's command:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=solo12-two-feet-IMU-teacher --agent=rsl_rl_cfg_entry_point --headless --num_envs=10000 --max_iterations=10000 --seed=230902 --symmetry-mode=augmentation '--run-name=[Cluster]-Does PPO learn if we train a Teacher-student architecture?' 'env.front_back_asymetry_curriculum=[True,True,True,True,True]' 'env.two_feet_above_height_reward_scale_curriculum=[2.0,1.2,1.5,1.5,1.5]' 'env.track_lin_vel_xy_reward_scale_curriculum=[1.2,1.8,1.5,1.5,1.5]' 'env.two_feet_above_height_alpha_curriculum=[10.0,20.0,25.0,25.0,25.0]' env.rear_feet_in_contact_for_twofeet=False env.base_collision_terminal_penalty=-10.0 env.tracking_std=0.223 env.soft_qlim_penalty_reward_scale=-0.5 env.kp=9.0 env.kd=0.2 env.effort_limit_sim=2.25 'env.joint_physical_limit_calf=[-179.0,179.0]' 'env.joint_physical_limit_hip=[-65.0,65.0]' env.use_asymmetric_thigh_limits=True 'env.joint_physical_limit_front_thigh=[-145.0,55.0]' 'env.joint_physical_limit_rear_thigh=[-55.0,145.0]' env.joint_soft_limit_hip_delta=20.0 env.joint_soft_limit_thigh_delta=20.0 env.joint_soft_limit_calf_delta=24.0 env.three_or_more_feet_contact_triggers_reset=True env.three_or_more_feet_contact_triggers_after=1.0 'env.forces_applied_to_base_curriculum_by_phase=[0.0,0.0,0.0,3.0,6.0]' agent.algorithm.value_loss_coef=1.0 agent.algorithm.entropy_coef=0.005 agent.algorithm.learning_rate=1.0e-3 agent.algorithm.gamma=0.97 agent.algorithm.max_grad_norm=1.0 'agent.policy.critic_hidden_dims=[512,256,128]' 'agent.policy.actor_hidden_dims=[512,256,128]' agent.policy.actor_obs_normalization=True agent.policy.critic_obs_normalization=True agent.local_redundancy.interval=50 agent.local_redundancy.batch_size=64 agent.adam_beta2=0.95 agent.weight_decay=0.001 agent.optimizer=adamW agent.algorithm.torch_compile=True env.noisy_critic_obs=False env.compliant_contact_damping=60.0 env.compliant_contact_stiffness=3000.0 'env.delta_x_floor_stiffness_range_mm=[0.2,5.0]'
```

## Verification

Completed checks:

- 63 simulator-free tests: uniform indentation, inverse conversion, validation, repeatable independent sampling, face-to-tile mapping, IMU pipeline, and source-snapshot handling.
- 37 task/configuration/live-history tests, including disabled fixed stiffness and flat-versus-rough spawn-pool behavior.
- Disabled-path comparison against pre-change commit `af40db6`: identical terrain origins and bit-identical 40-step traces for four environments, including root and joint states, processed actions, rewards and teacher observations.
- A 32-environment live scene with ten loaded spheres on different flat tiles. Measured sink-in matched reference indentation within 0.02 mm. Physics materials kept damping 60, and remained unchanged after partial resets and curriculum robot-material randomization.
- Public headless playback reloaded the new teacher checkpoint and ran with the floor range enabled. The custom actor/critic widths must also be passed to the player, as in training.
- The actual supplied teacher-training entry point, reduced to four environments and two iterations. It kept symmetry augmentation, AdamW, compilation and the requested network widths. Teacher encoder, actor and critic weights changed; losses and saved checkpoint tensors stayed finite.

Reproduce the physical material check from the repository with:

```bash
./isaaclab.sh -p source/isaaclab_tasks/test/direct/solo12/check_floor_contacts.py --headless --num_envs=32 --out=/absolute/path/floor-check.json 'env.delta_x_floor_stiffness_range_mm=[0.2,5.0]' env.compliant_contact_stiffness=3000.0 env.compliant_contact_damping=60.0 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
``` Local simulator checks do not establish IRICluster compatibility or 10,000-environment throughput. No cluster job or synchronization is part of this change.
