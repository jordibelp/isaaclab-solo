"""Local weight-only gait-edit experiment. No training and no hardware control.

Run with env_isaaclab Python. All experiment outputs go to --output.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import faulthandler

faulthandler.dump_traceback_later(60, repeat=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", help="JSON list of direct final-layer edits")
parser.add_argument("--replicas", type=int, default=8)
parser.add_argument("--seconds", type=float, default=32.0)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--robust", action="store_true")
parser.add_argument("--nominal-plant", action="store_true", help="Keep nominal physical properties in the noisy/delayed/pushed test")
parser.add_argument("--center", help="NPZ baseline trace used to center common-mode edits")
parser.add_argument("--commands", default="0.1,-0.1,0.3,-0.3,0.0,0.5,-0.5,0.1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import isaaclab_tasks
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12TwoFeetEnvCfg
from isaaclab_tasks.direct.solo12.solo12_env import _world_velocity_in_heading_frame_xy


def final_layer_edit(state, spec, center):
    """Scale common/differential rear-joint outputs about a recorded center."""
    w, b = state["actor.6.weight"].clone(), state["actor.6.bias"].clone()
    for left, right, joint in ((7, 10, "thigh"), (8, 11, "calf")):
        common = float(spec.get("common_" + joint, spec.get("common", 1.0)))
        diff = float(spec.get("diff", 1.0))
        old_w, old_b = w[[left, right]].clone(), b[[left, right]].clone()
        mean_w, delta_w = old_w.mean(0), (old_w[0] - old_w[1]) * 0.5
        mean_b, delta_b = old_b.mean(), (old_b[0] - old_b[1]) * 0.5
        center_mean = center[[left, right]].mean()
        bias = float(spec.get("bias_" + joint, 0.0))
        w[left], w[right] = common * mean_w + diff * delta_w, common * mean_w - diff * delta_w
        b[left] = common * mean_b + (1 - common) * center_mean + diff * delta_b + bias
        b[right] = common * mean_b + (1 - common) * center_mean - diff * delta_b - bias
    if "project_file" in spec:
        projection = np.load(spec["project_file"])
        rank = int(spec.get("project_rank", 1))
        start = int(spec.get("project_start", 0))
        u = torch.as_tensor(projection["directions"][:, start:start+rank], device=w.device, dtype=w.dtype)
        hmean = torch.as_tensor(projection["mean"], device=w.device, dtype=w.dtype)
        change = float(spec.get("project_alpha", 1.0)) * (w @ u) @ u.T
        w -= change
        b += change @ hmean
    return w, b


def first_layer_edit(state, spec):
    """Fold physical-unit left/right input transformations into normalized weights."""
    w, b = state["actor.0.weight"].clone(), state["actor.0.bias"].clone()
    mean = state["actor_obs_normalizer._mean"][0]
    std = state["actor_obs_normalizer._std"][0] + 0.01
    for columns, name in (((6,9,12,15), "imu_acc_x_gain"), ((8,11,14,17), "imu_acc_z_gain"), ((1,), "pitch_rate_gain")):
        if name in spec:
            gain = float(spec[name])
            old = w[:, columns].clone()
            w[:, columns] *= gain
            b += ((gain-1)*old*(mean[list(columns)]/std[list(columns)])[None,:]).sum(1)
    for offset, name in ((21, "position"), (33, "velocity"), (45, "memory")):
        gain = float(spec.get("input_" + name, 1.0))
        if gain == 1:
            continue
        for left, right in ((offset+7, offset+10), (offset+8, offset+11)):
            wl, wr = w[:, left].clone()/std[left], w[:, right].clone()/std[right]
            nl = ((1+gain)*wl + (1-gain)*wr)/2
            nr = ((1-gain)*wl + (1+gain)*wr)/2
            w[:, left], w[:, right] = nl*std[left], nr*std[right]
            b += (nl-wl)*mean[left] + (nr-wr)*mean[right]
    return w, b


def edited_state(state, spec, center):
    if not (set(spec)-{"name"}):
        # A no-op control must preserve tensor bytes, not just algebraic equivalence.
        return dict(state)
    result = dict(state)
    for index in (0, 2, 4, 6):
        for kind in ("weight", "bias"):
            key = f"actor.{index}.{kind}"
            result[key] = state[key].clone()
    result["actor.0.weight"], result["actor.0.bias"] = first_layer_edit(state, spec)
    result["actor.6.weight"], result["actor.6.bias"] = final_layer_edit(state, spec, center)
    if "circuit_file" not in spec:
        return result
    # Reserve low-influence units, preserving their baseline mean contribution.
    circuit = np.load(spec["circuit_file"])
    ids = {}
    for layer in (0, 2, 4):
        ids[layer] = [int(x) for x in circuit[f"ids{layer}"]]
        nxt_w, nxt_b = result[f"actor.{layer+2}.weight"], result[f"actor.{layer+2}.bias"]
        mean = torch.as_tensor(circuit[f"mean{layer}"][ids[layer]], device=nxt_w.device, dtype=nxt_w.dtype)
        nxt_b += nxt_w[:, ids[layer]] @ mean
        nxt_w[:, ids[layer]] = 0
        result[f"actor.{layer}.weight"][ids[layer]] = 0
        result[f"actor.{layer}.bias"][ids[layer]] = 0
    # A bounded two-state rotation, using previous rear-action differences as its state.
    # It lives entirely in four/two/two existing ELU units; no runtime clock is added.
    amp = float(spec.get("circuit_amplitude", 0.3))
    ratio = float(spec.get("circuit_ratio", 1.0))
    theta = 2*np.pi*float(spec.get("circuit_hz", 2.0))*0.02
    radius = float(spec.get("circuit_radius", 1.02))
    rot = radius*np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    std = state["actor_obs_normalizer._std"][0]+0.01
    mean = state["actor_obs_normalizer._mean"][0]
    for phase in (0, 1):
        coef = torch.zeros(57, device=std.device)
        for j, left, right, scale in ((0, 52, 55, amp), (1, 53, 56, amp*ratio)):
            coef[left], coef[right] = 10*rot[phase,j]/(2*scale), -10*rot[phase,j]/(2*scale)
        for branch in (0, 1):
            unit = ids[0][2*phase+branch]
            result["actor.0.weight"][unit] = coef*std
            result["actor.0.bias"][unit] = coef@mean + (9 if branch==0 else -11) + (0.01 if phase==0 else 0)
        unit = ids[2][phase]
        result["actor.2.weight"][unit, ids[0][2*phase]] = 0.1
        result["actor.2.weight"][unit, ids[0][2*phase+1]] = -0.1
        result["actor.2.bias"][unit] = 1.0  # bounded state + 2, positive through ELU
        result["actor.4.weight"][ids[4][phase], unit] = 1.0
    w, b = result["actor.6.weight"], result["actor.6.bias"]
    retain = float(spec.get("circuit_retain", 0.25))
    for phase, left, right, scale in ((0, 7, 10, amp), (1, 8, 11, amp*ratio)):
        mw, dw = (w[left]+w[right])/2, (w[left]-w[right])/2
        mb, db = (b[left]+b[right])/2, (b[left]-b[right])/2
        w[left], w[right] = mw+retain*dw, mw-retain*dw
        b[left], b[right] = mb+retain*db, mb-retain*db
        scale *= float(spec.get("circuit_gain", 1.0))
        w[left,ids[4][phase]] += scale
        w[right,ids[4][phase]] -= scale
        b[left] -= 2*scale
        b[right] += 2*scale
    hip = float(spec.get("circuit_hip", 0.0))
    hip_phase = int(spec.get("circuit_hip_phase", 1))
    for joint in (6, 9):
        w[joint, ids[4][hip_phase]] += hip
        b[joint] -= 2*hip
    return result


def run():
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.candidates).read_text()) if args.candidates else [{"name": "original"}]
    count = len(specs) * args.replicas
    cfg = Solo12TwoFeetEnvCfg()
    # Match the user-supplied public-player command, including its default clean-play switches.
    cfg.scene.num_envs = count
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = 80.0
    cfg.tricky_terrain = False
    cfg.curriculum_tricky_terrain_idx = 0
    cfg.curriculum_two_feet = False
    cfg.base_obs_from_imu = True
    cfg.remove_root_lin_vel_b_from_actor_obs = True
    cfg.include_lin_acc_steps_in_imu_obs = 4
    cfg.enabled_self_collisions = True
    cfg.base_filtered_pairs = ["hip"]
    cfg.extra_mass_on_front_feet = 0.0
    cfg.joint_physical_limit_calf = [-179.0, 179.0]
    cfg.joint_physical_limit_hip = [-60.0, 60.0]
    cfg.use_asymmetric_thigh_limits = True
    cfg.joint_physical_limit_front_thigh = [-145.0, 55.0]
    cfg.joint_physical_limit_rear_thigh = [-55.0, 145.0]
    cfg.joint_soft_limit_hip_delta = 20.0
    cfg.joint_soft_limit_thigh_delta = 20.0
    cfg.joint_soft_limit_calf_delta = 24.0
    cfg.kp, cfg.kd, cfg.effort_limit_sim = 9.0, 0.2, 2.25
    cfg.robot.actuators["legs"].stiffness = 9.0
    cfg.robot.actuators["legs"].damping = 0.2
    cfg.robot.actuators["legs"].effort_limit_sim = 2.25
    cfg.bringup_clip_rad = 0.0
    cfg.command_resampling_time_s = 1e9
    cfg.standing_env_prob = 0.0
    cfg.command_lin_vel_x_range = [-0.6, 0.6]
    cfg.command_lin_vel_y_range = [0.0, 0.0]
    cfg.command_ang_vel_z_range = [0.0, 0.0]
    # Extract the actual clean-play transformation so behavior follows the public player.
    player = Path(__file__).parents[1] / "rsl_rl/play_direct_0325.py"
    tree = ast.parse(player.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_disable_training_stochasticity_for_play")
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(player), "exec"), scope)
    if not args.robust:
        scope[func.name](cfg)
    else:
        if args.nominal_plant:
            cfg.events = None
        cfg.actuation_delay_range = (0, 3)
        cfg.flexed_initial_joint_pos_noise_range = (-0.04, 0.04)
        cfg.base_push_force_xy_range = (-2.0, 2.0)
        cfg.forces_applied_to_base_curriculum = []
        cfg.base_push_interval_range_s = (4.0, 6.0)
        cfg.base_push_duration_range_s = (0.1, 0.2)
    env = gym.make("solo12-two-feet", cfg=cfg)
    raw = env.unwrapped
    device = raw.device
    if args.robust:
        # Paired physical draws: every edit gets the same replica-specific plant.
        view = raw._robot.root_physx_view
        idx = torch.arange(count, dtype=torch.int32)
        matched = {}
        for field in ("masses", "inertias", "coms", "material_properties", "dof_friction_properties"):
            values = getattr(view, "get_"+field)()
            values[:] = values[:args.replicas].repeat((len(specs),)+(1,)*(values.ndim-1))
            getattr(view, "set_"+field)(values, idx)
            readback = getattr(view, "get_"+field)()
            torch.testing.assert_close(readback, values)
            matched[field] = readback.cpu().numpy().tolist()
        (out/"matched_plant_properties.json").write_text(json.dumps(matched))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    actor = ActorCritic(TensorDict({"policy": torch.zeros(count, 57, device=device), "critic": torch.zeros(count, 60, device=device)}, batch_size=[count]),
                        {"policy": ["policy"], "critic": ["critic"]}, 12,
                        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu",
                        actor_obs_normalization=True, critic_obs_normalization=True, noise_std_type="log").to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    center = torch.zeros(12, device=device)
    if args.center:
        center = torch.as_tensor(np.load(args.center)["action_center"], device=device)
    edits = [edited_state(state, spec, center) for spec in specs]
    layer_weights = {i: torch.stack([s[f"actor.{i}.weight"] for s in edits]) for i in (0,2,4,6)}
    layer_biases = {i: torch.stack([s[f"actor.{i}.bias"] for s in edits]) for i in (0,2,4,6)}
    (out / "candidates.json").write_text(json.dumps(specs, indent=2))
    # Retain full source and resolved configuration. This is not an online W&B run.
    (out / "resolved_cfg.txt").write_text(repr(cfg.to_dict()))
    (out / "provenance.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
        "sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "iteration": checkpoint.get("iter"),
        "seed": args.seed, "robust": args.robust, "nominal_plant": args.nominal_plant, "replicas": args.replicas,
        "seconds": args.seconds, "physics_dt": raw.physics_dt, "policy_dt": raw.step_dt,
        "joint_names": cfg.joint_names, "feet": raw._feet_body_names,
        "feet_height_order": raw._feet_robot_body_names}, indent=2))
    print("EDIT_RESET_BEGIN", flush=True)
    obs, _ = env.reset(seed=args.seed)
    print("EDIT_RESET_DONE", flush=True)
    if args.robust:
        delays = raw._action_delay_steps[:args.replicas].repeat(len(specs))
        raw._action_delay_steps[:] = delays
        raw._action_delay_buffer.set_time_lag(delays)
        (out/"initial_delay_steps.json").write_text(json.dumps(delays.cpu().tolist()))
    observations, actions, hidden, states, dones = [], [], [], [], []
    substep = []
    disturbances = []
    original_record = raw._record_base_imu_history_sample
    def record():
        original_record()
        forces = raw._contact_sensor.data.net_forces_w[:, raw._feet_body_ids]
        contact = (forces.norm(dim=-1) > 1.0).float()
        heights = raw._get_foot_positions_w()[:, :, 2] - raw.scene.env_origins[:, 2:3]
        vel = _world_velocity_in_heading_frame_xy(raw._robot.data.root_lin_vel_w, raw._robot.data.root_quat_w)
        substep.append(torch.cat((contact, heights, vel, raw._robot.data.root_lin_vel_w[:, 2:3],
            raw._robot.data.root_pos_w[:, 2:3] - raw.scene.env_origins[:, 2:3]), dim=1).detach().clone())
    raw._record_base_imu_history_sample = record
    steps = round(args.seconds / raw.step_dt)
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(steps):
            # Symmetric, then reversed order across replicas: all edits see both directions.
            seq = [float(v) for v in args.commands.split(",")]
            segment = min(int(step * raw.step_dt / (args.seconds / len(seq))), len(seq) - 1)
            sign = torch.where(torch.arange(count, device=device) % 2 == 0, 1.0, -1.0)
            raw._commands[:, 0] = seq[segment] * sign
            raw._commands[:, 1:] = 0.0
            raw._command_steps_left[:] = 100000000
            # Do not read observations again: that would draw fresh noise in robust mode.
            obs["policy"][:, 18:21] = raw._commands
            x = actor.actor_obs_normalizer(obs["policy"])
            x = x.reshape(len(specs), args.replicas, -1)
            for layer in (0,2,4):
                x = torch.nn.functional.elu(torch.einsum("gri,goi->gro", x, layer_weights[layer]) + layer_biases[layer][:,None,:])
            h = x.reshape(count, 128)
            a = torch.einsum("gri,goi->gro", x, layer_weights[6]) + layer_biases[6][:, None, :]
            a = a.reshape(count, 12)
            if step == 0 and len(specs) == 1 and specs[0].get("name") == "original":
                torch.testing.assert_close(a, actor.act_inference(TensorDict({"policy": obs["policy"]}, batch_size=[count])), atol=1e-5, rtol=1e-5)
            observations.append(obs["policy"].clone())
            actions.append(a.clone())
            hidden.append(h.clone())
            obs, rew, terminated, truncated, info = env.step(a)
            disturbances.append(torch.cat((raw._base_push_forces_b[:,0,:],raw._action_delay_steps[:,None]),dim=-1).clone())
            dones.append(torch.stack((terminated, truncated), dim=-1).clone())
            states.append(torch.cat((raw._robot.data.joint_pos[:, raw._joint_ids], raw._robot.data.joint_vel[:, raw._joint_ids], raw._commands,
                raw._robot.data.root_pos_w-raw.scene.env_origins, raw._robot.data.root_quat_w), dim=-1).clone())
            if step % 250 == 0:
                print(f"EDIT_PROGRESS {step}/{steps} envs={count} elapsed={time.monotonic()-started:.1f}s", flush=True)
    arrays = {"observations": torch.stack(observations).cpu().numpy(), "actions": torch.stack(actions).cpu().numpy(),
        "hidden": torch.stack(hidden).cpu().numpy(), "states": torch.stack(states).cpu().numpy(),
        "dones": torch.stack(dones).cpu().numpy(), "physics": torch.stack(substep).cpu().numpy(),
        "disturbances": torch.stack(disturbances).cpu().numpy(),
        "action_center": torch.stack(actions)[round(2/raw.step_dt):].mean((0, 1)).cpu().numpy()}
    np.savez_compressed(out / "trace.npz", **arrays)
    print(f"EDIT_SAVED {out} elapsed={time.monotonic()-started:.1f}s", flush=True)
    env.close()


try:
    run()
except BaseException:
    # Kit's close path can hang while unwinding an exception; preserve the actual cause.
    import traceback
    import os
    import sys
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close()
