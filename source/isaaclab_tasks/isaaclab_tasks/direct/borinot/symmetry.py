# SPDX-License-Identifier: BSD-3-Clause
"""Borinot command-task symmetries for RSL PPO/SAC (no simulator imports).

Only identity and a 180-degree WORLD-heading rotation are exposed. This moves
all physical parts together; it is not a body mirror or a rotor permutation.
The audit below explains why the requested body mirrors do not preserve the
configured rotor/leg model. See SYMMETRY.md for initial-state limitations.
"""
import math

import torch
from tensordict import TensorDict


SIGNS = {"left_right": (1., -1., 1.), "front_back": (-1., 1., 1.), "rotate_180": (-1., -1., 1.)}


def rotor_symmetry_audit(cfg):
    """Check both rotor positions and axial yaw signs, not the hexagon alone."""
    positions = torch.tensor(cfg.rotor_pos_b, dtype=torch.float64)
    spins = torch.tensor(cfg.rotor_spin, dtype=torch.float64)
    ratio = getattr(cfg, "km_over_kf", cfg.km / cfg.kf)
    allocation = torch.stack((torch.ones(6), positions[:, 1], -positions[:, 0], spins*ratio))
    result = {}
    for name, signs in SIGNS.items():
        s = positions.new_tensor(signs)
        distances = torch.cdist(positions*s, positions)
        permutation = distances.argmin(-1)
        det = math.prod(signs)
        wrench_signs = positions.new_tensor((1., det*signs[0], det*signs[1], det))
        residual = allocation[:, permutation] - wrench_signs[:, None]*allocation
        geometry_error = float(distances.min(-1).values.max())
        wrench_error = float(residual.abs().max())
        result[name] = {"permutation": permutation.tolist(), "geometry_error_m": geometry_error,
                        "wrench_error_per_n": wrench_error,
                        "valid": len(set(permutation.tolist())) == 6 and geometry_error < 1e-6 and wrench_error < 1e-8}
    return result


class BorinotSymmetry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.legacy = hasattr(cfg, "des_joint_out") and not hasattr(cfg, "control_mode")
        if not self.legacy and not hasattr(cfg, "command_lin_vel_x_range"):
            raise ValueError("Borinot symmetry currently supports command-tracking tasks only. "
                             "The fixed-goal goalkeeper geometry/reset distribution is not the same task under these mirrors.")
        self.audit = rotor_symmetry_audit(cfg)
        self.transforms = ("identity", "world_yaw_180")
        self.mode = "legacy" if self.legacy else cfg.control_mode
        quat = getattr(cfg, "robot_home_quat", (1., 0., 0., 0.))
        w, x, y, z = quat
        self.home_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self.action_slices = {}
        self._action_layout()
        if (self.mode in ("indi", "wrench") and cfg.sac_action_mapping
                and not cfg.predict_yaw_as_sincos):
            controller = cfg.indi if self.mode == "indi" else cfg.wrench
            turns = controller.heading_range_degs / 180.
            if turns < 1 or not math.isclose(turns, round(turns), abs_tol=1e-8):
                raise ValueError("World-heading augmentation needs a whole number of full-circle SAC heading ranges "
                                 "(heading_range_degs=180, 360, ...), or predict_yaw_as_sincos=True.")
        self.obs_slices = {}
        self._observation_layout()

    def _action_layout(self):
        offset = 0

        def add(name, size):
            nonlocal offset
            self.action_slices[name] = slice(offset, offset+size)
            offset += size

        if self.legacy or self.mode.startswith("thrust_and_"):
            add("rotors", 6)
        elif self.mode == "PI_rotors_PD_joint_pos_control":
            add("collective", 1)
            add("rates", 3)
        elif self.mode in ("cbrt", "cbrt_indi"):
            add("rates", 3)
            add("collective", 1)
        elif self.mode in ("indi", "wrench"):
            add("acceleration", 3)
            if self.mode == "indi":
                add("rates", 3)
            add("yaw", 2 if self.cfg.predict_yaw_as_sincos else 1)
            if self.mode == "wrench":
                add("rates", 3)
        else:
            raise ValueError(f"Unsupported Borinot control mode: {self.mode}")
        for joint in ("thigh", "calf"):
            pair = self.cfg.des_joint_out if self.legacy else (
                self.mode != "thrust_and_torque_control" and getattr(self.cfg, f"predict_{joint}_joint_pos_as_sincos", False))
            add(joint, 2 if pair else 1)
        self.action_size = offset
        if offset != self.cfg.action_space:
            raise ValueError("Resolve Borinot action_space after control/angle overrides before enabling symmetry.")

    def _observation_layout(self):
        offset = 0

        def add(name, size):
            nonlocal offset
            self.obs_slices[name] = slice(offset, offset+size)
            offset += size

        if self.legacy:
            add("height", 1)
            add("rotation", 9)
        add("velocity", 3)
        add("omega", 3)
        if not self.legacy and self.cfg.obs_include_base_lin_acc:
            add("acceleration", 3)
        add("joint_position", 2)
        add("joint_velocity", 2)
        if self.legacy and self.cfg.include_force_info_in_obs:
            add("force_norm", 1)
        if not self.legacy or self.cfg.obs_include_prev_action:
            add("actions", self.action_size)
        if not self.legacy:
            add("gravity", 3)
            add("rotation", 9)
        add("commands", 3 if self.legacy else 4)
        if not self.legacy:
            if self.mode == "PI_rotors_PD_joint_pos_control":
                add("rotors", 6)
                add("rate_integral", 3)
            elif self.mode in ("indi", "wrench", "cbrt", "cbrt_indi"):
                add("rotors", 6)
                if self.mode == "indi":
                    add("joint_integral", 2)
        self.observation_size = offset

    def actions(self, value, name):
        if value.shape[-1] != self.action_size:
            raise ValueError(f"Expected {self.action_size} Borinot actions, got {value.shape[-1]}.")
        if name == "identity":
            return value.clone()
        result = value.clone()
        if name == "world_yaw_180":
            layout = self.action_slices
            if "acceleration" in layout:
                result[:, layout["acceleration"].start:layout["acceleration"].start+2] *= -1
            if "yaw" in layout:
                slot = layout["yaw"]
                if slot.stop-slot.start == 2:
                    result[:, slot] *= -1
                    # The environment defines the singular zero pair as zero
                    # radians. Its rotated representative must therefore be pi.
                    zero = (value[:, slot] == 0).all(-1)
                    result[zero, slot.start] = -1.
                else:
                    scale, centre, half_range = 1., 0., math.pi
                    if self.cfg.sac_action_mapping:
                        controller = self.cfg.indi if self.mode == "indi" else self.cfg.wrench
                        scale = half_range = math.radians(controller.heading_range_degs)
                    elif self.mode == "wrench":
                        centre = self.home_yaw
                    angle = value[:, slot]*scale
                    # Swap adjacent pi-wide cells. This is an involution on raw R,
                    # not a many-to-one wrap of unbounded PPO samples into [-pi, pi].
                    parity = torch.floor((angle-centre+half_range)/math.pi).remainder(2)
                    if self.cfg.sac_action_mapping:
                        # +range and -range are duplicate circle endpoints.
                        # Keep an exactly saturated +1 inside the SAC bounds.
                        parity = torch.where(angle == half_range, 1., parity)
                    result[:, slot] = (angle+(1-2*parity)*math.pi)/scale
            return result
        raise ValueError(f"Unsupported Borinot symmetry: {name}")

    def observations(self, value, name):
        if value.shape[-1] != self.observation_size:
            raise ValueError(f"Expected {self.observation_size} Borinot observations, got {value.shape[-1]}.")
        if name == "identity":
            return value.clone()
        result = value.clone()
        if name == "world_yaw_180":
            layout = self.obs_slices
            rotation = value[:, layout["rotation"]].reshape(-1, 3, 3)
            world = value.new_tensor((-1., -1., 1.))
            result[:, layout["rotation"]] = (world[None, :, None]*rotation).flatten(1)
            if "actions" in layout:
                result[:, layout["actions"]] = self.actions(value[:, layout["actions"]], name)
            return result
        raise ValueError(f"Unsupported Borinot symmetry: {name}")


def _get_transform(env):
    task = getattr(env, "unwrapped", env)
    cache_name = "_borinot_symmetry"
    transform = getattr(task, cache_name, None)
    if transform is None:
        transform = BorinotSymmetry(task.cfg)
        setattr(task, cache_name, transform)
    return transform


@torch.no_grad()
def compute_symmetric_observations_actions(env=None, obs=None, actions=None, obs_type="policy"):
    transform = _get_transform(env)
    augmented_obs = None
    if isinstance(obs, TensorDict):
        unknown = set(obs.keys())-{"policy", "critic"}
        if unknown:
            raise ValueError(f"Unmapped Borinot observation groups: {unknown}")
        data = {key: torch.cat([transform.observations(value, name) for name in transform.transforms], 0)
                for key, value in obs.items()}
        augmented_obs = TensorDict(data, batch_size=[obs.batch_size[0]*len(transform.transforms)], device=obs.device)
    elif obs is not None:
        augmented_obs = torch.cat([transform.observations(obs, name) for name in transform.transforms], 0)
    augmented_actions = None if actions is None else torch.cat(
        [transform.actions(actions, name) for name in transform.transforms], 0)
    return augmented_obs, augmented_actions
