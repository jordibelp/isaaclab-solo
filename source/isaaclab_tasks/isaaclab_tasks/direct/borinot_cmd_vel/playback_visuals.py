# SPDX-License-Identifier: BSD-3-Clause
"""Read-only, robot-0 flight overlays. No markers are created until enabled."""
import torch

from isaaclab_tasks.controllers.borinot_indi import rotation_from_quat
from .task_math import heading_rotation

# label, RGB, metres per physical unit, value units
OVERLAYS = {
    "rotor_command": ("Rotor commands / orange / 0.05 m/N", (1.0, 0.45, 0.05), 0.05, "N"),
    "rotor_actual": ("Realised thrust / cyan / 0.05 m/N", (0.05, 0.85, 1.0), 0.05, "N"),
    "policy_rate": ("Policy rate / magenta / 0.25 m/(rad/s)", (1.0, 0.1, 0.8), 0.25, "rad/s body"),
    "actual_rate": ("Measured rate / violet / 0.25 m/(rad/s)", (0.55, 0.3, 1.0), 0.25, "rad/s body"),
    "command_velocity": ("Velocity command / green / 1 m/(m/s)", (0.15, 1.0, 0.15), 1.0, "m/s heading"),
    "actual_velocity": ("Measured velocity / blue / 1 m/(m/s)", (0.15, 0.4, 1.0), 1.0, "m/s heading"),
    "disturbance": ("External push / red / 0.03 m/N", (1.0, 0.1, 0.1), 0.03, "N world Z"),
}


def overlay_vectors(env, enabled):
    """Return world-space arrow origins, scaled vectors, and unscaled readouts.

    Rotor commands are final clipped allocator outputs, before speed lag.
    Policy rates are the latest environment-processed policy outputs, BEFORE any
    optional actuator delay. They are not a world-Z yaw command or INDI alpha.
    """
    data, cfg = env.robot.data, env.cfg
    position = data.root_pos_w[0]
    rotation = rotation_from_quat(data.root_quat_w[:1])[0]
    heading = heading_rotation(data.root_quat_w[:1])[0]
    result = {}
    for key in enabled:
        origin = position.unsqueeze(0)
        if key in ("rotor_command", "rotor_actual"):
            values = (env._actuator_commands[0, :6] * (env._hover_per_rotor * cfg.thrust_to_weight)
                      if key == "rotor_command" else env._rotor_thrust[0])
            hubs = position.new_tensor(cfg.rotor_pos_b)
            # Separate paired command/realised arrows slightly along body X.
            hubs[:, 0] += -0.012 if key == "rotor_command" else 0.012
            hubs[:, 2] += 0.025
            origin = position + hubs @ rotation.T
            vector = values[:, None] * rotation[:, 2]
        elif key == "policy_rate":
            if cfg.control_mode not in ("cbrt", "cbrt_indi"):
                continue
            values = env.actions[0, :3]
            if cfg.sac_action_mapping:
                values = values.clamp(-1, 1) * values.new_tensor(cfg.body_rate_scale)
            vector = (rotation @ values).unsqueeze(0)
        elif key == "actual_rate":
            values = data.root_ang_vel_b[0]
            vector = (rotation @ values).unsqueeze(0)
        elif key == "command_velocity":
            values = env._commands[0, :3]
            vector = (heading @ values).unsqueeze(0)
        elif key == "actual_velocity":
            values = heading.T @ data.root_lin_vel_w[0]
            vector = data.root_lin_vel_w[:1]
        elif key == "disturbance":
            values = env._base_push_force_z_w[:1]
            origin = (position + rotation @ env._base_push_points_b[0]).unsqueeze(0)
            vector = position.new_zeros(1, 3)
            vector[:, 2] = values
        result[key] = (origin, vector * OVERLAYS[key][2], values)
    return result


def arrow_instances(origins, vectors, size=1.0):
    """Cylinder + cone transforms, local +Z. Zero arrows have zero scale.

    End-to-end length is size * |vector|. Short arrows shrink their heads rather
    than imposing a minimum length that would misrepresent near-zero signals.
    """
    vectors = vectors * size
    length = vectors.norm(dim=-1)
    direction = vectors / length.clamp_min(1e-12)[:, None]
    direction = torch.where((length > 1e-12)[:, None], direction,
                            direction.new_tensor((0., 0., 1.)))
    # Shortest rotation from local +Z; handle exactly -Z without a zero quaternion.
    axis = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(length)), -1)
    sine = axis.norm(dim=-1)
    half_angle = torch.atan2(sine, direction[:, 2]) / 2
    quaternion = torch.cat((half_angle.cos()[:, None],
                             axis / sine.clamp_min(1e-12)[:, None] * half_angle.sin()[:, None]), -1)
    down = (sine < 1e-12) & (direction[:, 2] < 0)
    quaternion[down] = quaternion.new_tensor((0., 1., 0., 0.))
    quaternion /= quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    head = torch.minimum(length * 0.3, length.new_full(length.shape, 0.045 * size))
    shaft = length - head
    positions = torch.cat((origins + direction * (shaft / 2)[:, None],
                            origins + direction * (length - head / 2)[:, None]))
    radius = torch.minimum(length * 0.12, length.new_full(length.shape, 0.004 * size))
    shaft_scale = torch.stack((radius, radius, shaft), -1)
    head_scale = torch.stack((radius * 2.8, radius * 2.8, head), -1)
    return positions, quaternion.repeat(2, 1), torch.cat((shaft_scale, head_scale)), [0]*len(length) + [1]*len(length)


class FlightVisuals:
    def __init__(self, env):
        self.env, self.size = env, 1.0
        self.enabled = {key: False for key in OVERLAYS}
        self.markers, self.values = {}, {}

    def set_enabled(self, key, enabled):
        self.enabled[key] = bool(enabled)
        if not enabled:
            self.values.pop(key, None)
            if key in self.markers:
                self.markers[key].set_visibility(False)

    def update(self):
        enabled = [key for key, active in self.enabled.items() if active]
        if not enabled:
            return
        # Local primitive prototypes need no downloads, collisions, or debug-draw
        # singleton. Toggling these cannot erase another feature's debug lines.
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        for key, (origins, vectors, values) in overlay_vectors(self.env, enabled).items():
            if key not in self.markers:
                material = sim_utils.PreviewSurfaceCfg(diffuse_color=OVERLAYS[key][1],
                                                       emissive_color=OVERLAYS[key][1])
                self.markers[key] = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path=f"/Visuals/BorinotFlight/{key}", markers={
                        "shaft": sim_utils.CylinderCfg(radius=1., height=1., visual_material=material),
                        "head": sim_utils.ConeCfg(radius=1., height=1., visual_material=material),
                    }))
            marker = self.markers[key]
            marker.set_visibility(True)
            positions, orientations, scales, indices = arrow_instances(origins, vectors, self.size)
            marker.visualize(translations=positions, orientations=orientations, scales=scales, marker_indices=indices)
            self.values[key] = values.detach()

    def readout(self):
        return "\n".join(f"{key.replace('_', ' ')}: " +
                         ", ".join(f"{v:+.2f}" for v in values.cpu().tolist()) + f" {OVERLAYS[key][3]}"
                         for key, values in self.values.items())

    def close(self):
        for marker in self.markers.values():
            marker.set_visibility(False)
        self.values.clear()
