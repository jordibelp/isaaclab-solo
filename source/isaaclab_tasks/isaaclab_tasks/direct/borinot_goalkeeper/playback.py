# SPDX-License-Identifier: BSD-3-Clause
"""Play-only scenery. The registered training environment is never modified."""

import torch
from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject

from .borinot_goalkeeper_env import BorinotGoalkeeperEnv


class BorinotGoalkeeperPlayEnv(BorinotGoalkeeperEnv):
    """Continue resolved trajectories without letting old balls affect the policy.

    At most ``keep_balls`` balls per environment are on the field, including the
    live shot. Retired balls collide with scenery, not the robot or other balls.
    """

    def __init__(self, cfg, render_mode=None, keep_balls=5, **kwargs):
        if not 0 <= keep_balls <= 20:
            raise ValueError("keep_balls must be between 0 and 20.")
        self.keep_balls = keep_balls
        self._reset_to_home = False
        super().__init__(cfg, render_mode, **kwargs)
        self._history_used = torch.zeros((self.num_envs, keep_balls), dtype=torch.bool, device=self.device)
        self._history_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._history_properties = [
            (ball.root_physx_view.get_masses(), ball.root_physx_view.get_inertias(),
             ball.root_physx_view.get_material_properties()) for ball in self.history_balls
        ]

    def _setup_scene(self):
        super()._setup_scene()
        stage = sim_utils.get_current_stage()
        # Visual-only frame and translucent goal: the original solid collider
        # stays in place, but orbit/keeper views do not hide behind an opaque wall.
        gx, gy, gz = self.cfg.goal_center
        depth, width, height = self.cfg.goal_size
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.26, 0.32), opacity=0.22)
        material.func("/World/PlayGoalMaterial", material)
        floor_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.045, 0.065, 0.095), roughness=0.8)
        floor_material.func("/World/PlayFloorMaterial", floor_material)
        stage.GetPrimAtPath("/World/Light").GetAttribute("inputs:intensity").Set(650.0)
        stage.GetPrimAtPath("/World/Light").GetAttribute("visibleInPrimaryRay").Set(False)
        for i in range(self.num_envs):
            sim_utils.bind_visual_material(f"/World/envs/env_{i}/Goal", "/World/PlayGoalMaterial")
            sim_utils.bind_visual_material(f"/World/envs/env_{i}/Floor", "/World/PlayFloorMaterial")
        for name, size, offset in (
            ("Left", (0.025, 0.025, height), (0., -width/2, 0.)),
            ("Right", (0.025, 0.025, height), (0., width/2, 0.)),
            ("Top", (0.025, width, 0.025), (0., 0., height/2)),
            ("Bottom", (0.025, width, 0.025), (0., 0., -height/2)),
        ):
            frame = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.70, 0.66)))
            frame.func(f"/World/envs/env_.*/GoalFrame{name}", frame,
                       translation=(gx + depth/2 + 0.01, gy + offset[1], gz + offset[2]))
        self.history_balls = []
        for slot in range(self.keep_balls):
            cfg = self.cfg.ball.copy()
            cfg.prim_path = f"/World/envs/env_.*/HistoryBall_{slot}"
            cfg.spawn.activate_contact_sensors = False
            cfg.spawn.visual_material.diffuse_color = (0.28, 0.68, 0.86)
            cfg.init_state.pos = (0.0, 0.0, -20.0 - slot)
            ball = RigidObject(cfg)
            self.history_balls.append(ball)
            self.scene.rigid_objects[f"history_ball_{slot}"] = ball
        # Filter pairs before PhysX starts. Keeping floor/goal collisions gives
        # post-shot bounces, while old balls cannot push the keeper or new shot.
        for env_index in range(self.num_envs):
            prefix = f"/World/envs/env_{env_index}"
            robot_bodies = [prim.GetPath() for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix + "/Robot"))
                            if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            history_paths = [f"{prefix}/HistoryBall_{i}" for i in range(self.keep_balls)]
            for path in history_paths:
                prim = stage.GetPrimAtPath(path)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableSpeculativeCCDAttr().Set(True)
                pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                for target in [*robot_bodies, prefix + "/Ball", *history_paths]:
                    if str(target) != path:
                        pairs.AddTarget(target)

    def _retire_ball(self, env_ids):
        """Copy the *post-contact* pose, velocity and randomized physical properties."""
        if not self.keep_balls:
            return
        slots = self._history_cursor[env_ids].clone()
        for slot, ball in enumerate(self.history_balls):
            ids = env_ids[slots == slot]
            if not len(ids):
                continue
            ids_cpu = ids.cpu()
            masses, inertias, materials = self._history_properties[slot]
            masses[ids_cpu] = self._ball_masses[ids_cpu]
            inertias[ids_cpu] = self._ball_inertias[ids_cpu]
            materials[ids_cpu] = self._ball_materials[ids_cpu]
            ball.root_physx_view.set_masses(masses, ids_cpu)
            ball.root_physx_view.set_inertias(inertias, ids_cpu)
            ball.root_physx_view.set_material_properties(materials, ids_cpu)
            ball.write_root_state_to_sim(self.ball.data.root_state_w[ids].clone(), ids)
            self._history_used[ids, slot] = True
        self._history_cursor[env_ids] = (slots + 1) % self.keep_balls

    def _park_ball(self, env_ids):
        self._retire_ball(env_ids)
        super()._park_ball(env_ids)

    def _launch_ball(self, env_ids):
        # A launch evicts the oldest ball only when the field is full. No
        # wall-clock expiry: slow-motion and pauses keep the same last N balls.
        if self.keep_balls:
            full = self._history_used[env_ids].sum(-1) >= self.keep_balls
            for slot, ball in enumerate(self.history_balls):
                ids = env_ids[full & (self._history_cursor[env_ids] == slot)]
                if len(ids):
                    state = ball.data.default_root_state[ids].clone()
                    state[:, :3] = self.scene.env_origins[ids]
                    state[:, 2] -= 20 + slot
                    state[:, 7:] = 0
                    ball.write_root_state_to_sim(state, ids)
                    self._history_used[ids, slot] = False
        super()._launch_ball(env_ids)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Preserve an unfinished trajectory on an automatic timeout or manual
        # reset, but do not turn an abandoned shot into a save or goal.
        live = env_ids[self._ball_active[env_ids]]
        self._retire_ball(live)
        super()._reset_idx(env_ids)
        if self._reset_to_home:
            root = self.robot.data.root_state_w[env_ids].clone()
            root[:, :3] = self._home + self.scene.env_origins[env_ids]
            self.robot.write_root_state_to_sim(root, env_ids)

    def reset_borinot(self):
        """Reset pose, velocity, joints, controller and current shot; keep history."""
        self._reset_to_home = True
        try:
            return self.reset()
        finally:
            self._reset_to_home = False
