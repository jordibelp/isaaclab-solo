"""VisualizationMarkers stand-in: decoration geoms drawn into the MuJoCo viewer scene."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from . import backend as mj_backend
from . import usd_scene


def _rows(value, count=None):
    if value is None:
        return None
    array = value.detach().cpu().double().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, float)
    return array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(1, -1)


class VisualizationMarkers:
    """Prototype shapes (sphere, cylinder, cone, cuboid) placed by ``visualize``; not physical."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._prototypes = [self._prototype(shape) for shape in cfg.markers.values()]
        self._translations = np.zeros((0, 3))
        self._orientations = self._scales = self._indices = None
        self._visible = True
        mj_backend.current().markers.append(self)

    @staticmethod
    def _prototype(shape):
        material = getattr(shape, "visual_material", None)
        color = tuple(getattr(material, "diffuse_color", (0.6, 0.6, 0.6)))
        emissive = tuple(getattr(material, "emissive_color", (0.0, 0.0, 0.0)))
        rgba = np.array((*color, float(getattr(material, "opacity", 1.0))), dtype=np.float32)
        kind = type(shape).__name__.removesuffix("Cfg").lower()
        if kind == "sphere":
            size = (shape.radius,) * 3
        elif kind in ("cylinder", "capsule", "cone"):
            if getattr(shape, "axis", "Z").upper() != "Z":
                raise NotImplementedError("Markers support Z-axis cylinders and cones only.")
            size = (shape.radius, shape.radius, shape.height)
        elif kind == "cuboid":
            size = tuple(0.5 * np.asarray(shape.size))
        else:
            raise NotImplementedError(f"Marker shape {type(shape).__name__} has no MuJoCo stand-in.")
        return kind, np.asarray(size, float), rgba, float(max(emissive) > 0.0) * 0.8

    def set_visibility(self, visible: bool):
        self._visible = bool(visible)

    def is_visible(self):
        return self._visible

    def visualize(self, translations=None, orientations=None, scales=None, marker_indices=None):
        """Update the instances. As in Isaac Lab, omitted arguments keep their previous values."""
        if translations is not None:
            self._translations = _rows(translations)
        if orientations is not None:
            self._orientations = _rows(orientations)
        if scales is not None:
            self._scales = _rows(scales)
        if marker_indices is not None:
            self._indices = np.asarray(marker_indices.cpu() if isinstance(marker_indices, torch.Tensor)
                                       else marker_indices, dtype=int).reshape(-1)

    def geoms(self):
        """(type, size, pos, mat, rgba, emission) for every visible instance."""
        if not self._visible:
            return
        for i, position in enumerate(self._translations):
            kind, size, rgba, emission = self._prototypes[0 if self._indices is None else self._indices[i]]
            scale = np.ones(3) if self._scales is None else self._scales[i]
            quat = np.array((1.0, 0.0, 0.0, 0.0)) if self._orientations is None else self._orientations[i]
            rotation = usd_scene.quat_to_matrix(quat)
            size = size * scale
            if not np.all(size > 1e-9):
                continue
            if kind == "sphere":
                yield mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, position, rotation.reshape(9), rgba, emission
            elif kind == "cuboid":
                yield mujoco.mjtGeom.mjGEOM_BOX, size, position, rotation.reshape(9), rgba, emission
            elif kind in ("cylinder", "capsule"):
                geom = mujoco.mjtGeom.mjGEOM_CYLINDER if kind == "cylinder" else mujoco.mjtGeom.mjGEOM_CAPSULE
                yield geom, np.array((size[0], size[1], 0.5 * size[2])), position, rotation.reshape(9), rgba, emission
            else:  # cone centred on its axis, like a USD cone: MuJoCo draws an arrow from its base
                base = position - 0.5 * size[2] * rotation[:, 2]
                yield (mujoco.mjtGeom.mjGEOM_ARROW, np.array((size[0], size[1], size[2])), base,
                       rotation.reshape(9), rgba, emission)
