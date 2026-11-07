"""Stand-ins for the Isaac Lab modules that need a running Isaac Sim.

Config classes here are plain keyword bags with Isaac Lab's defaults: enough for the
Borinot ``@configclass`` files, ``to_dict``/``from_dict`` and Hydra overrides. Runtime
classes (environment, scene, assets, sensors, markers) come from the MuJoCo backend.
Real Isaac Lab modules are used wherever they import without the simulator:
``isaaclab.utils``, ``isaaclab.envs.common``, ``isaaclab.envs.utils.spaces`` and the
``DirectRLEnvCfg`` definition itself.
"""

from __future__ import annotations

import copy
import importlib
import sys
import types


class _CfgMeta(type):
    def __getattr__(cls, name):
        if name.endswith("Cfg"):  # nested config classes such as ImuCfg.OffsetCfg
            nested = _cfg(name, cls.__module__)
            setattr(cls, name, nested)
            return nested
        raise AttributeError(f"{cls.__name__}.{name}")


class _Cfg(metaclass=_CfgMeta):
    """Keyword-only config bag. Class-level ``_defaults`` are deep-copied per instance."""

    _defaults: dict = {}

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError(f"{type(self).__name__} accepts keyword arguments only.")
        for klass in reversed(type(self).__mro__):
            for key, value in klass.__dict__.get("_defaults", {}).items():
                self.__dict__[key] = copy.deepcopy(value)
        self.__dict__.update(kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, **changes):
        new = copy.deepcopy(self)
        new.__dict__.update(changes)
        return new

    def to_dict(self):
        from isaaclab.utils.dict import class_to_dict

        return class_to_dict(self)

    def from_dict(self, data):
        from isaaclab.utils.dict import update_class_from_dict

        update_class_from_dict(self, data)

    def __repr__(self):
        fields = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{type(self).__name__}({fields})"


def _cfg(name: str, module: str, defaults: dict | None = None, base: type = _Cfg, func=None) -> type:
    attributes = {"_defaults": dict(defaults or {}), "__module__": module, "__qualname__": name}
    if func is not None:
        attributes["func"] = staticmethod(func)
    return type(name, (base,), attributes)


def _module(name: str, **attributes) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__dict__["__isaac_stub__"] = True
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, module)
    module.__dict__.update(attributes)
    return module


def _missing_attribute(module_name: str):
    def __getattr__(name):
        if name.endswith("Cfg"):
            # Unknown config bags are harmless placeholders; runtime classes are not.
            klass = _cfg(name, module_name)
            setattr(sys.modules[module_name], name, klass)
            return klass
        raise AttributeError(
            f"{module_name}.{name} is not available in the MuJoCo backend. "
            "Add a stand-in to mujoco/borinot_sim2sim/isaac_stubs.py if the task needs it.")
    return __getattr__


def _stub_function(module: str, name: str):
    def function(*args, **kwargs):
        raise NotImplementedError(f"{module}.{name} has no MuJoCo implementation.")
    function.__module__, function.__name__, function.__qualname__ = module, name, name
    return function


def install() -> None:
    """Insert every stand-in module into ``sys.modules``. Safe to call once."""
    if "isaaclab.sim" in sys.modules and getattr(sys.modules["isaaclab.sim"], "__isaac_stub__", False):
        return
    import isaaclab
    from . import usd_scene

    sim_name = "isaaclab.sim"
    shape_defaults = {
        "visual_material_path": "material", "visual_material": None,
        "physics_material_path": "material", "physics_material": None,
        "rigid_props": None, "collision_props": None, "mass_props": None,
        "activate_contact_sensors": False, "semantic_tags": None, "copy_from_source": True, "visible": True,
    }
    sim = _module(
        sim_name,
        SimulationCfg=_cfg("SimulationCfg", sim_name, {
            "physics_prim_path": "/physicsScene", "device": "cuda:0", "dt": 1.0 / 60.0, "render_interval": 1,
            "gravity": (0.0, 0.0, -9.81), "enable_scene_query_support": False, "use_fabric": True,
            "physx": None, "physics_material": None, "render": None, "create_stage_in_memory": False,
            "logging_level": "WARNING", "save_logs_to_file": True, "log_dir": None}),
        PhysxCfg=_cfg("PhysxCfg", sim_name, {"solver_type": 1, "bounce_threshold_velocity": 0.5,
                                              "enable_ccd": False, "enable_stabilization": False}),
        RenderCfg=_cfg("RenderCfg", sim_name),
        RigidBodyPropertiesCfg=_cfg("RigidBodyPropertiesCfg", sim_name),
        ArticulationRootPropertiesCfg=_cfg("ArticulationRootPropertiesCfg", sim_name),
        CollisionPropertiesCfg=_cfg("CollisionPropertiesCfg", sim_name),
        MassPropertiesCfg=_cfg("MassPropertiesCfg", sim_name, {"mass": None, "density": None}),
        RigidBodyMaterialCfg=_cfg("RigidBodyMaterialCfg", sim_name, {
            "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0, "improve_patch_friction": True,
            "friction_combine_mode": "average", "restitution_combine_mode": "average",
            "compliant_contact_stiffness": 0.0, "compliant_contact_damping": 0.0},
            func=usd_scene.spawn_physics_material),
        PreviewSurfaceCfg=_cfg("PreviewSurfaceCfg", sim_name, {
            "diffuse_color": (0.18, 0.18, 0.18), "emissive_color": (0.0, 0.0, 0.0), "roughness": 0.5,
            "metallic": 0.0, "opacity": 1.0}, func=usd_scene.spawn_preview_surface),
        UsdFileCfg=_cfg("UsdFileCfg", sim_name, {
            "usd_path": None, "scale": None, "variants": None, "rigid_props": None, "collision_props": None,
            "mass_props": None, "articulation_props": None, "fixed_tendons_props": None, "joint_drive_props": None,
            "visual_material_path": "material", "visual_material": None, "activate_contact_sensors": False,
            "semantic_tags": None, "copy_from_source": True, "visible": True}, func=usd_scene.spawn_from_usd),
        SphereCfg=_cfg("SphereCfg", sim_name, {**shape_defaults, "radius": None}, func=usd_scene.spawn_sphere),
        CylinderCfg=_cfg("CylinderCfg", sim_name, {**shape_defaults, "radius": None, "height": None, "axis": "Z"},
                         func=usd_scene.spawn_cylinder),
        CapsuleCfg=_cfg("CapsuleCfg", sim_name, {**shape_defaults, "radius": None, "height": None, "axis": "Z"},
                        func=usd_scene.spawn_capsule),
        ConeCfg=_cfg("ConeCfg", sim_name, {**shape_defaults, "radius": None, "height": None, "axis": "Z"},
                     func=usd_scene.spawn_cone),
        CuboidCfg=_cfg("CuboidCfg", sim_name, {**shape_defaults, "size": None}, func=usd_scene.spawn_cuboid),
        GroundPlaneCfg=_cfg("GroundPlaneCfg", sim_name, {
            "usd_path": None, "color": (0.0, 0.0, 0.0), "size": (100.0, 100.0), "physics_material": None,
            "visible": True}, func=usd_scene.spawn_ground_plane),
        DomeLightCfg=_cfg("DomeLightCfg", sim_name, {"intensity": 1.0, "color": (1.0, 1.0, 1.0)},
                          func=usd_scene.spawn_light),
        DistantLightCfg=_cfg("DistantLightCfg", sim_name, {"intensity": 1.0, "color": (1.0, 1.0, 1.0)},
                             func=usd_scene.spawn_light),
        get_current_stage=usd_scene.get_current_stage,
        bind_visual_material=usd_scene.bind_visual_material,
        bind_physics_material=usd_scene.bind_physics_material,
    )
    sim.__getattr__ = _missing_attribute(sim_name)

    actuators_name = "isaaclab.actuators"
    actuator_defaults = {
        "joint_names_expr": None, "effort_limit": None, "velocity_limit": None, "effort_limit_sim": None,
        "velocity_limit_sim": None, "stiffness": None, "damping": None, "armature": None, "friction": None,
        "dynamic_friction": None, "viscous_friction": None}
    actuators = _module(
        actuators_name,
        ActuatorBaseCfg=_cfg("ActuatorBaseCfg", actuators_name, actuator_defaults),
        ImplicitActuatorCfg=_cfg("ImplicitActuatorCfg", actuators_name, actuator_defaults),
        IdealPDActuatorCfg=_cfg("IdealPDActuatorCfg", actuators_name, actuator_defaults),
        DCMotorCfg=_cfg("DCMotorCfg", actuators_name, actuator_defaults),
        IdealPDActuator=type("IdealPDActuator", (), {"__module__": actuators_name}),
    )
    actuators.__getattr__ = _missing_attribute(actuators_name)

    assets_name = "isaaclab.assets"
    articulation_state = _cfg("InitialStateCfg", assets_name, {
        "pos": (0.0, 0.0, 0.0), "rot": (1.0, 0.0, 0.0, 0.0), "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0), "joint_pos": {".*": 0.0}, "joint_vel": {".*": 0.0}})
    rigid_state = _cfg("InitialStateCfg", assets_name, {
        "pos": (0.0, 0.0, 0.0), "rot": (1.0, 0.0, 0.0, 0.0), "lin_vel": (0.0, 0.0, 0.0), "ang_vel": (0.0, 0.0, 0.0)})
    articulation_cfg = _cfg("ArticulationCfg", assets_name, {
        "prim_path": None, "spawn": None, "init_state": articulation_state(), "collision_group": 0,
        "debug_vis": False, "articulation_root_prim_path": None, "soft_joint_pos_limit_factor": 1.0,
        "actuators": {}, "actuator_value_resolution_debug_print": False})
    articulation_cfg.InitialStateCfg = articulation_state
    rigid_cfg = _cfg("RigidObjectCfg", assets_name, {
        "prim_path": None, "spawn": None, "init_state": rigid_state(), "collision_group": 0, "debug_vis": False})
    rigid_cfg.InitialStateCfg = rigid_state
    assets = _module(assets_name, ArticulationCfg=articulation_cfg, RigidObjectCfg=rigid_cfg,
                     AssetBaseCfg=_cfg("AssetBaseCfg", assets_name, {"prim_path": None, "spawn": None}))

    scene_name = "isaaclab.scene"
    scene = _module(scene_name, InteractiveSceneCfg=_cfg("InteractiveSceneCfg", scene_name, {
        "num_envs": 1, "env_spacing": 0.0, "lazy_sensor_update": True, "replicate_physics": True,
        "filter_collisions": True, "clone_in_fabric": False}))

    sensors_name = "isaaclab.sensors"
    sensors = _module(sensors_name, ContactSensorCfg=_cfg("ContactSensorCfg", sensors_name, {
        "prim_path": None, "update_period": 0.0, "history_length": 0, "track_pose": False,
        "track_air_time": False, "force_threshold": 1.0, "debug_vis": False, "filter_prim_paths_expr": []}))

    managers_name = "isaaclab.managers"

    class SceneEntityCfg(_Cfg):
        _defaults = {"name": None, "joint_names": None, "body_names": None, "preserve_order": False}

        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)

    SceneEntityCfg.__module__ = managers_name
    _module(managers_name, SceneEntityCfg=SceneEntityCfg, EventTermCfg=_cfg("EventTermCfg", managers_name, {
        "func": None, "params": {}, "mode": None, "interval_range_s": None, "is_global_time": False,
        "min_step_count_between_reset": 0}))

    devices = _module("isaaclab.devices")
    _module("isaaclab.devices.openxr", XrCfg=_cfg("XrCfg", "isaaclab.devices.openxr"))
    devices.openxr = sys.modules["isaaclab.devices.openxr"]

    # isaaclab.envs keeps its real directory so common.py, utils/spaces.py and
    # direct_rl_env_cfg.py load from source; only the simulator-bound classes are replaced.
    envs_name = "isaaclab.envs"
    envs = _module(envs_name)
    envs.__path__ = [str(sys.modules["isaaclab"].__path__[0] + "/envs")]
    _module(envs_name + ".ui", BaseEnvWindow=type("BaseEnvWindow", (), {"__module__": envs_name + ".ui"}))
    mdp_name = envs_name + ".mdp"
    mdp = _module(mdp_name)
    # Other tasks' configs (e.g. Solo12, imported by the Borinot agent configs) name more terms.
    def mdp_attribute(name):
        if name.startswith("__"):  # module protocol lookups (__file__, __path__, ...)
            raise AttributeError(name)
        return _stub_function(mdp_name, name)

    mdp.__getattr__ = mdp_attribute
    for module in (assets, scene, sensors, sys.modules[managers_name], _module("isaaclab.terrains")):
        module.__getattr__ = _missing_attribute(module.__name__)
    common = importlib.import_module(envs_name + ".common")
    direct_cfg = importlib.import_module(envs_name + ".direct_rl_env_cfg")
    envs.ViewerCfg = common.ViewerCfg
    envs.VecEnvObs = common.VecEnvObs
    envs.VecEnvStepReturn = common.VecEnvStepReturn
    envs.DirectRLEnvCfg = direct_cfg.DirectRLEnvCfg
    envs.ManagerBasedRLEnvCfg = type("ManagerBasedRLEnvCfg", (), {"__module__": envs_name})
    envs.ManagerBasedEnvCfg = envs.ManagerBasedRLEnvCfg
    envs.ManagerBasedRLEnv = type("ManagerBasedRLEnv", (), {"__module__": envs_name})
    envs.ManagerBasedEnv = envs.ManagerBasedRLEnv

    from . import assets as mj_assets
    from . import env as mj_env
    from . import markers as mj_markers
    from . import omni_ui_tk

    # Startup terms of the Borinot event configs (opt-in in the MuJoCo players).
    for name in ("randomize_rigid_body_mass", "randomize_rigid_body_com", "randomize_joint_parameters"):
        setattr(mdp, name, getattr(mj_env, name))
    sim.SimulationContext = mj_env.SimulationContext
    assets.Articulation = mj_assets.Articulation
    assets.RigidObject = mj_assets.RigidObject
    scene.InteractiveScene = mj_env.InteractiveScene
    sensors.ContactSensor = mj_assets.ContactSensor
    envs.DirectRLEnv = mj_env.DirectRLEnv
    _module("isaaclab.markers", VisualizationMarkers=mj_markers.VisualizationMarkers,
            VisualizationMarkersCfg=_cfg("VisualizationMarkersCfg", "isaaclab.markers",
                                         {"prim_path": None, "markers": {}}))
    omni = _module("omni")
    omni.__path__ = []
    omni.ui = omni_ui_tk.load_module()
    sys.modules["omni.ui"] = omni.ui
    for name in ("sim", "actuators", "assets", "scene", "sensors", "managers", "devices", "envs", "markers",
                 "terrains"):
        setattr(isaaclab, name, sys.modules[f"isaaclab.{name}"])
