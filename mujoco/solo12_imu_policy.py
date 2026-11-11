"""Simulator-free loading of the production Solo12 teacher and history policies."""

from __future__ import annotations

import importlib.util
import json
from functools import cache
from pathlib import Path

import numpy as np
import torch


TASK_MODELS = {
    "solo12-two-feet-IMU-teacher": "base_imu_teacher",
    "solo12-two-feet-IMU-student-dagger": "base_imu_student_dagger",
    "solo12-two-feet-IMU-student-rl": "base_imu_student_rl",
}
DEFAULT_CONTRACT = {
    "include_foot_height_obs": False,
    "include_lin_acc_steps_in_imu_obs": 4,
    "teacher_include_base_force": True,
    "teacher_foot_force_obs": "norm",
    "action_obs_is_last_executed": False,
    "base_imu_history_action_is_last_executed": False,
    "imu_ekf_processed_inputs": True,
    "use_rotMat_on_imu_encoder": False,
    "apply_tanh_to_latent": True,
    "decimation": 4,
    "physics_dt": 0.005,
}


@cache
def production_modules():
    """Import the real networks/contracts by file, without importing Isaac Sim packages."""
    root = Path(__file__).resolve().parents[1]
    modules = []
    for name, path in (
        ("solo12_mujoco_imu_models", root / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/base_imu_actor_critic.py"),
        ("solo12_mujoco_imu_contract", root / "source/scripts/rsl_rl/solo12_imu_pipeline.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def saved_contract(path: Path, payload: dict) -> tuple[dict, dict]:
    saved = {}
    for parent in (path.parent, path.parent.parent):
        sidecar = parent / "params/solo12_imu_contract.json"
        if sidecar.is_file():
            saved = json.loads(sidecar.read_text())
            break
    contract = {**saved.get("observation_contract", {}), **payload.get("observation_contract", {})}
    contract.update((payload.get("infos") or {}).get("solo12_imu_latent") or {})
    return contract, {**saved.get("policy", {}), **payload.get("teacher_policy", {})}


class EncodedPolicy(torch.nn.Module):
    """Use production PPO/DAgger arithmetic, normalization and TCN padding unchanged."""

    def __init__(self, checkpoint: Path, task: str, overrides: dict, teacher_checkpoint: Path | None = None):
        super().__init__()
        models, pipeline = production_modules()
        self.policy_model = TASK_MODELS[task]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        is_adapter = "adapter_state_dict" in payload
        if is_adapter != (self.policy_model == "base_imu_student_dagger"):
            raise ValueError("Select the DAgger task for an adapter checkpoint, or the teacher/student-rl task for PPO.")
        saved, policy_cfg = saved_contract(checkpoint, payload)
        if is_adapter:
            if payload.get("layout", {}).get("kind", "base_imu") != "base_imu":
                raise ValueError("The two-feet IMU player requires a base_imu adapter, not a race adapter.")
            saved["apply_tanh_to_latent"] = pipeline.adapter_latent_tanh(payload)
        if "teacher_latent_activation" in saved:
            bounded = saved["teacher_latent_activation"] == "tanh"
            if saved["teacher_latent_activation"] not in ("tanh", "identity"):
                raise ValueError("Unknown saved teacher latent activation.")
            if "apply_tanh_to_latent" in saved and saved["apply_tanh_to_latent"] != bounded:
                raise ValueError("Saved teacher and student latent activation settings disagree.")
            saved["apply_tanh_to_latent"] = bounded
        for name, value in saved.items():
            if name in overrides and name != "apply_tanh_to_latent" and overrides[name] != value:
                raise ValueError(f"env.{name}={overrides[name]} conflicts with the saved observation contract ({value}).")
        # Saved parameter-free activation wins, just as in the Isaac public player.
        self.contract = {**DEFAULT_CONTRACT, **overrides, **saved}
        if self.contract["decimation"] != 4 or self.contract["physics_dt"] != 0.005:
            raise ValueError("MuJoCo playback requires the trained 200 Hz physics / 50 Hz policy clock.")
        if overrides.get("base_obs_from_imu", False) or overrides.get("remove_root_lin_vel_b_from_actor_obs", False):
            raise ValueError("base_obs_from_imu/remove_root_lin_vel_b_from_actor_obs are simple_mlp-only flags.")
        if overrides.get("remove_root_lin_vel_b_from_obs", False):
            raise ValueError("Encoded teacher layouts keep base linear velocity; remove_root_lin_vel_b_from_obs is unsupported.")
        if not self.contract["imu_ekf_processed_inputs"] or self.contract["use_rotMat_on_imu_encoder"]:
            raise ValueError("MuJoCo history playback supports processed gyro/acceleration/projected-gravity IMU inputs only.")
        if self.contract["teacher_foot_force_obs"] not in ("none", "norm", "vector"):
            raise ValueError("teacher_foot_force_obs must be none, norm or vector.")
        if type(self.contract["apply_tanh_to_latent"]) is not bool:
            raise ValueError("apply_tanh_to_latent must be a boolean.")
        self.contract["teacher_latent_activation"] = "tanh" if self.contract["apply_tanh_to_latent"] else "identity"
        self.teacher_checkpoint = None
        activation = policy_cfg.get("activation", overrides.get("policy_activation", "elu"))
        if is_adapter:
            teacher_path = Path(teacher_checkpoint or payload.get("teacher_checkpoint", "")).expanduser()
            if not teacher_path.is_file():
                raise ValueError("DAgger playback needs its frozen teacher checkpoint. Pass --teacher-checkpoint=/local/teacher.pt.")
            self.teacher_checkpoint = str(teacher_path.resolve())
            teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
            _, teacher_cfg = saved_contract(teacher_path, teacher_payload)
            state = teacher_payload["model_state_dict"]
            self.network = self._teacher(models, pipeline, state, activation=policy_cfg.get("activation", teacher_cfg.get("activation", activation)))
            tcn_state = payload["adapter_state_dict"]
            self.history_encoder = self._tcn(models, tcn_state, payload.get("layout", {}))
            dims = payload["dims"]
            if (dims["history_len"] != self.history_len or dims["history_sample_dim"] != 45
                    or dims["latent_dim"] != self.history_encoder.to_latent[1].out_features
                    or dims["teacher_obs_dim"] != self.network.teacher_encoder_obs_dim + 3):
                raise ValueError("DAgger adapter dimensions disagree with its teacher or history encoder.")
            from rsl_rl.networks import EmpiricalNormalization

            normalizer_state = payload.get("history_normalizer_state_dict")
            self.history_normalizer = EmpiricalNormalization(self.history_len * 45) if normalizer_state else torch.nn.Identity()
            if normalizer_state:
                self.history_normalizer.load_state_dict(normalizer_state, strict=True)
            self.input_dim = self.history_len * 45 + 3
            self.checkpoint_format = "dagger"
        else:
            state = payload.get("model_state_dict", {})
            if self.policy_model == "base_imu_teacher":
                self.network = self._teacher(models, pipeline, state, activation=activation)
                self.input_dim = self.network.teacher_encoder_obs_dim + 3
            else:
                tcn_state = {key.removeprefix("actor_history_encoder."): value for key, value in state.items()
                             if key.startswith("actor_history_encoder.")}
                encoder = self._tcn(models, tcn_state, {"activation": self.contract.get("base_imu_tcn_activation", "elu")})
                channels, _, kernel = tcn_state["conv_stack.0.conv.weight"].shape
                actor_in, actor_hidden, _ = pipeline.mlp_shape(state, "actor")
                critic_in, critic_hidden, _ = pipeline.mlp_shape(state, "critic")
                feed_history = any(key.startswith("critic_history_encoder.") for key in state)
                latent = encoder.to_latent[1].out_features
                if actor_in != latent + 3:
                    raise ValueError("Student latent and actor dimensions disagree.")
                teacher_dim = critic_in - (latent if feed_history else 0)
                self.input_dim = self.history_len * 45 + 3
                obs = {"policy": torch.zeros(1, self.input_dim),
                       "critic": torch.zeros(1, teacher_dim + (self.history_len * 45 if feed_history else 0))}
                self.network = models.Solo12BaseImuStudentActorCritic(
                    obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
                    actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden, activation=activation,
                    actor_obs_normalization="actor_obs_normalizer._mean" in state,
                    critic_obs_normalization="critic_obs_normalizer._mean" in state,
                    noise_std_type="log" if "log_std" in state else "scalar",
                    history_len=self.history_len, history_sample_dim=45, teacher_critic_obs_dim=teacher_dim,
                    tcn_channels=channels, tcn_latent_dim=latent, tcn_kernel_size=kernel,
                    tcn_activation=self.contract["base_imu_tcn_activation"],
                    apply_tanh_to_latent=self.contract["apply_tanh_to_latent"], feed_history_encoding_to_critic=feed_history,
                )
                self.network.load_state_dict(state, strict=True)
            self.checkpoint_format = "ppo"
        if self.policy_model != "base_imu_teacher":
            if (self.history_len - 1) % 4:
                raise ValueError("Student history must contain both endpoints of whole 50 Hz policy intervals.")
            requested = overrides.get("base_imu_history_policy_steps")
            if requested is not None and 4 * requested + 1 != self.history_len:
                raise ValueError("base_imu_history_policy_steps disagrees with the checkpoint's TCN.")
            self.contract["history_len"] = self.history_len
        self.eval()

    def _teacher(self, models, pipeline, state, *, activation):
        encoder_in, encoder_hidden, latent = pipeline.mlp_shape(state, "teacher_encoder")
        actor_in, actor_hidden, _ = pipeline.mlp_shape(state, "actor")
        critic_in, critic_hidden, _ = pipeline.mlp_shape(state, "critic")
        c = self.contract
        expected = (45 + 4 * c["include_foot_height_obs"] + 3 * c["include_lin_acc_steps_in_imu_obs"]
                    + 3 * c["teacher_include_base_force"] + {"none": 0, "norm": 4, "vector": 12}[c["teacher_foot_force_obs"]])
        if encoder_in != expected or critic_in != encoder_in + 3 or actor_in != latent + 3:
            raise ValueError(f"Teacher observation contract mismatch: checkpoint encoder={encoder_in}, playback={expected}.")
        obs = {key: torch.zeros(1, encoder_in + 3) for key in ("policy", "critic")}
        network = models.Solo12BaseImuTeacherActorCritic(
            obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
            teacher_encoder_obs_dim=encoder_in, teacher_latent_dim=latent, teacher_encoder_hidden_dims=encoder_hidden,
            actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden, activation=activation,
            teacher_latent_activation=c["teacher_latent_activation"],
            actor_obs_normalization="actor_obs_normalizer._mean" in state,
            critic_obs_normalization="critic_obs_normalizer._mean" in state,
            noise_std_type="log" if "log_std" in state else "scalar",
        )
        network.load_state_dict(state, strict=True)
        return network

    def _tcn(self, models, state, layout):
        if "conv_stack.0.conv.weight" not in state:
            raise ValueError("Student checkpoint has no base-IMU TCN weights; select a matching task/checkpoint.")
        channels, sample_dim, kernel = state["conv_stack.0.conv.weight"].shape
        latent, flat_dim = state["to_latent.1.weight"].shape
        if sample_dim != 45 or flat_dim % channels:
            raise ValueError("Student TCN does not use the 45-value processed-IMU history layout.")
        self.history_len = flat_dim // channels
        self.contract["base_imu_tcn_activation"] = layout.get("activation", "elu")
        encoder = models.BaseImuTcnEncoder(
            history_len=self.history_len, sample_dim=sample_dim, channels=channels, latent_dim=latent,
            kernel_size=kernel, activation=self.contract["base_imu_tcn_activation"],
            apply_tanh_to_latent=self.contract["apply_tanh_to_latent"],
        )
        encoder.load_state_dict(state, strict=True)
        return encoder

    @torch.inference_mode()
    def forward(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self.checkpoint_format == "dagger":
            teacher_obs = torch.zeros(1, self.network.teacher_encoder_obs_dim + 3)
            teacher_obs[:, -3:] = obs[:, -3:]
            command = self.network.actor_obs_normalizer(teacher_obs)[:, -3:]
            latent = self.history_encoder(self.history_normalizer(obs[:, :-3]))
            result = self.network.actor(torch.cat((latent, command), dim=-1))
        else:
            result = self.network.act_inference({"policy": obs})
        return result.squeeze(0).numpy()
