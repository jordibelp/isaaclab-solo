# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


def _register(task_id: str, cfg_entry_point: str, rsl_rl_cfg_name: str = "Solo12RacePPORunnerCfg"):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.solo12_race_env:Solo12RaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": cfg_entry_point,
            "skrl_cfg_entry_point": "isaaclab_tasks.direct.solo12.agents:skrl_ppo_cfg_fast.yaml",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{rsl_rl_cfg_name}",
        },
    )


_register("Isaac-Solo12-Race-Direct-v0", f"{__name__}.solo12_race_env_cfg:Solo12RaceEnvCfg")
_register(
    "Isaac-Solo12-Race-IMU-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceIMUEnvCfg",
    "Solo12RaceIMUPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-JointStateTCN-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceJointStateTcnEnvCfg",
    "Solo12RaceJointStateTcnPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-JointStates_IMU_TCN-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceJointStateImuTcnEnvCfg",
    "Solo12RaceJointStateImuTcnPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-ParamsConditioned-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceParamsConditionedEnvCfg",
    "Solo12RaceParamsConditionedPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-ParamsConditionedEnc-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceParamsConditionedEncEnvCfg",
    "Solo12RaceParamsConditionedEncPPORunnerCfg",
)
_register(
    "Solo12-Race-ParamsConditionedEnc-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceParamsConditionedEncEnvCfg",
    "Solo12RaceParamsConditionedEncPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-ParamsDaggerJointStateTCN-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceParamsDaggerJointStateTcnEnvCfg",
    "Solo12RaceParamsConditionedEncPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-ParamsDaggerJointStates_IMU_TCN-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceParamsDaggerJointStateImuTcnEnvCfg",
    "Solo12RaceParamsConditionedEncPPORunnerCfg",
)
_register("Isaac-Solo12-Race-EvalCamera-Direct-v0", f"{__name__}.solo12_race_env_cfg:Solo12RaceEvalCameraEnvCfg")
_register(
    "Isaac-Solo12-Race-IMU-EvalCamera-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceIMUEvalCameraEnvCfg",
    "Solo12RaceIMUPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-JointStateTCN-EvalCamera-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceJointStateTcnEvalCameraEnvCfg",
    "Solo12RaceJointStateTcnPPORunnerCfg",
)
_register(
    "Isaac-Solo12-Race-JointStates_IMU_TCN-EvalCamera-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceJointStateImuTcnEvalCameraEnvCfg",
    "Solo12RaceJointStateImuTcnPPORunnerCfg",
)

# Backward-compatible aliases for the earlier vision naming.
_register("Isaac-Solo12-Race-Vision-Direct-v0", f"{__name__}.solo12_race_env_cfg:Solo12RaceEvalCameraEnvCfg")
_register(
    "Isaac-Solo12-Race-IMU-Vision-Direct-v0",
    f"{__name__}.solo12_race_env_cfg:Solo12RaceIMUEvalCameraEnvCfg",
    "Solo12RaceIMUPPORunnerCfg",
)
