# SPDX-License-Identifier: BSD-3-Clause
"""Direct-actuator Borinot policies with a replaceable perception encoder."""
import gymnasium as gym

for task, family in (("goto", "GoTo"), ("cmd-vel", "Velocity")):
    for stage, config in (("teacher", "Teacher"), ("student-dagger", "Dagger"), ("student-rl", "Student")):
        gym.register(
            id=f"Borinot-{task}-{stage}",
            entry_point=f"{__name__}.env:{family}AdaptiveEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.cfg:{family}{config}Cfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:{family}{'Student' if config == 'Student' else 'Teacher'}Runner",
            },
        )
