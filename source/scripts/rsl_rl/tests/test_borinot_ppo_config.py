# SPDX-License-Identifier: BSD-3-Clause
"""Check the matched PPO recipe and both task bindings without importing Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[4]
TASKS = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct"


def configs():
    # Execute the real class definitions; simulator-backed imports are not needed here.
    namespace = {"configclass": lambda cls: cls, "RslRlOnPolicyRunnerCfg": object,
                 "RslRlPpoActorCriticCfg": SimpleNamespace, "RslRlLocalRedundancyCfg": SimpleNamespace,
                 "Solo12PpoAlgorithmCfg": SimpleNamespace}
    for name in ("borinot", "borinot_goalkeeper"):
        tree = ast.parse((TASKS / name / "agents/rsl_rl_ppo_cfg.py").read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=classes, type_ignores=[]), name, "exec"), namespace)
    return namespace["BorinotPPORunnerCfg"](), namespace["BorinotGoalkeeperPPORunnerCfg"]()


@pytest.mark.parametrize("index", [0, 1])
def test_matches_reference_update_batch_and_epochs(index):
    cfg = configs()[index]
    assert cfg.num_steps_per_env == 24
    assert 10000 * cfg.num_steps_per_env == 240000
    assert 10000 * cfg.num_steps_per_env // cfg.algorithm.num_mini_batches == 60000
    assert cfg.algorithm.num_learning_epochs * cfg.algorithm.num_mini_batches == 20
    assert cfg.max_iterations == 10000


@pytest.mark.parametrize("index", [0, 1])
def test_requested_flags_are_exposed_with_reference_defaults(index):
    cfg = configs()[index]
    assert cfg.policy.actor_obs_normalization and cfg.policy.critic_obs_normalization
    assert cfg.policy.actor_hidden_dims == cfg.policy.critic_hidden_dims == [512, 256, 128]
    assert cfg.local_redundancy.enabled
    assert (cfg.local_redundancy.interval, cfg.local_redundancy.batch_size) == (50, 64)
    assert cfg.optimizer.lower() == "adamw"
    assert (cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay) == (.9, .95, .001)
    assert cfg.algorithm.torch_compile
    assert cfg.algorithm.gamma == .99 > .97
    assert (cfg.algorithm.learning_rate, cfg.algorithm.entropy_coef, cfg.algorithm.value_loss_coef) == (.001, .005, 1.)


def test_observation_routing_and_log_directories_stay_task_specific():
    original, keeper = configs()
    assert original.obs_groups == {"policy": ["policy"], "critic": ["policy"]}
    assert keeper.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    assert original.experiment_name != keeper.experiment_name
    assert original.wandb_project == keeper.wandb_project == "borinotIsaacLab"


def test_original_registration_keeps_skrl_and_adds_rsl_rl():
    tree = ast.parse((TASKS / "borinot/__init__.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "register")
    kwargs = next(k.value for k in call.keywords if k.arg == "kwargs")
    keys = {ast.literal_eval(k) for k in kwargs.keys}
    assert {"skrl_cfg_entry_point", "rsl_rl_cfg_entry_point"} <= keys


def test_original_asset_paths_resolve_in_the_canonical_layout():
    filename = TASKS / "borinot/borinot_env_cfg.py"
    cls = next(n for n in ast.parse(filename.read_text()).body if isinstance(n, ast.ClassDef))
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("usd_path", "ground_usd_path"):
                path = eval(compile(ast.Expression(node.value), "asset_path", "eval"),
                            {"Path": Path, "__file__": str(filename)})
                assert Path(path).is_file(), path
