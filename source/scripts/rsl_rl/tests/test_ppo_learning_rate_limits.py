# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only checks for configurable RSL-RL PPO limits; no Isaac Sim or Slurm."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rsl_rl.algorithms import PPO
import rsl_rl.algorithms.ppo as ppo_module
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


REPO = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("lr_installer", REPO / "scripts/tools/install_rsl_rl_ppo_lr_limits.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def make_algorithm(cls=PPO, **kwargs):
    torch.manual_seed(43)
    obs = TensorDict({"policy": torch.zeros(4, 3)}, batch_size=[4])
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 2,
                         actor_hidden_dims=[4], critic_hidden_dims=[4], noise_std_type="log")
    alg = cls(policy, device="cpu", num_learning_epochs=1, num_mini_batches=1,
              learning_rate=kwargs.pop("learning_rate", 5e-4), **kwargs)
    alg.init_storage("rl", 4, 2, obs, [2])
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.arange(4, dtype=torch.float32), torch.zeros(4), {})
        alg.compute_returns(obs)
    return alg


def shift_policy_for_kl(alg, kl):
    # Two unit-std actions shifted equally: sum_i (delta_mu_i**2 / 2) = kl.
    with torch.no_grad():
        alg.policy.actor[-1].bias.add_(kl ** 0.5)


@pytest.mark.parametrize("initial,lower,upper,kl,expected", [
    (4e-4, 1e-5, 5e-4, .001, 5e-4),       # cap at custom ceiling
    (6e-5, 5e-5, 5e-4, .03, 5e-5),        # cap at custom floor
    (5e-4, 1e-5, 1e-3, .01, 5e-4),        # dead band
    (.02, 1e-5, .03, .001, .03),           # can exceed the old hardcoded ceiling
    (5e-7, 1e-7, 1e-3, .03, 5e-7 / 1.5), # can go below the old hardcoded floor
    (5e-4, 5e-4, 5e-4, .001, 5e-4),       # equal bounds pin the LR
    (5e-4, 5e-4, 5e-4, .03, 5e-4),
])
def test_real_ppo_update_respects_custom_limits(initial, lower, upper, kl, expected):
    alg = make_algorithm(learning_rate=initial, min_learning_rate=lower, max_learning_rate=upper)
    shift_policy_for_kl(alg, kl)
    # All groups, including a separate action-noise group, must receive the LR.
    params = alg.optimizer.param_groups[0]["params"]
    alg.optimizer = torch.optim.Adam([{"params": params[:-1]}, {"params": params[-1:]}], lr=initial)
    seen_lrs = []
    alg.optimizer.register_step_pre_hook(lambda optim, args, kwargs: seen_lrs.append(
        [group["lr"] for group in optim.param_groups]))
    losses = alg.update()
    assert alg.learning_rate == pytest.approx(expected)
    assert seen_lrs == [pytest.approx([expected, expected])]
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_fixed_schedule_does_not_adapt():
    alg = make_algorithm(schedule="fixed", learning_rate=.02, max_learning_rate=5e-4)
    shift_policy_for_kl(alg, .03)
    alg.update()
    assert alg.learning_rate == .02
    assert alg.optimizer.param_groups[0]["lr"] == .02


@pytest.mark.parametrize("kl,expected", [(0., 5e-4), (.005, 5e-4), (.02, 5e-4),
                                         (.001, 7.5e-4), (.03, 5e-4 / 1.5)])
def test_exact_kl_thresholds(kl, expected):
    tree = ast.parse(Path(ppo_module.__file__).read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "kl_mean > self.desired_kl * 2.0")
    namespace = {"self": SimpleNamespace(learning_rate=5e-4, desired_kl=.01,
                                        min_learning_rate=1e-5, max_learning_rate=.01), "kl_mean": kl}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "scheduler", "exec"), namespace)
    assert namespace["self"].learning_rate == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    {"min_learning_rate": 0}, {"min_learning_rate": -1}, {"min_learning_rate": float("nan")},
    {"max_learning_rate": float("inf")}, {"max_learning_rate": True},
    {"min_learning_rate": 1e-3, "max_learning_rate": 1e-4},
    {"max_learning_rate": 1e-4}, {"min_learning_rate": 1e-3},
    {"learning_rate": float("nan")}, {"learning_rate": 0},
])
def test_invalid_limits_fail_before_optimizer_construction(kwargs):
    with pytest.raises(ValueError, match="learning_rate"):
        PPO(None, **{"learning_rate": 5e-4, **kwargs})


def test_default_update_is_bit_identical_to_original_dependency():
    original = Path(ppo_module.__file__ + ".before-configurable-lr.bak").read_text()
    namespace = dict(vars(ppo_module))
    exec(compile(original, "original_ppo.py", "exec"), namespace)
    for kl in (.001, .01, .03):
        outputs = []
        for cls in (namespace["PPO"], PPO):
            alg = make_algorithm(cls)
            shift_policy_for_kl(alg, kl)
            losses = alg.update()
            outputs.append((losses, copy.deepcopy(alg.policy.state_dict()), alg.optimizer.state_dict(),
                            torch.random.get_rng_state()))
        assert outputs[0][0] == outputs[1][0]
        for key in outputs[0][1]:
            assert torch.equal(outputs[0][1][key], outputs[1][1][key])
        assert outputs[0][2]["param_groups"] == outputs[1][2]["param_groups"]
        for key, state in outputs[0][2]["state"].items():
            for name, value in state.items():
                assert torch.equal(value, outputs[1][2]["state"][key][name])
        assert torch.equal(outputs[0][3], outputs[1][3])


def test_root_config_hydra_merge_and_actual_runner_forwarding():
    filename = REPO / "source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
    tree = ast.parse(filename.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RslRlOnPolicyRunnerCfg")
    defaults = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)
                and n.target.id in ("min_learning_rate", "max_learning_rate")}
    cfg = OmegaConf.create({"agent": defaults})
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["agent.min_learning_rate=5e-5",
                                                    "agent.max_learning_rate=5e-4"]))
    obs = TensorDict({"policy": torch.zeros(4, 3)}, batch_size=[4])
    env = SimpleNamespace(get_observations=lambda: obs, num_actions=2, num_envs=4)
    train_cfg = {**OmegaConf.to_container(cfg.agent), "num_steps_per_env": 2, "save_interval": 1,
                 "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
                 "algorithm": {"class_name": "PPO", "learning_rate": 5e-4},
                 "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [4], "critic_hidden_dims": [4]}}
    runner = OnPolicyRunner(env, train_cfg, device="cpu")
    assert runner.alg.min_learning_rate == 5e-5
    assert runner.alg.max_learning_rate == 5e-4
    assert runner.alg.learning_rate == 5e-4


def test_installer_preview_backup_and_idempotence(tmp_path):
    package_dir = Path(ppo_module.__file__).parents[1]
    for relative in installer.PATCHES:
        source = package_dir / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(str(source) + ".before-configurable-lr.bak").read_bytes())
    originals = {relative: (tmp_path / relative).read_bytes() for relative in installer.PATCHES}
    assert len(installer.install(tmp_path)) == 2
    assert all((tmp_path / relative).read_bytes() == content for relative, content in originals.items())
    assert len(installer.install(tmp_path, apply=True)) == 2
    for relative, content in originals.items():
        assert Path(str(tmp_path / relative) + ".before-configurable-lr.bak").read_bytes() == content
    assert installer.install(tmp_path, apply=True) == []


def test_installer_checks_all_files_before_writing(tmp_path):
    package_dir = Path(ppo_module.__file__).parents[1]
    for relative in installer.PATCHES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "algorithms/ppo.py":
            target.write_bytes(Path(str(package_dir / relative) + ".before-configurable-lr.bak").read_bytes())
        else:
            target.write_text("# Unknown upstream version\n")
    with pytest.raises(RuntimeError, match="Unsupported RSL-RL source"):
        installer.install(tmp_path, apply=True)
    assert not list(tmp_path.rglob("*.bak"))
