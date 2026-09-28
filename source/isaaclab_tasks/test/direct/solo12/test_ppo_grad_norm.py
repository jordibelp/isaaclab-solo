"""PPO gradient metrics match the installed RSL-RL update without changing its step."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def _solo12_ppo_class():
    module_path = Path(__file__).parents[3] / "isaaclab_tasks/direct/solo12/agents/solo12_ppo.py"
    spec = importlib.util.spec_from_file_location("solo12_ppo_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solo12PPO


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"))]
)
def test_ppo_logs_pre_clip_actor_and_critic_norms_without_changing_update(monkeypatch, device):
    torch.manual_seed(19)
    obs = TensorDict({"policy": torch.randn(8, 6, device=device)}, batch_size=[8], device=device)
    policy = ActorCritic(
        obs, {"policy": ["policy"], "critic": ["policy"]}, num_actions=2,
        actor_hidden_dims=[16], critic_hidden_dims=[16], noise_std_type="log",
    ).to(device)
    actions = policy.act(obs).detach()
    old_log_prob = policy.get_actions_log_prob(actions).detach()
    old_mu = policy.action_mean.detach()
    old_sigma = policy.action_std.detach()
    target_values = policy.evaluate(obs).detach()
    batch = (
        obs, actions, target_values, torch.ones(8, 1, device=device), target_values + 2.0,
        old_log_prob, old_mu, old_sigma, (None, None), None,
    )

    def make_algorithm(cls):
        policy_copy = ActorCritic(
            obs, {"policy": ["policy"], "critic": ["policy"]}, num_actions=2,
            actor_hidden_dims=[16], critic_hidden_dims=[16], noise_std_type="log",
        ).to(device)
        policy_copy.load_state_dict(policy.state_dict())
        alg = cls(
            policy_copy, num_learning_epochs=1, num_mini_batches=2,
            max_grad_norm=0.01, schedule="fixed", desired_kl=None, device=device,
        )
        # train.py replaces upstream PPO's Adam with AdamW for the pasted command.
        alg.optimizer = torch.optim.AdamW(alg.policy.parameters(), lr=1e-3, weight_decay=0.001)
        alg.storage = SimpleNamespace(mini_batch_generator=lambda *args: iter((batch, batch)), clear=lambda: None)
        return alg

    baseline = make_algorithm(PPO)
    observed = {"actor": [], "critic": []}
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_before_combined_clip(parameters, max_norm):
        for name in observed:
            grads = [
                p.grad for param_name, p in baseline.policy.named_parameters()
                if (param_name.startswith("critic") == (name == "critic")) and p.grad is not None
            ]
            observed[name].append(torch.nn.utils.get_total_norm(grads).item())
        return original_clip(parameters, max_norm)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", capture_before_combined_clip)
        baseline.update()

    instrumented = make_algorithm(_solo12_ppo_class())
    losses = instrumented.update()

    for name, norms in observed.items():
        assert len(norms) == 2
        assert losses[f"grad_norm_{name}"] == pytest.approx(sum(norms) / len(norms), rel=1e-5)
        assert max(norms) > instrumented.max_grad_norm
    assert instrumented.max_grad_norm == 0.01
    assert not instrumented.optimizer._optimizer_step_pre_hooks
    for expected, actual in zip(baseline.policy.parameters(), instrumented.policy.parameters()):
        torch.testing.assert_close(actual, expected)


def test_ppo_gradient_tags_reach_the_wandb_gradients_section(monkeypatch):
    from rsl_rl.utils import wandb_utils
    from torch.utils.tensorboard import SummaryWriter

    train_path = Path(__file__).parents[4] / "scripts/rsl_rl/train.py"
    function = next(
        node for node in ast.parse(train_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_rsl_rl_wandb_writer_for_single_stream"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(train_path), "exec"), namespace)
    original_writer = wandb_utils.WandbSummaryWriter
    with monkeypatch.context() as patch:
        tensorboard_tags, wandb_tags = [], []
        patch.setattr(wandb_utils, "WandbSummaryWriter", original_writer)
        patch.setattr(SummaryWriter, "add_scalar", lambda self, tag, *args, **kwargs: tensorboard_tags.append(tag))
        patch.setattr(wandb_utils.wandb, "log", lambda values, step: wandb_tags.append((values, step)))
        namespace["_patch_rsl_rl_wandb_writer_for_single_stream"]()
        writer = wandb_utils.WandbSummaryWriter.__new__(wandb_utils.WandbSummaryWriter)
        writer.add_scalar("Loss/grad_norm_actor", 1.0, 1)
        writer.add_scalar("Loss/grad_norm_critic", 2.0, 1)
        writer.add_scalar("Loss/surrogate", 3.0, 1)
        expected = ["Gradients/grad_norm_actor", "Gradients/grad_norm_critic", "Loss/surrogate"]
        assert tensorboard_tags == expected
        assert wandb_tags == [({tag: value}, 1) for tag, value in zip(expected, (1.0, 2.0, 3.0))]
