"""Raw-Q diagnostics use actual atoms and do not change the SAC update."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.algorithms.sac import Q_STAT_NAMES
from rsl_rl_sac.models import SACActorModel, SACCriticModel


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU"))])
def device(request):
    return request.param


def algorithm(device, spacing, method="min"):
    obs = TensorDict({"policy": torch.randn(4, 3, device=device)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[8]).to(device)
    loss = spacing if spacing in ("mse", "mse_target_norm_popart", "c51") else "two_hot"
    critic = SACCriticModel(
        obs, groups, "critic", 1, num_actions=2, hidden_dims=[8], distributional_loss=loss,
        distributional_num_bins=101, distributional_symlog_limit=5,
        distributional_linear_limit=20 if spacing == "linear" else 0,
    ).to(device)
    # Include a terminal, a timeout, and different n-step discounts.
    batch = (obs, torch.randn(4, 2, device=device), torch.randn(4, 1, device=device), obs,
             torch.tensor([[0.], [1.], [1.], [0.]], device=device),
             torch.tensor([[0.], [0.], [1.], [0.]], device=device),
             torch.tensor([[1.], [1.], [2.], [3.]], device=device))
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch] * 3))
    alg = SAC(actor, critic, replay, device=device, num_mini_batches=3,
              policy_frequency=2, q_reduction_method=method)
    return alg


@pytest.mark.parametrize("spacing", ["symexp", "linear"])
@pytest.mark.parametrize("method", ["min", "mean", "mean_pi_q_none"])
def test_raw_q_uses_atoms_not_inverse_of_symlog_mean(device, spacing, method):
    alg = algorithm(device, spacing, method)
    atoms = alg.critic.value_support
    p1 = torch.zeros(2, atoms.numel(), device=device)
    p2 = torch.zeros_like(p1)
    # Rare negative tail plus positive mass, with the critics reversing order across states.
    p1[0, 0], p1[0, 60], p1[1, 75] = 0.01, 0.99, 1.0
    p2[0, 75], p2[1, 0], p2[1, 60] = 1.0, 0.01, 0.99
    tiny = torch.finfo(p1.dtype).tiny
    stats = alg._critic_q_stats(p1.clamp_min(tiny).log().requires_grad_(),
                               p2.clamp_min(tiny).log().requires_grad_())
    assert not stats.requires_grad
    a = 0.01 * atoms[0].double() + 0.99 * atoms[60].double()
    b = atoms[75].double()
    expected = torch.stack(((a + b) / 2, (a + b) / 2, (a + b) / 2,
                            torch.minimum(a, b) if method == "min" else (a + b) / 2))
    torch.testing.assert_close(stats.double(), expected, rtol=2e-6, atol=2e-6)
    x = atoms.sign() * atoms.abs().log1p()
    mean_x = (0.01 * x[0] + 0.99 * x[60] + x[75]) / 2
    wrong_q = mean_x.sign() * mean_x.abs().expm1()
    assert abs(stats[2] - wrong_q) > 0.1


@pytest.mark.parametrize("spacing", ["symexp", "linear", "mse", "mse_target_norm_popart", "c51"])
@pytest.mark.parametrize("method", ["min", "mean", "mean_pi_q_none"])
def test_update_averages_replay_and_policy_q_on_their_own_clocks(device, spacing, method):
    alg = algorithm(device, spacing, method)
    with torch.no_grad():
        for network in (alg.critic.critic1, alg.critic.critic2):
            network[-1].weight.normal_(0, 0.02)
    expected_replay, expected_policy = [], []
    original_losses, original_actor = alg._critic_losses_fn, alg._actor_objective_fn

    def losses(*args):
        result = original_losses(*args)
        with torch.no_grad():
            if alg.critic.distributional_critic_ce:
                support = alg.critic.value_support.double()
                q1, q2 = [(out.detach().double().softmax(-1) * support).sum(-1)
                          for out in result[2:]]
            else:
                q1, q2 = [alg.critic.q_from_output(out.detach()).double() for out in result[2:]]
            reduced = torch.minimum(q1, q2) if method == "min" else (q1 + q2) / 2
            expected_replay.append(torch.stack((q1.mean(), q2.mean(), ((q1 + q2) / 2).mean(), reduced.mean())))
        return result

    def actor(*args):
        result = original_actor(*args)
        expected_policy.append(result[1].detach().double().mean())
        return result

    alg._critic_losses_fn, alg._actor_objective_fn = losses, actor
    metrics = alg.update()
    torch.testing.assert_close(torch.tensor([metrics[key] for key in Q_STAT_NAMES], device=device).double(),
                               torch.stack(expected_replay).mean(0), rtol=2e-4, atol=3e-6)
    assert len(expected_replay) == 3 and len(expected_policy) == 2
    assert metrics["CriticQ/policy_mean"] == pytest.approx(
        torch.stack(expected_policy).mean().item(), rel=2e-6, abs=1e-6)
    # An update phase with no actor update must not log a fabricated policy value.
    alg.actor_optimizer = None
    assert "CriticQ/policy_mean" not in alg.update()


def test_raw_q_reaches_tensorboard_and_wandb_verbatim(monkeypatch):
    from rsl_rl_sac.utils import wandb_utils
    from rsl_rl_sac.utils.logger import Logger

    tags, uploaded = [], []
    monkeypatch.setattr(wandb_utils.SummaryWriter, "add_scalar", lambda self, tag, value, **kw: tags.append(tag))
    monkeypatch.setattr(wandb_utils.wandb, "log", lambda values, **kw: uploaded.append(values))
    writer = wandb_utils.WandbSummaryWriter.__new__(wandb_utils.WandbSummaryWriter)
    logger = Logger.__new__(Logger)
    logger.writer = writer
    logger.cfg = {"num_steps_per_env": 2, "algorithm": {"rnd_cfg": None}}
    logger.log_dir, logger.logger_type = None, "tensorboard"
    logger.num_envs, logger.gpu_world_size = 4, 1
    logger.tot_timesteps, logger.tot_time = 0, 1.0
    logger.ep_extras, logger.rewbuffer, logger.lenbuffer = [], [], []
    metrics = {key: value for value, key in enumerate((*Q_STAT_NAMES, "CriticQ/policy_mean"))}
    logger.log(it=1, start_it=0, total_it=2, collect_time=0.1, learn_time=0.1,
               loss_dict=metrics, learning_rate=1e-3, action_std=torch.zeros(2), rnd_weight=None)
    assert set(metrics) <= set(tags)
    assert not any(tag.startswith("Loss/CriticQ/") for tag in tags)
    logged = {key: value for values in uploaded for key, value in values.items()}
    assert {key: logged[key] for key in metrics} == metrics
