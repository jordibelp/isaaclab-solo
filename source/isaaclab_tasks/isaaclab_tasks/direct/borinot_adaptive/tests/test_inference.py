"""CPU-only: from tests/, run PYTHONPATH=.. python -m pytest --rootdir=. --confcutdir=."""
import pytest
import torch

from inference import HeldLatentInference, encoder_period


class InstrumentedPolicy:
    command_dim = 2
    actor_obs_normalizer = staticmethod(lambda x: x * 2)
    get_actor_obs = staticmethod(lambda x: x)
    actor = staticmethod(lambda x: x)

    def __init__(self):
        self.encoded_batches = []

    def _actor_input(self, obs):
        self.encoded_batches.append(len(obs))
        return obs


@pytest.mark.parametrize(("actor", "encoder", "period"), [(200, 20, 10), (100, 20, 5), (200, 0, 1), (200, 200, 1)])
def test_encoder_period(actor, encoder, period):
    assert encoder_period(actor, encoder) == period


@pytest.mark.parametrize("encoder", [-1, 30, 300])
def test_invalid_rate(encoder):
    with pytest.raises(ValueError, match="positive divisor"):
        encoder_period(200, encoder)


@torch.inference_mode()
def test_cadence_and_partial_reset():
    model = InstrumentedPolicy()
    policy = HeldLatentInference(model, encoder_period(200, 20))
    for tick in range(22):
        if tick == 3:
            policy.reset(torch.tensor([[1], [0]]))
        obs = torch.tensor([[tick, tick + 100, -tick], [tick + 10, tick + 200, tick]], dtype=torch.float32)
        actual = policy(obs)
        first = max(t for t in (0, 3, 13) if t <= tick)
        second = tick // 10 * 10 + 10
        torch.testing.assert_close(actual[:, 0], torch.tensor([first, second], dtype=torch.float32) * 2)
        torch.testing.assert_close(actual[:, 1:], obs[:, 1:] * 2)
    assert model.encoded_batches == [2, 1, 1, 1, 1]


@torch.inference_mode()
def test_equal_rate_matches_normal_forward():
    model = InstrumentedPolicy()
    policy = HeldLatentInference(model, 1)
    for _ in range(3):
        obs = torch.randn(4, 3)
        torch.testing.assert_close(policy(obs), model.actor(model._actor_input(model.actor_obs_normalizer(obs))))
