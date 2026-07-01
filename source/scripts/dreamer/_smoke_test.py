"""Standalone correctness smoke test for dreamer_core (no Isaac Sim needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
from dreamer_core.agent import DreamerAgent, DreamerConfig
from dreamer_core.buffer import SequenceReplayBuffer

torch.manual_seed(0)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# Solo12-shaped, small for a fast smoke test.
OBS, CMD, ACT = 36, 0, 12
B, T = 64, 24
cfg = DreamerConfig(
    obs_dim=OBS, command_dim=CMD, action_dim=ACT, device=dev,
    deter=128, stoch=32, discrete=32, model_hidden=256, blocks=8,
    encoder_hidden_dims=[128, 128], head_hidden_dims=[256, 256],
    actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
    imag_horizon=15, discount=0.99,
    use_amp=False, use_compile=False,
)

def make_batch(B, T):
    return {
        "obs": torch.randn(B, T, OBS, device=dev),
        "command": torch.zeros(B, T, 0, device=dev),
        "prev_action": torch.tanh(torch.randn(B, T, ACT, device=dev)),
        "reward": torch.randn(B, T, device=dev),
        "is_first": (torch.rand(B, T, device=dev) < 0.05),
        "is_terminal": (torch.rand(B, T, device=dev) < 0.02),
        "is_last": (torch.rand(B, T, device=dev) < 0.03),
        "initial_stoch": torch.zeros(B, 32, 32, device=dev),
        "initial_deter": torch.zeros(B, 128, device=dev),
    }

class MockBuf:
    def update_latents(self, wb, s, d):
        assert s.shape[0] == B and d.shape[0] == B

for use_amp, use_compile in [(False, False), (True, False)]:
    cfg.use_amp, cfg.use_compile = use_amp, use_compile
    agent = DreamerAgent(cfg)
    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    batch = make_batch(B, T)
    wb = (torch.zeros(B, T, dtype=torch.long, device=dev), torch.arange(B, device=dev))
    m = agent.update(batch, wb, MockBuf())
    torch.cuda.synchronize() if dev == "cuda:0" else None
    print(f"[amp={use_amp} compile={use_compile}] params={n_params:,} "
          f"total_loss={float(m['total_loss']):.4f} "
          f"dyn={float(m['loss/dyn']):.3f} rep={float(m['loss/rep']):.3f} "
          f"decoder={float(m['loss/decoder']):.3f} reward={float(m['loss/reward']):.3f} "
          f"policy={float(m['loss/policy']):.4f} value={float(m['loss/value']):.3f} "
          f"repval={float(m['loss/repval']):.3f}")

# a few consecutive steps to confirm stability
agent = DreamerAgent(cfg)
for i in range(5):
    m = agent.update(make_batch(B, T), (torch.zeros(B, T, dtype=torch.long, device=dev), torch.arange(B, device=dev)), MockBuf())
print("5-step total_loss:", float(m["total_loss"]))

# test act() path
st = agent.initial_state(8)
a, st2 = agent.act(st, torch.zeros(8, ACT, device=dev), torch.zeros(8, 0, device=dev),
                   torch.ones(8, dtype=torch.bool, device=dev), torch.randn(8, OBS, device=dev))
print("act -> action", tuple(a.shape), "range", (float(a.min()), float(a.max())))
print("SMOKE OK")
