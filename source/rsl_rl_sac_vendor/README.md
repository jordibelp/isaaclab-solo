# Vendored RSL-RL-SAC

This directory contains the `rsl_rl` Python package from
`leggedrobotics/rsl_rl_sac` commit
`e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`, renamed to the
`rsl_rl_sac` namespace.

The namespace isolation is intentional: IsaacLab's existing and heavily
customized PPO workflow remains on `rsl-rl-lib==3.1.2`, while SAC uses the
paper authors' 4.0.1 implementation. This prevents SAC support from silently
changing PPO behavior or checkpoint compatibility.

Upstream: https://github.com/leggedrobotics/rsl_rl_sac

Paper: https://arxiv.org/abs/2605.24975

License: BSD-3-Clause (the source files retain their upstream headers).

Local opt-in extension: [categorical CE critics](../scripts/rsl_rl/SAC_DISTRIBUTIONAL_CRITIC.md)
on a symexp support, selected with `agent.critic.distributional_loss=two_hot` or
`=hl_gauss` (default `mse`, the original scalar heads).
