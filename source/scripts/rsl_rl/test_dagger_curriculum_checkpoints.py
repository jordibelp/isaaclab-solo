"""CPU checks for phase-specific DAgger checkpoint selection."""

from types import SimpleNamespace

import pytest
import torch

from dagger_curriculum_checkpoints import CurriculumBestTracker, curriculum_state


def state(idx=0, phase=1):
    return {"global_idx": idx, "two_feet_phase": phase}


def record(tracker, rewards, dones, before=None, after=None, excluded=None):
    before = state() if before is None else before
    tracker.update(torch.tensor(rewards), torch.tensor(dones, dtype=torch.bool), before,
                   before if after is None else after,
                   None if excluded is None else torch.tensor(excluded))


def test_episode_returns_span_rollouts_and_ignore_unfinished_episodes():
    tracker = CurriculumBestTracker()
    record(tracker, [2., 100.], [False, False])
    assert tracker.pop_improvements() == []
    record(tracker, [3., 100.], [True, False])
    result, = tracker.pop_improvements()
    assert result["best_model_value"] == 5.
    assert result["best_model_episode_count"] == 1
    assert result["filename"] == "adapter_best_curriculum_idx_0_two_feet_phase_1.pt"


def test_each_phase_has_an_independent_best_even_with_lower_return():
    tracker = CurriculumBestTracker()
    record(tracker, [10.], [True], after=state(1, 2))
    first, = tracker.pop_improvements()
    record(tracker, [1.], [True], before=state(1, 2))
    second, = tracker.pop_improvements()
    assert first["best_model_value"] == 10.
    assert second["best_model_value"] == 1.
    assert second["filename"] == "adapter_best_curriculum_idx_1_two_feet_phase_2.pt"


def test_mid_rollout_promotion_keeps_old_stage_and_rejects_mixed_episodes():
    tracker = CurriculumBestTracker()
    record(tracker, [10., 500.], [True, False], after=state(1, 2))
    record(tracker, [2., 500.], [True, True], before=state(1, 2))
    old, new = tracker.pop_improvements()
    assert old["curriculum_state"] == state()
    assert old["best_model_value"] == 10.
    assert new["best_model_value"] == 2.
    assert new["best_model_episode_count"] == 1
    record(tracker, [0., 4.], [False, True], before=state(1, 2))
    improved, = tracker.pop_improvements()
    assert improved["best_model_value"] == 3.


def test_only_fresh_improvements_replace_stage_best():
    tracker = CurriculumBestTracker()
    record(tracker, [5.], [True])
    assert len(tracker.pop_improvements()) == 1
    assert tracker.pop_improvements() == []
    record(tracker, [1.], [True])
    assert tracker.pop_improvements() == []
    record(tracker, [20.], [True])
    result, = tracker.pop_improvements()
    assert result["best_model_value"] == pytest.approx(26 / 3)


def test_window_uses_last_100_episodes_and_excludes_posed_starts():
    tracker = CurriculumBestTracker()
    record(tracker, [999., *range(101)], [True] * 102, excluded=[True, *([False] * 101)])
    result, = tracker.pop_improvements()
    assert result["best_model_value"] == 50.5
    assert result["best_model_episode_count"] == 100


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_returns_do_not_create_best(value):
    tracker = CurriculumBestTracker()
    record(tracker, [value], [True])
    assert tracker.pop_improvements() == []


@pytest.mark.parametrize("idx,phase,suffix", [(3, None, "curriculum_idx_3"),
                                            (None, 2, "two_feet_phase_2"),
                                            (0, 2, "curriculum_idx_0_two_feet_phase_2")])
def test_filename_stage_identity(idx, phase, suffix):
    tracker = CurriculumBestTracker()
    record(tracker, [1.], [True], before=state(idx, phase))
    result, = tracker.pop_improvements()
    assert result["filename"] == f"adapter_best_{suffix}.pt"


def test_no_curriculum_creates_no_additional_checkpoints():
    env = SimpleNamespace(cfg=SimpleNamespace(curriculum_two_feet=False),
                          get_curriculum_global_idx=lambda: None)
    assert curriculum_state(env) is None
    tracker = CurriculumBestTracker()
    tracker.update(torch.ones(2), torch.ones(2, dtype=torch.bool), None, None)
    assert tracker.pop_improvements() == []


def test_state_reads_runtime_phase_not_configured_start_phase():
    env = SimpleNamespace(cfg=SimpleNamespace(curriculum_two_feet=True, two_feet_curriculum_start_phase=1),
                          _two_feet_curriculum_phase=4, get_curriculum_global_idx=lambda: 3)
    assert curriculum_state(env) == state(3, 4)


def test_tracking_preserves_training_rng_and_step_tensors():
    tracker = CurriculumBestTracker()
    rewards = torch.tensor([1., 2.])
    dones = torch.tensor([True, False])
    excluded = torch.tensor([False, True])
    copies = [x.clone() for x in (rewards, dones, excluded)]
    rng = torch.random.get_rng_state()
    before, after = state(), state(1, 2)
    tracker.update(rewards, dones, before, after, excluded)
    tracker.pop_improvements()
    assert torch.equal(rng, torch.random.get_rng_state())
    for value, original in zip((rewards, dones, excluded), copies):
        assert torch.equal(value, original)
    assert before == state() and after == state(1, 2)
