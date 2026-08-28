"""Tests for the bounded per-frame intervention checkpoint.

Strategy: no model and no GPU. These exercise the pieces every
resume depends on directly, because the property that matters is
not visible in a single run: a checkpoint is only worth keeping if
restoring it reproduces the draws the frame was about to make, and
that can only be shown by drawing, walking the generator forward,
restoring, and drawing again.

Passing proves the random state round-trips exactly, that the byte
accounting counts what it claims to, and that a run past its
ceiling degrades to no random state rather than dropping frames or
refusing to record. The last one is the one worth stating: dropping
old checkpoints would take away the early frames an edit is most
likely to re-enter.
"""

from __future__ import annotations

import torch

from src.inference.checkpoint import (
    CHECKPOINT_RNG_BYTES_MAX,
    CheckpointBudget,
    DgemmaFrame,
    FrameCheckpoint,
    LladaFrame,
    rng_capture,
    rng_restore,
)

CANVAS_LENGTH = 6


def _llada_checkpoint() -> FrameCheckpoint:
    return FrameCheckpoint(
        ids=torch.zeros((1, CANVAS_LENGTH), dtype=torch.long),
        canvas_index=0,
        rng=None,
        extra=LladaFrame(
            reveal_conf=torch.zeros(
                CANVAS_LENGTH, dtype=torch.float
            )
        ),
    )


# -- the random state --


def test_a_restored_state_repeats_the_draw() -> None:
    torch.manual_seed(7)
    state = rng_capture()
    first = torch.rand(4)

    rng_restore(state)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_work_in_between_does_not_move_the_restored_draw() -> None:
    """The whole point of the checkpoint.

    Re-seeding cannot do this: it would reproduce the run's first
    step, not the step the user branched from, so an edit made
    after another generation would diverge from the same edit made
    before it.
    """
    torch.manual_seed(7)
    state = rng_capture()
    expected = torch.rand(4)

    rng_restore(state)
    _ = torch.rand(1024)
    torch.manual_seed(99)
    _ = torch.rand(32)

    rng_restore(state)
    assert torch.equal(torch.rand(4), expected)


def test_two_captures_of_a_moving_generator_differ() -> None:
    """Negative space: a capture reads the generator, not a
    constant. Without this the test above would pass against an
    implementation that captured nothing at all."""
    torch.manual_seed(7)
    first = rng_capture()
    _ = torch.rand(8)
    second = rng_capture()

    assert not torch.equal(first.cpu, second.cpu)


def test_a_capture_is_not_disturbed_by_later_draws() -> None:
    torch.manual_seed(7)
    state = rng_capture()
    snapshot = state.cpu.clone()

    _ = torch.rand(64)

    assert torch.equal(state.cpu, snapshot)


# -- byte accounting --


def test_a_state_reports_a_positive_size() -> None:
    state = rng_capture()
    assert state.nbytes() > 0


def test_a_checkpoint_counts_every_part_it_holds() -> None:
    bare = _llada_checkpoint()
    with_rng = FrameCheckpoint(
        ids=bare.ids,
        canvas_index=bare.canvas_index,
        rng=rng_capture(),
        extra=bare.extra,
    )

    assert with_rng.nbytes() > bare.nbytes()


def test_a_dgemma_payload_grows_with_its_canvas() -> None:
    small = DgemmaFrame(
        stable=(0, 1), seen_revealed=frozenset({0})
    )
    large = DgemmaFrame(
        stable=(0, 1, 2, 3),
        seen_revealed=frozenset({0, 1, 2}),
    )

    assert large.nbytes() > small.nbytes()


# -- the ceiling --


def test_a_run_within_its_budget_keeps_every_state() -> None:
    budget = CheckpointBudget()
    for _ in range(8):
        assert budget.capture_rng() is not None
    assert budget.frames_without_rng == 0


def test_a_run_past_its_ceiling_stops_capturing() -> None:
    """Degrades rather than evicts.

    The frames keep coming and keep their canvas; only the random
    state stops, so such a frame still resumes, just without the
    bit-for-bit repeat.
    """
    budget = CheckpointBudget(limit_bytes=1)

    first = budget.capture_rng()
    second = budget.capture_rng()

    assert first is not None, "the ceiling is tested before a draw"
    assert second is None
    assert budget.frames_without_rng == 1


def test_the_overshoot_is_one_state_at_most() -> None:
    budget = CheckpointBudget(limit_bytes=1)
    state = budget.capture_rng()
    assert state is not None
    assert budget.spent_bytes == state.nbytes()

    for _ in range(4):
        assert budget.capture_rng() is None
    assert budget.spent_bytes == state.nbytes()


def test_the_default_ceiling_clears_a_long_run() -> None:
    """A 128-step LLaDA run spends well under the cap, so the
    bound is a rail against an unfamiliar step count rather than
    something an ordinary run negotiates with."""
    one_state = rng_capture().nbytes()
    assert one_state * 129 < CHECKPOINT_RNG_BYTES_MAX
