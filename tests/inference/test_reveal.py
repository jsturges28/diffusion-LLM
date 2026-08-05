"""Tests for the first-time reveal signal shared by the samplers.

Strategy: ``newly_revealed`` is pure and torch-free, so these drive it
directly with plain booleans, replaying the three shapes the samplers
actually produce. LLaDA only ever adds resolved positions; a resume
starts with most of the canvas already written; DiffusionGemma lets a
draft token settle, churn, and settle again.

Passing proves the signal is monotone per canvas: every position is
reported born exactly once, never re-reported when it merely stays
resolved, and never re-reported after it churns and re-settles. Both
features downstream depend on that. The birth glow would strobe
without it, and tokens per second would over-count.
"""

from __future__ import annotations

from src.inference.reveal import newly_revealed


def test_nothing_is_born_on_an_all_masked_canvas() -> None:
    assert newly_revealed([False, False, False], set()) == []


def test_a_first_resolution_is_reported() -> None:
    assert newly_revealed([False, True, False], set()) == [1]


def test_positions_are_reported_in_ascending_order() -> None:
    fresh = newly_revealed([True, False, True, True], set())
    assert fresh == [0, 2, 3]


def test_an_already_seen_position_is_not_reborn() -> None:
    # The token is still resolved on this frame, as it will be on
    # every later frame, but it was born once already.
    assert newly_revealed([True, True], {0, 1}) == []


def test_only_the_unseen_half_is_reported() -> None:
    assert newly_revealed([True, True, True], {0}) == [1, 2]


def test_the_seen_set_is_not_mutated() -> None:
    # The sampler owns that state and folds the result in itself, so
    # a helper that quietly updated it would give two owners.
    seen = {0}
    newly_revealed([True, True], seen)
    assert seen == {0}


def test_a_llada_run_reports_each_position_once() -> None:
    # Steps reveal one or two positions and never take one back.
    seen: set[int] = set()
    frames = [
        [False, False, False, False],
        [False, True, False, False],
        [True, True, False, False],
        [True, True, True, True],
    ]
    born = []
    for frame in frames:
        fresh = newly_revealed(frame, seen)
        seen.update(fresh)
        born.append(fresh)
    assert born == [[], [1], [0], [2, 3]]
    assert sum(len(b) for b in born) == 4


def test_a_resume_does_not_rebirth_the_surviving_prefix() -> None:
    # The frame-0 seed the resume path builds: everything already
    # written by the run it branched from counts as seen.
    start = [True, True, False, False]
    seen = set(newly_revealed(start, set()))
    assert seen == {0, 1}
    assert newly_revealed([True, True, True, False], seen) == [2]


def test_a_churning_draft_is_born_only_once() -> None:
    # DiffusionGemma: position 1 settles, changes again (so it reads
    # unresolved), then re-settles. Only the first settle is a birth.
    seen: set[int] = set()
    frames = [
        [False, True],
        [False, False],
        [False, True],
    ]
    born = []
    for frame in frames:
        fresh = newly_revealed(frame, seen)
        seen.update(fresh)
        born.append(fresh)
    assert born == [[1], [], []]


def test_an_empty_canvas_reports_nothing() -> None:
    assert newly_revealed([], set()) == []
