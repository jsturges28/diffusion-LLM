"""Convergence and throughput count positions, not characters.

Strategy: drive the two pure functions directly with hand-built
frames, so the arithmetic is checkable by reading the fixture. The
end-to-end path through the endpoint is covered by
`tests/web/test_run_metrics.py`; what this file pins is the maths.

The defect being fixed is a measurement error dressed as a label.
Convergence divided mask glyphs by decoded characters while the axis
read "% Resolved", so a position resolving into a ten-character token
advanced the curve ten times as far as one resolving into a single
character. Two runs with the same token schedule could therefore show
different convergence purely because their words were longer, which
reverses the comparison the chart exists to support.

The throughput numerator had a second, narrower version of the same
problem: it subtracted every frame's mask count from the *first*
frame's, one baseline for a whole run. That is right while there is
one canvas. DiffusionGemma commits a canvas and starts the next from
fresh noise, so the mask count jumps back up while the baseline stays
behind and the series falls, undercounting a whole committed canvas.

Passing proves convergence is invariant to how long the tokens are,
that production accumulates across canvases and never goes backwards,
and that a run with no usable records still gets a curve, reported as
the weaker measure rather than passed off as the stronger one.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.analytics.metrics import (
    CONVERGENCE_BASES,
    CONVERGENCE_BASIS_CHARACTERS,
    CONVERGENCE_BASIS_TOKENS,
    compute_convergence,
    convergence_from_records,
    records_match_frames,
    tokens_produced_series,
)
from src.inference.reveal import newly_revealed

MASK = "\u2591"


def _record(text: str, masked: bool) -> Dict[str, Any]:
    return {"t": MASK if masked else text, "m": masked, "id": 1}


def _frames(schedule: List[List[bool]], text: str) -> List[Any]:
    """Token frames from a mask schedule and one token spelling.

    The schedule is what the model did; the text is only how it
    reads. Convergence must depend on the first and not the second,
    which is what lets the two callers below differ in text alone.
    """
    return [
        [_record(text, masked) for masked in frame]
        for frame in schedule
    ]


# Three positions resolving one per frame.
SCHEDULE = [
    [True, True, True],
    [False, True, True],
    [False, False, True],
    [False, False, False],
]


# -- convergence counts positions --


def test_convergence_does_not_depend_on_token_length() -> None:
    """The finding, in one assertion.

    Same schedule, same positions, different spellings. The curves
    must be identical, and under the old character measure they were
    not even close.
    """
    short = convergence_from_records(_frames(SCHEDULE, "a"))
    long = convergence_from_records(
        _frames(SCHEDULE, "extraordinarily")
    )

    assert short == long


def test_the_ratio_is_resolved_positions_over_all_of_them(
) -> None:
    series = convergence_from_records(_frames(SCHEDULE, "a"))

    assert [point["resolved_ratio"] for point in series] == [
        0.0,
        round(1 / 3, 6),
        round(2 / 3, 6),
        1.0,
    ]
    assert [point["mask_count"] for point in series] == [
        3,
        2,
        1,
        0,
    ]
    assert series[0]["total_tokens"] == 3


def test_the_character_measure_really_does_differ() -> None:
    """Without this the fix would be indistinguishable from a no-op.

    A long token inflates the character denominator as well as the
    numerator, so the two measures land in different places.
    """
    texts = ["".join(
        MASK if masked else "extraordinarily"
        for masked in frame
    ) for frame in SCHEDULE]

    by_chars = compute_convergence(texts)
    by_tokens = convergence_from_records(
        _frames(SCHEDULE, "extraordinarily")
    )

    assert by_chars[1]["resolved_ratio"] != (
        by_tokens[1]["resolved_ratio"]
    )


def test_an_empty_frame_reads_as_resolved() -> None:
    # Matches the character path, which reports 1.0 for empty text:
    # a canvas holding nothing has nothing left to resolve.
    series = convergence_from_records([[]])

    assert series[0]["resolved_ratio"] == 1.0
    assert series[0]["mask_count"] == 0


def test_a_frame_of_nulls_does_not_crash() -> None:
    # A run can store null for a frame it captured nothing for.
    series = convergence_from_records([None, [_record("a", False)]])

    assert series[0]["total_tokens"] == 0
    assert series[1]["resolved_ratio"] == 1.0


def test_convergence_needs_at_least_one_frame() -> None:
    with pytest.raises(AssertionError):
        convergence_from_records([])


# -- which measure a run is allowed to use --


def test_records_of_a_different_length_are_refused() -> None:
    """Pairing frame i of one stream with frame i of another would
    read plausibly and describe neither run."""
    assert records_match_frames(_frames(SCHEDULE, "a"), 4) is True
    assert records_match_frames(_frames(SCHEDULE, "a"), 3) is False
    assert records_match_frames([], 0) is False
    assert records_match_frames(None, 4) is False
    assert records_match_frames("not a list", 4) is False


def test_the_two_bases_are_distinct_names() -> None:
    assert CONVERGENCE_BASIS_TOKENS in CONVERGENCE_BASES
    assert CONVERGENCE_BASIS_CHARACTERS in CONVERGENCE_BASES
    assert CONVERGENCE_BASIS_TOKENS != CONVERGENCE_BASIS_CHARACTERS


# -- throughput accumulates across canvases --


def test_production_counts_up_within_one_canvas() -> None:
    series = convergence_from_records(_frames(SCHEDULE, "a"))

    assert tokens_produced_series(series, [0, 0, 0, 0]) == [
        0,
        1,
        2,
        3,
    ]


def test_a_run_with_no_canvas_index_is_one_canvas() -> None:
    # LLaDA never leaves canvas 0, and an older run may record no
    # index at all. Both must read as a single canvas.
    series = convergence_from_records(_frames(SCHEDULE, "a"))

    assert tokens_produced_series(series, None) == [0, 1, 2, 3]


def test_a_second_canvas_adds_to_the_first(
) -> None:
    """The multi-canvas defect, directly.

    Two canvases of two positions each. The old numerator subtracted
    from the first frame's mask count, so when canvas two started
    fully masked the series fell back to zero and the two tokens
    canvas one had committed stopped being counted.
    """
    schedule = [
        [True, True],
        [False, False],
        [True, True],
        [False, False],
    ]
    series = convergence_from_records(_frames(schedule, "a"))

    produced = tokens_produced_series(series, [0, 0, 1, 1])

    assert produced == [0, 2, 2, 4]
    assert produced[-1] == 4, "both canvases are counted"


def test_production_never_goes_backwards() -> None:
    """A draft can un-resolve a position; that is churn, not undoing.

    The resolution trace is the convergence chart's job. Letting the
    throughput numerator sawtooth would say more about the sampler's
    schedule than about how fast the run produced anything.
    """
    schedule = [
        [True, True, True],
        [False, False, True],
        [False, True, True],
        [False, False, False],
    ]
    series = convergence_from_records(_frames(schedule, "a"))

    produced = tokens_produced_series(series, [0, 0, 0, 0])

    for i in range(1, len(produced)):
        assert produced[i] >= produced[i - 1], produced


def test_production_works_off_the_character_measure_too() -> None:
    # A run with no records still gets a throughput curve; it is
    # simply counted in the weaker unit, like its convergence.
    texts = ["".join(
        MASK if masked else "a" for masked in frame
    ) for frame in SCHEDULE]
    series = compute_convergence(texts)

    assert tokens_produced_series(series, None) == [0, 1, 2, 3]


def test_production_needs_at_least_one_frame() -> None:
    with pytest.raises(AssertionError):
        tokens_produced_series([], None)


# -- the live readout and the chart agree --


def _live_produced(
    schedule: List[List[bool]], canvas_index: List[int]
) -> List[int]:
    """What the generator's live Tokens/s counts, in Python.

    The browser sums each frame's ``revealed`` list, which the
    samplers build with ``newly_revealed`` and reset per canvas.
    Reproducing it here is the only way to compare the two readouts,
    since one of them lives in JavaScript.
    """
    produced: List[int] = []
    running = 0
    seen: set = set()
    current = canvas_index[0]
    for i, frame in enumerate(schedule):
        if canvas_index[i] != current:
            seen = set()
            current = canvas_index[i]
        resolved = [not masked for masked in frame]
        fresh = newly_revealed(resolved, seen)
        seen.update(fresh)
        running += len(fresh)
        produced.append(running)
    return produced


@pytest.mark.parametrize(
    ("schedule", "canvas_index"),
    [
        (SCHEDULE, [0, 0, 0, 0]),
        (
            [
                [True, True],
                [False, False],
                [True, True],
                [False, False],
            ],
            [0, 0, 1, 1],
        ),
    ],
)
def test_the_two_throughput_readouts_agree(
    schedule: List[List[bool]], canvas_index: List[int]
) -> None:
    """The generator and Analytics must count the same run the same.

    They did not. The live readout sums per-frame reveals, which
    accumulates across canvases correctly; Analytics subtracted from
    a single first-frame baseline, which does not. On a multi-canvas
    run the same run read as two different speeds depending on which
    page you were looking at.
    """
    series = convergence_from_records(_frames(schedule, "a"))

    assert tokens_produced_series(series, canvas_index) == (
        _live_produced(schedule, canvas_index)
    )
