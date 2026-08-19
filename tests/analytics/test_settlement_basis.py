"""Which convergence measure a run gets, and why it is that one.

Strategy: hand-built token frames driven straight through the pure
functions, so every number here is checkable by reading the fixture.
The endpoint's choice between them is covered in
`tests/web/test_run_metrics.py`.

The problem is that "resolved" means two different things depending
on the model, and only one of them was being counted. LLaDA masks a
position with a real vocabulary entry, so its mask flag is ground
truth. DiffusionGemma has no mask token; it renoises unsettled
positions to fresh real tokens and the sampler infers resolution from
a position holding still. Stability is not settlement, and measured
on a real run the gap is tenfold: a canvas reading 90.2% resolved had
8.6% of positions holding what it eventually committed.

So the reader picks per run. The interesting half of this file is the
negative space: agreement-with-committed looks like a single measure
that would serve both models, and it is wrong for LLaDA, because
positions still masked at the end agree with the end from frame zero
and lift the curve off the floor before anything has resolved. That
counter-example is pinned below, because without it a future session
would reasonably try to simplify these two paths into one.

Passing proves the id test separates the two kinds of run, that
settlement is exact and measured per canvas rather than per run, that
the LLaDA path is untouched, and that the throughput numerator keeps
counting what the sampler resolved even when the chart stops.
"""

from __future__ import annotations

from typing import Any, Dict

from src.analytics.metrics import (
    convergence_from_records,
    convergence_from_settlement,
    masks_are_real,
    tokens_produced_series,
)

MASK_GLYPH = "\u2591"
MASK_ID = 126336


def _real_mask(masked: bool, token_id: int) -> Dict[str, Any]:
    """A LLaDA-shaped record: one mask token, one id."""
    if masked:
        return {"t": MASK_GLYPH, "m": True, "id": MASK_ID}
    return {"t": "x", "m": False, "id": token_id}


def _inferred_mask(
    masked: bool, token_id: int
) -> Dict[str, Any]:
    """A DiffusionGemma-shaped record: a real token either way."""
    return {"t": "x", "m": masked, "id": token_id}


# -- telling the two kinds of run apart --


def test_one_shared_id_means_a_real_mask() -> None:
    frames = [
        [_real_mask(True, 0), _real_mask(True, 0)],
        [_real_mask(False, 7), _real_mask(True, 0)],
    ]

    assert masks_are_real(frames) is True


def test_many_ids_mean_the_flag_was_inferred() -> None:
    """DiffusionGemma's masked positions hold real tokens, so they
    do not share an id. On a real run there were 526 of them."""
    frames = [
        [_inferred_mask(True, 11), _inferred_mask(True, 12)],
        [_inferred_mask(True, 13), _inferred_mask(False, 14)],
    ]

    assert masks_are_real(frames) is False


def test_a_run_with_nothing_masked_reads_as_real() -> None:
    # Both measures agree when nothing is marked, so the cheaper
    # answer is fine and the caller needs no third branch.
    frames = [[_real_mask(False, 1), _real_mask(False, 2)]]

    assert masks_are_real(frames) is True


def test_malformed_frames_do_not_derail_the_test() -> None:
    frames = [None, "not a frame", [_real_mask(True, 0)], []]

    assert masks_are_real(frames) is True


# -- settlement is exact, and per canvas --


def test_settlement_counts_agreement_with_the_end() -> None:
    # Three positions, each reaching its final id one frame later
    # than the last. Final ids are 10, 20, 30.
    frames = [
        [{"id": 0}, {"id": 0}, {"id": 0}],
        [{"id": 10}, {"id": 0}, {"id": 0}],
        [{"id": 10}, {"id": 20}, {"id": 0}],
        [{"id": 10}, {"id": 20}, {"id": 30}],
    ]

    series = convergence_from_settlement(frames, [0, 0, 0, 0])

    assert [p["resolved_ratio"] for p in series] == [
        0.0,
        round(1 / 3, 6),
        round(2 / 3, 6),
        1.0,
    ]


def test_each_canvas_is_measured_against_its_own_end() -> None:
    """The reason this takes canvas_index at all.

    A run-wide final would measure canvas 1 against canvas 2's
    content, which it never contained, so canvas 1 would read as
    barely settled however cleanly it converged.
    """
    frames = [
        [{"id": 0}, {"id": 0}],
        [{"id": 10}, {"id": 20}],
        [{"id": 0}, {"id": 0}],
        [{"id": 30}, {"id": 40}],
    ]

    series = convergence_from_settlement(frames, [0, 0, 1, 1])

    assert [p["resolved_ratio"] for p in series] == [
        0.0, 1.0, 0.0, 1.0,
    ]


def test_without_a_canvas_index_it_is_one_canvas() -> None:
    frames = [[{"id": 0}], [{"id": 9}]]

    series = convergence_from_settlement(frames, None)

    assert [p["resolved_ratio"] for p in series] == [0.0, 1.0]


def test_the_unsettled_count_shares_the_other_paths_key() -> None:
    # Downstream reads mask_count without caring which measure
    # produced it, so the key has to mean "not done yet" in all three.
    frames = [[{"id": 0}, {"id": 0}], [{"id": 1}, {"id": 2}]]

    series = convergence_from_settlement(frames, None)

    assert series[0]["mask_count"] == 2
    assert series[1]["mask_count"] == 0
    assert series[0]["total_tokens"] == 2


def test_an_empty_canvas_reads_as_resolved() -> None:
    series = convergence_from_settlement([[], [{"id": 1}]], None)

    assert series[0]["resolved_ratio"] == 1.0


def test_an_empty_tail_does_not_erase_the_canvas() -> None:
    # The final frame is the last one with records, so a run that
    # captured nothing for its last frame still has a target.
    frames = [[{"id": 0}], [{"id": 5}], []]

    series = convergence_from_settlement(frames, None)

    assert series[0]["resolved_ratio"] == 0.0
    assert series[1]["resolved_ratio"] == 1.0


# -- why it is not used where the mask is real --


def test_settlement_would_lift_a_llada_curve_off_the_floor(
) -> None:
    """The counter-example, kept so nobody merges the two paths.

    A LLaDA run can end with positions still masked; the real one
    measured ended at 81.9% resolved. Those positions hold the mask
    token at the end and held it at the start, so settlement scores
    them agreeing from frame zero and the curve opens at 18.1%
    instead of nothing.
    """
    # Ten positions; two never resolve and stay masked throughout.
    frames = []
    for resolved in range(0, 9):
        frame = []
        for position in range(10):
            never = position >= 8
            done = (not never) and position < resolved
            frame.append(_real_mask(not done, 100 + position))
        frames.append(frame)

    by_mask = convergence_from_records(frames)
    by_settlement = convergence_from_settlement(frames, None)

    assert by_mask[0]["resolved_ratio"] == 0.0
    assert by_settlement[0]["resolved_ratio"] == 0.2
    assert masks_are_real(frames) is True


# -- throughput keeps counting what the sampler resolved --


def test_throughput_reads_the_mask_series_not_settlement() -> None:
    """The generator's footer counts the sampler's own reveals and
    cannot know settlement, which is retrospective. Feeding this the
    settlement series would make one run read as two speeds."""
    # Position 0 holds filler (id 99) steadily at frame 1, so the
    # sampler calls it resolved while it is nothing of the sort: the
    # canvas commits id 3 there. That gap is the whole phenomenon.
    frames = [
        [_inferred_mask(True, 1), _inferred_mask(True, 2)],
        [_inferred_mask(False, 99), _inferred_mask(True, 4)],
        [_inferred_mask(False, 3), _inferred_mask(False, 5)],
    ]

    by_mask = convergence_from_records(frames)
    by_settlement = convergence_from_settlement(frames, None)

    from_mask = tokens_produced_series(by_mask, None)
    from_settlement = tokens_produced_series(
        by_settlement, None
    )

    # The two disagree, which is exactly why the caller must pass
    # the mask series rather than whichever one the chart shows.
    assert from_mask == [0, 1, 2]
    assert from_settlement == [0, 0, 2]
