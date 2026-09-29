"""A diffusion run's candidates stay inside their budget, evenly.

Strategy: offer synthetic steps to `CandidateCapture` with budgets a
few steps wide, so the thinning shows within a handful of steps, and
read back the frames it kept and the message it builds. No model and
no tokenizer: a step is a few tensors, and decoding is a stub that
counts its calls.

Passing proves the policy the ROADMAP settled. Every step is kept
until the budget binds, which a default LLaDA run never makes it do;
past it the kept steps stay evenly spaced at a doubling stride and
never pass the budget; the final step is kept whatever the stride;
and the message decodes each distinct id once, marks the token each
position held, and appends it with its rank exactly where the
candidates omit it.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import pytest
import torch

from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    CANDIDATES_PER_POSITION,
)
from src.inference.ar_sampler import TOP_K_ALTERNATIVES
from src.inference.candidate_capture import (
    MESSAGE_TYPE,
    CandidateCapture,
    StepCandidates,
    step_candidates,
)
from src.inference.logit_signals import Candidates

K = CANDIDATES_PER_POSITION
POSITIONS = 4
STEP_RECORDS = POSITIONS * K


def _step(
    frame: int,
    *,
    positions: int = POSITIONS,
    held: List[int] | None = None,
) -> StepCandidates:
    """A step whose candidates at position p are ids 10p..10p+4,
    holding the likeliest unless ``held`` says otherwise."""
    ids = torch.tensor(
        [[10 * p + c for c in range(K)] for p in range(positions)]
    )
    probs = torch.tensor([[0.4, 0.2, 0.1, 0.05, 0.01]] * positions)
    tokens = ids[:, 0] if held is None else torch.tensor(held)
    candidates = Candidates(
        ids=ids,
        probs=probs,
        held_probs=torch.full((positions,), 0.4),
        held_ranks=torch.ones(positions, dtype=torch.long),
    )
    return step_candidates(frame, tokens, candidates)


def _capture(budget_steps: int) -> CandidateCapture:
    return CandidateCapture(
        k=K, budget_records=budget_steps * STEP_RECORDS
    )


def _offer(capture: CandidateCapture, count: int) -> None:
    for frame in range(1, count + 1):
        capture.offer(_step(frame))


# -- the budget --


def test_every_step_is_kept_while_the_budget_allows() -> None:
    capture = _capture(budget_steps=10)

    _offer(capture, 10)
    capture.finish()

    assert capture.stride == 1
    assert capture.frames() == list(range(1, 11))


def test_passing_the_budget_halves_what_was_kept() -> None:
    """Four steps of room and eight offered: the fifth halves the
    sample to every other step, and the final step takes the last
    kept step's place rather than halving it again."""
    capture = _capture(budget_steps=4)

    _offer(capture, 8)
    capture.finish()

    assert capture.stride == 2
    assert capture.frames() == [1, 3, 5, 8]


@pytest.mark.parametrize(
    ("budget_steps", "offered"),
    [(1, 9), (3, 7), (4, 33), (5, 5), (5, 6), (7, 100), (16, 257)],
)
def test_kept_steps_never_pass_the_budget(
    budget_steps: int, offered: int
) -> None:
    capture = _capture(budget_steps)

    for frame in range(1, offered + 1):
        capture.offer(_step(frame))
        assert capture.records() <= capture.budget_records
    capture.finish()

    assert capture.records() <= capture.budget_records
    assert capture.frames()[-1] == offered


@pytest.mark.parametrize("offered", [9, 17, 40, 131])
def test_kept_steps_are_evenly_spaced_before_the_final(
    offered: int,
) -> None:
    """Uniform in time is the point of a stride: a reader scrubbing
    through the kept steps sees the run at an even pace, not dense
    where it started and sparse where it ended."""
    capture = _capture(budget_steps=6)

    _offer(capture, offered)
    capture.finish()
    frames = capture.frames()

    assert frames[0] == 1
    gaps = [
        b - a
        for a, b in zip(frames[:-2], frames[1:-1], strict=True)
    ]
    assert gaps
    assert all(gap == capture.stride for gap in gaps)


def test_the_final_step_is_kept_whatever_the_stride() -> None:
    capture = _capture(budget_steps=3)

    _offer(capture, 12)
    capture.finish()

    assert capture.stride > 1
    assert capture.frames()[-1] == 12


def test_a_default_llada_run_is_captured_at_every_step() -> None:
    """160 positions for 128 steps is the budget exactly, so the run
    people make most often is never thinned."""
    capture = CandidateCapture()

    for frame in range(1, 129):
        capture.offer(_step(frame, positions=160))
    capture.finish()

    assert capture.budget_records == CANDIDATE_BUDGET_RECORDS
    assert capture.stride == 1
    assert len(capture.frames()) == 128


def test_frames_out_of_order_are_a_programmer_error() -> None:
    capture = _capture(budget_steps=4)
    capture.offer(_step(3))

    with pytest.raises(AssertionError):
        capture.offer(_step(3))


def test_a_step_wider_than_the_budget_is_a_programmer_error() -> None:
    """No stride can fit a step that alone passes the budget, and a
    capture that silently kept nothing would read as a run without
    candidates."""
    capture = _capture(budget_steps=1)

    with pytest.raises(AssertionError):
        capture.offer(_step(1, positions=POSITIONS + 1))


# -- the message --


def test_nothing_offered_flushes_nothing() -> None:
    assert _capture(budget_steps=2).flush(str) is None


def test_the_message_names_its_frames_and_marks_held_tokens() -> None:
    capture = _capture(budget_steps=8)
    _offer(capture, 3)

    message = capture.flush(lambda token: f"<{token}>")

    assert message is not None
    assert message["type"] == MESSAGE_TYPE
    assert message["k"] == K
    assert message["stride"] == 1
    assert message["frames"] == [1, 2, 3]
    assert len(message["sets"]) == 3
    first = message["sets"][0][2]
    assert first["h"] == 20
    assert [row["id"] for row in first["c"]] == [20, 21, 22, 23, 24]
    assert first["c"][1] == {"id": 21, "t": "<21>", "p": 0.2}


def test_flush_decodes_each_distinct_id_once() -> None:
    """Decoding is the slow part of building the message, and a run's
    candidates repeat the same few thousand ids at every step."""
    calls: Counter[int] = Counter()

    def decode(token: int) -> str:
        calls[token] += 1
        return str(token)

    capture = _capture(budget_steps=8)
    _offer(capture, 6)
    capture.flush(decode)

    assert calls
    assert max(calls.values()) == 1
    assert len(calls) == POSITIONS * K


def test_a_held_token_outside_the_candidates_is_appended() -> None:
    """The row the popover could not otherwise explain is the token
    on screen, so it is appended with its own rank and its exact
    probability, which is routinely what rounds to zero."""
    step = _step(1, held=[0, 999, 20, 30])
    step.candidates.held_probs[1] = 0.000012
    step.candidates.held_ranks[1] = 4321
    capture = _capture(budget_steps=2)
    capture.offer(step)

    message = capture.flush(str)

    assert message is not None
    rows = message["sets"][0][1]["c"]
    assert len(rows) == K + 1
    assert message["sets"][0][1]["h"] == 999
    assert rows[-1]["id"] == 999
    assert rows[-1]["rank"] == 4321
    assert rows[-1]["p"] == pytest.approx(0.000012)


def test_a_held_token_among_the_candidates_is_not_repeated() -> None:
    capture = _capture(budget_steps=2)
    capture.offer(_step(1, held=[2, 11, 20, 30]))

    message = capture.flush(str)

    assert message is not None
    for position in range(POSITIONS):
        rows = message["sets"][0][position]["c"]
        assert len(rows) == K
        assert all("rank" not in row for row in rows)


def test_candidate_probabilities_round_to_four_places() -> None:
    step = _step(1)
    step.candidates.probs[0, 0] = 0.123456
    capture = _capture(budget_steps=2)
    capture.offer(step)

    message = capture.flush(str)

    assert message is not None
    assert message["sets"][0][0]["c"][0]["p"] == 0.1235


def test_five_candidates_match_the_autoregressive_popover() -> None:
    """One number for both kinds of run, so a popover reads the same
    whichever model produced it."""
    assert CANDIDATES_PER_POSITION == TOP_K_ALTERNATIVES
