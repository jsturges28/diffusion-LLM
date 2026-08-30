"""Tests that a LLaDA frame records what a masked position holds.

Strategy: two levels. ``_build_token_list`` is a pure function over a
tensor and a tokenizer, so it is called directly for the boundary
cases, and a real ``streaming_resume`` over the stub model from
``test_llada_resume_conf`` proves the sampler actually threads its
prediction that far rather than the parameter merely existing.

LLaDA computes an argmax over every position on every step and used
to drop it one line after using it, writing the mask glyph into the
frame instead. So the app has been reporting how confident the model
is (that number drives mask opacity) about a guess it could not name,
and the metrics strip showed a guess for DiffusionGemma and a block
for LLaDA at the same kind of position.

Passing proves a masked position carries the model's current pick,
that a resolved position still carries what it actually settled on
rather than the pick, that ``m`` keeps saying unsettled either way,
and that the opening frame keeps the glyph, since at that point there
is no prediction to name.
"""

from __future__ import annotations

import torch

from src.inference.streaming_sampler import (
    MASK_ID,
    _build_token_list,
)
from tests.inference.test_llada_resume_conf import (
    GEN_LENGTH,
    PROMPT_LEN,
    _resume,
)

MASK_GLYPH = "\u2591"

# Two settled positions and two masked, so a frame can be read for
# both halves of the convention at once.
FRAME_IDS = [5, MASK_ID, 7, MASK_ID]
# What the model would put at each position this step. The entries
# under the settled ones differ from what is there, which is what
# proves the guess is not written over a committed token.
GUESS_IDS = [1, 3, 2, 4]

assert len(FRAME_IDS) == len(GUESS_IDS), "one guess per position"
assert GUESS_IDS[0] != FRAME_IDS[0], (
    "a settled position must disagree with its guess"
)


class _DigitTokenizer:
    """Decodes an id to its digits, so a frame is readable as text."""

    def decode(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join(str(int(i)) for i in ids)


def _tokens(*, with_guess: bool) -> list[dict]:
    x = torch.tensor(
        [[0] * PROMPT_LEN + FRAME_IDS], dtype=torch.long
    )
    guess = None
    if with_guess:
        guess = torch.tensor(GUESS_IDS, dtype=torch.long)
    return _build_token_list(
        x,
        PROMPT_LEN,
        _DigitTokenizer(),
        None,
        None,
        guess,
    )


# -- what a frame records --


def test_a_masked_position_names_the_current_pick() -> None:
    tokens = _tokens(with_guess=True)

    assert tokens[1]["t"] == "3"
    assert tokens[3]["t"] == "4"


def test_a_named_pick_is_still_marked_unsettled() -> None:
    """The reveal is a display choice downstream of this. If ``m``
    softened here, every consumer would read a guess as an answer:
    the convergence basis, the diff, the metrics strip."""
    tokens = _tokens(with_guess=True)

    assert tokens[1]["m"] is True
    assert tokens[3]["m"] is True


def test_a_masked_position_keeps_the_mask_id() -> None:
    """``t`` gains the guess; ``id`` stays the truth about the
    canvas, which is what resume and remasking work off."""
    tokens = _tokens(with_guess=True)

    assert tokens[1]["id"] == MASK_ID
    assert tokens[3]["id"] == MASK_ID


def test_a_settled_position_ignores_the_pick() -> None:
    """The model has an opinion about a position it already wrote,
    and the frame must show what is there, not that opinion."""
    tokens = _tokens(with_guess=True)

    assert tokens[0]["t"] == "5"
    assert tokens[2]["t"] == "7"
    assert tokens[0]["m"] is False


def test_the_opening_frame_keeps_the_glyph() -> None:
    """Frame 0 is emitted before the first step, so there is no
    prediction. A blank canvas of blocks is the honest drawing of
    that, and it is a boundary rather than a gap."""
    tokens = _tokens(with_guess=False)

    assert tokens[1]["t"] == MASK_GLYPH
    assert tokens[3]["t"] == MASK_GLYPH
    assert tokens[0]["t"] == "5"


# -- and that the sampler gets it there --


def test_a_streamed_frame_carries_the_pick() -> None:
    """The stub model is biased toward token 3, so every masked
    position in a mid-run frame should name it."""
    frames = [
        item
        for item in _resume(
            remask_positions=[1, 3], remaining_steps=2
        )
        if item.get("type") == "frame"
    ]
    assert len(frames) >= 2, "need a frame after a step"

    tokens = frames[1]["tokens"]
    still_masked = [tok for tok in tokens if tok["m"]]
    assert still_masked, "the fixture must leave one unsettled"
    for tok in still_masked:
        assert tok["t"] == "3"


def test_the_streams_first_frame_keeps_the_glyph() -> None:
    frames = [
        item
        for item in _resume(
            remask_positions=[1, 3], remaining_steps=2
        )
        if item.get("type") == "frame"
    ]

    first = frames[0]["tokens"]
    assert first[1]["t"] == MASK_GLYPH
    assert first[3]["t"] == MASK_GLYPH


def test_a_pick_never_lengthens_the_frame() -> None:
    """Every position emits one entry either way, because the two
    stacked token layers align by index."""
    with_guess = _tokens(with_guess=True)
    without = _tokens(with_guess=False)

    assert len(with_guess) == GEN_LENGTH
    assert len(without) == GEN_LENGTH
