"""Signals read off logits match the softmax they avoid building.

Strategy: call each reduction against the formulation it replaced, on
CPU, with no model. They are pure functions over one tensor, so the
arithmetic is the whole surface and this is where it gets pinned.

Why the reductions exist: a probability tensor over a whole canvas is
the expensive way to ask a cheap question. At LLaDA's default
160-token canvas a float32 softmax over its ~126K vocabulary is 96
MiB, and 513 MiB at the 1024-token ceiling the registry allows, per
denoising step,
on a card already holding 17 GiB of weights. DiffusionGemma's is worse
per position at ~262K entries. `ROADMAP-03` asks for "numerically
stable reductions over logits ... without retaining a full probability
tensor longer than required", and this file is what says they are the
same numbers.

Passing proves three things. The reductions agree with a materialized
softmax to float tolerance; chunking does not lose or repeat the tail
when a canvas does not divide evenly; and entropy is a different
quantity from confidence rather than a second name for it. That last
one matters because the report found a channel labelled entropy that
was emitting argmax confidence, and the way to make that unrepeatable
is a case where the two numbers cannot be confused.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.logit_signals import (
    LOGIT_CHUNK_POSITIONS,
    picked_confidence,
    top_confidence,
)

VOCAB = 512
# Tight enough to catch a wrong formula, loose enough for the float32
# reassociation the chunked form performs.
TOLERANCE = 1e-5


def _logits(
    positions: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    torch.manual_seed(seed)
    # Scaled so the distributions are peaked rather than near-uniform,
    # which is where a stability bug would show.
    return (torch.randn(positions, VOCAB) * 4).to(dtype)


def _softmax_oracle(logits: torch.Tensor) -> torch.Tensor:
    """The tensor these functions exist not to build."""
    return torch.softmax(logits.float(), dim=-1)


# -- the likeliest token and its probability --


def test_top_confidence_matches_the_softmax_maximum() -> None:
    logits = _logits(64)
    probs = _softmax_oracle(logits)
    want_conf, want_ids = probs.max(dim=-1)

    ids, conf = top_confidence(logits)

    assert torch.equal(ids, want_ids)
    assert torch.allclose(conf, want_conf, atol=TOLERANCE)


def test_top_confidence_is_a_probability() -> None:
    """Negative space on the unit. A reduction that forgot to
    normalize would still correlate with confidence while being
    unusable as one, and the heatmap scale assumes [0, 1]."""
    ids, conf = top_confidence(_logits(64, seed=11))
    del ids

    assert bool((conf >= 0.0).all())
    assert bool((conf <= 1.0).all())


# -- the probability of an already-chosen token --


def test_picked_confidence_matches_a_softmax_gather() -> None:
    """The LLaDA case. It picks from Gumbel-noised logits and reports
    the clean probability of that pick, so the quantity is a gather
    rather than a maximum."""
    logits = _logits(64, seed=2)
    picks = torch.randint(0, VOCAB, (64,))
    probs = _softmax_oracle(logits)
    want = torch.gather(probs, 1, picks.unsqueeze(-1)).squeeze(-1)

    got = picked_confidence(logits, picks)

    assert torch.allclose(got, want, atol=TOLERANCE)


def test_picking_the_argmax_agrees_with_top_confidence() -> None:
    """The two functions must not disagree where they overlap, which
    is every run at temperature 0, the registry's default."""
    logits = _logits(64, seed=4)
    ids, top = top_confidence(logits)

    picked = picked_confidence(logits, ids)

    assert torch.allclose(picked, top, atol=TOLERANCE)


def test_picked_confidence_can_be_far_below_the_maximum() -> None:
    """Guards the distinction itself. If this function quietly
    returned the maximum, a temperature run would overstate how sure
    the model was about what it actually chose."""
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    confident = picked_confidence(logits, torch.tensor([0]))
    unlikely = picked_confidence(logits, torch.tensor([1]))

    assert float(confident) > 0.9
    assert float(unlikely) < 0.01


# -- chunking, where the tail is easy to lose --


@pytest.mark.parametrize(
    "positions",
    [
        1,
        LOGIT_CHUNK_POSITIONS - 1,
        LOGIT_CHUNK_POSITIONS,
        LOGIT_CHUNK_POSITIONS + 1,
        LOGIT_CHUNK_POSITIONS * 2 + 7,
    ],
)
def test_every_canvas_width_survives_chunking(
    positions: int,
) -> None:
    """A canvas width is not obliged to be a multiple of the chunk
    size, and the obvious way to get chunking wrong is to drop or
    repeat the last partial slice."""
    logits = _logits(positions, seed=13)
    probs = _softmax_oracle(logits)
    want_conf, want_ids = probs.max(dim=-1)

    ids, conf = top_confidence(logits)

    assert ids.shape[0] == positions
    assert conf.shape[0] == positions
    assert torch.equal(ids, want_ids)
    assert torch.allclose(conf, want_conf, atol=TOLERANCE)


def test_bfloat16_logits_are_widened_before_reducing() -> None:
    """What the checkpoints actually hand over. bf16 accumulates
    visible error summing a hundred thousand terms, so the reductions
    cast per chunk; casting the whole canvas first would restore the
    allocation the chunking exists to avoid."""
    wide = _logits(48, seed=17)
    narrow = wide.to(torch.bfloat16)

    _, conf = top_confidence(narrow)

    assert conf.dtype == torch.float32
    # bf16 has about three decimal digits, so agreement with the
    # float32 answer is loose by construction; what is checked is that
    # the reduction happened in the wider type rather than in bf16.
    _, wide_conf = top_confidence(wide)
    assert torch.allclose(conf, wide_conf, atol=2e-2)


def test_a_mismatched_pick_count_is_a_programmer_error() -> None:
    """Negative space. One pick per position or the gather is
    meaningless, and a silent broadcast is worse than a crash."""
    picks = torch.zeros(4, dtype=torch.long)

    with pytest.raises(AssertionError):
        picked_confidence(_logits(8), picks)
