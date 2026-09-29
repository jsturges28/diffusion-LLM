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

The draft pass gets the same checks, plus its own: DiffusionGemma
reads confidence, entropy and its candidates off one walk, and the
token a draft shows has to be its first candidate even where two
logits tie.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.logit_signals import (
    DRAFT_CHUNK_POSITIONS,
    LOGIT_CHUNK_POSITIONS,
    draft_candidates,
    draft_signals,
    entropy_nats,
    picked_confidence,
    top_candidates,
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


def test_the_draft_reads_the_softmax_maximum() -> None:
    logits = _logits(64)
    probs = _softmax_oracle(logits)
    want_conf, want_ids = probs.max(dim=-1)

    signals = draft_signals(logits, 1)

    assert torch.equal(signals.ids, want_ids)
    assert torch.allclose(
        signals.confidence, want_conf, atol=TOLERANCE
    )


def test_draft_confidence_is_a_probability() -> None:
    """Negative space on the unit. A reduction that forgot to
    normalize would still correlate with confidence while being
    unusable as one, and the heatmap scale assumes [0, 1]."""
    conf = draft_signals(_logits(64, seed=11), 1).confidence

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


def test_picking_the_argmax_agrees_with_the_draft() -> None:
    """The two functions must not disagree where they overlap, which
    is every LLaDA run at temperature 0, the registry's default."""
    logits = _logits(64, seed=4)
    signals = draft_signals(logits, 1)

    picked = picked_confidence(logits, signals.ids)

    assert torch.allclose(picked, signals.confidence, atol=TOLERANCE)


def test_picked_confidence_can_be_far_below_the_maximum() -> None:
    """Guards the distinction itself. If this function quietly
    returned the maximum, a temperature run would overstate how sure
    the model was about what it actually chose."""
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    confident = picked_confidence(logits, torch.tensor([0]))
    unlikely = picked_confidence(logits, torch.tensor([1]))

    assert float(confident) > 0.9
    assert float(unlikely) < 0.01


# -- entropy, which is not confidence --


def test_entropy_of_a_uniform_distribution_is_log_vocab() -> None:
    """Hand-computable: four equally likely tokens carry ln 4 nats,
    and confidence is 0.25. Two numbers that cannot be mistaken for
    each other, which is the point of the case."""
    logits = torch.zeros((1, 4))

    entropy = float(entropy_nats(logits))
    conf = draft_signals(logits, 1).confidence

    assert entropy == pytest.approx(math.log(4), abs=1e-5)
    assert float(conf) == pytest.approx(0.25, abs=1e-5)


def test_entropy_of_a_peaked_distribution_is_near_zero() -> None:
    """The other end, and the pair to the test above: high confidence
    with low entropy. A channel emitting confidence under an entropy
    label would get these two backwards."""
    logits = torch.tensor([[40.0, 0.0, 0.0, 0.0]])

    entropy = float(entropy_nats(logits))
    conf = draft_signals(logits, 1).confidence

    assert entropy == pytest.approx(0.0, abs=1e-5)
    assert float(conf) == pytest.approx(1.0, abs=1e-5)


def test_entropy_matches_the_direct_definition() -> None:
    """Against `-sum(p log p)` computed from a materialized softmax,
    which is the formula the stable rearrangement replaces."""
    logits = _logits(64, seed=7)
    probs = _softmax_oracle(logits)
    want = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    got = entropy_nats(logits)

    assert torch.allclose(got, want, atol=1e-4)


def test_entropy_is_never_negative() -> None:
    """The arithmetic can land a hair below zero on a
    near-deterministic distribution, and a negative entropy would
    render as a colour outside the scale rather than as an error."""
    logits = torch.tensor([[100.0, -100.0, -100.0]])

    assert float(entropy_nats(logits)) >= 0.0


def test_entropy_and_confidence_are_not_the_same_series() -> None:
    """The report found a channel labelled entropy that emitted
    argmax confidence. Stated as a property so no future channel can
    be wired to the wrong reduction and still pass."""
    logits = _logits(32, seed=9)

    entropy = entropy_nats(logits)
    conf = draft_signals(logits, 1).confidence

    assert not torch.allclose(entropy, conf, atol=0.1)


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
    picks = torch.randint(0, VOCAB, (positions,))
    want = probs.gather(-1, picks.unsqueeze(-1)).squeeze(-1)

    picked = picked_confidence(logits, picks)
    entropy = entropy_nats(logits)

    assert picked.shape[0] == positions
    assert entropy.shape[0] == positions
    assert torch.allclose(picked, want, atol=TOLERANCE)


def test_bfloat16_logits_are_widened_before_reducing() -> None:
    """What the checkpoints actually hand over. bf16 accumulates
    visible error summing a hundred thousand terms, so the reductions
    cast per chunk; casting the whole canvas first would restore the
    allocation the chunking exists to avoid."""
    wide = _logits(48, seed=17)
    narrow = wide.to(torch.bfloat16)

    conf = picked_confidence(narrow, wide.argmax(dim=-1))
    entropy = entropy_nats(narrow)

    assert conf.dtype == torch.float32
    assert entropy.dtype == torch.float32
    # bf16 has about three decimal digits, so agreement with the
    # float32 answer is loose by construction; what is checked is that
    # the reduction happened in the wider type rather than in bf16.
    wide_conf = picked_confidence(wide, wide.argmax(dim=-1))
    assert torch.allclose(conf, wide_conf, atol=2e-2)


def test_a_mismatched_pick_count_is_a_programmer_error() -> None:
    """Negative space. One pick per position or the gather is
    meaningless, and a silent broadcast is worse than a crash."""
    picks = torch.zeros(4, dtype=torch.long)

    with pytest.raises(AssertionError):
        picked_confidence(_logits(8), picks)


# -- what a step was weighing, and where its held token stood --

CANDIDATES = 5


def _held(positions: int, *, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (positions,))


def test_top_candidates_match_the_softmax_top_k() -> None:
    """The candidates, likeliest first, and the held token's
    probability and rank, against a materialized softmax. The canvas
    crosses a chunk boundary, so the tail is covered too."""
    positions = LOGIT_CHUNK_POSITIONS * 2 + 5
    logits = _logits(positions, seed=23)
    held = _held(positions, seed=29)
    probs = _softmax_oracle(logits)
    want_probs, want_ids = torch.topk(probs, CANDIDATES, dim=-1)
    want_held = probs.gather(-1, held.unsqueeze(-1))
    want_ranks = (probs > want_held).sum(dim=-1) + 1

    got = top_candidates(logits, CANDIDATES, held)

    assert got.ids.shape == (positions, CANDIDATES)
    assert torch.equal(got.ids, want_ids)
    assert torch.allclose(got.probs, want_probs, atol=TOLERANCE)
    assert torch.allclose(
        got.held_probs, want_held.squeeze(-1), atol=TOLERANCE
    )
    assert torch.equal(got.held_ranks, want_ranks)


def test_a_held_argmax_ranks_first() -> None:
    """A position holding the model's favourite ranks 1 and carries
    the top candidate's probability, which is the case the popover
    marks without appending a row."""
    logits = _logits(40, seed=31)
    held = logits.argmax(dim=-1)

    got = top_candidates(logits, CANDIDATES, held)

    assert torch.equal(got.held_ranks, torch.ones_like(held))
    assert torch.allclose(got.held_probs, got.probs[:, 0])


def test_tied_logits_share_the_better_rank() -> None:
    """Counting strictly greater tokens, as the autoregressive
    sampler does, so two popovers never disagree about a tie."""
    logits = torch.zeros(1, VOCAB)
    logits[0, 3] = 5.0
    logits[0, 7] = 5.0
    logits[0, 9] = 2.0

    tied = top_candidates(logits, CANDIDATES, torch.tensor([7]))
    third = top_candidates(logits, CANDIDATES, torch.tensor([9]))

    assert int(tied.held_ranks[0]) == 1
    assert int(third.held_ranks[0]) == 3


def test_top_candidates_widen_bfloat16() -> None:
    """The checkpoints hand over bf16; the probabilities come back in
    float32, reduced per chunk like the other signals."""
    narrow = _logits(48, seed=37).to(torch.bfloat16)

    got = top_candidates(narrow, CANDIDATES, _held(48, seed=41))

    assert got.probs.dtype == torch.float32
    assert got.held_probs.dtype == torch.float32


@pytest.mark.parametrize("k", [0, VOCAB + 1])
def test_k_outside_the_vocabulary_is_a_programmer_error(
    k: int,
) -> None:
    with pytest.raises(AssertionError):
        top_candidates(_logits(4), k, _held(4, seed=43))


def test_a_mismatched_held_count_is_a_programmer_error() -> None:
    """One held token per position, for the same reason as a pick."""
    with pytest.raises(AssertionError):
        top_candidates(_logits(8), CANDIDATES, _held(4, seed=47))


# -- the draft pass: every DiffusionGemma signal off one walk --


def _float64_oracle(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The distribution and its entropy in double precision, which is
    what both float32 forms are compared against."""
    wide = logits.double()
    probs = torch.softmax(wide, dim=-1)
    entropy = -(probs * torch.log_softmax(wide, dim=-1)).sum(dim=-1)
    return probs, entropy


@pytest.mark.parametrize(
    "positions",
    [
        1,
        DRAFT_CHUNK_POSITIONS - 1,
        DRAFT_CHUNK_POSITIONS,
        DRAFT_CHUNK_POSITIONS + 1,
        LOGIT_CHUNK_POSITIONS * 2 + 7,
    ],
)
def test_the_draft_pass_matches_the_softmax_it_avoids(
    positions: int,
) -> None:
    """Every field against a double-precision softmax, at widths that
    cross the draft pass's own chunk boundaries."""
    logits = _logits(positions, seed=53)
    probs, want_entropy = _float64_oracle(logits)
    want_probs, want_ids = torch.topk(probs, CANDIDATES, dim=-1)

    signals = draft_signals(logits, CANDIDATES)

    assert torch.equal(signals.candidate_ids, want_ids)
    assert torch.equal(signals.ids, want_ids[:, 0])
    assert torch.allclose(
        signals.candidate_probs.double(), want_probs, atol=TOLERANCE
    )
    assert torch.allclose(
        signals.confidence.double(), want_probs[:, 0], atol=TOLERANCE
    )
    assert torch.allclose(
        signals.entropy.double(), want_entropy, atol=TOLERANCE
    )


def test_the_draft_entropy_is_the_quantity_the_others_read() -> None:
    """One signal, two implementations: the draft pass and
    `entropy_nats` must not disagree about what a run's entropy is."""
    logits = _logits(64, seed=59)

    signals = draft_signals(logits, 1)

    assert torch.allclose(
        signals.entropy, entropy_nats(logits), atol=1e-4
    )


def test_the_draft_pass_widens_bfloat16() -> None:
    """What DiffusionGemma hands over: bf16, on the host. The signals
    come back in float32, close to the float32 answer."""
    wide = _logits(48, seed=61)
    narrow = wide.to(torch.bfloat16)

    signals = draft_signals(narrow, CANDIDATES)

    assert signals.confidence.dtype == torch.float32
    assert signals.entropy.dtype == torch.float32
    assert signals.candidate_probs.dtype == torch.float32
    exact = draft_signals(wide, CANDIDATES)
    assert torch.allclose(
        signals.confidence, exact.confidence, atol=2e-2
    )


def test_a_draft_shows_its_first_candidate_even_at_a_tie() -> None:
    """Where logits tie exactly, ``argmax`` and ``topk`` choose
    differently: on the CPU this three-way tie comes back from
    ``topk`` as 301, 402, 17, and ``argmax`` says 17. bf16 makes ties
    common at DiffusionGemma's vocabulary. The shown token comes from
    the same call as the list, so the marked row is always the first,
    and it is an argmax."""
    tied = [17, 301, 402]
    logits = torch.zeros(3, VOCAB)
    logits[:, tied] = 6.0

    signals = draft_signals(logits, CANDIDATES)

    assert torch.equal(signals.ids, signals.candidate_ids[:, 0])
    shown = logits.gather(-1, signals.ids.unsqueeze(-1)).squeeze(-1)
    assert bool((shown == 6.0).all())
    listed = signals.candidate_ids[:, :3].sort(dim=-1).values
    assert bool((listed == torch.tensor(tied)).all())


def test_one_candidate_is_enough_without_alternatives() -> None:
    """A run with Alternatives off asks for one, which is still the
    likeliest token and its probability."""
    logits = _logits(20, seed=67)

    signals = draft_signals(logits, 1)

    assert signals.candidate_ids.shape == (20, 1)
    assert torch.equal(signals.ids, signals.candidate_ids[:, 0])
    assert torch.allclose(
        signals.confidence, signals.candidate_probs[:, 0]
    )


def test_draft_candidates_agree_with_the_general_reduction() -> None:
    """The shortcut that skips counting: a draft holds its own
    argmax, so its rank is 1 and its probability the first
    candidate's, exactly what `top_candidates` finds by counting."""
    logits = _logits(40, seed=71)
    signals = draft_signals(logits, CANDIDATES)

    got = draft_candidates(signals)
    want = top_candidates(logits, CANDIDATES, signals.ids)

    assert torch.equal(got.ids, want.ids)
    assert torch.allclose(got.probs, want.probs, atol=TOLERANCE)
    assert torch.allclose(
        got.held_probs, want.held_probs, atol=TOLERANCE
    )
    assert torch.equal(got.held_ranks, want.held_ranks)
    assert bool((got.held_ranks == 1).all())


def test_the_draft_extremes_read_as_themselves() -> None:
    """Hand-computable ends: a flat row is confidence 1/V at ln V
    nats, a decided one confidence 1 at zero, never below it."""
    flat = draft_signals(torch.zeros(2, VOCAB), 1)
    decided = torch.full((2, VOCAB), -100.0)
    decided[:, 3] = 100.0
    sure = draft_signals(decided, 1)

    floor = torch.full((2,), 1 / VOCAB)
    assert torch.allclose(flat.confidence, floor)
    assert torch.allclose(
        flat.entropy, torch.full((2,), math.log(VOCAB)), atol=1e-5
    )
    assert torch.allclose(sure.confidence, torch.ones(2))
    assert bool((sure.entropy >= 0.0).all())
    assert float(sure.entropy.max()) < 1e-5


def test_a_draft_entropy_never_shows_below_zero() -> None:
    """A peaked row far from zero, where float32 spacing is coarse:
    the arithmetic lands at -0.00018 nats before the clamp, and a
    negative entropy would draw outside the colour scale."""
    torch.manual_seed(0)
    logits = torch.randn(1, VOCAB) * 0.3 + 1000.0
    logits[0, 7] = 1020.0

    entropy = draft_signals(logits, 1).entropy

    assert float(entropy[0]) >= 0.0


def test_the_draft_pass_leaves_its_input_alone() -> None:
    """float32 logits are reduced without being copied, so an
    in-place step on them would write into the caller's tensor."""
    logits = _logits(16, seed=73)
    before = logits.clone()

    draft_signals(logits, CANDIDATES)

    assert torch.equal(logits, before)


@pytest.mark.parametrize("k", [0, VOCAB + 1])
def test_the_draft_pass_refuses_k_outside_the_vocabulary(
    k: int,
) -> None:
    with pytest.raises(AssertionError):
        draft_signals(_logits(4), k)
