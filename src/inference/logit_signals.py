"""Read XAI signals off logits without materializing a softmax.

Every signal this project shows comes from one forward pass's logits:
how confident the model was, how spread its distribution was, and
eventually what else it was considering. The obvious way to get any of
them is `softmax(logits)`, and the obvious way does not fit: a
probability tensor over the whole canvas is hundreds of megabytes on a
card already holding the model.

So the reductions live here instead, in the numerically stable form
and chunked along positions, so the transient is bounded by the chunk
rather than the canvas. `ROADMAP-03` asks for exactly this:
"numerically stable reductions over logits for max probability,
entropy, and optional top-k without retaining a full probability
tensor longer than required".

Shared by both diffusion samplers rather than living with either one.
DiffusionGemma proved the technique first on its own confidence read;
LLaDA needs the same thing for a picked token rather than the argmax,
and both now need entropy. A module of its own is what stops the
constant below being written twice and drifting.

The two read differently. LLaDA's logits are on the card, where a
pass is cheap, so it calls one reduction per signal. DiffusionGemma's
arrive on the host, copied there by transformers before the streamer
sees them, so every pass is paid in CPU time; ``draft_signals`` reads
all of its signals off one.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

import torch

# Positions reduced at a time. The transient is this many rows of the
# vocabulary in float32: 32 rows of DiffusionGemma's ~262K vocabulary
# is about 33 MiB, against the 256 MiB a whole-canvas softmax holds,
# and 32 rows of LLaDA's ~126K is about 16 MiB against 96 MiB at that
# model's default canvas. Large enough that per-chunk overhead is
# noise, small enough that the peak is not.
LOGIT_CHUNK_POSITIONS = 32

assert LOGIT_CHUNK_POSITIONS > 0, "a chunk must hold a position"

# Positions per chunk for the draft pass, which runs on the host,
# where a chunk that stays in cache is what makes it fast: at a
# draft's 256 by 262,144 shape in bf16, 8 rows (8 MiB of float32) took
# about 55 ms on the maintainer's machine, against 66 at 16 rows and
# 98 at the 32 above. Those suit the card, where a chunk is a round of
# kernel launches.
DRAFT_CHUNK_POSITIONS = 8

assert 0 < DRAFT_CHUNK_POSITIONS <= LOGIT_CHUNK_POSITIONS, (
    "the draft pass may only shrink the transient"
)


def picked_confidence(
    logits: torch.Tensor, picks: torch.Tensor
) -> torch.Tensor:
    """The probability of an already-chosen token per position.

    Separate from ``draft_signals`` because the choice is not always
    the argmax. LLaDA picks from Gumbel-noised logits and then reports
    the clean probability of what it picked, so with a temperature the
    pick and the maximum part company, and reading the maximum would
    quietly overstate how sure the model was.

    ``logits`` is (positions, vocabulary) and ``picks`` is
    (positions,). Returns (positions,).
    """
    assert logits.dim() == 2, "expected (positions, vocabulary)"
    assert picks.dim() == 1, "expected one pick per position"
    assert picks.shape[0] == logits.shape[0], "pick per position"
    conf_chunks = []
    pick_chunks = torch.split(picks, LOGIT_CHUNK_POSITIONS, dim=0)
    logit_chunks = torch.split(
        logits, LOGIT_CHUNK_POSITIONS, dim=0
    )
    for chunk, chosen in zip(
        logit_chunks, pick_chunks, strict=True
    ):
        wide = _widen(chunk)
        taken = torch.gather(
            wide, dim=-1, index=chosen.unsqueeze(-1)
        ).squeeze(-1)
        spread = torch.logsumexp(wide, dim=-1)
        conf_chunks.append(torch.exp(taken - spread))
    return torch.cat(conf_chunks, dim=0)


def entropy_nats(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of each position's distribution, in nats.

    ``H = logsumexp(z) - sum(softmax(z) * z)``, which is the stable
    rearrangement of ``-sum(p log p)``. The softmax still exists, but
    only for one chunk at a time, which is the whole difference.

    Nats rather than bits to match the autoregressive sampler, which
    has reported nats since entropy first appeared there. Two units
    for one quantity would make the Analytics scale a guess.
    """
    assert logits.dim() == 2, "expected (positions, vocabulary)"
    chunks = []
    for chunk in torch.split(
        logits, LOGIT_CHUNK_POSITIONS, dim=0
    ):
        wide = _widen(chunk)
        spread = torch.logsumexp(wide, dim=-1)
        probs = torch.softmax(wide, dim=-1)
        weighted = (probs * wide).sum(dim=-1)
        chunks.append(spread - weighted)
    value = torch.cat(chunks, dim=0)
    # Clamped because the arithmetic can land a hair below zero on a
    # near-deterministic distribution, and a negative entropy would
    # render as a colour outside the scale rather than as an error.
    return value.clamp_min(0.0)


class Candidates(NamedTuple):
    """What a step was weighing at each position, and where the token
    each position holds stood in it."""

    ids: torch.Tensor  # (positions, k), likeliest first
    probs: torch.Tensor  # (positions, k)
    held_probs: torch.Tensor  # (positions,)
    held_ranks: torch.Tensor  # (positions,), 1 for the likeliest


def top_candidates(
    logits: torch.Tensor, k: int, held: torch.Tensor
) -> Candidates:
    """The k likeliest tokens per position, and the held token's
    standing among all of them.

    ``logits`` is (positions, vocabulary) and ``held`` is
    (positions,), the token each position holds at this step. The
    held token's rank
    is one plus how many tokens the model preferred, counted strictly,
    so ties share the better rank: the convention the autoregressive
    sampler's ``_token_rank`` uses, which keeps the two popovers
    agreeing about what a rank means.
    """
    assert logits.dim() == 2, "expected (positions, vocabulary)"
    assert held.dim() == 1, "expected one held token per position"
    assert held.shape[0] == logits.shape[0], "held per position"
    assert 0 < k <= logits.shape[1], "k within the vocabulary"
    parts: Tuple[List[torch.Tensor], ...] = ([], [], [], [])
    held_chunks = torch.split(held, LOGIT_CHUNK_POSITIONS, dim=0)
    logit_chunks = torch.split(logits, LOGIT_CHUNK_POSITIONS, dim=0)
    for chunk, chosen in zip(logit_chunks, held_chunks, strict=True):
        wide = _widen(chunk)
        spread = torch.logsumexp(wide, dim=-1, keepdim=True)
        top, top_ids = torch.topk(
            wide, k, dim=-1, largest=True, sorted=True
        )
        taken = torch.gather(
            wide, dim=-1, index=chosen.unsqueeze(-1)
        )
        parts[0].append(top_ids)
        parts[1].append(torch.exp(top - spread))
        parts[2].append(torch.exp(taken - spread).squeeze(-1))
        parts[3].append((wide > taken).sum(dim=-1) + 1)
    return Candidates(*(torch.cat(part, dim=0) for part in parts))


class DraftSignals(NamedTuple):
    """Everything a DiffusionGemma draft reads off its logits."""

    ids: torch.Tensor  # (positions,), the argmax the draft shows
    confidence: torch.Tensor  # (positions,), its probability
    entropy: torch.Tensor  # (positions,), in nats
    candidate_ids: torch.Tensor  # (positions, k), likeliest first
    candidate_probs: torch.Tensor  # (positions, k)


def draft_signals(logits: torch.Tensor, k: int) -> DraftSignals:
    """Confidence, entropy and the k likeliest tokens, in one pass.

    For a model whose draft shows its argmax. Reading the three
    separately widened and exponentiated every logit three times,
    which on the host cost more than the capture of candidates it
    was paying for; here each chunk is widened once and exponentiated
    once, and the rest is one ``topk`` and one dot product per row.
    ``k`` is 1 when no candidates are wanted, which still leaves the
    likeliest token and its probability.

    ``ids`` is the first of the top k, so the token a frame shows is
    always its first candidate, even where two logits tie exactly.

    The logits must be finite, as DiffusionGemma's temperature
    schedule leaves them: an ``-inf`` would make the entropy NaN, as
    it would in ``entropy_nats``. Not checked per element, because
    that would be a pass over every logit, the cost this removes.
    """
    assert logits.dim() == 2, "expected (positions, vocabulary)"
    assert logits.shape[0] > 0, "a draft has positions"
    assert 0 < k <= logits.shape[1], "k within the vocabulary"
    rows = min(DRAFT_CHUNK_POSITIONS, logits.shape[0])
    wide = torch.empty(
        (rows, logits.shape[1]),
        dtype=torch.float32,
        device=logits.device,
    )
    weights = torch.empty_like(wide)
    chunks = []
    for chunk in torch.split(logits, DRAFT_CHUNK_POSITIONS, dim=0):
        count = chunk.shape[0]
        chunks.append(
            _draft_chunk(chunk, k, wide[:count], weights[:count])
        )
    fields = zip(*chunks, strict=True)
    return DraftSignals(
        *(torch.cat(field, dim=0) for field in fields)
    )


def draft_candidates(signals: DraftSignals) -> Candidates:
    """A draft's candidates, with the token it shows as the held one.

    That token is the first candidate, so it needs no count over the
    vocabulary to be ranked: it ranks first, at the first candidate's
    probability.
    """
    assert signals.candidate_ids.dim() == 2, "a set per position"
    assert signals.candidate_ids.shape[0] == signals.ids.shape[0]
    return Candidates(
        ids=signals.candidate_ids,
        probs=signals.candidate_probs,
        held_probs=signals.candidate_probs[:, 0],
        held_ranks=torch.ones_like(signals.ids),
    )


def _draft_chunk(
    chunk: torch.Tensor,
    k: int,
    wide: torch.Tensor,
    weights: torch.Tensor,
) -> DraftSignals:
    """One chunk's signals, worked in two float32 buffers the caller
    reuses from chunk to chunk.

    Reused because allocating them per chunk made the pass hostage to
    the allocator: at a draft's shape the same arithmetic took 39 to
    53 ms with the buffers reused and 111 to 149 ms without, as freed
    memory went back to the system and came back faulted in. Copying
    into ``wide`` is also the widening, and nothing is written to the
    caller's logits. Nothing returned is a view of either buffer, or
    the next chunk would overwrite it.

    Entropy as ``peak + log(total) - E[z]``, where ``total`` sums
    ``exp(z - peak)`` and the expectation is a dot product of those
    weights with the logits: the quantity ``entropy_nats`` reads, with
    one exponential where that takes three. Against float64 at a
    draft's shape it is within 3e-6 nats, where ``entropy_nats``
    drifts by 3e-4.
    """
    assert wide.shape == chunk.shape, "a buffer row per position"
    assert weights.shape == chunk.shape, "a buffer row per position"
    wide.copy_(chunk)
    top, top_ids = torch.topk(
        wide, k, dim=-1, largest=True, sorted=True
    )
    peak = top[:, :1]
    torch.sub(wide, peak, out=weights)
    weights.exp_()
    total = weights.sum(dim=-1, keepdim=True)
    log_total = torch.log(total)
    dot = torch.linalg.vecdot(weights, wide, dim=-1).unsqueeze(-1)
    entropy = peak + log_total - dot / total
    return DraftSignals(
        ids=top_ids[:, 0],
        confidence=total.reciprocal().squeeze(-1),
        entropy=entropy.squeeze(-1).clamp_min(0.0),
        candidate_ids=top_ids,
        candidate_probs=torch.exp(top - peak - log_total),
    )


def _widen(chunk: torch.Tensor) -> torch.Tensor:
    """One chunk in float32, cast per chunk and not up front.

    bf16 accumulates visible error summing a hundred thousand terms,
    so the reductions need the wider type. Casting the whole canvas
    first would restore exactly the allocation this module exists to
    avoid, which is why the cast is here rather than at the caller.
    """
    if chunk.dtype in (torch.float16, torch.bfloat16):
        return chunk.float()
    return chunk
