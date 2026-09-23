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
"""

from __future__ import annotations

from typing import Tuple

import torch

# Positions reduced at a time. The transient is this many rows of the
# vocabulary in float32: 32 rows of DiffusionGemma's ~262K vocabulary
# is about 33 MiB, against the 256 MiB a whole-canvas softmax holds,
# and 32 rows of LLaDA's ~126K is about 16 MiB against 96 MiB at that
# model's default canvas. Large enough that per-chunk overhead is
# noise, small enough that the peak is not.
LOGIT_CHUNK_POSITIONS = 32

assert LOGIT_CHUNK_POSITIONS > 0, "a chunk must hold a position"


def top_confidence(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The likeliest token per position, and its probability.

    ``logits`` is (positions, vocabulary). Returns (ids, probability),
    both (positions,), on the input's device.

    ``exp(max - logsumexp)`` is ``softmax(...).max()`` from two
    reductions rather than a materialized distribution.
    """
    assert logits.dim() == 2, "expected (positions, vocabulary)"
    id_chunks = []
    conf_chunks = []
    for chunk in torch.split(
        logits, LOGIT_CHUNK_POSITIONS, dim=0
    ):
        wide = _widen(chunk)
        top, top_ids = wide.max(dim=-1)
        spread = torch.logsumexp(wide, dim=-1)
        id_chunks.append(top_ids)
        conf_chunks.append(torch.exp(top - spread))
    return (
        torch.cat(id_chunks, dim=0),
        torch.cat(conf_chunks, dim=0),
    )


def picked_confidence(
    logits: torch.Tensor, picks: torch.Tensor
) -> torch.Tensor:
    """The probability of an already-chosen token per position.

    Separate from ``top_confidence`` because the choice is not always
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
