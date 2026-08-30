"""DiffusionGemma reads confidence off the logits, cheaply.

Strategy: call `_from_logits` directly against the formulation it
replaced. It is a static method over one tensor, so this runs on CPU
with no model, no streamer and no queue. The frame-level consequences
of the change live in `test_dgemma_resume.py`; what is pinned here is
the arithmetic and the reason it was worth rewriting.

The old form built a float32 softmax over the whole canvas to read
one number per position. At 256 positions by roughly 262K vocabulary
entries that probability tensor is about 256 MiB, and the cast in
front of it copies, so the peak was near half a gigabyte per
denoising step. That number is the entire reason the measurement was
ever optional: a toggle existed so a user could decline to pay it,
which meant a run could be recorded with no confidence at all.

`exp(max - logsumexp)` is the same quantity from two reductions, and
taking a slice of positions at a time bounds what exists at once.
Passing proves the rewrite is arithmetically identical to the softmax
it replaced, that chunking does not lose the tail when the canvas
does not divide evenly, and that the result is a probability rather
than something that merely correlates with one.
"""

from __future__ import annotations

import inspect

import torch

from src.backends.registry import REGISTRY
from src.inference.dgemma_sampler import (
    LOGIT_CHUNK_POSITIONS,
    FrameQueueStreamer,
    streaming_generate,
    streaming_resume,
)

VOCAB = 512


def _reference(
    logits: torch.Tensor,
) -> tuple[list[int], list[float]]:
    """The formulation this replaced, kept as the oracle."""
    probs = torch.softmax(logits.float(), dim=-1)
    conf, ids = probs.max(dim=-1)
    return ids.tolist(), conf.tolist()


def _logits(positions: int, *, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    # bf16 because that is what the checkpoint hands over, and the
    # cast is exactly what the chunking is careful about.
    return (torch.randn(positions, VOCAB) * 4).to(torch.bfloat16)


# -- the same answer as before --


def test_the_argmax_is_unchanged() -> None:
    logits = _logits(64)

    ids, _ = FrameQueueStreamer._from_logits(logits)

    assert ids == _reference(logits)[0]


def test_the_probability_is_unchanged() -> None:
    logits = _logits(64)

    _, conf = FrameQueueStreamer._from_logits(logits)

    expected = _reference(logits)[1]
    # strict, so a chunking bug that returned the wrong number of
    # positions fails here rather than passing on a short zip.
    for got, want in zip(conf, expected, strict=True):
        assert abs(got - want) < 1e-5, f"{got} vs {want}"


def test_a_ragged_tail_is_not_dropped() -> None:
    """A canvas that does not divide evenly by the chunk size. The
    obvious way to get chunking wrong is to lose or repeat the last
    partial slice, and a canvas width is not obliged to be round."""
    positions = LOGIT_CHUNK_POSITIONS * 2 + 7
    logits = _logits(positions, seed=3)

    ids, conf = FrameQueueStreamer._from_logits(logits)

    assert len(ids) == positions
    assert len(conf) == positions
    assert ids == _reference(logits)[0]


def test_a_canvas_smaller_than_one_chunk_works() -> None:
    logits = _logits(3, seed=5)

    ids, conf = FrameQueueStreamer._from_logits(logits)

    assert len(ids) == 3
    assert ids == _reference(logits)[0]


def test_a_batched_leading_dimension_is_squeezed() -> None:
    """The model hands over (batch, positions, vocab); the frame is
    built from one canvas."""
    logits = _logits(16, seed=7)

    flat, _ = FrameQueueStreamer._from_logits(logits)
    batched, _ = FrameQueueStreamer._from_logits(logits.unsqueeze(0))

    assert flat == batched


# -- and it is a probability --


def test_every_value_is_a_probability() -> None:
    conf = FrameQueueStreamer._from_logits(_logits(48, seed=11))[1]

    for value in conf:
        assert 0.0 < value <= 1.0, value


def test_a_decided_position_reads_near_one() -> None:
    """Negative space for the test above: a distribution with one
    dominant logit must come back near 1.0, so "in range" is not
    passing on a function that returns a constant."""
    logits = torch.zeros(4, VOCAB)
    logits[:, 2] = 30.0

    conf = FrameQueueStreamer._from_logits(logits)[1]

    for value in conf:
        assert value > 0.99, value


def test_a_uniform_position_reads_near_the_floor() -> None:
    """The other end. A flat distribution over the vocabulary puts
    max probability at 1/VOCAB, which is the least sure the model
    can be."""
    logits = torch.zeros(4, VOCAB)

    conf = FrameQueueStreamer._from_logits(logits)[1]

    for value in conf:
        assert abs(value - 1.0 / VOCAB) < 1e-6, value


# -- and it is not optional --


def test_no_parameter_offers_to_turn_it_off() -> None:
    """The parameter panel is generated from `param_specs`, so the
    absence of a spec is the absence of the control. Asserted
    against the whole registry rather than one model, because the
    reason to reject this measurement was never model-specific."""
    for entry in REGISTRY.values():
        names = [spec.name for spec in entry.param_specs]
        assert "entropy_signal" not in names, entry.id


def test_neither_entry_point_takes_a_gate() -> None:
    """The spec is what the browser sees; these are what the worker
    calls. A spec removed while the plumbing stayed would leave the
    measurement switchable by anything that talks to the worker
    directly."""
    for entry_point in (streaming_generate, streaming_resume):
        params = inspect.signature(entry_point).parameters
        assert "entropy_signal" not in params, entry_point.__name__
