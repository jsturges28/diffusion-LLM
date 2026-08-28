"""Bounded per-frame checkpoints for reproducible interventions.

A diffusion run keeps one of these per streamed frame so a later
edit can re-enter that exact frame: the canvas it held, the
confidence state the display was reporting, and the random state
the sampler was about to draw from.

The random state is the point of the module. Re-seeding from the
run's seed reproduces the run's *first* step, not the step the user
picked, so the same edit made before and after other random work
would diverge. Restoring the frame's own generator state is what
makes two attempts at one intervention agree.

The two diffusion backends checkpoint different things because
their confidence means different things: LLaDA records the softmax
probability each position was revealed at, DiffusionGemma records
how many consecutive steps a position has held its prediction.
``extra`` carries whichever applies, so the shared fields stay
shared without pretending the models are the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple, Union

import torch

# Ceiling on the random state a single run may retain, far above
# what one produces: a CPU generator state is 5,056 bytes, so a
# 129-frame LLaDA run spends about 650 KiB and a very long
# DiffusionGemma run a few MiB. Present so an unfamiliar step count
# cannot quietly pin an unbounded amount of host memory for a
# session, not because any real run approaches it.
CHECKPOINT_RNG_BYTES_MAX = 64 * 1024 * 1024

# Python ints in a tuple or set cost far more than this, but the
# budget exists to bound torch state, and counting the bookkeeping
# at machine-word width keeps the estimate honest without pretending
# to measure interpreter overhead.
_INT_BYTES_APPROX = 8


@dataclass(frozen=True)
class RngState:
    """Torch's generator state at one instant, CPU and CUDA.

    Both are kept because which one a step consumes depends on where
    its tensors live: LLaDA's gumbel noise and random remasking draw
    on the device holding the logits, so a CPU host exercises the
    first field and a GPU host the second.
    """

    cpu: torch.Tensor
    cuda: Tuple[torch.Tensor, ...]

    def nbytes(self) -> int:
        total = self.cpu.numel() * self.cpu.element_size()
        for state in self.cuda:
            total += state.numel() * state.element_size()
        assert total > 0, "a captured state occupies something"
        return total


def rng_capture() -> RngState:
    """The generator state the next step will draw from."""
    cpu = torch.get_rng_state()
    assert cpu.dtype == torch.uint8, "rng state is a byte tensor"
    cuda: Tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda = tuple(torch.cuda.get_rng_state_all())
    return RngState(cpu=cpu, cuda=cuda)


def rng_restore(state: RngState) -> bool:
    """Put the generators back where the frame left them.

    Returns whether the CUDA half was restored too, so a caller can
    tell a fully reproducible re-entry from one that only pinned the
    host generator.

    A checkpoint can outlive the device layout it was taken under
    (captured on a two-GPU host, restored after one was claimed
    elsewhere). Restoring a mismatched list raises, and a resume that
    still re-enters the right canvas is worth more than bit-identical
    noise, so the device states are skipped instead of insisted on.
    """
    assert isinstance(state, RngState), "restore takes a capture"
    torch.set_rng_state(state.cpu)
    if len(state.cuda) == 0:
        return True
    if not torch.cuda.is_available():
        return False
    if len(state.cuda) != torch.cuda.device_count():
        return False
    torch.cuda.set_rng_state_all(list(state.cuda))
    return True


@dataclass(frozen=True)
class LladaFrame:
    """LLaDA's per-position reveal confidence at one frame.

    Shape ``(gen_length,)`` on CPU. Held so a resumed branch reports
    the confidence each surviving token was actually revealed at,
    rather than the flat 1.0 that made an edited run's heatmap read
    as uniformly certain.
    """

    reveal_conf: torch.Tensor

    def nbytes(self) -> int:
        conf = self.reveal_conf
        return conf.numel() * conf.element_size()


@dataclass(frozen=True)
class DgemmaFrame:
    """DiffusionGemma's stability window and born positions.

    ``stable`` is the per-position count of consecutive steps holding
    the same prediction, which is what its confidence is derived from
    when the Entropy Signal is off. ``seen_revealed`` is the set of
    positions already reported as born on this canvas, so a resume
    does not re-birth the prefix it inherited.
    """

    stable: Tuple[int, ...]
    seen_revealed: FrozenSet[int]

    def nbytes(self) -> int:
        count = len(self.stable) + len(self.seen_revealed)
        return count * _INT_BYTES_APPROX


FramePayload = Union[LladaFrame, DgemmaFrame]


@dataclass(frozen=True)
class FrameCheckpoint:
    """One frame's resumable state.

    ``ids`` is the canvas as token ids on CPU, the same tensor the
    display was built from. ``rng`` is absent when the run exhausted
    its random-state budget, in which case a resume from this frame
    re-seeds as it did before rather than refusing to run.
    """

    ids: torch.Tensor
    canvas_index: int
    rng: Optional[RngState]
    extra: FramePayload

    def nbytes(self) -> int:
        total = self.ids.numel() * self.ids.element_size()
        total += self.extra.nbytes()
        if self.rng is not None:
            total += self.rng.nbytes()
        return total


class CheckpointBudget:
    """Tracks what a run's random states have cost so far.

    Degrades rather than evicts. Dropping old checkpoints would take
    away the early frames an edit is most likely to re-enter, and
    refusing to record would end the run, so past the ceiling the
    canvas and confidence keep being captured and only the random
    state stops. Such a frame still resumes; it just resumes the way
    every frame did before this existed.
    """

    def __init__(
        self, limit_bytes: int = CHECKPOINT_RNG_BYTES_MAX
    ) -> None:
        assert limit_bytes > 0, "a budget has room for something"
        self._limit_bytes = limit_bytes
        self._spent_bytes = 0
        self._frames_without_rng = 0

    def capture_rng(self) -> Optional[RngState]:
        """The next frame's random state, or None once spent.

        The ceiling is tested before the capture rather than after,
        so the total may overshoot by one state. That is bounded by
        a few kilobytes and keeps the accounting to one branch.
        """
        if self._spent_bytes >= self._limit_bytes:
            self._frames_without_rng += 1
            return None
        state = rng_capture()
        self._spent_bytes += state.nbytes()
        assert self._spent_bytes > 0, "a capture costs something"
        return state

    @property
    def spent_bytes(self) -> int:
        return self._spent_bytes

    @property
    def frames_without_rng(self) -> int:
        return self._frames_without_rng
