"""What a run cost the device, and when the worker declines to say.

Strategy: two halves. First the CPU half, run for real against the
installed torch on whatever host this is, because the failure it
guards is not hypothetical: without CUDA,
``torch.cuda.reset_peak_memory_stats()`` raises RuntimeError rather
than returning something falsy, so an unguarded reset in
``begin_run`` would break every SmolLM3-on-CPU generation. Second the
CUDA half, with the four ``torch.cuda`` entry points replaced, since
the sandbox has no card.

The numbers in the CUDA half are deliberately shaped like a real
LLaDA run: about 17 GiB of resident weights and a small transient on
top. That is the case the measurement exists for, and it is what makes
the baseline load-bearing: the transient is a fraction of a percent of
the peak, so a peak reported alone says almost nothing about whether a
change to the sampler helped.

Passing proves a CPU run records nothing rather than zeros, a CUDA run
reports the baseline with both peaks, a resume keeps the measurement
its generation started, and any failure to read leaves the block
absent instead of half filled.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pytest

from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    provenance_envelope,
    reset_vram_peak,
    vram_peak_bytes,
)
from tests.backends.test_worker_provenance import _StubBackend

# A resident LLaDA-8B in bf16, and the per-step transient the chunked
# reduction brought down. Real figures rather than round ones, so an
# assertion that accidentally compared the wrong pair would not pass
# by arithmetic coincidence.
WEIGHTS_BYTES = 17 * 1024**3
TRANSIENT_BYTES = 15 * 1024**2
RESERVED_SLACK_BYTES = 256 * 1024**2


class _Backend(Backend):
    """The smallest thing that can begin a run.

    Deliberately not one of the three real backends: this is about the
    base class's contract, which all three inherit, and a stub keeps a
    model out of a test about bookkeeping.
    """

    def __init__(self, device: str) -> None:
        self.model_info = None  # type: ignore[assignment]
        self.effective_device = device

    def load(self, *, device: str = "cuda") -> None:
        raise NotImplementedError

    async def handle_generate(
        self,
        ws: Any,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """What every real handle_generate does, minus the model."""
        self.begin_run()


def _generate(backend: _Backend) -> None:
    asyncio.run(
        backend.handle_generate(
            None, {}, asyncio.Event(), None  # type: ignore[arg-type]
        )
    )


class _FakeCuda:
    """The four calls the worker makes, with settable answers."""

    def __init__(
        self,
        *,
        available: bool = True,
        allocated: int = WEIGHTS_BYTES,
        peak: Optional[int] = None,
        reserved: Optional[int] = None,
    ) -> None:
        self.available = available
        self.allocated = allocated
        self.peak = allocated if peak is None else peak
        self.reserved = (
            self.peak if reserved is None else reserved
        )
        self.resets = 0

    def is_available(self) -> bool:
        return self.available

    def reset_peak_memory_stats(self) -> None:
        # Mirrors the real behaviour that makes the baseline
        # necessary: the peak drops to the *current* allocation, not
        # to zero.
        self.resets += 1
        self.peak = self.allocated

    def memory_allocated(self) -> int:
        return self.allocated

    def max_memory_allocated(self) -> int:
        return self.peak

    def max_memory_reserved(self) -> int:
        return self.reserved


def _install(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeCuda
) -> None:
    """Point torch.cuda's four entry points at *fake*.

    Patching the real module rather than injecting a seam, so the
    function-local ``import torch`` in the worker is exercised exactly
    as it runs in production.
    """
    import torch

    for name in (
        "is_available",
        "reset_peak_memory_stats",
        "memory_allocated",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            torch.cuda, name, getattr(fake, name)
        )


# -- the CPU half, real --


def test_a_cpu_generation_does_not_raise() -> None:
    """The regression this guard exists for.

    ``reset_peak_memory_stats`` raises without CUDA, so a reset that
    did not check would take down the one model a GPU-less host can
    run. Deliberately unpatched: the point is the behaviour of the
    torch that is actually installed.
    """
    backend = _Backend("cpu")

    _generate(backend)

    assert backend.run_counter == 1


def test_a_cpu_generation_records_no_baseline() -> None:
    backend = _Backend("cpu")

    _generate(backend)

    assert backend.vram_start_bytes is None


def test_the_envelope_omits_the_block_on_cpu() -> None:
    """Absent rather than zeroed.

    On a CUDA-less host ``max_memory_allocated`` returns 0 without
    complaint, so the tempting shape here is a block of zeros. A run
    that peaked at zero bytes is not a thing, and a reader cannot tell
    that claim from a real measurement.
    """
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert "resources" not in envelope


def test_reading_without_cuda_answers_none() -> None:
    """Both helpers, against the installed torch, on this host."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("this asserts the no-CUDA path")

    assert reset_vram_peak() is None
    assert vram_peak_bytes() is None


# -- the CUDA half, patched --


def test_a_cuda_generation_records_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuda(allocated=WEIGHTS_BYTES)
    _install(monkeypatch, fake)
    backend = _Backend("cuda")

    _generate(backend)

    assert fake.resets == 1
    assert backend.vram_start_bytes == WEIGHTS_BYTES


def test_the_envelope_reports_the_baseline_and_both_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole block, with the three figures distinguishable.

    Each is a different number, so a wiring mistake that reported one
    of them twice cannot pass.
    """
    peak = WEIGHTS_BYTES + TRANSIENT_BYTES
    reserved = peak + RESERVED_SLACK_BYTES
    _install(
        monkeypatch,
        _FakeCuda(
            allocated=WEIGHTS_BYTES, peak=peak, reserved=reserved
        ),
    )
    backend = _StubBackend("cuda", vram_start_bytes=WEIGHTS_BYTES)

    envelope = provenance_envelope(
        backend  # type: ignore[arg-type]
    )

    assert envelope["resources"] == {
        "vram_allocated_start_bytes": WEIGHTS_BYTES,
        "vram_allocated_peak_bytes": peak,
        "vram_reserved_peak_bytes": reserved,
    }


def test_the_transient_is_the_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the baseline is recorded at all.

    Stated as the subtraction a reader performs, because the peak
    alone cannot answer the question the measurement was added for:
    17.01 GiB against 17.09 GiB is not a difference anyone can see,
    and 15 MiB against 96 MiB is.
    """
    peak = WEIGHTS_BYTES + TRANSIENT_BYTES
    _install(
        monkeypatch,
        _FakeCuda(allocated=WEIGHTS_BYTES, peak=peak),
    )
    block = provenance_envelope(
        _StubBackend(  # type: ignore[arg-type]
            "cuda", vram_start_bytes=WEIGHTS_BYTES
        )
    )["resources"]

    above = (
        block["vram_allocated_peak_bytes"]
        - block["vram_allocated_start_bytes"]
    )

    assert above == TRANSIENT_BYTES


def test_a_resume_keeps_its_generation_s_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peak spans a run and every resume of it.

    ``begin_run`` is called only by generation, so a resume inherits
    the baseline its run started with. That is the behaviour wanted,
    and it comes free from the existing token boundary rather than
    from a rule of its own, which is exactly why it needs a test: no
    line of code says it.
    """
    fake = _FakeCuda(allocated=WEIGHTS_BYTES)
    _install(monkeypatch, fake)
    backend = _Backend("cuda")
    _generate(backend)

    # A resume does whatever it does without beginning a run.
    backend.last_run_state = {"resumed": True}

    assert fake.resets == 1
    assert backend.vram_start_bytes == WEIGHTS_BYTES


def test_a_second_generation_starts_a_new_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the boundary above."""
    fake = _FakeCuda(allocated=WEIGHTS_BYTES)
    _install(monkeypatch, fake)
    backend = _Backend("cuda")

    _generate(backend)
    fake.allocated = WEIGHTS_BYTES + TRANSIENT_BYTES
    _generate(backend)

    assert fake.resets == 2
    assert backend.vram_start_bytes == (
        WEIGHTS_BYTES + TRANSIENT_BYTES
    )


def test_a_cpu_placed_model_on_a_cuda_host_measures_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second guard, which the availability check does not cover.

    CUDA is present and would answer, but this model is on the CPU, so
    the device's numbers belong to something else. Reporting them
    would attribute another process's memory to this run.
    """
    fake = _FakeCuda(allocated=WEIGHTS_BYTES)
    _install(monkeypatch, fake)
    backend = _Backend("cpu")

    _generate(backend)

    assert fake.resets == 0
    assert backend.vram_start_bytes is None


# -- failures leave no half-written block --


def test_a_reset_that_fails_measures_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCuda(allocated=WEIGHTS_BYTES)
    _install(monkeypatch, fake)

    def _explode() -> None:
        raise RuntimeError("no device")

    import torch

    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", _explode
    )
    backend = _Backend("cuda")

    _generate(backend)

    assert backend.vram_start_bytes is None


def test_a_peak_read_that_fails_omits_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three or none.

    A block carrying a baseline with no peak invites the subtraction
    it cannot support, so a failed read drops the whole thing rather
    than reporting the half that worked.
    """
    _install(monkeypatch, _FakeCuda(allocated=WEIGHTS_BYTES))

    def _explode() -> int:
        raise RuntimeError("device fell over")

    import torch

    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", _explode
    )

    envelope = provenance_envelope(
        _StubBackend(  # type: ignore[arg-type]
            "cuda", vram_start_bytes=WEIGHTS_BYTES
        )
    )

    assert "resources" not in envelope
