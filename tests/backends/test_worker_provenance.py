"""A worker says where it actually is, and every run carries it.

Strategy: two halves, both without a model. First, `FrameStreamer`
against a stub socket, checking that a terminal frame acquires the
envelope and an ordinary frame does not, through both of the paths a
`done` can leave by. Second, `provenance_envelope` against a stub
backend, checking it reports the loaded placement rather than the
requested one.

What passing proves is the missing link in `DATA-04`. The supervisor
records the device it *asked* for. Two of the three backends fall back
to CPU when CUDA was requested and is unavailable, and nothing
downstream could tell: a run that took four minutes on a CPU was saved
as a GPU run, which is a misleading record precisely in the fields
that exist to make timings comparable.

The other half, that the save prefers the run's envelope over the
supervisor's current state, is in `tests/web/test_run_provenance.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.backends.protocol import ModelCapabilities, ModelInfo
from src.backends.worker_base import (
    FrameStreamer,
    library_versions,
    provenance_envelope,
)

ENVELOPE: Dict[str, Any] = {
    "model_id": "stub",
    "device": "cpu",
    "versions": {"torch": "0.0.0"},
    "tokenizer": {},
}


class _StubSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


async def _frames(
    *frames: Dict[str, Any],
) -> AsyncGenerator[Dict[str, Any], None]:
    for frame in frames:
        yield frame


def _streamer(socket: _StubSocket) -> FrameStreamer:
    return FrameStreamer(
        socket,  # type: ignore[arg-type]
        provenance=lambda: dict(ENVELOPE),
    )


# -- what a terminal frame carries --


def test_a_done_frame_carries_the_envelope() -> None:
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.run(
            _frames(
                {"type": "frame", "text": "a"},
                {"type": "done", "final_text": "a"},
            ),
            0.0,
        )
    )

    assert socket.sent[-1]["provenance"] == ENVELOPE


def test_an_ordinary_frame_does_not() -> None:
    """Once per run, not once per step. A diffusion run emits
    hundreds of frames and the envelope re-describes the same
    worker every time."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.run(
            _frames(
                {"type": "frame", "text": "a"},
                {"type": "done", "final_text": "a"},
            ),
            0.0,
        )
    )

    assert "provenance" not in socket.sent[0]


def test_a_worker_sent_done_carries_it_too() -> None:
    """The path a guided edit takes. Its terminal frame is built by
    the worker rather than the sampler, and it used to be assembled
    by hand at each call site with whatever fields that site
    remembered."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.send_done(
            {"type": "done", "final_text": "stopped here"}, 0.0
        )
    )

    assert socket.sent[-1]["provenance"] == ENVELOPE
    assert socket.sent[-1]["final_text"] == "stopped here"


def test_a_worker_sent_done_is_timed_like_any_other() -> None:
    """The LLaDA site omitted elapsed and the DiffusionGemma one
    included it. Routing both through here settles it."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.send_done({"type": "done"}, 0.0)
    )

    assert isinstance(socket.sent[-1]["elapsed"], float)


def test_send_done_refuses_a_non_terminal_frame() -> None:
    """It stamps provenance, so sending an ordinary frame through
    it would attach the envelope to every step."""
    socket = _StubSocket()
    stream = _streamer(socket)

    with pytest.raises(AssertionError):
        asyncio.run(
            stream.send_done({"type": "frame"}, 0.0)
        )


def test_a_streamer_without_provenance_stamps_nothing() -> None:
    """The default. Tests and any future caller that has no backend
    to attest for must still be able to stream."""
    socket = _StubSocket()
    stream = FrameStreamer(socket)  # type: ignore[arg-type]

    asyncio.run(
        stream.run(_frames({"type": "done"}), 0.0)
    )

    assert "provenance" not in socket.sent[-1]


# -- what the envelope says --


class _StubBackend:
    """A loaded backend, minus the model."""

    def __init__(self, effective_device: str) -> None:
        self.model_info = ModelInfo(
            id="stub",
            display_name="Stub",
            param_specs=[],
            capabilities=ModelCapabilities(),
            worker_module="none",
            venv_python="none",
            checkpoint="org/stub-checkpoint",
        )
        self.effective_device = effective_device
        self.tokenizer = None
        self.model = None


def test_the_envelope_reports_where_the_model_landed() -> None:
    """The finding in one assertion. The supervisor would say cuda
    here, because cuda is what it asked for."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert envelope["device"] == "cpu"


def test_a_backend_that_has_not_loaded_says_unknown() -> None:
    """Not a device. A blank must not read as a placement, or the
    default would quietly become whatever "" happens to compare
    equal to downstream."""
    backend = _StubBackend("cpu")
    backend.effective_device = None

    envelope = provenance_envelope(
        backend  # type: ignore[arg-type]
    )

    assert envelope["device"] == "unknown"


def test_the_envelope_names_the_model_and_checkpoint() -> None:
    envelope = provenance_envelope(
        _StubBackend("cuda")  # type: ignore[arg-type]
    )

    assert envelope["model_id"] == "stub"
    assert envelope["checkpoint"] == "org/stub-checkpoint"


def test_the_envelope_reports_this_worker_s_libraries() -> None:
    """Read in the worker, not the supervisor: the three venvs hold
    deliberately incompatible versions, so the supervisor's own
    imports describe the wrong environment."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert envelope["versions"] == library_versions()
    assert "torch" in envelope["versions"]


def test_the_envelope_omits_an_unreadable_context_window() -> None:
    """Omitted rather than null, matching /health, so a consumer's
    "is there a ceiling" test stays a plain key check."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert "context_length" not in envelope
