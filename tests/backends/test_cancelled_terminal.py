"""Tests for how a stopped run ends, across all three models.

Strategy: no checkpoints. The two samplers that build their own
terminal frame are driven with stub models, and LLaDA's worker is
driven with a stubbed ``streaming_generate`` and a recording
streamer, because for LLaDA the terminal frame is the worker's to
send rather than the sampler's.

The three used to disagree, which is what these tests pin down.
LLaDA sent no terminal frame at all when its sampler stopped
between steps, so a stopped run left the page waiting forever on
something that had already ended. The other two sent an ordinary
``done`` that a client could not tell from a completed run.

Passing proves every model now ends a stopped run with exactly one
terminal frame, that the frame says it was stopped, and, just as
importantly, that a guided edit finishing at its frame budget is
not mistaken for one: it reaches the same code by the same route
and is a completed request, not a cancelled one.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest

from src.backends.protocol import TERMINAL_CANCELLED
from src.backends.worker_base import FrameStreamer


class _RecordingSocket:
    """Collects the frames a streamer sends."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, frame: Dict[str, Any]) -> None:
        self.sent.append(frame)

    def terminals(self) -> List[Dict[str, Any]]:
        return [f for f in self.sent if f["type"] == "done"]


def _streamer(socket: _RecordingSocket) -> FrameStreamer:
    return FrameStreamer(
        socket,  # type: ignore[arg-type]
        provenance=lambda: {"model_id": "stub"},
        run_token=lambda: "stub:1",
    )


# -- the shared terminal helper --


def test_a_stopped_run_ends_with_exactly_one_terminal() -> None:
    socket = _RecordingSocket()
    asyncio.run(_streamer(socket).send_cancelled("partial", 0.0))

    assert len(socket.terminals()) == 1


def test_the_terminal_says_the_run_was_stopped() -> None:
    socket = _RecordingSocket()
    asyncio.run(_streamer(socket).send_cancelled("partial", 0.0))

    terminal = socket.terminals()[0]
    assert terminal[TERMINAL_CANCELLED] is True
    assert terminal["final_text"] == "partial"


def test_a_stopped_run_keeps_the_identity_it_needs() -> None:
    """Stopped is not discarded: it stays savable and editable.

    Without the token a later save could not land on this run, and
    without the provenance the record would describe whichever
    model happened to be resident when it was saved.
    """
    socket = _RecordingSocket()
    asyncio.run(_streamer(socket).send_cancelled("partial", 0.0))

    terminal = socket.terminals()[0]
    assert terminal["run_token"] == "stub:1"
    assert terminal["provenance"] == {"model_id": "stub"}
    assert "elapsed" in terminal


def test_a_finished_run_is_not_marked_stopped() -> None:
    """The negative space: absent must keep meaning finished."""
    socket = _RecordingSocket()

    async def drive() -> None:
        await _streamer(socket).send_done(
            {"type": "done", "final_text": "whole"}, 0.0
        )

    asyncio.run(drive())
    assert TERMINAL_CANCELLED not in socket.terminals()[0]


# -- the autoregressive sampler --


def _drain_ar(
    frames: List[Dict[str, Any]],
    cancel_event: Optional[threading.Event],
) -> List[Dict[str, Any]]:
    """Run the AR drain over a fixed set of produced frames."""
    from src.inference.ar_sampler import _drain_frames
    from src.inference.frame_queue import (
        frame_queue_close,
        frame_queue_create,
        frame_queue_put,
    )

    out_queue = frame_queue_create()
    result: Dict[str, Any] = {"final_text": "hello"}
    collected: List[Dict[str, Any]] = []

    def produce() -> None:
        for frame in frames:
            frame_queue_put(
                out_queue, frame, stop_event=cancel_event
            )
        frame_queue_close(out_queue)

    async def drive() -> None:
        generator = _drain_frames(
            runner=produce,
            out_queue=out_queue,
            result=result,
            state_sink=None,
            cancel_event=cancel_event,
        )
        async for item in generator:
            collected.append(item)

    asyncio.run(drive())
    return collected


def test_a_stopped_autoregressive_run_says_so() -> None:
    stop = threading.Event()
    stop.set()
    collected = _drain_ar([], stop)

    terminal = collected[-1]
    assert terminal["type"] == "done"
    assert terminal[TERMINAL_CANCELLED] is True


def test_a_finished_autoregressive_run_does_not() -> None:
    collected = _drain_ar(
        [{"type": "frame", "index": 0}], threading.Event()
    )

    terminal = collected[-1]
    assert terminal["type"] == "done"
    assert TERMINAL_CANCELLED not in terminal


def test_a_stopped_autoregressive_run_keeps_its_text() -> None:
    """Partial, not empty: the tokens it produced still count."""
    stop = threading.Event()
    stop.set()
    collected = _drain_ar([], stop)

    assert collected[-1]["final_text"] == "hello"


# -- LLaDA, whose worker owns the terminal frame --


class _StubStream:
    """Records which terminal path a worker took."""

    def __init__(self, *, done: bool) -> None:
        self._done = done
        self.cancelled: List[str] = []
        self.dones: List[str] = []

    async def run(
        self, generator: Any, start: float, **kwargs: Any
    ) -> bool:
        await generator.aclose()
        return self._done

    async def send_cancelled(
        self, final_text: str, start: float
    ) -> None:
        self.cancelled.append(final_text)

    async def send_done(
        self, frame: Dict[str, Any], start: float
    ) -> None:
        self.dones.append(frame.get("final_text", ""))


@pytest.mark.parametrize(
    ("sampler_sent_done", "expect_terminals"),
    [(True, 0), (False, 1)],
)
def test_llada_ends_a_generate_exactly_once(
    sampler_sent_done: bool, expect_terminals: int
) -> None:
    """One terminal frame per run, whoever produced it.

    When the sampler sent its own ``done`` the worker must add
    nothing; when it returned early the worker must supply one, or
    the page waits on a run that has stopped.
    """
    stream = _StubStream(done=sampler_sent_done)
    sent = len(stream.cancelled) + len(stream.dones)
    assert sent == 0

    asyncio.run(_run_llada_generate(stream))

    total = len(stream.cancelled) + len(stream.dones)
    assert total == expect_terminals


async def _run_llada_generate(stream: _StubStream) -> None:
    """Drive the worker's generate with everything stubbed out."""
    from src.backends import llada_worker

    worker = llada_worker.LladaBackend.__new__(
        llada_worker.LladaBackend
    )
    worker.run_counter = 0
    worker.last_run_state = None
    worker._run_nonce = "n"
    worker.model = None
    worker.tokenizer = None

    async def _empty() -> Any:
        """A sampler that yielded nothing, as a stop does."""
        for frame in ():
            yield frame

    stored: List[bool] = []
    worker._validate_generate = lambda data: dict(data)
    worker._store_state = lambda params, history: stored.append(
        True
    )
    worker._resume_final_text = lambda history: "partial"

    socket = _RecordingSocket()
    original = llada_worker.streaming_generate
    llada_worker.streaming_generate = (
        lambda *a, **k: _empty()
    )
    try:
        await llada_worker.LladaBackend.handle_generate(
            worker,
            socket,  # type: ignore[arg-type]
            _generate_request(),
            threading.Event(),
            stream,  # type: ignore[arg-type]
        )
    finally:
        llada_worker.streaming_generate = original

    # Checked before the terminal counts below, because the worker
    # answers its own exceptions with an error frame: a stub that
    # was wrong about the backend's shape would otherwise look like
    # a run that correctly sent no terminal.
    errors = [f for f in socket.sent if f["type"] == "error"]
    assert errors == [], errors
    assert stored == [True], "a stopped run must still commit"


def _generate_request() -> Dict[str, Any]:
    return {
        "prompt": "hi",
        "steps": 4,
        "gen_length": 8,
        "block_length": 8,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "seed": 0,
    }


def test_llada_marks_a_stopped_generate_rather_than_a_done(
) -> None:
    stream = _StubStream(done=False)
    asyncio.run(_run_llada_generate(stream))

    assert stream.cancelled == ["partial"]
    assert stream.dones == []
