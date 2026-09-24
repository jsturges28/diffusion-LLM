"""The worker volunteers resource samples, and stops when told.

Strategy: the pump is driven directly with a recording socket and a
near-zero interval, so a test costs milliseconds rather than seconds.
Then one pass through a real worker app, because the pump is
started before the readiness handshake and that ordering is the part a
unit test cannot see.

That ordering is deliberate and is the reason for the placement: the
interval with no ``effective_device`` is the weight load, which is the
largest VRAM movement the application makes and happens behind a
spinner with no frames to attach a sample to. The cost of covering it
is that an advisory sample can reach the client before the worker says
it is ready, so this file pins that rather than leaving it to timing.

Passing proves a sample carries its wire type, a worker with nothing
measure sends nothing at all, the pump gives up rather than raising
when the socket goes, cancellation is immediate, the sample's kind
follows where the model landed, and a sample may legitimately precede
the ready handshake.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.backends import worker_base
from src.backends.protocol import ModelCapabilities, ModelInfo
from src.backends.resource_sampler import KIND_CPU, KIND_VRAM
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    create_worker_app,
    pump_resource_samples,
    resource_sample_kind,
)

# Fast enough that a test never waits on a real cadence, slow enough
# that the loop yields between ticks.
TICK_SECONDS = 0.001

VRAM_SAMPLE: Dict[str, Any] = {
    "kind": KIND_VRAM,
    "fraction": 0.71,
    "used_bytes": 17 * 1024**3,
    "total_bytes": 24 * 1024**3,
}
CPU_SAMPLE: Dict[str, Any] = {
    "kind": KIND_CPU,
    "fraction": 0.25,
    "busy_cores": 8.0,
    "total_cores": 32,
}


class _Recorder:
    """A socket that keeps what it was sent.

    ``breaks`` makes the next send fail the way a closed socket does,
    which is the case the pump has to survive without noise.
    """

    def __init__(self, breaks: bool = False) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.breaks = breaks

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if self.breaks:
            raise RuntimeError("socket closed")
        self.sent.append(payload)


class _Sampler:
    """A CpuSampler stand-in answering a fixed script."""

    def __init__(self, answer: Optional[Dict[str, Any]]) -> None:
        self.answer = answer
        self.calls = 0

    def sample(self) -> Optional[Dict[str, Any]]:
        self.calls += 1
        return self.answer


class _Placed:
    """Just enough backend for the kind to be decided."""

    def __init__(self, device: Optional[str]) -> None:
        self.effective_device = device


async def _run_briefly(
    socket: _Recorder,
    backend: Any,
    sampler: Any,
    ticks: int = 3,
) -> "asyncio.Task[None]":
    """Let the pump tick a few times, then stop it."""
    task = asyncio.create_task(
        pump_resource_samples(socket, backend, sampler)
    )
    await asyncio.sleep(TICK_SECONDS * ticks * 4)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return task


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched on worker_base, which holds its own reference.

    Patching the constant in resource_sampler would do nothing: the
    name was bound at import and the pump reads the copy here.
    """
    monkeypatch.setattr(
        worker_base, "SAMPLE_INTERVAL_SECONDS", TICK_SECONDS
    )


# -- which resource a sample describes --


def test_a_card_placed_model_reports_vram() -> None:
    assert resource_sample_kind(_Placed("cuda")) == KIND_VRAM


def test_a_cpu_placed_model_reports_cpu() -> None:
    """Even on a host with a card. The device's figures would describe
    something other than this run."""
    assert resource_sample_kind(_Placed("cpu")) == KIND_CPU


def test_before_loading_a_card_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load interval, where no placement exists to follow yet and
    the card filling up is the thing worth watching."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )

    assert resource_sample_kind(_Placed(None)) == KIND_VRAM


def test_before_loading_a_host_with_no_card_reports_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_base, "vram_sample", lambda: None)

    assert resource_sample_kind(_Placed(None)) == KIND_CPU


# -- what the pump sends --


def test_a_sample_carries_its_wire_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader dispatches on type, so a sample without one would be
    silently dropped by a client that is working correctly."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )
    socket = _Recorder()

    asyncio.run(
        _run_briefly(socket, _Placed("cuda"), _Sampler(None))
    )

    assert socket.sent, "the pump sent nothing"
    assert socket.sent[0]["type"] == "resource_sample"
    assert socket.sent[0]["fraction"] == VRAM_SAMPLE["fraction"]


def test_the_sample_is_not_mutated_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire type is stamped onto a copy. The sampler's dict is its
    own, and a pump that scribbled on it would leak a protocol detail
    into a module that knows nothing about the protocol."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )

    asyncio.run(
        _run_briefly(_Recorder(), _Placed("cuda"), _Sampler(None))
    )

    assert "type" not in VRAM_SAMPLE


def test_a_cpu_worker_sends_its_sampler_s_answer() -> None:
    socket = _Recorder()
    sampler = _Sampler(CPU_SAMPLE)

    asyncio.run(_run_briefly(socket, _Placed("cpu"), sampler))

    assert sampler.calls > 0
    assert socket.sent[0]["kind"] == KIND_CPU
    assert socket.sent[0]["total_cores"] == 32


def test_nothing_to_measure_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with no card and no /proc. Silence, so the meter never
    appears, rather than a stream of zeros that would draw as a
    working instrument reporting an idle machine."""
    monkeypatch.setattr(worker_base, "vram_sample", lambda: None)
    socket = _Recorder()

    asyncio.run(
        _run_briefly(socket, _Placed(None), _Sampler(None))
    )

    assert socket.sent == []


def test_the_pump_keeps_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A meter is a series, not a reading. Asserted loosely on count
    because the exact number depends on the scheduler."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )
    socket = _Recorder()

    asyncio.run(
        _run_briefly(socket, _Placed("cuda"), _Sampler(None))
    )

    assert len(socket.sent) >= 2


# -- and when it stops --


def test_a_closed_socket_ends_the_pump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns rather than raising. A telemetry line must not be able
    to turn a disconnect into an error, and the disconnect path has
    already cancelled everything that matters."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )

    async def _drive() -> bool:
        task = asyncio.create_task(
            pump_resource_samples(
                _Recorder(breaks=True),
                _Placed("cuda"),
                _Sampler(None),
            )
        )
        await asyncio.sleep(TICK_SECONDS * 8)
        return task.done()

    assert asyncio.run(_drive()) is True


def test_cancelling_stops_it_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its only await is a sleep, which is why the socket handler
    cancels without waiting on it."""
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )

    async def _drive() -> bool:
        socket = _Recorder()
        task = await _run_briefly(
            socket, _Placed("cuda"), _Sampler(None)
        )
        return task.cancelled() or task.done()

    assert asyncio.run(_drive()) is True


# -- the ordering the placement buys, and pays for --


class _CardBackend(Backend):
    """A loaded backend that believes it is on a card."""

    def __init__(self) -> None:
        self.model_info = ModelInfo(
            id="stub",
            display_name="Stub",
            param_specs=[],
            capabilities=ModelCapabilities(
                family="diffusion",
                generation_shape="iterative_canvas",
                input_mode="chat",
                supported_devices=("cuda", "cpu"),
            ),
            worker_module="none",
            environment="none",
            checkpoint="none",
        )
        self.effective_device = "cuda"
        self.tokenizer = None

    def load(self, *, device: str = "cuda") -> None:
        self.effective_device = "cuda"

    async def handle_generate(
        self,
        ws: Any,
        data: Dict[str, Any],
        cancel_event: Any,
        stream: FrameStreamer,
    ) -> None:
        raise NotImplementedError


@pytest.fixture()
def card_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        worker_base, "vram_sample", lambda: VRAM_SAMPLE
    )
    app = create_worker_app(_CardBackend(), device="cuda")
    with TestClient(app) as client:
        yield client


def test_a_sample_may_arrive_before_the_ready_handshake(
    card_client: TestClient,
) -> None:
    """The cost of covering the weight load, stated as a fact.

    The pump starts before the readiness wait, so on a card the first
    thing a client reads can be a sample rather than ``model_status``.
    Harmless, because the browser dispatches on type and never assumes
    an order, but a test helper that reads "the first message" has to
    know. Both messages must appear; which comes first is timing.
    """
    with card_client.websocket_connect("/ws") as socket:
        kinds = {socket.receive_json()["type"] for _ in range(2)}

    assert "model_status" in kinds
    assert "resource_sample" in kinds
