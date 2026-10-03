"""Every activation is numbered, so no window can act for another.

Strategy: drive a real `ModelManager` with the fake process and
scripted probe from `tests/web/test_worker_lifecycle.py`, and drive
the endpoints with `TestClient` against that manager. Two browser
windows are simulated as two callers of the same supervisor, which is
all they are: the whole failure class comes from one process serving
both.

What passing proves is the finding's claim in reverse. Activation was
singleton global state with no owner. A second window could supersede
the first window's load and the first would navigate when the
*replacement* became ready, because "is it ready" was a question
about the supervisor rather than about the activation that window
started. Cancel was worse: it stopped whatever worker was loading,
so either window could kill the other's load with a button that gave
no sign of having reached across.

The reconnect half, that a page whose worker was replaced is told so,
is the `resident` frame checked near the end. After it comes the
relay itself (`A2-QUALITY-02`): a message each way through the real
route, and a teardown that finishes the half it cancels before the
worker's connection is let go.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.web import server
from src.web.server import ModelManager
from tests.web.test_worker_lifecycle import (
    LOADING,
    READY,
    FakeProcess,
)


class Harness:
    """A manager on fakes, plus a client speaking to its endpoints."""

    def __init__(
        self, health: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.processes: List[FakeProcess] = []
        self.health = health or [READY]

        def spawn(
            command: Any, *, cwd: Any, env: Any
        ) -> FakeProcess:
            made = FakeProcess(pid=5000 + len(self.processes))
            self.processes.append(made)
            return made

        async def probe(url: str) -> Optional[Dict[str, Any]]:
            return self.health[-1]

        self.manager = ModelManager(
            spawn=spawn,  # type: ignore[arg-type]
            probe=probe,
            start_timeout_s=0.05,
            stop_timeout_s=0.01,
            kill_timeout_s=0.01,
            health_poll_s=0.001,
            progress_poll_s=0.001,
            vram_settle_timeout_s=0.01,
        )
        self.previous = server.manager
        server.manager = self.manager
        self.client = TestClient(server.app)

    def close(self) -> None:
        server.manager = self.previous

    async def settle(self) -> None:
        task = self.manager._monitor_task
        if task is None:
            return
        # Shielded so a monitor that finalizes itself is observed
        # rather than cancelled out from under the assertion.
        with contextlib.suppress(
            asyncio.TimeoutError, asyncio.CancelledError
        ):
            await asyncio.wait_for(
                asyncio.shield(task), timeout=0.4
            )


@pytest.fixture
def harness() -> Any:
    made = Harness()
    try:
        yield made
    finally:
        made.close()


@pytest.fixture(autouse=True)
def _plausible_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "_gpu_name", lambda: "Fake GPU"
    )
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: 99.0
    )
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)


def _activate(harness: Harness, model_id: str) -> Dict[str, Any]:
    response = harness.client.post(
        f"/api/models/{model_id}/activate"
    )
    assert response.status_code == 200, response.text
    body: Dict[str, Any] = response.json()
    return body


def _status(harness: Harness) -> Dict[str, Any]:
    body: Dict[str, Any] = harness.client.get(
        "/api/models/activation"
    ).json()
    return body


# -- an activation has a number --


def test_an_activation_is_given_an_operation(
    harness: Harness,
) -> None:
    body = _activate(harness, "llada")

    assert isinstance(body["operation"], int)
    assert body["operation"] > 0


def test_the_next_activation_gets_a_later_number(
    harness: Harness,
) -> None:
    """Monotonic, because the comparison a client makes is identity
    rather than ordering, and a reused number would make one
    window's load indistinguishable from another's."""
    first = _activate(harness, "llada")["operation"]
    second = _activate(harness, "smollm3")["operation"]

    assert second > first


def test_the_status_reports_the_current_operation(
    harness: Harness,
) -> None:
    """The poll is where a client compares, so the number has to be
    on the thing it polls."""
    operation = _activate(harness, "llada")["operation"]

    assert _status(harness)["operation"] == operation


def test_a_second_activation_moves_the_status_on(
    harness: Harness,
) -> None:
    """What the first window's poll sees after being superseded: not
    its own operation, which is how it knows to stop."""
    first = _activate(harness, "llada")["operation"]
    _activate(harness, "smollm3")

    assert _status(harness)["operation"] != first


def test_reselecting_the_resident_model_reuses_its_number(
    harness: Harness,
) -> None:
    """A no-op activation starts nothing, so there is no new
    activation to number; the caller is handed the one that produced
    the worker already running."""

    async def scenario() -> int:
        first = await harness.manager.activate(
            "llada", device="cuda"
        )
        await harness.settle()
        again = await harness.manager.activate(
            "llada", device="cuda"
        )
        return first if first == again else -1

    assert asyncio.run(scenario()) > 0


def test_the_number_survives_the_failure_it_describes(
    harness: Harness,
) -> None:
    """A client polls for the outcome of its own load, so a failure
    that dropped its number would be reported to nobody."""
    harness.health = [LOADING, {"status": "error",
                                "id": "llada",
                                "message": "boom"}]

    async def scenario() -> None:
        await harness.manager.activate("llada", device="cuda")
        await harness.settle()

    asyncio.run(scenario())
    status = _status(harness)

    assert status["state"] == "error"
    assert status["operation"] == harness.manager.activation_id


# -- cancel reaches only your own load --


def test_a_cancel_naming_the_load_stops_it(
    harness: Harness,
) -> None:
    harness.health = [LOADING]
    operation = _activate(harness, "llada")["operation"]

    response = harness.client.post(
        "/api/models/activate/cancel",
        json={"operation": operation},
    )

    assert response.status_code == 200
    assert not harness.processes[0].alive


def test_a_cancel_for_another_window_s_load_is_refused(
    harness: Harness,
) -> None:
    """The half of the finding a user would notice: window B's
    Cancel used to stop window A's load."""
    harness.health = [LOADING]
    stale = _activate(harness, "llada")["operation"]
    _activate(harness, "smollm3")

    response = harness.client.post(
        "/api/models/activate/cancel",
        json={"operation": stale},
    )

    assert response.status_code == 409
    assert harness.processes[-1].alive


def test_a_refused_cancel_says_who_owns_the_load(
    harness: Harness,
) -> None:
    """A button that silently does nothing reads as broken, so the
    refusal has to be worth showing."""
    harness.health = [LOADING]
    stale = _activate(harness, "llada")["operation"]
    _activate(harness, "smollm3")

    body = harness.client.post(
        "/api/models/activate/cancel",
        json={"operation": stale},
    ).json()

    assert "SmolLM3-3B" in body["message"]
    assert "started somewhere else" in body["message"]


def test_a_cancel_naming_nothing_is_refused(
    harness: Harness,
) -> None:
    """Not naming an activation is not the same as owning one. This
    is exactly what every caller sent before this change."""
    harness.health = [LOADING]
    _activate(harness, "llada")

    response = harness.client.post(
        "/api/models/activate/cancel"
    )

    assert response.status_code == 409
    assert harness.processes[0].alive


def test_cancelling_when_nothing_loads_is_harmless(
    harness: Harness,
) -> None:
    """There is nothing to protect, and a stale window tidying up
    after itself should not be told off for it."""
    response = harness.client.post(
        "/api/models/activate/cancel"
    )

    assert response.status_code == 200


# -- two windows, interleaved --


def test_the_first_window_cannot_act_on_the_second_s_load(
    harness: Harness,
) -> None:
    """Both clauses at once, which is how they actually occur: A
    starts X, B replaces it with Y, and A can neither recognise Y's
    readiness as its own nor cancel it."""
    harness.health = [LOADING]
    window_a = _activate(harness, "llada")["operation"]
    window_b = _activate(harness, "smollm3")["operation"]

    status = _status(harness)
    refused = harness.client.post(
        "/api/models/activate/cancel",
        json={"operation": window_a},
    )

    assert status["operation"] == window_b
    assert status["operation"] != window_a
    assert refused.status_code == 409
    assert harness.processes[-1].alive


# -- who the socket is actually talking to --


class _FakeWorkerSocket:
    """The worker end of the proxy, without a worker.

    The manager's process is a fake, so nothing is listening on the
    port the proxy would dial. Standing in here is what lets the
    handshake be tested at all; the alternative is a real subprocess,
    which is the thing `LIFE-02`'s seam exists to avoid.
    """

    def __init__(self, messages: List[str]) -> None:
        self._messages = list(messages)
        self.sent: List[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def __aiter__(self) -> Any:
        for message in self._messages:
            yield message


class _FakeConnect:
    def __init__(self, socket: _FakeWorkerSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeWorkerSocket:
        return self._socket

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


def _stub_worker_socket(
    monkeypatch: pytest.MonkeyPatch, messages: List[str]
) -> _FakeWorkerSocket:
    socket = _FakeWorkerSocket(messages)
    monkeypatch.setattr(
        server.websockets,
        "connect",
        lambda url, **kwargs: _FakeConnect(socket),
    )
    return socket


def _ready_worker(harness: Harness) -> None:
    async def scenario() -> None:
        await harness.manager.activate("llada", device="cuda")
        await harness.settle()

    asyncio.run(scenario())


def test_the_socket_says_which_model_answered(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready_worker(harness)
    _stub_worker_socket(monkeypatch, [])

    with harness.client.websocket_connect("/ws") as socket:
        first = socket.receive_json()

    assert first["type"] == "resident"
    assert first["model"] == "llada"
    assert first["device"] == "cuda"
    assert first["operation"] == harness.manager.activation_id


def test_the_handshake_precedes_worker_traffic(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page caches its model, device and whole parameter form at
    boot and only refreshes them by reloading, so it has to learn
    who it is talking to before it acts on anything that worker
    says."""
    _ready_worker(harness)
    _stub_worker_socket(
        monkeypatch,
        ['{"type": "model_status", "status": "ready"}'],
    )

    with harness.client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        second = socket.receive_json()

    assert first["type"] == "resident"
    assert second["type"] == "model_status"


def test_a_socket_with_no_serving_model_is_turned_away(
    harness: Harness,
) -> None:
    """The negative space: the handshake must not paper over the
    gate `LIFE-02` made honest."""
    with harness.client.websocket_connect("/ws") as socket:
        first = socket.receive_json()

    assert first["type"] == "error"


# -- the relay, both ways --


class _AnsweringWorkerSocket:
    """A worker end that answers each message with one of its own.

    The fake above only plays a script at the page, so nothing sent
    from the page was ever seen to arrive. This one records what it
    is sent and replies to it. With ``hang_up_after`` it ends its
    stream after that many replies, the way a worker that exits does;
    without, it runs until the proxy cancels it.

    ``settled`` turns true once its stream has finished, either way,
    and ``settled_when_released`` records whether that had happened
    by the time the proxy let go of the connection.
    """

    def __init__(self, hang_up_after: Optional[int] = None) -> None:
        self.received: List[str] = []
        self.hang_up_after = hang_up_after
        self.settled = False
        self.settled_when_released: Optional[bool] = None
        self.released = threading.Event()
        # Made on first use, inside the app's own event loop.
        self._replies: Optional[asyncio.Queue] = None

    def _queue(self) -> asyncio.Queue:
        if self._replies is None:
            self._replies = asyncio.Queue()
        return self._replies

    async def send(self, message: str) -> None:
        self.received.append(message)
        asked = json.loads(message).get("type")
        reply = json.dumps({"type": "answer", "to": asked})
        await self._queue().put(reply)
        if self.hang_up_after == len(self.received):
            await self._queue().put(None)

    async def __aiter__(self) -> Any:
        try:
            while True:
                reply = await self._queue().get()
                if reply is None:
                    return
                yield reply
        finally:
            self.settled = True


class _AnsweringConnect:
    def __init__(self, socket: _AnsweringWorkerSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _AnsweringWorkerSocket:
        return self._socket

    async def __aexit__(self, *_exc: Any) -> bool:
        self._socket.settled_when_released = self._socket.settled
        self._socket.released.set()
        return False


def _answering_worker(
    monkeypatch: pytest.MonkeyPatch,
    hang_up_after: Optional[int] = None,
) -> _AnsweringWorkerSocket:
    socket = _AnsweringWorkerSocket(hang_up_after)
    monkeypatch.setattr(
        server.websockets,
        "connect",
        lambda url, **kwargs: _AnsweringConnect(socket),
    )
    return socket


# Far beyond what a relay of one in-memory message takes.
RECEIVE_TIMEOUT_SECONDS = 5.0


def _receive(socket: Any) -> Dict[str, Any]:
    """The page's next message, failing rather than hanging.

    A test client's receive has no timeout, so a relay that dropped a
    direction would hang the suite. The reader is a daemon, and a
    failed test cancels the client's pending receive on its way out.
    """
    box: List[Dict[str, Any]] = []
    reader = threading.Thread(
        target=lambda: box.append(socket.receive_json()), daemon=True
    )
    reader.start()
    reader.join(timeout=RECEIVE_TIMEOUT_SECONDS)
    assert box, "nothing reached the page"
    return box[0]


def test_a_page_message_reaches_the_worker_and_is_answered(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready_worker(harness)
    worker = _answering_worker(monkeypatch)

    with harness.client.websocket_connect("/ws") as socket:
        resident = _receive(socket)
        socket.send_text('{"type": "generate"}')
        answer = _receive(socket)

    assert resident["type"] == "resident"
    assert worker.received == ['{"type": "generate"}']
    assert answer == {"type": "answer", "to": "generate"}


def test_a_page_leaving_settles_the_worker_stream_first(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page's half finishes, so the worker's is cancelled, and
    it has to have stopped by the time the connection it reads from
    is let go. Cancelling without waiting returned while it was still
    reading."""
    _ready_worker(harness)
    worker = _answering_worker(monkeypatch)

    with harness.client.websocket_connect("/ws") as socket:
        _receive(socket)
        socket.send_text('{"type": "generate"}')
        _receive(socket)

    released = worker.released.wait(RECEIVE_TIMEOUT_SECONDS)
    assert released, "the proxy never let go"
    assert worker.settled_when_released is True


def test_a_worker_hanging_up_ends_the_proxy(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction. Waited on rather than read off the page's
    socket: a test client never sees a close when a handler returns
    without sending one, where uvicorn would drop the connection."""
    _ready_worker(harness)
    worker = _answering_worker(monkeypatch, hang_up_after=1)

    with harness.client.websocket_connect("/ws") as socket:
        _receive(socket)
        socket.send_text('{"type": "generate"}')
        answer = _receive(socket)
        released = worker.released.wait(RECEIVE_TIMEOUT_SECONDS)

    assert answer["type"] == "answer"
    assert released, "the proxy outlived the worker it relayed"
    assert worker.settled_when_released is True
