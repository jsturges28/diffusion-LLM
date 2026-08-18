"""Tests for the worker's WebSocket message loop.

Strategy: build a real worker app around a stub backend and talk to it
over FastAPI's test client, which drives the actual dispatch in
``create_worker_app``. Everything the loop does, the ready handshake,
the generation lock, the busy refusals, cancel, the unknown-type
reply, happens here and nowhere else, and until now none of it was
tested at all: the existing backend tests call ``handle_*`` directly
and never reach the loop that decides which one runs.

That gap is why this file exists as part of `LIFE-01` and
`PROTOCOL-01` rather than after them. Both findings' Verification
clauses are about interleaving requests across two sockets, and both
turn on decisions the loop makes: whether a second socket's request is
refused as busy, and which run a stateful request is allowed to reach.
Neither is observable from a handler in isolation.

What passing proves, then: one worker serves two windows without
letting either act for the other. A second window's generation makes
the first window's follow-ups refused rather than answered from the
run that replaced theirs, including when both runs have identical
shapes, which is the case the report singles out because a bounds
check cannot catch it and the wrong answer looks right.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.backends.protocol import (
    ERROR_BUSY,
    ERROR_NO_TOKENIZER,
    ERROR_SCOPE_REQUEST,
    ERROR_SCOPE_RUN,
    ERROR_STALE_RUN,
    ERROR_UNKNOWN_MESSAGE,
    ModelCapabilities,
    ModelInfo,
)
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
    create_worker_app,
)

MODEL_ID = "stub"


def _model_info() -> ModelInfo:
    return ModelInfo(
        id=MODEL_ID,
        display_name="Stub",
        param_specs=[],
        capabilities=ModelCapabilities(),
        worker_module="none",
        venv_python="none",
        checkpoint="none",
    )


# No test may park a generation for longer than this. The lock tests
# below deliberately hold one open, and a bug that stopped the release
# arriving would otherwise hang the whole suite rather than fail. An
# earlier round of lifecycle tests in this repo did exactly that.
HOLD_TIMEOUT_SECONDS = 5.0


class _StubBackend(Backend):
    """A backend that runs no model but keeps the same bookkeeping.

    ``parks`` makes a generation wait inside the generation lock, so a
    second socket meets a genuinely busy worker rather than a
    simulated one.

    Releasing it takes more care than it looks. The obvious lever,
    sending ``cancel``, does not work: the message loop awaits the
    handler inline, so while a generation runs, the loop that would
    read a cancel from that same socket is not reading anything, and
    the next socket's cancel sets a different connection's event. And
    an ``asyncio.Event`` set straight from the test thread does not
    reliably wake a waiter in the loop's thread, which hangs rather
    than fails. So the gate is created in the loop and released
    through ``call_soon_threadsafe``, with a ``threading.Event`` in
    the other direction to tell the test when the park is real.
    """

    def __init__(self) -> None:
        self.model_info = _model_info()
        self.effective_device = "cpu"
        self.tokenizer = None
        self.parks = False
        # Set from the loop, waited on by the test: safe that way
        # round, and it removes the race where the second window asks
        # before the first has taken the lock.
        self.parked = threading.Event()
        self.probes: List[Dict[str, Any]] = []
        self.resumes: List[Dict[str, Any]] = []
        self._gate: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def load(self, *, device: str = "cuda") -> None:
        return None

    def release(self) -> None:
        """Let the parked generation run to its terminal frame."""
        assert self._loop is not None, "nothing is parked"
        assert self._gate is not None, "nothing is parked"
        gate = self._gate
        self._loop.call_soon_threadsafe(gate.set)

    async def handle_generate(
        self,
        ws: Any,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        self.begin_run()
        if self.parks:
            self._loop = asyncio.get_running_loop()
            self._gate = asyncio.Event()
            self.parked.set()
            await asyncio.wait_for(
                self._gate.wait(), timeout=HOLD_TIMEOUT_SECONDS
            )
        self.last_run_state = {"prompt": data.get("prompt", "")}
        await stream.send_done(
            {"type": "done", "final_text": data.get("prompt", "")},
            0.0,
        )

    async def handle_resume(
        self,
        ws: Any,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            self.check_run_token(data)
        except StaleRunError as exc:
            from src.backends.protocol import (
                MSG_RESUME,
                request_error,
            )

            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_STALE_RUN,
                    request_type=MSG_RESUME,
                )
            )
            return
        self.resumes.append(dict(self.last_run_state or {}))
        await stream.send_done(
            {"type": "done", "final_text": "resumed"}, 0.0
        )

    async def handle_probe(
        self, ws: Any, data: Dict[str, Any]
    ) -> None:
        try:
            self.check_run_token(data)
        except StaleRunError as exc:
            from src.backends.protocol import (
                MSG_PROBE,
                request_error,
                request_id_of,
            )

            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_STALE_RUN,
                    request_type=MSG_PROBE,
                    request_id=request_id_of(data),
                )
            )
            return
        self.probes.append(data)
        await ws.send_json(
            {
                "type": "probe_result",
                "request_id": data.get("request_id"),
            }
        )


@pytest.fixture()
def backend() -> _StubBackend:
    return _StubBackend()


@pytest.fixture()
def client(backend: _StubBackend) -> Iterator[TestClient]:
    with TestClient(create_worker_app(backend, device="cpu")) as c:
        yield c


@contextlib.contextmanager
def _window(client: TestClient) -> Iterator[Any]:
    """One browser, connected and past the ready handshake.

    A context manager rather than a bare connect, so a failing
    assertion still closes the socket; a leaked one keeps its server
    task alive and the client's shutdown waits on it.
    """
    with client.websocket_connect("/ws") as socket:
        ready = socket.receive_json()
        assert ready["type"] == "model_status", ready
        assert ready["status"] == "ready", ready
        yield socket


@contextlib.contextmanager
def _two_windows(client: TestClient) -> Iterator[Any]:
    """Two browsers on one worker, which is the whole subject here."""
    with _window(client) as first, _window(client) as second:
        yield first, second


def _generate(socket: Any, prompt: str) -> str:
    """Run a generation and return the token naming it."""
    socket.send_json({"type": "generate", "prompt": prompt})
    done = socket.receive_json()
    assert done["type"] == "done", done
    return done["run_token"]


def _park(backend: _StubBackend, socket: Any) -> None:
    """Start a generation and wait until it holds the lock.

    Waiting for the park rather than assuming it removes the race
    where the second window asks before the first has taken the lock,
    which would make a busy test pass or fail on scheduling.
    """
    backend.parks = True
    socket.send_json({"type": "generate", "prompt": "slow"})
    assert backend.parked.wait(timeout=HOLD_TIMEOUT_SECONDS), (
        "the generation never reached the lock"
    )


def _release(backend: _StubBackend, socket: Any) -> None:
    """Let the parked generation finish, and read its done frame."""
    backend.release()
    done = socket.receive_json()
    assert done["type"] == "done", done


# -- the handshake --


def test_a_ready_worker_says_so_on_connect(
    client: TestClient,
) -> None:
    with _window(client):
        pass


def test_the_handshake_names_the_model(client: TestClient) -> None:
    """The page compares this against the supervisor's own statement,
    which is the only way to catch a proxy pointed at the wrong
    worker."""
    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["model"] == MODEL_ID


# -- dispatch --


def test_an_unknown_type_is_refused_without_damage(
    client: TestClient,
) -> None:
    """A client bug, and scoped so it stays one: a page that just
    sent something unrecognisable is not helped by also losing
    whatever it was doing."""
    with _window(client) as socket:
        socket.send_json({"type": "sing"})
        reply = socket.receive_json()

    assert reply["type"] == "error"
    assert reply["code"] == ERROR_UNKNOWN_MESSAGE
    assert reply["scope"] == ERROR_SCOPE_REQUEST


def test_cancel_is_swallowed_rather_than_answered(
    client: TestClient,
) -> None:
    """It sets an event and replies with nothing, so the next frame
    on the socket belongs to whatever came after it. Were the branch
    missing, cancel would fall through to the unknown-type reply and
    the generation below would read that error instead of its own
    terminal frame.

    Worth knowing while reading this: cancel cannot actually
    interrupt a generation in flight. The message loop awaits the
    handler inline, so a cancel arriving on that socket is not read
    until the generation it was meant to stop has finished. That is
    `LIFE-04`'s to fix, and is recorded in the ledger.
    """
    with _window(client) as socket:
        socket.send_json({"type": "cancel"})

        token = _generate(socket, "after the cancel")

    assert token != ""


def test_a_generation_reaches_the_backend(
    client: TestClient,
) -> None:
    with _window(client) as socket:
        token = _generate(socket, "hello")

    assert token != ""


def test_every_terminal_frame_names_its_run(
    client: TestClient,
) -> None:
    with _window(client) as socket:
        first = _generate(socket, "one")
        second = _generate(socket, "two")

    assert first != second


# -- the generation lock --


def test_a_busy_worker_refuses_a_second_generation(
    backend: _StubBackend, client: TestClient
) -> None:
    """Run-scoped: the second window asked for a run and did not get
    one, so its run indicators have to come back down."""
    with _two_windows(client) as (first, second):
        _park(backend, first)

        second.send_json({"type": "generate", "prompt": "also"})
        refusal = second.receive_json()
        _release(backend, first)

    assert refusal["code"] == ERROR_BUSY
    assert refusal["scope"] == ERROR_SCOPE_RUN


def test_a_busy_worker_refuses_a_probe_without_ending_a_run(
    backend: _StubBackend, client: TestClient
) -> None:
    """`PROTOCOL-01`'s case, end to end. The same refusal as above,
    for a different request, and the scope is what stops it closing
    What If in the window that asked."""
    with _two_windows(client) as (first, second):
        _park(backend, first)

        second.send_json(
            {"type": "probe", "position": 0, "request_id": 9}
        )
        refusal = second.receive_json()
        _release(backend, first)

    assert refusal["code"] == ERROR_BUSY
    assert refusal["scope"] == ERROR_SCOPE_REQUEST
    assert refusal["request_id"] == 9


def test_one_refusal_two_scopes(
    backend: _StubBackend, client: TestClient
) -> None:
    """The pair above, stated as the comparison that matters. Same
    busy worker, same sentence, and the only difference is what the
    second window was asking for."""
    with _two_windows(client) as (first, second):
        _park(backend, first)

        second.send_json({"type": "resume", "frame_index": 1})
        for_run = second.receive_json()
        second.send_json({"type": "probe", "position": 0})
        for_probe = second.receive_json()
        _release(backend, first)

    assert for_run["message"] == for_probe["message"]
    assert for_run["scope"] == ERROR_SCOPE_RUN
    assert for_probe["scope"] == ERROR_SCOPE_REQUEST
    # Each names what it turned away. Every request in the streaming
    # branch happens to share a scope, so getting this wrong there
    # would not change any scope above; it would just have the frame
    # answer for a request the client never sent.
    assert for_run["request_type"] == "resume"
    assert for_probe["request_type"] == "probe"


def test_the_lock_is_released_after_a_run(
    backend: _StubBackend, client: TestClient
) -> None:
    """Otherwise the refusals above would be permanent."""
    with _two_windows(client) as (first, second):
        _park(backend, first)
        _release(backend, first)

        backend.parks = False
        token = _generate(second, "now mine")

    assert token != ""


def test_a_tokenizer_read_does_not_wait_for_the_lock(
    backend: _StubBackend, client: TestClient
) -> None:
    """Deliberate, and worth pinning: a prompt count is a tokenizer
    read costing microseconds, and stalling it behind a running model
    is exactly when the user is still typing.

    This stub loads no tokenizer, so the answer is that error rather
    than a count. Asserting the specific code is what makes the test
    bite: it proves the request reached `handle_count_prompt` while
    the lock was held, where merely checking that it was not refused
    as busy would also pass if the request had fallen through to the
    unknown-type branch.
    """
    with _two_windows(client) as (first, second):
        _park(backend, first)

        second.send_json(
            {"type": "count_prompt", "text": "hi", "request_id": 2}
        )
        reply = second.receive_json()
        _release(backend, first)

    assert reply["code"] == ERROR_NO_TOKENIZER, reply
    assert reply["request_id"] == 2


# -- two windows, one worker --


def test_a_second_windows_run_locks_the_first_out(
    client: TestClient,
) -> None:
    """`LIFE-01`'s clause. Window one completes a run, window two
    completes another, and window one's resume must be refused."""
    with _two_windows(client) as (window_one, window_two):
        token_one = _generate(window_one, "the first prompt")
        _generate(window_two, "the second prompt")

        window_one.send_json(
            {
                "type": "resume",
                "frame_index": 1,
                "run_token": token_one,
            }
        )
        refusal = window_one.receive_json()

    assert refusal["type"] == "error"
    assert refusal["code"] == ERROR_STALE_RUN


def test_the_second_window_can_still_work(
    client: TestClient,
) -> None:
    """The other half of the clause: refusing everything would also
    pass the test above."""
    with _two_windows(client) as (window_one, window_two):
        _generate(window_one, "the first prompt")
        token_two = _generate(window_two, "the second prompt")

        window_two.send_json(
            {
                "type": "resume",
                "frame_index": 1,
                "run_token": token_two,
            }
        )
        reply = window_two.receive_json()

    assert reply["type"] == "done"


def test_identical_runs_do_not_let_a_stale_request_through(
    client: TestClient,
) -> None:
    """The repeat the clause asks for, with equal shapes and equal
    output. Every check that works on shape passes here, which is
    what makes the wrong answer look like a right one."""
    with _two_windows(client) as (window_one, window_two):
        token_one = _generate(window_one, "identical")
        _generate(window_two, "identical")

        window_one.send_json(
            {
                "type": "resume",
                "frame_index": 1,
                "run_token": token_one,
            }
        )
        refusal = window_one.receive_json()

    assert refusal["code"] == ERROR_STALE_RUN


def test_a_stale_probe_is_refused_across_sockets(
    backend: _StubBackend, client: TestClient
) -> None:
    """Same fence, scoped to the request, so the first window loses a
    measurement rather than its edit session."""
    with _two_windows(client) as (window_one, window_two):
        token_one = _generate(window_one, "one")
        _generate(window_two, "two")

        window_one.send_json(
            {
                "type": "probe",
                "position": 0,
                "request_id": 3,
                "run_token": token_one,
            }
        )
        refusal = window_one.receive_json()

    assert refusal["code"] == ERROR_STALE_RUN
    assert refusal["scope"] == ERROR_SCOPE_REQUEST
    assert backend.probes == []


def test_the_admitted_request_reaches_the_run_it_named(
    backend: _StubBackend, client: TestClient
) -> None:
    """Not just that the stale one is turned away: the one let
    through has to act on the run it asked about."""
    with _two_windows(client) as (window_one, window_two):
        _generate(window_one, "replaced")
        token_two = _generate(window_two, "current")

        window_two.send_json(
            {
                "type": "resume",
                "frame_index": 1,
                "run_token": token_two,
            }
        )
        assert window_two.receive_json()["type"] == "done"

    assert backend.resumes == [{"prompt": "current"}]
