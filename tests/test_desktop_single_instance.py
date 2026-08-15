"""A second desktop launch joins the first rather than rivalling it.

Strategy: run real HTTP servers from the standard library on
ephemeral ports and point the launcher's probe at them. A real socket
is the point: the question the probe answers is what is listening
over there, and a mocked answer would only assert that the mock was
consulted. No GUI is touched, because the decision is made before any
window library is.

What passing proves is the narrow slice of `LIFE-05` that was taken.
Launching the desktop app twice used to start two supervisors: the
second finds port 8760 busy, shrugs, and takes an ephemeral one
instead. Each then enforces "one resident model" over a GPU neither
knows it shares, so both spawn a worker and the second dies of CUDA
out of memory after the user has already waited out its load. The
maintainer hit exactly that, which is what pulled this forward.

The distinction the probe has to draw is the reason it exists. A
failed bind alone cannot tell a second copy of this app from an
unrelated process holding the same port, and those want opposite
responses: stand down for the first, step around for the second.

What is deliberately not covered here: a host-level lease across the
browser supervisor and a manually launched `main.py`. That is the
rest of `LIFE-05` and it is recorded as deferred.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator, Optional

import pytest

import desktop
from src.web.server import APP_IDENTITY


def _serve(payload: Optional[Dict[str, Any]], status: int = 200):
    """A one-route HTTP server on an ephemeral port.

    Yields its port. ``payload`` of None serves a body that is not
    JSON at all, standing in for an unrelated service.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib's name
            body = (
                json.dumps(payload).encode()
                if payload is not None
                else b"<html>not us</html>"
            )
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            pass  # Silence the default stderr logging.

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True
    )
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def ours() -> Iterator[int]:
    yield from _serve({"app": APP_IDENTITY, "pid": 4321})


@pytest.fixture
def foreign() -> Iterator[int]:
    yield from _serve({"app": "something-else"})


@pytest.fixture
def not_http() -> Iterator[int]:
    yield from _serve(None)


def _dead_port() -> int:
    """A port with nothing on it: bound, read, and released."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    return port


# -- telling our own supervisor from everything else --


def test_our_supervisor_is_recognised(ours: int) -> None:
    found = desktop.probe_supervisor(ours)

    assert found is not None
    assert found["app"] == APP_IDENTITY


def test_the_identity_carries_the_owner_s_pid(ours: int) -> None:
    """So the message can name what is already running rather than
    just asserting that something is."""
    found = desktop.probe_supervisor(ours)

    assert found is not None
    assert found["pid"] == 4321


def test_another_service_is_not_mistaken_for_ours(
    foreign: int,
) -> None:
    """The case that makes a bind test insufficient. Something else
    on 8760 must send this app around it, not stop it."""
    assert desktop.probe_supervisor(foreign) is None


def test_a_non_json_listener_is_not_ours(
    not_http: int,
) -> None:
    assert desktop.probe_supervisor(not_http) is None


def test_an_error_response_is_not_ours() -> None:
    generator = _serve({"app": APP_IDENTITY}, status=500)
    port = next(generator)
    try:
        assert desktop.probe_supervisor(port) is None
    finally:
        generator.close()


def test_an_empty_port_is_not_ours() -> None:
    assert desktop.probe_supervisor(_dead_port()) is None


def test_the_probe_gives_up_quickly() -> None:
    """A launch runs this while the user waits for a window, so
    something listening that will not answer must not hold it."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # accepts, never replies
    port = int(listener.getsockname()[1])
    try:
        found = desktop.probe_supervisor(
            port, timeout_seconds=0.2
        )
    finally:
        listener.close()

    assert found is None


# -- what the launcher does with that --


def test_nothing_listening_means_nothing_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common path, and it must not cost a network round trip:
    the local bind test answers it."""
    probed = []
    monkeypatch.setattr(
        desktop, "_port_available", lambda port: True
    )
    monkeypatch.setattr(
        desktop,
        "probe_supervisor",
        lambda *a, **k: probed.append(1),
    )

    assert desktop.find_running_instance(1234) is None
    assert probed == []


def test_a_busy_port_is_asked_who_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop, "_port_available", lambda port: False
    )
    monkeypatch.setattr(
        desktop,
        "probe_supervisor",
        lambda port, **k: {"app": APP_IDENTITY, "pid": 7},
    )

    found = desktop.find_running_instance(1234)

    assert found is not None
    assert found["pid"] == 7


def _fenced_main(
    monkeypatch: pytest.MonkeyPatch,
    running: Optional[Dict[str, Any]],
) -> list:
    """Arm `main` so that going past the guard fails immediately.

    Everything downstream of the early return raises rather than
    being counted. Without this the failure mode is a hang: `main`
    would build a uvicorn server, start a thread and wait thirty
    seconds for a port that the test never opened, and a hanging
    suite is worse than a failing one because nothing says why.
    """
    reached: list = []

    def refuse(name: str):
        def boom(*_args: Any, **_kwargs: Any) -> Any:
            reached.append(name)
            raise AssertionError(
                f"main() reached {name} with an instance already"
                " running; a second supervisor would have started"
            )

        return boom

    monkeypatch.setattr(
        desktop, "find_running_instance", lambda: running
    )
    monkeypatch.setattr(desktop, "_select_gui", refuse("_select_gui"))
    monkeypatch.setattr(
        desktop, "_resolve_port", refuse("_resolve_port")
    )
    monkeypatch.setattr(
        desktop.uvicorn, "Server", refuse("uvicorn.Server")
    )
    return reached


def test_a_second_launch_starts_no_second_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point. `main` must return before it builds a
    uvicorn server, which is what used to put two of them on one
    GPU."""
    reached = _fenced_main(
        monkeypatch, {"app": APP_IDENTITY, "pid": 99}
    )
    monkeypatch.setattr(
        desktop, "focus_running_window", lambda: True
    )

    desktop.main()

    assert reached == []


def test_a_second_launch_tries_to_raise_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focused: list = []
    _fenced_main(monkeypatch, {"app": APP_IDENTITY, "pid": 99})
    monkeypatch.setattr(
        desktop,
        "focus_running_window",
        lambda: focused.append(True) or True,
    )

    desktop.main()

    assert focused == [True]


def test_a_launch_says_what_it_found(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Raising another process's window is the window manager's to
    allow and often refused under Wayland, so the message is the
    part that always works. Exiting silently would read as the
    launcher being broken."""
    _fenced_main(monkeypatch, {"app": APP_IDENTITY, "pid": 99})
    monkeypatch.setattr(
        desktop, "focus_running_window", lambda: False
    )

    desktop.main()

    said = capsys.readouterr().err
    assert "already running" in said
    assert "99" in said


def test_a_foreign_port_still_lets_the_app_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated process on 8760 must not lock this app out. The
    existing ephemeral fallback is the right answer there, and it is
    only reachable because the probe said "not ours"."""
    monkeypatch.setattr(
        desktop, "_port_available", lambda port: False
    )
    monkeypatch.setattr(
        desktop, "probe_supervisor", lambda port, **k: None
    )

    assert desktop.find_running_instance() is None
    assert desktop._resolve_port() != desktop.DESKTOP_PORT


# -- raising the window is best-effort, and says so --


def test_focusing_reports_failure_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither tool installed there is nothing to try. The
    caller has already printed its message and must not be handed an
    exception on the way out."""
    monkeypatch.setattr(
        desktop.shutil, "which", lambda name: None
    )

    assert desktop.focus_running_window() is False
