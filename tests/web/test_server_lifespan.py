"""The supervisor starts and stops in a lifespan (`A2-DEPS-01`).

Strategy: enter the app through FastAPI's test client with recorders
standing in for the orphan sweep and the model manager, and note
where a request falls between them. The warning check runs in a
fresh interpreter, because the app is built when its module is
imported, and this process imported it before any test ran.

The startup and shutdown hooks this replaces were deprecated, and
their warnings were most of the suite's. Passing proves the move kept
what they did: the sweep finishes before anything is served, and the
manager stops after the last request, even when the lifespan ends in
an error. It also proves that building and entering the app records
no lifecycle deprecation, and that no module under ``src`` registers
a hook the deprecated way.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import List

import pytest
from fastapi.testclient import TestClient

from src.web import model_manager, server

# Served by the static mount, so the request touches neither the
# manager nor the data root, and its place in the order is all it
# can show.
STATIC_PATH = "/app.js"

# Run in the fresh interpreter: build the app by importing it, enter
# its lifespan with the sweep and the manager stubbed, and print each
# lifecycle deprecation recorded on the way.
SUPERVISOR_PROBE = """
import asyncio
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from src.web import model_manager, server

    class _Stopped:
        async def stop(self):
            return None

    def _swept():
        return None

    model_manager.sweep_orphan_workers = _swept
    server.manager = _Stopped()

    async def _enter():
        async with server.app.router.lifespan_context(server.app):
            return None

    asyncio.run(_enter())

for warning in caught:
    text = str(warning.message)
    if not issubclass(warning.category, DeprecationWarning):
        continue
    if "on_event" in text or "lifespan" in text:
        print(text.strip())
"""


class _RecordingManager:
    """Notes the one call the app makes to it, on the way out."""

    def __init__(self, events: List[str]) -> None:
        self._events = events

    async def stop(self) -> None:
        self._events.append("stop")


class _ServerFailed(RuntimeError):
    """Raised inside the lifespan, standing in for a server that
    stops on an error rather than on request."""


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """The sweep and the manager's stop, in the order they ran."""
    recorded: List[str] = []

    def record_sweep() -> None:
        recorded.append("sweep")

    monkeypatch.setattr(
        model_manager, "sweep_orphan_workers", record_sweep
    )
    monkeypatch.setattr(
        server, "manager", _RecordingManager(recorded)
    )
    return recorded


def test_the_sweep_runs_before_serving_and_the_stop_after(
    events: List[str],
) -> None:
    with TestClient(server.app) as client:
        events.append("entered")
        response = client.get(STATIC_PATH)
        events.append("served")
    events.append("exited")

    assert response.status_code == 200, response.status_code
    assert events == [
        "sweep",
        "entered",
        "served",
        "stop",
        "exited",
    ]


def test_the_manager_stops_when_the_lifespan_ends_in_an_error(
    events: List[str],
) -> None:
    """A worker left running keeps its VRAM after the supervisor is
    gone, so the stop may not depend on the server ending cleanly."""

    async def enter_and_fail() -> None:
        async with server.app.router.lifespan_context(server.app):
            events.append("entered")
            raise _ServerFailed("the server stopped on an error")

    with pytest.raises(_ServerFailed):
        asyncio.run(enter_and_fail())

    assert events == ["sweep", "entered", "stop"]


def test_building_and_entering_the_app_warns_nothing() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SUPERVISOR_PROBE],
        cwd=server.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout


def test_no_module_registers_a_hook_the_deprecated_way() -> None:
    """Every app under ``src``, not only the two there are today, so
    a third cannot bring the deprecated form back."""
    source_root = server.REPO_ROOT / "src"
    scanned = sorted(source_root.rglob("*.py"))
    offenders = [
        str(path.relative_to(source_root))
        for path in scanned
        if "on_event(" in path.read_text(encoding="utf-8")
    ]

    assert source_root / "web" / "server.py" in scanned
    assert source_root / "backends" / "worker_app.py" in scanned
    assert offenders == [], offenders
