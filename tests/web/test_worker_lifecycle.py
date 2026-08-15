"""A worker that stops is verified gone before anything replaces it.

Strategy: drive a real `ModelManager` with a fake process and a
scripted health probe, both injected at construction, with the
timeouts turned down to milliseconds. No subprocess, no socket, no
model. That is the whole reason the seam in
`src/web/worker_process.py` exists: these are the five scenarios
`LIFE-02` asks for, and none of them was reachable before.

What passing proves is the finding's claim in reverse. Three exits in
the startup monitor used to record an error and return with the worker
still running, so it kept its VRAM while the supervisor called itself
idle, and the page gates, which asked only whether a process existed,
went on admitting traffic to a model that would never answer. The stop
path escalated to SIGKILL and cleared every field without waiting, so
a replacement could be spawned against memory whose release nothing
had confirmed; the eight-second settle window in `_preflight_vram` was
standing in for that missing wait.

The two assertions the finding names are
`test_no_spawn_before_the_previous_process_exits` and the cases below
that compare a terminal manager snapshot against what the fake
process actually did.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.web import server
from src.web.server import ModelManager

# Any real id, so REGISTRY lookups inside activate resolve. Which
# model it is does not matter here; the lifecycle is identical.
MODEL_ID = "llada"


class FakeProcess:
    """A worker whose behavior the test chooses.

    Models the three ways a real one can respond to being stopped:
    exit on SIGTERM (the normal case), ignore SIGTERM and die on
    SIGKILL (wedged in a driver call), or ignore both (stuck in
    uninterruptible I/O).
    """

    def __init__(
        self,
        *,
        pid: int = 1000,
        exits_on_terminate: bool = True,
        exits_on_kill: bool = True,
    ) -> None:
        self.pid = pid
        self._exits_on_terminate = exits_on_terminate
        self._exits_on_kill = exits_on_kill
        self._code: Optional[int] = None
        self.calls: List[str] = []

    def poll(self) -> Optional[int]:
        return self._code

    def terminate(self) -> None:
        self.calls.append("terminate")
        if self._exits_on_terminate:
            self._code = 0

    def kill(self) -> None:
        self.calls.append("kill")
        if self._exits_on_kill:
            self._code = -9

    def wait(self, timeout: float) -> int:
        self.calls.append("wait")
        if self._code is None:
            raise TimeoutError(f"still running after {timeout}s")
        return self._code

    # -- what the test drives from the outside --

    def exit_with(self, code: int) -> None:
        """The worker died on its own, as a crash during load."""
        self._code = code

    @property
    def alive(self) -> bool:
        return self._code is None


class Harness:
    """A manager wired to fakes, plus what they recorded."""

    def __init__(
        self,
        *,
        health: Optional[List[Dict[str, Any]]] = None,
        process: Optional[FakeProcess] = None,
    ) -> None:
        self.processes: List[FakeProcess] = []
        self.spawns: List[List[FakeProcess]] = []
        self._next = process
        # Consumed one per probe; the last entry repeats, so a test
        # can park the worker in "loading" indefinitely.
        self.health = health or []
        self.probes = 0

        def spawn(
            command: Any, *, cwd: Any, env: Any
        ) -> FakeProcess:
            # Snapshot every earlier process's liveness at the moment
            # of the spawn. This is what proves a replacement never
            # starts while its predecessor is still running.
            self.spawns.append(
                [p for p in self.processes if p.alive]
            )
            made = self._next or FakeProcess(
                pid=1000 + len(self.processes)
            )
            self._next = None
            self.processes.append(made)
            return made

        async def probe(url: str) -> Optional[Dict[str, Any]]:
            self.probes += 1
            if not self.health:
                return None
            index = min(self.probes - 1, len(self.health) - 1)
            return self.health[index]

        self.manager = ModelManager(
            spawn=spawn,  # type: ignore[arg-type]
            probe=probe,
            start_timeout_s=0.05,
            stop_timeout_s=0.01,
            kill_timeout_s=0.01,
            health_poll_s=0.001,
            progress_poll_s=0.001,
        )

    async def settle(self, seconds: float = 0.4) -> None:
        """Let the startup monitor run to its conclusion."""
        task = self.manager._monitor_task
        if task is None:
            return
        # Shielded so a monitor that finalizes itself is observed
        # rather than cancelled out from under the assertion.
        with contextlib.suppress(
            asyncio.TimeoutError, asyncio.CancelledError
        ):
            await asyncio.wait_for(
                asyncio.shield(task), timeout=seconds
            )


READY = {
    "status": "ready",
    "id": MODEL_ID,
    "versions": {"torch": "2.4.0"},
    "tokenizer": {"name_or_path": "x"},
}
LOADING = {"status": "loading", "id": MODEL_ID}
FAILED = {
    "status": "error",
    "id": MODEL_ID,
    "message": "CUDA out of memory",
}


@pytest.fixture(autouse=True)
def _no_vram_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activate on "cuda" without asking a GPU that is not here.

    The pre-flight is `LIFE-06`'s subject, not this file's; leaving
    it live would make every case here depend on nvidia-smi.
    """
    monkeypatch.setattr(
        server, "_gpu_name", lambda: "Fake GPU"
    )
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: 99.0
    )


@pytest.fixture(autouse=True)
def _interpreter_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The venv the registry names is not installed in CI."""
    monkeypatch.setattr(Path, "exists", lambda self: True)


def _activate(harness: Harness, device: str = "cuda") -> None:
    asyncio.run(
        harness.manager.activate(MODEL_ID, device=device)
    )


async def _activate_and_settle(
    harness: Harness, device: str = "cuda"
) -> None:
    await harness.manager.activate(MODEL_ID, device=device)
    await harness.settle()


# -- the five ways startup ends --


def test_a_ready_worker_becomes_the_resident_model() -> None:
    """The happy path, first, so the failures below are read against
    something known to work."""
    harness = Harness(health=[READY])

    asyncio.run(_activate_and_settle(harness))

    assert harness.manager.load_state == "ready"
    assert harness.manager.active_id == MODEL_ID
    assert harness.processes[0].alive


def test_a_startup_timeout_terminates_the_worker() -> None:
    """The finding's headline. A worker that never answers used to
    be recorded as an error and left running, holding its VRAM."""
    harness = Harness(health=[])  # never answers

    asyncio.run(_activate_and_settle(harness))

    assert not harness.processes[0].alive
    assert "terminate" in harness.processes[0].calls


def test_a_startup_timeout_says_so(
) -> None:
    harness = Harness(health=[])

    asyncio.run(_activate_and_settle(harness))

    assert harness.manager.load_state == "error"
    assert "did not start in time" in str(
        harness.manager.load_error
    )


def test_a_health_reported_failure_terminates_the_worker() -> None:
    """The second exit that used to leave a worker alive: the model
    itself failed to load and the process stayed up around it."""
    harness = Harness(health=[LOADING, FAILED])

    asyncio.run(_activate_and_settle(harness))

    assert not harness.processes[0].alive
    assert harness.manager.load_state == "error"
    assert "CUDA out of memory" in str(
        harness.manager.load_error
    )


def test_a_worker_that_exits_during_startup_is_reaped() -> None:
    """The third exit. Here the process is already gone, so the only
    thing to get right is that the manager agrees."""
    process = FakeProcess()
    harness = Harness(health=[LOADING], process=process)

    async def scenario() -> None:
        await harness.manager.activate(MODEL_ID, device="cuda")
        await asyncio.sleep(0.01)
        process.exit_with(1)
        await harness.settle()

    asyncio.run(scenario())

    assert harness.manager.load_state == "error"
    assert "code 1" in str(harness.manager.load_error)
    assert harness.manager.active_id is None


def test_a_graceful_stop_needs_no_kill() -> None:
    """The negative space around the escalation below: a worker that
    honors SIGTERM must never be killed."""
    harness = Harness(health=[READY])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.stop()

    asyncio.run(scenario())

    assert harness.processes[0].calls.count("terminate") == 1
    assert "kill" not in harness.processes[0].calls


def test_a_worker_that_ignores_sigterm_is_killed() -> None:
    process = FakeProcess(exits_on_terminate=False)
    harness = Harness(health=[READY], process=process)

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.stop()

    asyncio.run(scenario())

    assert "kill" in process.calls
    assert not process.alive


def test_a_worker_surviving_sigkill_is_reported_not_hidden(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing further can be done, so the requirement is honesty.
    Reporting a clean stop here is what would let a replacement be
    spawned against memory that was never released."""
    process = FakeProcess(
        exits_on_terminate=False, exits_on_kill=False
    )
    harness = Harness(health=[READY], process=process)

    async def scenario() -> None:
        await _activate_and_settle(harness)
        with caplog.at_level("ERROR"):
            await harness.manager.stop()

    asyncio.run(scenario())

    assert "survived SIGKILL" in caplog.text


# -- the ordering the finding asks for --


def test_no_spawn_before_the_previous_process_exits() -> None:
    """The assertion `LIFE-02` names. A second activation must find
    the first worker already gone, not merely signalled."""
    harness = Harness(health=[READY])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.activate(
            "smollm3", device="cpu"
        )
        await harness.settle()

    asyncio.run(scenario())

    assert len(harness.spawns) == 2
    # Nothing was alive at the moment of either spawn.
    assert harness.spawns[0] == []
    assert harness.spawns[1] == []


def test_a_stop_waits_for_a_stubborn_worker_to_die() -> None:
    """The wait after the kill, which is what the eight-second VRAM
    settle window was standing in for."""
    process = FakeProcess(exits_on_terminate=False)
    harness = Harness(health=[READY], process=process)

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.stop()

    asyncio.run(scenario())

    # terminate, wait, kill, wait: the escalation in order.
    assert process.calls == [
        "terminate",
        "wait",
        "kill",
        "wait",
    ]


# -- the manager's story matches the process --


@pytest.mark.parametrize(
    "health",
    [[], [LOADING, FAILED]],
    ids=["timeout", "reported-error"],
)
def test_a_terminal_snapshot_agrees_with_process_reality(
    health: List[Dict[str, Any]],
) -> None:
    """The other assertion the finding names. Whatever the manager
    says after a failure, the process must actually be in that
    state."""
    harness = Harness(health=health)

    asyncio.run(_activate_and_settle(harness))

    assert harness.manager.load_state == "error"
    assert harness.manager.active_id is None
    assert harness.manager._proc is None
    assert not harness.processes[0].alive


def test_a_stopped_manager_holds_no_worker() -> None:
    harness = Harness(health=[READY])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.stop()

    asyncio.run(scenario())

    assert harness.manager.active_id is None
    assert harness.manager.load_state == "idle"
    assert not harness.processes[0].alive


def test_a_cancelled_activation_frees_the_worker() -> None:
    """Cancel arrives mid-load, which is the case the lock is left
    free for.

    Cancelling names the activation since `LIFE-03`, which is why
    the operation the activate returned is carried here.
    """
    harness = Harness(health=[LOADING])

    async def scenario() -> None:
        operation = await harness.manager.activate(
            MODEL_ID, device="cuda"
        )
        await asyncio.sleep(0.01)
        await harness.manager.cancel_activation(operation)

    asyncio.run(scenario())

    assert not harness.processes[0].alive
    assert harness.manager.load_state == "idle"


# -- the failure outlives the process, on purpose --


def test_the_reason_survives_the_worker() -> None:
    """The page that would show it is usually a redirect away, so
    clearing it during finalization left the menu with nothing to
    say about why it was there."""
    harness = Harness(health=[LOADING, FAILED])

    asyncio.run(_activate_and_settle(harness))

    assert harness.manager.load_state == "error"
    assert harness.manager.load_error is not None


def test_a_deliberate_stop_reports_nothing() -> None:
    """The other half: switching or cancelling is not a failure and
    must not leave an error on the menu."""
    harness = Harness(health=[READY])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        await harness.manager.stop()

    asyncio.run(scenario())

    assert harness.manager.load_state == "idle"
    assert harness.manager.load_error is None


def test_trying_again_clears_the_previous_failure() -> None:
    harness = Harness(health=[])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        assert harness.manager.load_error is not None
        harness.health = [READY]
        await _activate_and_settle(harness)

    asyncio.run(scenario())

    assert harness.manager.load_state == "ready"
    assert harness.manager.load_error is None


# -- a late failure must not clobber a newer worker --


def test_a_superseded_finalize_leaves_the_new_worker_alone(
) -> None:
    """The process identity check. A slow termination can outlast
    the activation that replaced it, and clearing the manager's
    fields then would delete a healthy worker's identity while it
    kept running."""
    harness = Harness(health=[READY])

    async def scenario() -> None:
        await _activate_and_settle(harness)
        stale = harness.processes[0]
        # A second worker takes over.
        await harness.manager.activate(
            "smollm3", device="cpu"
        )
        await harness.settle()
        # The first worker's monitor finally reports its failure.
        await harness.manager._finalize(
            stale, error="too late to matter"
        )

    asyncio.run(scenario())

    assert harness.manager.active_id == "smollm3"
    assert harness.manager.load_state == "ready"
    assert harness.manager.load_error is None


# -- what the page gates now ask --


def test_a_loading_worker_cannot_serve_yet() -> None:
    """`is_serving` versus `status`: the process exists, so the menu
    still counts its VRAM, but no request may reach it."""
    harness = Harness(health=[LOADING])

    async def scenario() -> None:
        await harness.manager.activate(MODEL_ID, device="cuda")
        await asyncio.sleep(0.01)

    asyncio.run(scenario())

    assert harness.manager.status(MODEL_ID) == "active"
    assert not harness.manager.is_serving(MODEL_ID)


def test_a_failed_worker_cannot_serve() -> None:
    """The gate the finding is about. This used to be true and let
    traffic through anyway, because the gate asked only whether a
    process was alive."""
    harness = Harness(health=[LOADING, FAILED])

    asyncio.run(_activate_and_settle(harness))

    assert not harness.manager.is_serving(MODEL_ID)


def test_a_ready_worker_can_serve() -> None:
    harness = Harness(health=[READY])

    asyncio.run(_activate_and_settle(harness))

    assert harness.manager.is_serving(MODEL_ID)


def test_only_the_resident_model_can_serve() -> None:
    harness = Harness(health=[READY])

    asyncio.run(_activate_and_settle(harness))

    assert not harness.manager.is_serving("smollm3")
