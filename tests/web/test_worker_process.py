"""The seam that lets the worker lifecycle be tested at all.

Strategy: exercise `worker_process` against real short-lived
subprocesses, because the whole point of the module is what it does
to an operating-system process, and a test that mocked that away
would prove nothing. The processes are `sleep` and `true`, so this
stays fast and needs no model.

What passing proves is narrow but load-bearing. `SubprocessHandle`
reports the same things `Popen` does, so swapping the manager onto
the handle changed no behavior. The command builder produces the argv
the supervisor has always launched, and it carries the marker the
orphan sweep matches on, so the sweep cannot quietly stop recognizing
its own workers. And the module imports without a web framework,
which is what keeps `tests/web/test_worker_lifecycle.py` able to run
five process scenarios in milliseconds.

Two deliberate limits. The real processes here all exit on their own,
because an agent sandbox refuses to signal a process in a new session
and `spawn_worker` deliberately puts workers in one; signalling is
therefore checked by delegation against a stand-in instead. And
whether terminate-then-kill is the right *sequence* is not this
file's question at all, it is the manager's, and it is answered in
`tests/web/test_worker_lifecycle.py` where the process is a fake.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.web.worker_process import (
    SubprocessHandle,
    download_command,
    spawn_worker,
    worker_command,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_IMPORTS = ("fastapi", "torch", "transformers", "httpx")


# -- the isolation the lifecycle tests depend on --


def test_the_seam_needs_neither_a_framework_nor_a_model() -> None:
    """Reimport the module with the heavy packages poisoned.

    Same constraint `run_store` carries, for the same reason: a test
    that has to stand up a web framework to check process handling is
    a test nobody runs in a loop, and the five `LIFE-02` scenarios
    are only cheap while this holds.
    """
    import importlib

    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".", 1)[0]
        if root in FORBIDDEN_IMPORTS:
            raise AssertionError(
                f"worker_process must not import {root}"
            )
        return real_import(name, *args, **kwargs)

    sys.modules.pop("src.web.worker_process", None)
    try:
        builtins.__import__ = guarded
        importlib.import_module("src.web.worker_process")
    finally:
        builtins.__import__ = real_import


# -- the argv the supervisor launches --


def test_the_command_names_the_worker_module() -> None:
    command = worker_command(
        python=Path("/venv/bin/python"),
        model_id="LLaDA-8B-Instruct",
        port=41234,
        device="cuda",
    )

    assert command[0] == "/venv/bin/python"
    assert command[1:3] == ["-m", "src.backends.run_worker"]


def test_the_command_carries_every_argument() -> None:
    """All four are load-bearing: the wrong model, port or device
    each produce a worker that looks healthy and is not the one that
    was asked for."""
    command = worker_command(
        python=Path("/venv/bin/python"),
        model_id="SmolLM3-3B",
        port=41234,
        device="cpu",
    )

    assert "--model" in command
    assert command[command.index("--model") + 1] == "SmolLM3-3B"
    assert command[command.index("--port") + 1] == "41234"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_the_download_command_names_its_module() -> None:
    """`TRUST-04` runs a fetch through this same seam, so a download
    is a child the supervisor can name and end rather than a daemon
    thread nothing could reach."""
    command = download_command(
        python=Path("/venv/bin/python"),
        repo_id="GSAI-ML/LLaDA-8B-Instruct",
    )

    assert command[0] == "/venv/bin/python"
    assert command[1:3] == ["-m", "src.inference.download_main"]
    assert (
        command[command.index("--repo") + 1]
        == "GSAI-ML/LLaDA-8B-Instruct"
    )


def test_a_download_inherits_the_orphan_guards() -> None:
    """Not a separate spawn path, which is the point: a download
    gets its own session and PDEATHSIG for free, so even a
    hard-killed supervisor takes the fetch with it. The old
    in-process daemon thread could not be reached at all."""
    from src.web.worker_process import _popen_orphan_guards

    guards = _popen_orphan_guards()

    if sys.platform.startswith("linux"):
        assert guards["start_new_session"] is True
        assert "preexec_fn" in guards
    else:
        assert guards == {}


def test_the_command_is_what_the_orphan_sweep_looks_for() -> None:
    """The sweep matches a marker against /proc cmdline to decide
    what is one of ours. If the two drift, a crashed supervisor's
    worker keeps its VRAM and nothing reclaims it."""
    from src.web.server import _WORKER_CMD_MARKER

    command = worker_command(
        python=Path("/venv/bin/python"),
        model_id="LLaDA-8B-Instruct",
        port=1,
        device="cuda",
    )

    assert _WORKER_CMD_MARKER in " ".join(command)


# -- the handle over a real process --


def test_a_running_process_polls_as_running() -> None:
    handle = spawn_worker(
        ["sleep", "0.4"], cwd=REPO_ROOT, env={}
    )

    assert handle.poll() is None
    assert handle.pid > 0

    handle.wait(5.0)


def test_a_finished_process_reports_its_exit_code() -> None:
    handle = spawn_worker(["true"], cwd=REPO_ROOT, env={})

    handle.wait(5.0)

    assert handle.poll() == 0


def test_a_failed_process_reports_a_nonzero_code() -> None:
    """Distinct from a clean exit because the startup monitor puts
    the code in the message the user reads."""
    handle = spawn_worker(["false"], cwd=REPO_ROOT, env={})

    handle.wait(5.0)

    assert handle.poll() != 0


def test_wait_raises_rather_than_hanging_on_a_survivor() -> None:
    """The behavior the termination path is built on: a worker that
    does not exit has to surface as an exception, not as a wait that
    never returns. `TimeoutExpired` specifically, since that is what
    the manager catches to decide to escalate."""
    handle = spawn_worker(
        ["sleep", "5"], cwd=REPO_ROOT, env={}
    )

    with pytest.raises(subprocess.TimeoutExpired):
        handle.wait(0.1)


def test_the_handle_reports_what_popen_reports() -> None:
    """Paired against the object it wraps, because the manager was
    swapped from one to the other and any disagreement between them
    is a behavior change nothing else would catch."""
    process = subprocess.Popen(["sleep", "0.4"])
    handle = SubprocessHandle(process)

    assert handle.pid == process.pid
    assert handle.poll() == process.poll()

    handle.wait(5.0)

    assert handle.poll() == process.poll()
    assert handle.poll() is not None


def test_the_worker_runs_in_its_own_session_on_linux() -> None:
    """The orphan guard the supervisor has always spawned with. A
    worker sharing the supervisor's session would take a Ctrl-C
    meant for the supervisor."""
    if not sys.platform.startswith("linux"):
        pytest.skip("session handling is Linux-only here")

    handle = spawn_worker(
        ["sleep", "0.4"], cwd=REPO_ROOT, env={}
    )

    assert os.getsid(handle.pid) != os.getsid(os.getpid())

    handle.wait(5.0)


# -- signalling, checked by delegation --


class _RecordingProcess:
    """A stand-in for `Popen` that records what it was asked to do.

    Used because an agent sandbox will not let this process signal a
    worker in its own session, and because `SubprocessHandle`'s whole
    job on these three calls is to forward them. Whether forwarding
    happens is the question; whether SIGTERM ends a process is
    CPython's.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.pid = 4321
        self.returncode: Any = None

    def poll(self) -> Any:
        self.calls.append("poll")
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float) -> int:
        self.calls.append(f"wait({timeout})")
        return 0


def test_terminate_reaches_the_process() -> None:
    process = _RecordingProcess()

    SubprocessHandle(process).terminate()  # type: ignore[arg-type]

    assert process.calls == ["terminate"]


def test_kill_reaches_the_process() -> None:
    process = _RecordingProcess()

    SubprocessHandle(process).kill()  # type: ignore[arg-type]

    assert process.calls == ["kill"]


def test_wait_passes_the_timeout_through() -> None:
    """A dropped timeout would turn the stop path's bounded wait
    into an unbounded one, which is the failure the bound exists to
    prevent."""
    process = _RecordingProcess()

    SubprocessHandle(process).wait(2.5)  # type: ignore[arg-type]

    assert process.calls == ["wait(2.5)"]
