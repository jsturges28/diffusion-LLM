"""Owning a model worker's operating-system process.

The supervisor's one job that is not HTTP: start a subprocess in
another virtual environment, and later prove it is gone. Both halves
live here so `ModelManager` can be driven by a fake in a test, which
it could not be while it called `subprocess.Popen` directly. There
are no tests of activation or termination today, and `LIFE-02` asks
for five scenarios (startup timeout, health-reported error, graceful
exit, terminate timeout, kill escalation) that need a process whose
behavior the test chooses.

Deliberately dependency-light: standard library only, no FastAPI and
no model libraries, the same constraint `run_store` carries and for
the same reason. A test that has to import a web framework to check
process handling is a test nobody runs in a loop.
"""

from __future__ import annotations

import ctypes
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

logger = logging.getLogger("diffusion_supervisor")


class WorkerHandle(Protocol):
    """One running worker, as much of it as the supervisor needs.

    A deliberately small surface: it is the whole vocabulary a fake
    has to implement, and every method here is one the termination
    path actually calls.
    """

    @property
    def pid(self) -> int:
        """The operating-system process id, for logging."""

    def poll(self) -> Optional[int]:
        """Exit code, or None while the process is still running."""

    def terminate(self) -> None:
        """Ask the process to exit (SIGTERM)."""

    def kill(self) -> None:
        """Stop the process without asking (SIGKILL)."""

    def wait(self, timeout: float) -> int:
        """Block until exit, or raise on timeout.

        Blocking on purpose. The caller runs it in a thread, which is
        what keeps the event loop free while a worker unloads several
        gigabytes.
        """


class SubprocessHandle:
    """A `WorkerHandle` backed by a real child process."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def wait(self, timeout: float) -> int:
        return self._process.wait(timeout)


def worker_command(
    *,
    python: Path,
    model_id: str,
    port: int,
    device: str,
    host: str = "127.0.0.1",
) -> List[str]:
    """The argv that starts one worker.

    Built here rather than inline at the spawn site so a test can
    assert what would be launched without launching it, and so the
    orphan sweep's command-line marker has one definition to match
    against.
    """
    return [
        str(python),
        "-m",
        "src.backends.run_worker",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
    ]


def download_command(
    *, python: Path, repo_id: str
) -> List[str]:
    """The argv that fetches one repository's weights.

    A download is spawned through the same seam as a worker, and for
    the same reason: it is long-running, it holds resources, and the
    supervisor has to be able to end it. Built here beside
    ``worker_command`` so both argv shapes are asserted in tests
    without launching anything.
    """
    return [
        str(python),
        "-m",
        "src.inference.download_main",
        "--repo",
        repo_id,
    ]


def spawn_worker(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> WorkerHandle:
    """Start a worker and return a handle to it."""
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        **_popen_orphan_guards(),
    )
    return SubprocessHandle(process)


def _popen_orphan_guards() -> Dict[str, object]:
    """Extra ``Popen`` kwargs to keep workers from being orphaned.

    On Linux, put the worker in its own session and arm PDEATHSIG.
    Elsewhere, return nothing (``preexec_fn`` is POSIX-only and the
    app targets Linux).
    """
    if sys.platform.startswith("linux"):
        return {
            "start_new_session": True,
            "preexec_fn": _set_pdeathsig,
        }
    return {}


def _set_pdeathsig() -> None:
    """Child-side: ask the kernel to SIGTERM this worker if the
    supervisor dies (Linux ``PR_SET_PDEATHSIG``).

    Belt-and-suspenders against orphaned workers holding VRAM: even if
    the supervisor is hard-killed (e.g. the desktop window closes mid
    load before the graceful stop can run), the worker is signalled.
    Best-effort and Linux-only; a failure here must not block spawn.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except Exception:  # noqa: BLE001 - best-effort orphan guard
        pass
