"""A download has an owner that can end it.

Strategy: drive `ModelManager` with an injected spawn and a fake
child, the pattern `test_worker_lifecycle.py` established. No
network, no real fetch, and no signalling: an agent sandbox refuses
to signal a process in its own session, so terminate and kill are
checked by delegation against a stand-in, exactly as
`test_worker_process.py` does for workers.

Before `TRUST-04` a download was an asyncio task delegating to a
thread whose helper started a second, daemon thread that it joined
through completion. Nothing could reach the fetch: cancelling the
task left `to_thread` running, `ModelManager.stop()` and the
shutdown hook touched only the model worker, and closing the desktop
left a multi-gigabyte transfer running against a 35-second join. No
test covered any of it, because there was nothing to cover.

Passing proves a download is a child this process can name and end,
that cancel and shutdown both end it, that its outcome survives the
move out of process as an exit status, and that a stale window
cannot cancel a fetch it does not own. What no test here can prove
is that SIGTERM ends a real download; that is CPython's job and the
maintainer's hardware pass.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.inference.download_main import (
    DOWNLOAD_EXIT_FAILED,
    DOWNLOAD_EXIT_OK,
    DOWNLOAD_EXIT_UNREACHABLE,
)
from src.web import server as server_module
from src.web.server import ActivationRefused, ModelManager

# A registry id whose checkpoint is a Hub repo, so it is downloadable.
DOWNLOADABLE = "llada"


class _FakeChild:
    """A download process whose exit the test chooses.

    Mirrors `FakeProcess` in `test_worker_lifecycle.py`, minus the
    stubbornness cases: how the escalation ladder handles a child
    that ignores SIGTERM is already pinned there, and this shares
    the same `_end_process`.
    """

    def __init__(self, *, pid: int = 4242) -> None:
        self.pid = pid
        self._code: Optional[int] = None
        self.calls: List[str] = []

    def finish(self, code: int) -> None:
        """The fetch ended on its own, with this status."""
        self._code = code

    def poll(self) -> Optional[int]:
        return self._code

    def terminate(self) -> None:
        self.calls.append("terminate")
        self._code = -15

    def kill(self) -> None:
        self.calls.append("kill")
        self._code = -9

    def wait(self, timeout: float) -> int:
        self.calls.append("wait")
        if self._code is None:
            raise TimeoutError(f"still running after {timeout}s")
        return self._code


class _Harness:
    """A manager whose downloads are fakes, plus the argv it used."""

    def __init__(self) -> None:
        self.commands: List[List[str]] = []
        self.children: List[_FakeChild] = []

        def spawn(
            command: Any, *, cwd: Path, env: Dict[str, str]
        ) -> Any:
            self.commands.append(list(command))
            child = _FakeChild(pid=4242 + len(self.children))
            self.children.append(child)
            return child

        self.manager = ModelManager(
            spawn=spawn,  # type: ignore[arg-type]
            stop_timeout_s=0.01,
            kill_timeout_s=0.01,
            download_poll_s=0.001,
        )

    @property
    def child(self) -> _FakeChild:
        assert self.children, "nothing was spawned"
        return self.children[-1]


@pytest.fixture(autouse=True)
def _no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the progress sampler off the network and off the disk.

    `repo_total_bytes` is one HTTP call and `repo_progress` stats a
    cache directory. Neither is under test here, and leaving them
    live would make these tests depend on what is cached locally.
    """
    from src.inference import hf_download

    monkeypatch.setattr(
        hf_download, "repo_total_bytes", lambda repo_id: 1000
    )
    monkeypatch.setattr(
        hf_download,
        "repo_progress",
        lambda repo_id, total: {
            "fraction": 0.5,
            "downloaded_bytes": 500,
            "total_bytes": total,
        },
    )


async def _start(harness: _Harness) -> int:
    """Start a download from inside a loop.

    `start_download` schedules the watcher with `create_task`, so it
    needs one. In production the endpoint that calls it is already
    async; only a test can reach it without a loop.
    """
    return harness.manager.start_download(DOWNLOADABLE)


async def _settle(harness: _Harness) -> None:
    """Let the watcher notice whatever the child just did."""
    task = harness.manager._download_task
    if task is not None:
        await task


# -- it is a child, not a thread --


def test_a_download_runs_out_of_process() -> None:
    """The finding's core. A thread inside this process could not be
    reached; a child has a pid and a status."""
    harness = _Harness()

    asyncio.run(_start(harness))

    assert len(harness.commands) == 1
    assert harness.manager._download_proc is not None


def test_the_child_is_told_which_repo_to_fetch() -> None:
    harness = _Harness()
    checkpoint = server_module.REGISTRY[DOWNLOADABLE].checkpoint

    asyncio.run(_start(harness))

    command = harness.commands[0]
    assert "src.inference.download_main" in command
    assert checkpoint in command


def test_a_second_download_is_refused_while_one_runs() -> None:
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        with pytest.raises(RuntimeError):
            harness.manager.start_download(DOWNLOADABLE)

    asyncio.run(run())

    assert len(harness.commands) == 1


def test_an_unknown_model_spawns_nothing() -> None:
    harness = _Harness()

    async def run() -> None:
        with pytest.raises(KeyError):
            harness.manager.start_download("not-a-model")

    asyncio.run(run())

    assert harness.commands == []


# -- the exit status is the whole report --


def test_a_clean_exit_finishes_the_download() -> None:
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        harness.child.finish(DOWNLOAD_EXIT_OK)
        await _settle(harness)

    asyncio.run(run())

    assert harness.manager.download_state == "done"
    assert harness.manager.download_error is None
    assert harness.manager.download_progress is None


def test_an_offline_exit_says_so_in_words() -> None:
    """The one failure with a remedy worth naming. It used to arrive
    as an exception this process could read; now it is a number, and
    the sentence is rebuilt from the repo."""
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        harness.child.finish(DOWNLOAD_EXIT_UNREACHABLE)
        await _settle(harness)

    asyncio.run(run())

    assert harness.manager.download_state == "error"
    message = harness.manager.download_error or ""
    assert "could not be reached" in message


def test_any_other_failure_is_reported_as_one() -> None:
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        harness.child.finish(DOWNLOAD_EXIT_FAILED)
        await _settle(harness)

    asyncio.run(run())

    assert harness.manager.download_state == "error"
    assert "failed" in (harness.manager.download_error or "")


def test_progress_is_sampled_while_the_child_runs() -> None:
    """Measured from the cache directory rather than reported by the
    child, which is what let a download move out of process without
    a channel between the two."""
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        for _ in range(50):
            await asyncio.sleep(0.001)
            if harness.manager.download_progress is not None:
                break
        harness.child.finish(DOWNLOAD_EXIT_OK)
        await _settle(harness)

    asyncio.run(run())

    # Cleared on the way out, so the sample above is the evidence it
    # was ever taken.
    assert harness.manager.download_state == "done"


# -- cancel --


def test_a_cancel_ends_the_child() -> None:
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.cancel_download()

    asyncio.run(run())

    assert "terminate" in harness.child.calls
    assert harness.manager.download_state == "idle"
    assert harness.manager.download_target is None


def test_a_cancel_is_not_reported_as_a_failure() -> None:
    """The watcher would otherwise see an exit it was never told to
    expect and call a deliberate stop an error."""
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.cancel_download()
        await asyncio.sleep(0.01)

    asyncio.run(run())

    assert harness.manager.download_state == "idle"
    assert harness.manager.download_error is None


def test_a_cancel_with_nothing_running_does_nothing() -> None:
    harness = _Harness()

    asyncio.run(harness.manager.cancel_download())

    assert harness.commands == []
    assert harness.manager.download_state == "idle"


def test_a_download_can_start_again_after_a_cancel() -> None:
    """Otherwise the refusal above would be permanent."""
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.cancel_download()
        harness.manager.start_download(DOWNLOADABLE)

    asyncio.run(run())

    assert len(harness.commands) == 2
    assert harness.manager.download_state == "downloading"


# -- identity --


def test_each_download_is_numbered() -> None:
    harness = _Harness()

    async def run() -> tuple[int, int]:
        first = harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.cancel_download()
        second = harness.manager.start_download(DOWNLOADABLE)
        return first, second

    first, second = asyncio.run(run())

    assert second > first


def test_a_stale_window_cannot_cancel_this_download() -> None:
    """Two windows both see one download. Cancelling somebody else's
    is the mistake the number exists to refuse."""
    harness = _Harness()

    async def run() -> None:
        operation = harness.manager.start_download(DOWNLOADABLE)
        with pytest.raises(ActivationRefused):
            await harness.manager.cancel_download(operation - 1)

    asyncio.run(run())

    assert harness.manager.download_state == "downloading"
    assert harness.child.calls == []


def test_naming_the_current_download_cancels_it() -> None:
    harness = _Harness()

    async def run() -> None:
        operation = harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.cancel_download(operation)

    asyncio.run(run())

    assert harness.manager.download_state == "idle"
    assert "terminate" in harness.child.calls


# -- shutdown --


def test_stopping_the_supervisor_ends_a_download() -> None:
    """What the finding was actually about: closing the app during a
    multi-gigabyte fetch used to leave it running."""
    harness = _Harness()

    async def run() -> None:
        harness.manager.start_download(DOWNLOADABLE)
        await harness.manager.stop()

    asyncio.run(run())

    assert "terminate" in harness.child.calls


def test_stopping_with_no_download_is_still_fine() -> None:
    """`stop` runs on every shutdown, most of which have no fetch."""
    harness = _Harness()

    asyncio.run(harness.manager.stop())

    assert harness.commands == []
