"""One writer at a time over part of the data root, in any process.

Strategy: every holder writes an enter line and an exit line to a
shared log while it holds the lock, with a pause between them. If two
ever held it at once, their lines interleave, so the log is the
evidence rather than any claim the lock makes about itself. Forked
processes stand in for the browser launcher and the desktop app, two
supervisors pointed at one data root, which is the case a
``threading.Lock`` cannot reach.

Passing proves the lock excludes across processes and across threads,
that exclusion comes from the sidecar file rather than from one lock
object, that differently named locks never wait on each other, and
that the sidecar is created once and never replaced, which is what
keeps every process contending on the same inode.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path
from typing import List

from src.web.data_root_lock import DataRootLock

HOLDERS = 6

# Long enough that holders overlapping would interleave their lines
# on any plausible scheduler.
HOLD_SECONDS = 0.02

LOCK = DataRootLock("test.lock")


def _hold_and_log(root: Path, holder: str) -> None:
    with LOCK.held(root):
        _log(root, f"enter {holder}")
        time.sleep(HOLD_SECONDS)
        _log(root, f"exit {holder}")


def _log(root: Path, line: str) -> None:
    with (root / "holders.log").open("a", encoding="utf-8") as log:
        log.write(line + "\n")


def _pairs(root: Path) -> List[List[str]]:
    lines = (root / "holders.log").read_text(encoding="utf-8")
    words = [line.split() for line in lines.splitlines()]
    starts = range(0, len(words), 2)
    return [words[start:start + 2] for start in starts]


def _assert_never_overlapped(root: Path, holders: int) -> None:
    pairs = _pairs(root)
    assert len(pairs) == holders
    for enter, leave in pairs:
        assert enter[0] == "enter", pairs
        assert leave[0] == "exit", pairs
        assert enter[1] == leave[1], "another holder got in between"


def test_processes_never_hold_it_at_once(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    procs = [
        context.Process(
            target=_hold_and_log, args=(tmp_path, f"process-{index}")
        )
        for index in range(HOLDERS)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)

    assert all(proc.exitcode == 0 for proc in procs)
    _assert_never_overlapped(tmp_path, HOLDERS)


def test_threads_never_hold_it_at_once(tmp_path: Path) -> None:
    threads = [
        threading.Thread(
            target=_hold_and_log, args=(tmp_path, f"thread-{index}")
        )
        for index in range(HOLDERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    _assert_never_overlapped(tmp_path, HOLDERS)


def _taken_while_held(
    root: Path, held: DataRootLock, wanted: DataRootLock
) -> bool:
    """Whether ``wanted`` can be taken while ``held`` is held."""
    taken = threading.Event()

    def take() -> None:
        with wanted.held(root):
            taken.set()

    with held.held(root):
        thread = threading.Thread(target=take)
        thread.start()
        got_in = taken.wait(timeout=0.3)
    thread.join(timeout=10)
    assert taken.is_set(), "the second lock was never granted"
    return got_in


def test_the_file_excludes_not_the_lock_object(
    tmp_path: Path,
) -> None:
    """Two locks over one sidecar exclude each other, as the old and
    the new code in two supervisors must during an upgrade, because
    neither shares the other's thread lock."""
    first = DataRootLock("shared.lock")
    second = DataRootLock("shared.lock")

    assert not _taken_while_held(tmp_path, first, second)


def test_differently_named_locks_never_wait_on_each_other(
    tmp_path: Path,
) -> None:
    """A save must not queue behind a settings write."""
    runs = DataRootLock("runs-test.lock")
    state = DataRootLock("state-test.lock")

    assert _taken_while_held(tmp_path, runs, state)


def test_the_sidecar_is_created_once_and_never_replaced(
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-yet-made"
    lock = DataRootLock("inode.lock")

    with lock.held(root):
        first = lock.path(root).stat().st_ino
    with lock.held(root):
        second = lock.path(root).stat().st_ino

    assert lock.path(root).parent == root
    assert first == second
