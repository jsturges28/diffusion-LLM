"""One writer at a time over part of the data root, in any process.

The browser launcher and the desktop app are separate supervisors
pointed at one data root, so anything that reads there, decides, and
then writes needs a lock that reaches across processes. A
``threading.Lock`` does not: the second supervisor walks straight
through it, which is how UI state lost writes until ``DATA-02``.

One mechanism for every such module, deliberately: an ``flock`` on a
sidecar file, taken after a thread lock. The kernel drops an flock
when its holder exits, however it exits, so a supervisor that dies
mid-write cannot wedge the other.

Standard library only, so a module held to it, as the run store is,
can take one.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX only; app is Linux
    fcntl = None  # type: ignore[assignment]


class DataRootLock:
    """An exclusive hold on one sidecar file under a data root.

    Each lock is named by its own sidecar, so locks over different
    things never contend: a save does not wait for a settings write.
    """

    def __init__(self, name: str) -> None:
        assert name, "a lock needs a sidecar name"
        assert "/" not in name, "the sidecar sits in the root itself"
        self._name = name
        # Queues this process's threads before any of them asks the
        # kernel, and is the only protection left on a host without
        # flock.
        self._thread_lock = threading.Lock()

    def path(self, root: Path) -> Path:
        """The sidecar this lock is taken on.

        Never the file it guards. Writers there go through
        ``os.replace``, which swaps a new inode into place, so a lock
        held on the file being replaced stops excluding anyone the
        moment the first writer finishes. The sidecar is only ever
        opened, never replaced, so every process contends on the same
        inode.
        """
        return root / self._name

    @contextlib.contextmanager
    def held(self, root: Path) -> Iterator[None]:
        """Hold the lock against every other holder, in any process.

        The thread lock first, so siblings in this process queue
        cheaply, then ``flock`` for a supervisor in another one.
        Closing the handle would release the lock on its own;
        unlocking explicitly says so.
        """
        assert isinstance(root, Path), "root must be a Path"
        root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            if fcntl is None:
                yield
                return
            with self.path(root).open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
