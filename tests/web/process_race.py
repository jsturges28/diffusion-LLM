"""The process context for tests that race two supervisors.

The browser launcher and the desktop app are separate processes over
one data root, so the locks that keep them apart are tested with real
processes. Those start from ``forkserver`` rather than ``fork``. By
the time the full suite reaches them, earlier tests have left threads
running, and forking a process with threads copies into the child
every lock one of them held at that instant, with no thread left to
release it. Python 3.12 warns about this on every such fork. A
forkserver child is forked instead from a server that has no other
threads.

What that costs: a child imports the test module afresh rather than
copying the test process, so it sees none of the test's patches, and
it reads the environment as it was when the server started. A race
that needs a step paused therefore installs the pause in the child.
Everything a child is handed is pickled, so targets live at module
level and events come from this context.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing.context import ForkServerContext


def race_context() -> ForkServerContext:
    """Where every cross-process race test gets its processes."""
    context = multiprocessing.get_context("forkserver")
    assert isinstance(context, ForkServerContext)
    return context
