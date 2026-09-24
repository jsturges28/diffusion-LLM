"""Suite-wide isolation from the machine the tests run on.

Only one thing so far, and it earns a file of its own because the
failure it prevents is confusing and intermittent.

The supervisor holds a host-wide lease while a model is resident
(`src/web/model_lease.py`), so that a browser instance and a desktop
instance cannot both load into one card. Tests activate models too,
and they do it on the same machine the maintainer runs the app on.
Without this fixture, running the suite with the app open means the
suite's first activation asks for a lease the app is holding, gets
refused, and a dozen tests that have nothing to do with leases fail
with a message about another instance. Worse, they pass again once the
app is closed, which is the shape of a bug nobody can reproduce.

Pointing the runtime directory somewhere disposable gives the suite
its own lease file. Nothing else in the codebase reads this variable,
so the blast radius is exactly the lease.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

from src.web.model_lease import RUNTIME_DIR_ENV


@pytest.fixture(autouse=True)
def isolated_model_lease(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Give each test a lease file nobody else can want.

    Per test rather than per session, for a second reason beyond the
    running app. A manager keeps its claim until its worker is
    finalized, which is correct for a supervisor and wrong for a test:
    tests build managers, activate, and abandon them without stopping.
    Sharing one lease file across the session would mean the first
    test to activate held it for the rest of the run and every later
    activation was refused, which is a failure with nothing to do with
    what those tests are about.

    Set on ``os.environ`` rather than through monkeypatch because the
    lease reads it when a manager first uses it, which may be inside
    code the test does not own. Restored afterwards so a run leaves
    the environment as it found it.
    """
    previous = os.environ.get(RUNTIME_DIR_ENV)
    os.environ[RUNTIME_DIR_ENV] = str(
        tmp_path_factory.mktemp("runtime")
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(RUNTIME_DIR_ENV, None)
        else:
            os.environ[RUNTIME_DIR_ENV] = previous
