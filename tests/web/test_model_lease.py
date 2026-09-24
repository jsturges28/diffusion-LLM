"""Only one process at a time may hold a model resident.

Strategy: contention is exercised inside this process, because a flock
lives on the open file description rather than on the process, so two
``PrimaryModelLease`` objects over one path genuinely fight. That
keeps the whole contention suite at milliseconds with no subprocess
machinery, and it is the same code path two supervisors take.

Exactly one test spends a subprocess, and it is the one that has to:
the stale-safety guarantee is that the *kernel* releases the lock when
the holder dies, which cannot be observed without a holder that really
dies. It exits through ``os._exit`` so no cleanup, no ``finally`` and
no atexit hook can run, which is what a crash looks like.

The trap this file exists to pin is the asymmetry between the lock and
the file. The kernel releases the lock on death and does not touch the
contents, so a crashed owner leaves its pid behind forever. Reading
those contents without first having been refused the lock would report
a process that died days ago as the live owner, in a message telling
somebody to go and close a window that is not open.

Passing proves a second claimant is refused while a first holds, that
it can name the holder, that a dead holder blocks nobody, that stale
or unparseable contents are never reported as an owner, and that the
lease declines to exist rather than blocking every activation when its
directory cannot be written.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.web import model_lease
from src.web.model_lease import (
    LEASE_FILE_NAME,
    RUNTIME_DIR_ENV,
    RUNTIME_DIR_FALLBACK,
    PrimaryModelLease,
    lease_path,
)

BROWSER = {"pid": 4001, "url": "http://127.0.0.1:8000"}
DESKTOP = {"pid": 4002, "url": "http://127.0.0.1:8760"}


@pytest.fixture()
def lease_file(tmp_path: Path) -> Path:
    """A lock path of this test's own, never the real one."""
    return tmp_path / LEASE_FILE_NAME


# -- where the file lives --


def test_the_runtime_directory_is_used_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Per-user runtime state, which is the honest scope: a lease
    cannot govern another account's processes."""
    monkeypatch.setenv(RUNTIME_DIR_ENV, str(tmp_path))

    assert lease_path() == (tmp_path / LEASE_FILE_NAME).resolve()


def test_tmp_is_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No runtime directory happens over plain ssh and in
    containers, where the app should still enforce one resident."""
    monkeypatch.delenv(RUNTIME_DIR_ENV, raising=False)

    assert lease_path().parent == RUNTIME_DIR_FALLBACK


def test_a_blank_runtime_directory_is_not_a_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary. An exported-but-empty variable is absence, not a
    request to write the lock into the current directory."""
    monkeypatch.setenv(RUNTIME_DIR_ENV, "   ")

    assert lease_path().parent == RUNTIME_DIR_FALLBACK


def test_the_name_says_which_resource_it_guards() -> None:
    """Stated because the finding asks for it: a later concurrent
    utility worker needs room for a separately named budget beside
    this one, which a lock called "app.lock" would not leave."""
    assert "primary-model" in LEASE_FILE_NAME


# -- contention --


def test_one_claimant_gets_it(lease_file: Path) -> None:
    lease = PrimaryModelLease(lease_file)

    assert lease.acquire(BROWSER) is True
    assert lease.held is True


def test_a_second_claimant_is_refused(lease_file: Path) -> None:
    """The finding in one assertion. Two supervisors, one card."""
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)

    second = PrimaryModelLease(lease_file)

    assert second.acquire(DESKTOP) is False
    assert second.held is False


def test_the_refused_claimant_can_name_the_holder(
    lease_file: Path
) -> None:
    """So the refusal can tell somebody which window to go to,
    rather than only that they cannot have it."""
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)
    second = PrimaryModelLease(lease_file)
    second.acquire(DESKTOP)

    assert second.owner() == BROWSER


def test_releasing_hands_it_over(lease_file: Path) -> None:
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)
    second = PrimaryModelLease(lease_file)
    assert second.acquire(DESKTOP) is False

    first.release()

    assert second.acquire(DESKTOP) is True


def test_acquiring_twice_is_not_self_competition(
    lease_file: Path
) -> None:
    """A supervisor switching models is not a rival. Making the
    caller track whether it already holds the lease would put that
    bookkeeping in the wrong place."""
    lease = PrimaryModelLease(lease_file)
    lease.acquire(BROWSER)

    assert lease.acquire(BROWSER) is True
    assert lease.held is True


def test_a_switch_updates_the_recorded_model(
    lease_file: Path
) -> None:
    """The second acquire rewrites the note, so a reader is told
    which model is resident now rather than which was first."""
    lease = PrimaryModelLease(lease_file)
    lease.acquire({**BROWSER, "model": "LLaDA-8B-Instruct"})

    lease.acquire({**BROWSER, "model": "SmolLM3-3B"})

    onlooker = PrimaryModelLease(lease_file)
    onlooker.acquire(DESKTOP)
    owner = onlooker.owner()
    assert owner is not None
    assert owner["model"] == "SmolLM3-3B"


def test_releasing_what_was_never_held_is_harmless(
    lease_file: Path
) -> None:
    """Called from the one terminal path, which runs for workers that
    never started."""
    PrimaryModelLease(lease_file).release()


# -- the guarantee that needs a real death --


def test_a_dead_holder_blocks_nobody(lease_file: Path) -> None:
    """The stale-safety clause, and the reason this is a file lock.

    The holder exits through os._exit, so nothing it might have
    registered runs: no finally, no atexit, no signal handler. The
    lock goes because the kernel drops it with the process, which
    means there is no cleanup path in this module that has to survive
    a crash in order for the next launch to work.
    """
    holder = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys\n"
            "f = open(sys.argv[1], 'a+')\n"
            "fcntl.flock(f, fcntl.LOCK_EX)\n"
            "f.write('{\"pid\": 1}')\n"
            "f.flush()\n"
            "os._exit(9)\n",
            str(lease_file),
        ],
        capture_output=True,
        timeout=30,
    )
    assert holder.returncode == 9, holder.stderr

    assert PrimaryModelLease(lease_file).acquire(DESKTOP) is True


def test_a_dead_holder_is_never_named_as_the_owner(
    lease_file: Path
) -> None:
    """The trap. The kernel released the lock and left the contents,
    so the file still says pid 1 while nothing holds it.

    Acquiring first is what makes this safe: we win the lock, rewrite
    the note, and the stale line is gone before anybody could have
    quoted it. A reader that consulted the file without being refused
    would have told the user to close a window that is not open.
    """
    lease_file.write_text('{"pid": 1, "url": "http://stale"}')

    lease = PrimaryModelLease(lease_file)
    assert lease.acquire(DESKTOP) is True

    assert json.loads(lease_file.read_text()) == DESKTOP


# -- what it refuses to guess --


def test_unparseable_contents_name_nobody(lease_file: Path) -> None:
    """The holder rewrites in place, so a reader can catch a partial
    write. Reporting no owner degrades the message; reporting a
    guessed one would be a wrong instruction."""
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)
    lease_file.write_text('{"pid": 40')

    second = PrimaryModelLease(lease_file)
    second.acquire(DESKTOP)

    assert second.owner() is None


def test_empty_contents_name_nobody(lease_file: Path) -> None:
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)
    lease_file.write_text("")

    second = PrimaryModelLease(lease_file)
    second.acquire(DESKTOP)

    assert second.owner() is None


def test_a_non_object_names_nobody(lease_file: Path) -> None:
    """Valid JSON is not the same as a description of an owner."""
    first = PrimaryModelLease(lease_file)
    first.acquire(BROWSER)
    lease_file.write_text("[1, 2, 3]")

    second = PrimaryModelLease(lease_file)
    second.acquire(DESKTOP)

    assert second.owner() is None


def test_a_missing_file_names_nobody(tmp_path: Path) -> None:
    lease = PrimaryModelLease(tmp_path / "never-created.lock")

    assert lease.owner() is None


# -- when it cannot exist at all --


def test_an_unwritable_directory_does_not_block_activation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A deliberate choice, and the loud kind.

    If the runtime directory cannot be written there is no lease to
    take, and the options are to refuse every activation or to run
    without the cross-process guarantee. Refusing would break the app
    on a host where nothing is wrong with the model; running logs a
    warning saying exactly which guarantee is missing.
    """
    unwritable = tmp_path / "locked-out"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    lease = PrimaryModelLease(unwritable / LEASE_FILE_NAME)
    try:
        with caplog.at_level("WARNING"):
            granted = lease.acquire(BROWSER)
    finally:
        unwritable.chmod(0o700)

    assert granted is True
    assert lease.held is False
    assert "one resident" in caplog.text


def test_the_real_path_is_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing about this may depend on the working directory, for
    the same reason the data root may not."""
    monkeypatch.setenv(RUNTIME_DIR_ENV, str(tmp_path))

    assert lease_path().is_absolute()


def test_the_module_needs_no_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small and policy-only, like data_root. Asserted so a later
    convenience import cannot quietly pull FastAPI into a module the
    desktop launcher may one day want to read on its own."""
    import importlib

    monkeypatch.setitem(sys.modules, "src.web.server", None)

    importlib.reload(model_lease)

    assert model_lease.LEASE_FILE_NAME == LEASE_FILE_NAME


def test_the_lock_is_exclusive_not_advisory_only(
    lease_file: Path
) -> None:
    """Paired with the refusal above, from the other side: a raw
    flock must also be unable to take what the lease holds, so the
    guarantee does not depend on everybody using this class."""
    lease = PrimaryModelLease(lease_file)
    lease.acquire(BROWSER)

    with lease_file.open("a+") as outsider, pytest.raises(OSError):
        fcntl.flock(outsider, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_the_path_defaults_to_the_runtime_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constructed without an argument, which is how the supervisor
    builds it."""
    monkeypatch.setenv(RUNTIME_DIR_ENV, str(tmp_path))

    assert PrimaryModelLease().path.parent == tmp_path.resolve()


def test_the_owner_note_is_json_a_person_can_read(
    lease_file: Path
) -> None:
    """The file is a diagnostic as well as a lock: somebody looking
    at it with `cat` after a confusing refusal should be able to."""
    lease = PrimaryModelLease(lease_file)

    lease.acquire({**BROWSER, "model": "LLaDA-8B-Instruct"})

    written = json.loads(lease_file.read_text())
    assert written["pid"] == BROWSER["pid"]
    assert written["url"] == BROWSER["url"]
    assert lease_file.name == LEASE_FILE_NAME
