"""Who, on this machine, is allowed to hold a model resident.

"One model at a time" was enforced by one in-memory ``ModelManager``,
which is only one supervisor's promise. The two entry points bind
different ports on purpose (``main.py`` 8000, ``desktop.py`` 8760), so
running both gives two managers, two startup sweeps, and two
independent VRAM pre-flights that can each pass before the other's
allocation is visible. Two workers then launch into a device sized for
one, and either window can only switch or stop its own.

The lease guards *residency*, not the process. A second supervisor is
welcome to run: it serves its pages, browses Analytics, and reads
saved runs. What it cannot do is load a model while somebody else
has one,
and when it refuses it says who to go and talk to.

Its own module for the same reason as ``data_root``: it is small, it
is policy, and nothing here needs the server.

## Why a file lock

``fcntl.flock`` is released by the kernel when the holding process
dies, which is the whole of the stale-safety requirement. No pid to
validate, no liveness probe, no cleanup path that has to run on a
process that has already crashed, and no window in which a recycled
pid makes a dead owner look alive.

What flock does not carry is identity: it says "taken", not "taken by
whom". So the owner writes that into the file's contents, and a loser
reads them to name it.

## The one rule that is not obvious

**Never read the contents to name an owner without having failed to
acquire the lock first.** The kernel releases the lock on death but
does not touch the file, so a crashed owner leaves its own pid sitting
there indefinitely. Acquiring first and rewriting is what keeps a
stale line from being reported as a live instance; ``owner`` says so
too, and its only caller is the refusal path.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Where runtime state belongs on a Linux desktop, and it is per-user:
# a lease cannot govern another account's processes and should not
# pretend to. Cleared on logout, which is correct for something whose
# meaning expires with the session.
RUNTIME_DIR_ENV = "XDG_RUNTIME_DIR"

# The fallback when the session provides no runtime directory, which
# happens over plain ssh and in containers.
RUNTIME_DIR_FALLBACK = Path("/tmp")

# Named for the resource it guards rather than for the application,
# because the finding this closes asks that a future deliberately
# concurrent utility worker be able to take a separately named budget.
# "diffusion-llm" scopes it to this app; "primary-model" says which of
# the app's resources, leaving room for a second name beside it.
LEASE_FILE_NAME = "diffusion-llm-primary-model.lock"


def lease_path() -> Path:
    """The lock file for the primary model, as an absolute path."""
    raw = os.environ.get(RUNTIME_DIR_ENV, "").strip()
    directory = Path(raw) if raw else RUNTIME_DIR_FALLBACK
    resolved = (directory / LEASE_FILE_NAME).resolve()
    assert resolved.is_absolute(), "the lease path must be absolute"
    return resolved


class PrimaryModelLease:
    """This process's claim on being the one with a model loaded.

    One instance per supervisor, held open for as long as the claim
    lasts: the lock lives on the open file description, so closing the
    file is what releases it.

    Acquiring twice is a no-op rather than an error. A supervisor that
    already holds the lease is switching models, not competing with
    itself, and making the caller track that would put the bookkeeping
    in the wrong place.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        # Kept unresolved until first use. The supervisor builds its
        # manager at import, so resolving here would bake in whatever
        # the environment said before anything had a chance to set it,
        # which is the same trap the data root's docstring describes.
        # It also lets the test suite point the lease somewhere
        # harmless instead of fighting a running app for the real one.
        self._explicit = path
        self._resolved: Optional[Path] = None
        self._handle: Optional[Any] = None

    @property
    def path(self) -> Path:
        if self._resolved is None:
            self._resolved = (
                self._explicit
                if self._explicit is not None
                else lease_path()
            )
        return self._resolved

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self, owner: Dict[str, Any]) -> bool:
        """Claim it, and record who we are. True when it is ours.

        ``owner`` is written for somebody else to read, so it should
        carry what a person needs to act: which process, and where its
        window is.
        """
        assert isinstance(owner, dict), "owner must be a dict"
        if self.held:
            self.describe(owner)
            return True
        try:
            # Not a context manager, deliberately: the lock lives on
            # this open file description, so the claim lasts exactly
            # as long as the handle stays open. Closing it at the end
            # of a block is the one thing this must not do.
            handle = self.path.open(  # noqa: SIM115
                "a+", encoding="utf-8"
            )
        except OSError:
            # An unwritable runtime directory. Refusing every
            # activation over this would be worse than the race it
            # prevents, so the lease declines to exist and says so
            # loudly once.
            logger.warning(
                "no model lease at %s; cannot enforce one resident"
                " model across processes",
                self.path,
                exc_info=True,
            )
            return True
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Somebody else has it. Their contents are live, because a
            # dead holder's lock would not have refused us.
            handle.close()
            return False
        self._handle = handle
        self.describe(owner)
        return True

    def describe(self, owner: Dict[str, Any]) -> None:
        """Rewrite who we are, for a later reader.

        Called on every acquire, including the second one, so the
        recorded model follows a switch instead of naming whichever
        model happened to be first.

        In place rather than through a temporary file and a rename: a
        rename would give the path a new inode and leave this process
        holding a lock on a file nobody else will ever open.
        """
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            self._handle.truncate()
            json.dump(owner, self._handle)
            self._handle.flush()
        except OSError:
            # The claim is the lock, not the note beside it. A failed
            # write costs a later reader some detail and nothing else.
            logger.warning(
                "could not record the lease owner", exc_info=True
            )

    def release(self) -> None:
        """Give it up. Safe to call when it was never held."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            # Closing releases it regardless, which is the guarantee
            # that matters; an explicit unlock is only tidier.
            logger.warning(
                "could not unlock the lease", exc_info=True
            )
        try:
            handle.close()
        except OSError:
            logger.warning("could not close the lease", exc_info=True)

    def owner(self) -> Optional[Dict[str, Any]]:
        """Who holds it, for a refusal message.

        **Only meaningful after ``acquire`` returned False.** The
        contents outlive a crashed holder, so reading them without
        having been refused first would name a process that died days
        ago. Being refused is the proof that what is in the file
        belongs to something alive.

        None when there is nothing readable to report, so a caller
        says "another instance" rather than inventing a pid.
        """
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        if raw.strip() == "":
            return None
        try:
            parsed = json.loads(raw)
        except ValueError:
            # A partial write, since the holder rewrites in place.
            # Nothing to report is the honest answer; the caller's
            # message degrades rather than asserting a wrong pid.
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
