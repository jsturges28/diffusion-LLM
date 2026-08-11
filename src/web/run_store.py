"""Ownership of the saved-run directory: resolve, write, delete.

The supervisor used to do all of this inline, so path validation,
writing, and deletion had each drifted into their own slightly
different shape inside a two-thousand-line application module. Three
copies of the same traversal guard existed and a fourth call site had
none. This module is the one place that decides what a run directory
is and how one comes into or goes out of existence.

**It imports only the standard library, deliberately.** No FastAPI, no
torch, no Pydantic. That is not tidiness: it is what lets the tests
race publication in threads, inject write failures, and check every
traversal case without starting an app or loading a model. A test
enforces the constraint, because an import added in a hurry would
quietly take that ability away.

The caller keeps the HTTP shapes. This module speaks ``RunBundle``,
which is plain serializable content, and the supervisor translates
between that and its request models.

A run is published by moving its ``metadata.json`` into place last.
Every reader decides whether a directory is a run by looking for that
file, so a bundle appears whole or does not appear at all. See
``publish`` for why that beats renaming the directory.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# The file whose presence makes a directory a run. Every reader in the
# app already treats a folder without it as not-a-run.
METADATA_NAME = "metadata.json"

FINAL_TEXT_NAME = "final.txt"
HISTORY_NAME = "history.txt"

# Optional per-run signal files, keyed by the bundle attribute that
# supplies them. The order is fixed so a bundle is written the same
# way every time, which is what lets a test inject a failure at a
# named position and mean the same thing on every run.
SIDECAR_NAMES = (
    ("frame_tokens", "tokens.json"),
    ("original_frame_tokens", "original_tokens.json"),
    ("alternatives", "alternatives.json"),
    ("original_alternatives", "original_alternatives.json"),
)

# Working directories under the data root that are not runs.
# Dot-prefixed so ``is_run_dir`` skips them by the same rule it uses
# for anything else the user drops in there.
STAGING_DIR_NAME = ".staging"
TRASH_DIR_NAME = ".trash"

# Bumped on every write of a run, so a second editor holding an older
# view can be told its base moved. Runs saved before revisions existed
# have no such field and read as 0, which is why the whole corpus is
# editable without being rewritten.
REVISION_KEY = "revision"


class RunNotFoundError(FileNotFoundError):
    """No run by that id.

    Subclasses the builtin so the routes that already answer a
    missing run with a 404 keep working untouched. Naming the case
    separately is for callers that want to tell it apart from an
    unrelated missing file.
    """


class InvalidRunIdError(ValueError):
    """A run id that does not name a direct child of the data root.

    Subclasses ``ValueError`` for the same reason: the routes that
    turn a bad id into a 400 already catch that.
    """


class RevisionConflictError(Exception):
    """Someone else wrote this run since the caller last read it.

    Its own type, not a ``ValueError``, because the answer is a 409
    and a retry from a fresh read, which is a different conversation
    with the client than a malformed request.
    """

    def __init__(self, run_id: str, expected: int, actual: int):
        super().__init__(
            f"run {run_id} has moved on: expected revision"
            f" {expected}, found {actual}"
        )
        self.run_id = run_id
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class RunBundle:
    """Everything one saved run consists of, ready to write.

    Frozen because a bundle describes a finished run; anything that
    wants to change one builds another. Sidecars are ``None`` when the
    run has no such signal, which is distinct from an empty list:
    absent means the run never captured it, and the readers rely on
    that distinction.
    """

    metadata: Dict[str, Any]
    final_text: str
    frames: List[str]
    frame_tokens: Optional[List[Any]] = None
    original_frame_tokens: Optional[List[Any]] = None
    alternatives: Optional[List[Any]] = None
    original_alternatives: Optional[List[Any]] = None


def resolve_run_dir(root: Path, run_id: str) -> Path:
    """The directory for an existing run, or raise.

    The one guarded resolver. Every caller that used to build
    ``root / run_id`` by hand goes through this, including the metrics
    endpoint, which had no guard at all and would happily follow
    ``../`` out of the data root.

    Two distinct failures, because callers answer them differently: a
    malformed id is a client error worth a 400, while a valid id for a
    run that is not there is a 404.
    """
    assert isinstance(run_id, str), "run id must be a string"
    results_root = root.resolve()
    run_dir = (results_root / run_id).resolve()
    if run_dir.parent != results_root:
        raise InvalidRunIdError(f"invalid run id: {run_id}")
    if not run_dir.is_dir():
        raise RunNotFoundError(f"Run not found: {run_id}")
    return run_dir


def is_run_dir(path: Path) -> bool:
    """Whether a path is a saved run, for listing and counting.

    Metadata is the test, not the directory, because that is what
    every reader in the app already uses. A folder without it is a
    half-written save or something the user dropped there, and
    treating it as a run is how a partial save becomes visible.
    """
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    return (path / METADATA_NAME).is_file()


def list_run_ids(root: Path) -> List[str]:
    """Ids of every saved run, unsorted."""
    if not root.is_dir():
        return []
    return [
        child.name
        for child in root.iterdir()
        if is_run_dir(child)
    ]


def allocate(root: Path, model_id: str) -> str:
    """Reserve a unique run id and create its empty directory.

    The id keeps the human form it has always had, because it is shown
    in the Analytics table, the save status line, and the delete
    confirmation. A numeric suffix appears only when two runs of the
    same model really do land in the same second, which is the case
    that used to have them share one folder and interleave.

    Reservation is an exclusive ``mkdir``, so the filesystem picks the
    winner and two callers racing the same name cannot both proceed.
    The reserved directory holds no metadata, so no reader counts it
    as a run until ``publish`` puts one there.
    """
    assert isinstance(model_id, str) and model_id, "model id required"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{timestamp}_{model_id.replace('/', '_')}"
    for attempt in range(_ALLOCATE_MAX_ATTEMPTS):
        run_id = base if attempt == 0 else f"{base}-{attempt + 1}"
        try:
            (root / run_id).mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id
    raise RuntimeError(
        f"could not allocate a run id for {base} after"
        f" {_ALLOCATE_MAX_ATTEMPTS} attempts"
    )


# A second's worth of same-model saves. Reaching this means something
# is generating runs in a loop, which deserves an error rather than an
# unbounded search (TigerStyle: put a limit on everything).
_ALLOCATE_MAX_ATTEMPTS = 64


def stage(
    root: Path, run_id: str, bundle: RunBundle
) -> Path:
    """Write a complete bundle to a staging directory, and return it.

    The staging directory is private to this attempt, not shared by
    run id. Two callers replacing one run must not write into the
    same scratch space: the first version of this keyed staging on
    the run id alone, and a concurrent pair wiped each other's files
    mid-write.

    Nothing written here is visible to a reader. A failure part way
    through leaves an orphaned staging directory, which is inert: it
    sits under a dot-prefixed parent that listing skips.
    """
    staging = root / STAGING_DIR_NAME / f"{run_id}.{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        _write_json(
            staging / METADATA_NAME, bundle.metadata, indent=2
        )
        (staging / FINAL_TEXT_NAME).write_text(
            bundle.final_text, encoding="utf-8"
        )
        _write_history(staging / HISTORY_NAME, bundle.frames)
        for attribute, filename in SIDECAR_NAMES:
            payload = getattr(bundle, attribute)
            if payload is not None:
                _write_json(staging / filename, payload)
    except Exception:
        # Clean up after itself so a caller cannot leak scratch space
        # by failing before it has a path to clean.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def publish(root: Path, run_id: str, staging: Path) -> Path:
    """Move a staged bundle into place, metadata last.

    Metadata last is the whole mechanism. Every reader decides whether
    a directory is a run by looking for that file, so moving it in as
    the final step means a run appears complete or does not appear.
    There is no moment at which Analytics can see a bundle whose other
    files have not landed.

    Renaming the whole staging directory would be the obvious way to
    publish, and it is not available: the destination already exists
    as the reservation from ``allocate``, and renaming onto a
    non-empty directory fails. Removing the reservation first would
    open a window where the name is free for another caller to take.
    Moving file by file has no such window, and the last move is
    still a single atomic rename.

    Any file left from a previous bundle is removed first, so a
    replacement cannot inherit a sidecar the new one omitted.
    """
    if not (staging / METADATA_NAME).is_file():
        raise RunNotFoundError(
            f"nothing staged to publish for {run_id}"
        )
    target = root / run_id
    target.mkdir(parents=True, exist_ok=True)
    for existing in target.iterdir():
        if existing.is_file():
            existing.unlink()

    for item in sorted(staging.iterdir()):
        if item.name != METADATA_NAME:
            item.replace(target / item.name)
    (staging / METADATA_NAME).replace(target / METADATA_NAME)

    staging.rmdir()
    return target


def save(
    root: Path,
    bundle: RunBundle,
    *,
    model_id: str,
    run_id: Optional[str] = None,
    expected_revision: Optional[int] = None,
) -> Tuple[str, int]:
    """Publish a run, new or replacing, returning id and revision.

    The one entry point a caller needs. ``run_id`` names a run to
    replace, in which case ``expected_revision`` must match what is on
    disk; the check and the publication both happen here so a caller
    cannot read a revision, be overtaken, and then write anyway.

    Nothing is published if anything raises. The staging directory is
    discarded on failure, and the reservation for a new run is left
    behind empty, which no reader counts.
    """
    if run_id is None:
        return _publish_new(root, bundle, model_id)
    # Serialized against other replacements, because the revision
    # check and the write that acts on it have to be one step. Two
    # callers reading the same revision, both passing, and both
    # publishing is precisely the last-writer-wins the check exists
    # to prevent. One lock for all replacements rather than one per
    # run: a replacement is a user pressing Confirm, so there is no
    # contention worth a more complicated structure.
    with _REPLACE_LOCK:
        return _publish_replacement(
            root, bundle, run_id, expected_revision
        )


# Guards read-revision-then-publish. In-process only, which is the
# right scope today because one supervisor owns the data root;
# `LIFE-05` is where a second one becomes possible.
_REPLACE_LOCK = threading.Lock()


def _publish_new(
    root: Path, bundle: RunBundle, model_id: str
) -> Tuple[str, int]:
    run_id = allocate(root, model_id)
    _stage_and_publish(root, run_id, bundle, revision=1)
    return run_id, 1


def _publish_replacement(
    root: Path,
    bundle: RunBundle,
    run_id: str,
    expected_revision: Optional[int],
) -> Tuple[str, int]:
    actual = read_revision(root, run_id)
    expected = (
        actual if expected_revision is None else expected_revision
    )
    if expected != actual:
        raise RevisionConflictError(run_id, expected, actual)
    revision = actual + 1
    _stage_and_publish(root, run_id, bundle, revision=revision)
    return run_id, revision


def _stage_and_publish(
    root: Path, run_id: str, bundle: RunBundle, *, revision: int
) -> None:
    """Write the bundle aside, then move it in. All or nothing."""
    stamped = dict(bundle.metadata)
    stamped[REVISION_KEY] = revision
    staged = RunBundle(
        metadata=stamped,
        final_text=bundle.final_text,
        frames=bundle.frames,
        frame_tokens=bundle.frame_tokens,
        original_frame_tokens=bundle.original_frame_tokens,
        alternatives=bundle.alternatives,
        original_alternatives=bundle.original_alternatives,
    )
    staging = stage(root, run_id, staged)
    try:
        publish(root, run_id, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_revision(root: Path, run_id: str) -> int:
    """The revision of a saved run; 0 when it predates them.

    Every run saved before revisions existed reads as 0, which is what
    lets the whole existing corpus be edited without being rewritten.
    Unreadable metadata also reads as 0 rather than raising: the
    caller is about to replace the file anyway, and refusing to edit a
    run because its old metadata is corrupt would strand it.
    """
    run_dir = resolve_run_dir(root, run_id)
    try:
        raw = json.loads(
            (run_dir / METADATA_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return 0
    if not isinstance(raw, dict):
        return 0
    revision = raw.get(REVISION_KEY)
    if isinstance(revision, bool):
        return 0
    if isinstance(revision, int):
        return revision
    return 0


def delete(root: Path, run_id: str) -> None:
    """Remove a saved run without letting a reader see it torn.

    The visible directory is renamed out of the namespace first, so a
    concurrent read finds either a whole run or nothing at all. Only
    then is the content removed, which is the slow part and the part
    that used to happen in place.
    """
    run_dir = resolve_run_dir(root, run_id)
    trash_root = root / TRASH_DIR_NAME
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    condemned = trash_root / f"{run_id}-{stamp}"
    run_dir.replace(condemned)
    shutil.rmtree(condemned, ignore_errors=True)


def display_path(run_dir: Path, repo_root: Path) -> str:
    """Run folder as the UI should name it.

    Short and repo-relative when the data root is where it usually
    is. A ``--results-dir`` pointing elsewhere falls back to the full
    path, which is how the status line tells the user their runs are
    not in the usual place. An operating condition, not a broken
    invariant, so it degrades to a longer message rather than raising.
    """
    try:
        return str(run_dir.resolve().relative_to(repo_root))
    except ValueError:
        return str(run_dir)


def _write_json(
    path: Path, payload: Any, *, indent: Optional[int] = None
) -> None:
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_history(path: Path, frames: List[str]) -> None:
    """Frame texts with the delimiter the reader expects.

    The delimiter is unescaped and a model can emit it, which makes
    this a forgeable machine format; `DATA-05` replaces it with one
    that is not. Preserved verbatim here because this extraction
    changes no behavior.
    """
    with path.open("w", encoding="utf-8") as handle:
        for index, frame_text in enumerate(frames):
            handle.write(f"\n===== FRAME {index} =====\n")
            handle.write(frame_text)
            handle.write("\n")
