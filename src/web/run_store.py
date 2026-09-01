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

# Written for a person to read and no longer parsed by anything this
# code saves. Its frame delimiter is unescaped and appears verbatim
# around model output, so a model can emit the line and forge a frame
# boundary. Runs written before `frames.jsonl` existed are still read
# through it, because they are not being rewritten.
HISTORY_NAME = "history.txt"

# The machine-readable frame stream: one JSON object per line, so a
# delimiter cannot be forged by anything a model writes inside a
# string. Carries frame text and nothing else; `ANALYTICS-02` may add
# per-frame counts, which is what the schema version is for.
FRAMES_NAME = "frames.jsonl"

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

# The format this code writes. Absent means version 0, which is every
# run saved before this existed; readers dispatch on it rather than
# guessing a run's shape from which files happen to be present.
#
# Version 2 added the flat arrangement below. It is a version rather
# than another optional field because a v1 reader handed a flat
# `tokens.json` would see one frame per token record and draw a run
# that never happened, which is exactly the confidently-wrong answer
# `UnsupportedRunVersionError` exists to refuse.
SCHEMA_VERSION = 2
SCHEMA_VERSION_KEY = "schema_version"

# How `tokens.json` is arranged.
#
# APPEND is one record per position, in order, for a run that only
# ever grows: frame N is the first N+1 of them. SNAPSHOT, the default
# and the only arrangement before version 2, is one array per frame,
# each holding every position as it stood.
#
# The difference is N records against N(N+1)/2. The 2,048-token runs
# in this archive are 131 MiB of `tokens.json` for 2,048 tokens of
# text, and three such runs are half of everything saved here.
#
# Absent means SNAPSHOT, which is what every run written before
# version 2 is.
FRAME_SHAPE_KEY = "frame_shape"
FRAME_SHAPE_APPEND = "append"
FRAME_SHAPE_SNAPSHOT = "snapshot"

# Names the signals a run actually captured, so a reader is told
# rather than left to infer it from file presence and the type of the
# first token. Combinations of optional signals are not a version,
# which is why this is a separate field from the one above.
CAPTURE_KEY = "capture"

# Which generation produced this run, as the worker named it on the
# terminal frame (see `LIFE-01`). Present so a save can be published
# under the run's own identity rather than under whatever the client
# still remembers, which is the difference between saving a run twice
# and saving it once.
#
# Absent on every run saved before this existed, and on any run whose
# worker predates run tokens. Absent means "no identity", not "match
# anything": a save with no token creates, exactly as it always did.
RUN_TOKEN_KEY = "run_token"


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


class BundleInvalidError(Exception):
    """A staged bundle failed its check and was not published.

    Always raised before anything reaches the visible namespace, so
    the run that was already there, if any, is untouched.
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
        _write_frames(staging / FRAMES_NAME, bundle.frames)
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
    run_token: Optional[str] = None,
) -> Tuple[str, int]:
    """Publish a run, new or replacing, returning id and revision.

    The one entry point a caller needs. Which run is being written is
    decided here, in this order:

    ``run_token`` names the generation. If a run was already published
    for it, that run is the destination, whatever the caller believes.
    This is what makes a save idempotent per generation: a client that
    posted a save and never saw the answer, because the page navigated
    while the request was in flight, will post again and land on the
    run it already made rather than making a second one.

    ``run_id`` is the older way of saying the same thing, kept for a
    caller that has no token: every run saved before `LIFE-01`, and
    any worker too old to issue one.

    Neither means a new run.

    ``expected_revision`` still guards a replacement against a
    concurrent writer; the check and the publication happen here so a
    caller cannot read a revision, be overtaken, and write anyway.

    Nothing is published if anything raises. The staging directory is
    discarded on failure, and the reservation for a new run is left
    behind empty, which no reader counts.
    """
    # One lock over resolution and publication both, because they are
    # one decision. Two saves for the same generation arriving
    # together would otherwise each find no existing run and each
    # create one, which is the duplicate this exists to prevent, and
    # the same shape as the revision race below. Creates serialise
    # too as a result, which costs nothing: a save is a person
    # pressing a button.
    with _PUBLISH_LOCK:
        target = run_id
        published = find_run_by_token(root, run_token or "")
        if published is not None:
            target = published
        if target is None:
            return _publish_new(root, bundle, model_id, run_token)
        # A replacement from a client with no token must not erase the
        # identity the run already has, or the run becomes findable
        # only by an id that the next lost reply will forget again.
        token = run_token or read_run_token(root, target)
        return _publish_replacement(
            root, bundle, target, expected_revision, token
        )


# Guards resolve-identity-then-publish, and within that
# read-revision-then-publish. In-process only, which is the right
# scope today because one supervisor owns the data root; `LIFE-05` is
# where a second one becomes possible, and `ui_state.py` carries the
# interprocess pattern to copy if it ever does.
_PUBLISH_LOCK = threading.Lock()


def _publish_new(
    root: Path,
    bundle: RunBundle,
    model_id: str,
    run_token: Optional[str],
) -> Tuple[str, int]:
    run_id = allocate(root, model_id)
    _stage_and_publish(
        root, run_id, bundle, revision=1, run_token=run_token
    )
    return run_id, 1


def _publish_replacement(
    root: Path,
    bundle: RunBundle,
    run_id: str,
    expected_revision: Optional[int],
    run_token: Optional[str],
) -> Tuple[str, int]:
    actual = read_revision(root, run_id)
    expected = (
        actual if expected_revision is None else expected_revision
    )
    if expected != actual:
        raise RevisionConflictError(run_id, expected, actual)
    revision = actual + 1
    _stage_and_publish(
        root, run_id, bundle, revision=revision, run_token=run_token
    )
    return run_id, revision


def capture_manifest(bundle: RunBundle) -> Dict[str, bool]:
    """Which signals this run captured, stated rather than inferred.

    One entry per optional sidecar plus the frame stream. A reader of
    a versioned run consults this instead of testing for files and
    sniffing the type of the first token, which is what it has to do
    for runs that predate it.
    """
    manifest = {"frames": True}
    for attribute, _filename in SIDECAR_NAMES:
        manifest[attribute] = getattr(bundle, attribute) is not None
    return manifest


def _stage_and_publish(
    root: Path,
    run_id: str,
    bundle: RunBundle,
    *,
    revision: int,
    run_token: Optional[str] = None,
) -> None:
    """Write the bundle aside, validate it, then move it in.

    The token is stamped here, beside the revision, rather than by
    whoever built the bundle. Resolution and persistence are then one
    function's business and cannot disagree: a caller cannot ask to
    publish under a token and leave the run unfindable by it.
    """
    stamped = dict(bundle.metadata)
    stamped[REVISION_KEY] = revision
    stamped[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    stamped[CAPTURE_KEY] = capture_manifest(bundle)
    if run_token:
        stamped[RUN_TOKEN_KEY] = run_token
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
        validate_staged(staging)
        publish(root, run_id, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_staged(staging: Path) -> None:
    """Check a staged bundle before it is allowed to become a run.

    The point of staging is that what gets published was checked, and
    this is the check. It runs against the files as written rather
    than against the objects that produced them, so a serialization
    bug is caught here instead of by a reader months later.

    Deliberately narrow: structure and self-consistency, not content.
    Whether a confidence is plausible is not this function's business;
    whether the manifest describes the files that are actually here
    is.
    """
    metadata_path = staging / METADATA_NAME
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BundleInvalidError(
            "metadata must be an object, got"
            f" {type(raw).__name__}"
        )
    version = raw.get(SCHEMA_VERSION_KEY)
    if version != SCHEMA_VERSION:
        raise BundleInvalidError(
            f"staged metadata claims version {version!r},"
            f" this code writes {SCHEMA_VERSION}"
        )
    manifest = raw.get(CAPTURE_KEY)
    if not isinstance(manifest, dict):
        raise BundleInvalidError("capture manifest is missing")

    for name in (FINAL_TEXT_NAME, HISTORY_NAME, FRAMES_NAME):
        if not (staging / name).is_file():
            raise BundleInvalidError(f"{name} was not written")

    for attribute, filename in SIDECAR_NAMES:
        declared = bool(manifest.get(attribute))
        present = (staging / filename).is_file()
        if declared != present:
            raise BundleInvalidError(
                f"manifest says {attribute}={declared} but"
                f" {filename} is {'present' if present else 'absent'}"
            )
        if present:
            _require_json_list(staging / filename)


def _require_json_list(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise BundleInvalidError(
            f"{path.name} must be a list, got"
            f" {type(payload).__name__}"
        )


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


def read_run_token(root: Path, run_id: str) -> Optional[str]:
    """The generation a saved run came from, or None if unrecorded."""
    try:
        raw = json.loads(
            (root / run_id / METADATA_NAME).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    token = raw.get(RUN_TOKEN_KEY)
    return token if isinstance(token, str) and token else None


def find_run_by_token(root: Path, token: str) -> Optional[str]:
    """The run this generation was already published as, if any.

    A linear scan of the run directories, reading each metadata. That
    is affordable because ``list_runs`` in the analytics layer already
    does strictly more work on every Analytics page load, while this
    runs once per save. If it ever stops being affordable, an index is
    the answer, and it should be built for the listing first.

    An empty token matches nothing, deliberately. Runs saved before
    this field existed have no token, and treating "no identity" as a
    match would make one of them the destination for every save from a
    worker too old to issue tokens.
    """
    if not token:
        return None
    for run_id in list_run_ids(root):
        try:
            raw = json.loads(
                (root / run_id / METADATA_NAME).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            # A run whose metadata cannot be read is not a run this
            # save should silently overwrite.
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get(RUN_TOKEN_KEY) == token:
            return run_id
    return None


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
    """Frame texts in the human-readable transcript.

    Byte-identical to what the app has always written, because 182
    saved runs are read through this exact framing and none of them
    is being rewritten. Nothing written today parses it back: new
    runs carry `frames.jsonl` for that, and this file is kept because
    it is the one artifact a person can open and read.
    """
    with path.open("w", encoding="utf-8") as handle:
        for index, frame_text in enumerate(frames):
            handle.write(f"\n===== FRAME {index} =====\n")
            handle.write(frame_text)
            handle.write("\n")


def _write_frames(path: Path, frames: List[str]) -> None:
    """Frame texts as JSON lines, one object per frame.

    The container a reader can trust. Text lives inside a JSON string,
    so a model emitting the transcript's delimiter, or a newline, or
    anything else, cannot change where a frame begins.
    """
    with path.open("w", encoding="utf-8") as handle:
        for index, frame_text in enumerate(frames):
            line = json.dumps(
                {"i": index, "text": frame_text},
                ensure_ascii=False,
            )
            handle.write(line)
            handle.write("\n")
