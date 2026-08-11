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

This is the extraction only: every behavior here is what
``server.py`` did before, including the parts that are wrong.
Directories are still created with ``exist_ok=True`` so two saves in
one second still collide, files are still written straight into the
visible folder, and deletion still removes it in place. `DATA-01`
fixes those next, against tests that can finally reach them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def make_run_dir(root: Path, model_id: str) -> Path:
    """Create the directory for a new run and return it.

    One-second resolution and ``exist_ok=True``, which is what
    `DATA-01` is about to replace: two saves of the same model in the
    same second currently land in one folder and interleave.
    """
    assert isinstance(model_id, str) and model_id, "model id required"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = model_id.replace("/", "_")
    run_dir = root / f"{timestamp}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_bundle(run_dir: Path, bundle: RunBundle) -> None:
    """Write every file of a run into its directory.

    Straight into the visible folder, in order, exactly as the
    supervisor did. A failure part way through leaves a partial run
    that Analytics will happily list, which is the problem `DATA-01`
    solves by staging elsewhere and publishing atomically.
    """
    assert run_dir.is_dir(), f"run directory missing: {run_dir}"
    _write_json(run_dir / METADATA_NAME, bundle.metadata, indent=2)
    (run_dir / FINAL_TEXT_NAME).write_text(
        bundle.final_text, encoding="utf-8"
    )
    _write_history(run_dir / HISTORY_NAME, bundle.frames)
    for attribute, filename in SIDECAR_NAMES:
        payload = getattr(bundle, attribute)
        if payload is not None:
            _write_json(run_dir / filename, payload)


def delete(root: Path, run_id: str) -> None:
    """Remove a run directory and everything in it."""
    run_dir = resolve_run_dir(root, run_id)
    shutil.rmtree(run_dir)


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
