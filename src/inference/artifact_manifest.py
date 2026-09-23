"""Completion manifests for locally built model artifacts.

A Hub checkpoint says what it is through its commit. A locally built
one has no commit, so it has to say so some other way, and until now
it said nothing at all: the menu treated the existence of a directory
as proof a model was installed. A quantization run interrupted after
``mkdir`` and before the 16 GB ``torch.save`` finished left a
directory that looked ready and held a truncated state dict, which is
discovered on activation, minutes into a load.

This module is the other way. A manifest is written last, after every
file it names is on disk, so its presence is the completion signal and
its absence means "still building or abandoned". It also records what
the artifact was built from: the base checkpoint and its revision, the
commit of the code that did the building, and a digest of the state
dict. That is what makes a locally built model identifiable a year
later, which a directory name is not.

Shared by the builder (``scripts/quantize_diffusiongemma_nf4.py``) and
the reader (the supervisor's downloaded probe) because a manifest
format known to only one of them would be a format nobody enforces.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "artifact_manifest.json"

# Bumped when a reader would misread an older manifest, not when a
# field is added. A reader that finds a version it does not know
# refuses the artifact rather than guessing, so the number going up
# retires every directory built before it; that is a rebuild of a
# 16 GB file, and not something to spend on a new optional key.
MANIFEST_VERSION = 1

# Read in chunks so a 16 GB state dict is digested in bounded memory.
_DIGEST_CHUNK_BYTES = 8 * 1024 * 1024

# A repository has a handful of path components. Only bounds the walk.
_GIT_DIR_WALK_MAX = 64

assert MANIFEST_VERSION >= 1, "versions start at one"
assert _DIGEST_CHUNK_BYTES > 0, "a chunked read must make progress"


def staging_path(final: Path) -> Path:
    """Where an artifact is built before it becomes the real one.

    A sibling of the destination rather than a temporary directory
    elsewhere, because the promotion below is a rename and a rename is
    only atomic within one filesystem. ``/tmp`` is very often a
    different one, which would turn the atomic step into a 16 GB copy
    and reintroduce exactly the half-written state this avoids.
    """
    assert isinstance(final, Path), "destination path required"
    return final.with_name(final.name + ".incomplete")


def promote(staging: Path, final: Path) -> None:
    """Make a finished staging directory the real artifact.

    One rename, which either happened or did not. Refuses to overwrite
    an existing destination: replacing a working model with a new
    build is a decision for the caller, who is the only one who knows
    whether anything is currently loading from it.
    """
    assert staging.is_dir(), f"nothing staged at {staging}"
    if final.exists():
        raise FileExistsError(
            f"{final} already exists; move it aside first"
        )
    staging.rename(final)
    assert final.is_dir(), "the rename did not take"
    assert not staging.exists(), "the staging directory survived"


def file_digest(path: Path) -> str:
    """SHA-256 of one file, read in bounded chunks.

    Recorded at build time rather than verified at load time. Hashing
    16 GB takes a minute or so, which is acceptable once and not
    acceptable on every activation; what the reader checks instead is
    that the file is present and exactly the size the manifest says,
    which is what a truncated write gets wrong.
    """
    assert path.is_file(), f"not a file: {path}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    artifact: str,
    base_path: str,
    base_revision: Optional[str],
    state_dict_name: str,
    state_dict_bytes: int,
    state_dict_sha256: str,
    copied_files: List[str],
) -> Dict[str, Any]:
    """The record an artifact carries about its own construction."""
    assert artifact, "the artifact needs a name"
    assert state_dict_bytes > 0, "an empty state dict is not one"
    assert state_dict_sha256, "the digest is the integrity claim"
    return {
        "manifest_version": MANIFEST_VERSION,
        "artifact": artifact,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "base": {
            "path": base_path,
            # None when the base is a plain directory rather than a
            # Hub snapshot. Recorded as null rather than omitted, so a
            # reader can tell "unknown" from "this manifest predates
            # the field".
            "revision": base_revision,
        },
        "quantizer": {
            # The commit of this repository, which is the code that
            # decided what to quantize and how. Weights and the code
            # that produced them are one unit: the same base at a
            # different quantizer commit is a different artifact.
            "repo_commit": repo_commit(),
        },
        "state_dict": {
            "name": state_dict_name,
            "bytes": state_dict_bytes,
            "sha256": state_dict_sha256,
        },
        "files": sorted(copied_files),
    }


def write_manifest(
    directory: Path, manifest: Dict[str, Any]
) -> None:
    """Write the manifest, last, once everything it names is there.

    Ordering is the whole contract: written before the files it
    describes, it would be the same lie the bare directory was.
    """
    assert directory.is_dir(), f"no such directory: {directory}"
    state = manifest.get("state_dict") or {}
    weights = directory / str(state.get("name") or "")
    assert weights.is_file(), (
        f"refusing to attest a missing {weights.name}"
    )
    target = directory / MANIFEST_NAME
    target.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def read_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    """The manifest in ``directory``, or None if there is not one.

    None covers every way of not having a usable manifest: no file,
    unreadable, not JSON, not an object. They are one answer to the
    caller, which is going to treat the artifact as unbuilt either
    way, and distinguishing them would only invite a caller to act on
    a manifest it could not parse.
    """
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def is_complete_artifact(directory: Path) -> bool:
    """Whether this directory holds a finished, intact artifact.

    Checks presence and size rather than re-digesting, for the reason
    in ``file_digest``. Size is what a write that ran out of disk or
    was killed mid-save gets wrong, and it costs one stat.
    """
    if not directory.is_dir():
        return False
    manifest = read_manifest(directory)
    if manifest is None:
        return False
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        return False
    return _state_dict_intact(directory, manifest)


def _state_dict_intact(
    directory: Path, manifest: Dict[str, Any]
) -> bool:
    """Whether the weights named by a manifest are all there."""
    state = manifest.get("state_dict")
    if not isinstance(state, dict):
        return False
    name = state.get("name")
    expected = state.get("bytes")
    if not isinstance(name, str) or not name:
        return False
    if not isinstance(expected, int) or expected <= 0:
        return False
    weights = directory / name
    if not weights.is_file():
        return False
    return weights.stat().st_size == expected


def repo_commit() -> Optional[str]:
    """This repository's HEAD, in full, or None outside a checkout.

    Full rather than abbreviated: a short hash is a display
    convenience, and this value exists to be looked up years later
    when the repository is larger and short hashes have collided.
    """
    root = _repo_root()
    if root is None:
        return None
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    commit = finished.stdout.strip()
    return commit or None


def _repo_root() -> Optional[Path]:
    """The nearest ancestor holding a ``.git``, or None."""
    current = Path(__file__).resolve().parent
    for _ in range(_GIT_DIR_WALK_MAX):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return None
