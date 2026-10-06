"""Immutable selected-path conversation snapshots for Analytics.

The live conversation store owns editable branch history. This module
owns explicit research snapshots: copied turn-version names and pinned
saved-run files that survive later source deletion or replacement.

Only the standard library and the two durable stores are imported.
HTTP models and UI policy stay above this layer.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from src.web import _conversation_store_core as conversation_core
from src.web import conversation_store
from src.web import run_store
from src.web.data_root_lock import DataRootLock


SCHEMA_VERSION = 1
ROOT_NAME = "saved_conversations"
METADATA_NAME = "metadata.json"
PATH_NAME = "path.json"
LINKS_NAME = "links.json"
TURNS_DIR_NAME = "turns"
RUNS_DIR_NAME = "runs"
STAGING_DIR_NAME = ".staging"
TRASH_DIR_NAME = ".trash"
OPERATIONS_DIR_NAME = "operations"

SNAPSHOT_ID_PATTERN = r"^[0-9a-f]{32}$"
SNAPSHOT_SCAN_MAX = 10_000
PIN_FILE_COUNT_MAX = 32
PIN_COPY_BYTES_MAX = 512 * 1024 * 1024
PIN_FREE_MARGIN_BYTES = 16 * 1024 * 1024
TRASH_ATTEMPTS_MAX = 16

STATUS_PINNED = "pinned"
STATUS_TEXT_ONLY = "text_only"
STATUS_UNAVAILABLE = "unavailable"
STATUSES = (STATUS_PINNED, STATUS_TEXT_ONLY, STATUS_UNAVAILABLE)

_SNAPSHOT_ID = re.compile(SNAPSHOT_ID_PATTERN)
_LOCK = DataRootLock("saved_conversations.lock")

assert len(set(STATUSES)) == len(STATUSES)
assert len(run_store.SIDECAR_NAMES) + 4 <= PIN_FILE_COUNT_MAX
assert PIN_COPY_BYTES_MAX > 0


class SnapshotNotFoundError(FileNotFoundError):
    """No published snapshot has this identifier."""


class InvalidSnapshotIdError(ValueError):
    """The identifier cannot name a snapshot directory."""


class SnapshotCorruptError(RuntimeError):
    """A published snapshot violates its bounded schema."""


class SnapshotRevisionConflictError(Exception):
    """Mutable snapshot metadata changed after the caller read it."""

    def __init__(
        self, snapshot_id: str, expected: int, actual: int
    ) -> None:
        super().__init__(
            f"saved conversation {snapshot_id} has moved on:"
            f" expected title revision {expected}, found {actual}"
        )
        self.snapshot_id = snapshot_id
        self.expected = expected
        self.actual = actual


class SnapshotOperationConflictError(Exception):
    """One operation id was reused with different save semantics."""


@dataclass(frozen=True)
class SnapshotPreview:
    """Facts shown before snapshot publication."""

    default_title: str
    turn_count: int
    exchange_count: int
    xai_count: int
    text_only_count: int
    unavailable_count: int


@dataclass(frozen=True)
class SnapshotResult:
    """The identity and mutable-title revision of one snapshot."""

    snapshot_id: str
    title_revision: int
    replayed: bool


def preview_snapshot(
    results_dir: Path,
    *,
    conversation_id: str,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
) -> SnapshotPreview:
    """Inspect one exact path without publishing an artifact."""
    staging = _new_staging(results_dir, "preview")
    try:
        source = conversation_store.capture_snapshot_source(
            results_dir,
            conversation_id,
            branch_id=branch_id,
            branch_revision=branch_revision,
            turn_count=turn_count,
            tail_turn_id=tail_turn_id,
            tail_version=tail_version,
            turns_dir=staging / TURNS_DIR_NAME,
        )
        links = _inspect_run_links(results_dir, source)
        counts = _link_counts(links)
        return SnapshotPreview(
            default_title=_default_title(source),
            turn_count=source.turn_count,
            exchange_count=source.turn_count // 2,
            xai_count=counts[STATUS_PINNED],
            text_only_count=counts[STATUS_TEXT_ONLY],
            unavailable_count=counts[STATUS_UNAVAILABLE],
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def create_snapshot(
    results_dir: Path,
    *,
    operation_id: str,
    title: str,
    conversation_id: str,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
) -> SnapshotResult:
    """Publish one exact selected path, idempotent by operation id."""
    conversation_store.validate_operation_id(operation_id)
    clean_title = conversation_store.validate_title(title)
    digest = _request_digest(
        title=clean_title,
        conversation_id=conversation_id,
        branch_id=branch_id,
        branch_revision=branch_revision,
        turn_count=turn_count,
        tail_turn_id=tail_turn_id,
        tail_version=tail_version,
    )
    replay = _replay_result(results_dir, operation_id, digest)
    if replay is not None:
        return replay
    staging = _new_staging(results_dir, operation_id)
    try:
        source = conversation_store.capture_snapshot_source(
            results_dir,
            conversation_id,
            branch_id=branch_id,
            branch_revision=branch_revision,
            turn_count=turn_count,
            tail_turn_id=tail_turn_id,
            tail_version=tail_version,
            turns_dir=staging / TURNS_DIR_NAME,
        )
        links = _pin_run_links(
            results_dir, source, staging / RUNS_DIR_NAME
        )
        metadata = _write_staged_snapshot(
            staging=staging,
            snapshot_id=operation_id,
            title=clean_title,
            digest=digest,
            source=source,
            links=links,
        )
        return _publish_snapshot(
            results_dir,
            operation_id=operation_id,
            digest=digest,
            staging=staging,
            metadata=metadata,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def list_snapshots(results_dir: Path) -> List[Dict[str, object]]:
    """Return bounded summary rows newest first."""
    root = _root(results_dir, create=False)
    if root is None:
        return []
    rows: List[Dict[str, object]] = []
    scanned = 0
    for child in sorted(root.iterdir(), reverse=True):
        if child.name.startswith(".") or not child.is_dir():
            continue
        if child.name == OPERATIONS_DIR_NAME:
            continue
        scanned += 1
        if scanned > SNAPSHOT_SCAN_MAX:
            raise SnapshotCorruptError(
                "saved conversation scan exceeds its limit"
            )
        if not (child / METADATA_NAME).is_file():
            continue
        try:
            metadata = read_metadata(results_dir, child.name)
            rows.append(_summary(metadata))
        except (OSError, ValueError, SnapshotCorruptError) as exc:
            rows.append(_invalid_summary(child.name, str(exc)))
    rows.sort(
        key=lambda row: str(row.get("created_at", "")),
        reverse=True,
    )
    return rows


def read_metadata(
    results_dir: Path, snapshot_id: str
) -> Dict[str, object]:
    """Read and validate one snapshot's bounded metadata."""
    snapshot_dir = resolve_snapshot_dir(results_dir, snapshot_id)
    raw = _read_object(snapshot_dir / METADATA_NAME, "metadata")
    _validate_metadata(raw, expected_id=snapshot_id)
    return raw


def page_turns(
    results_dir: Path,
    snapshot_id: str,
    *,
    before: Optional[str] = None,
    limit: int = conversation_store.PAGE_SIZE_DEFAULT,
) -> Dict[str, object]:
    """Read one chronological page from an immutable snapshot."""
    conversation_core.validate_limit(
        limit,
        maximum=conversation_store.PAGE_SIZE_MAX,
        name="page limit",
    )
    snapshot_dir = resolve_snapshot_dir(results_dir, snapshot_id)
    metadata = read_metadata(results_dir, snapshot_id)
    turn_count = _stored_int(
        metadata.get("turn_count"), "turn_count", minimum=1
    )
    finish_before = _before_index(before, turn_count)
    finish = finish_before - 1
    start = max(1, finish - limit + 1)
    links = _read_links(snapshot_dir, turn_count)
    turns: List[Dict[str, object]] = []
    for index in range(start, finish + 1):
        turn = _read_snapshot_turn(snapshot_dir, index)
        payload = conversation_store.turn_to_payload(turn)
        payload["xai"] = links.get(turn.turn_id)
        turns.append(payload)
    has_more = start > 1
    return {
        "snapshot_id": snapshot_id,
        "title_revision": metadata["title_revision"],
        "turns": turns,
        "next_before": str(start) if has_more else None,
        "has_more": has_more,
    }


def rename_snapshot(
    results_dir: Path,
    snapshot_id: str,
    *,
    title: str,
    expected_revision: int,
) -> Dict[str, object]:
    """Rename metadata without changing snapshot content."""
    clean_title = conversation_store.validate_title(title)
    if expected_revision < 1:
        raise ValueError("expected title revision must be positive")
    with _LOCK.held(results_dir):
        metadata = read_metadata(results_dir, snapshot_id)
        actual = _stored_int(
            metadata.get("title_revision"),
            "title_revision",
            minimum=1,
        )
        if actual != expected_revision:
            raise SnapshotRevisionConflictError(
                snapshot_id, expected_revision, actual
            )
        metadata["title"] = clean_title
        metadata["title_revision"] = actual + 1
        metadata["updated_at"] = conversation_core.timestamp()
        snapshot_dir = resolve_snapshot_dir(results_dir, snapshot_id)
        conversation_core.write_json_atomic(
            snapshot_dir / METADATA_NAME, metadata
        )
        return metadata


def delete_snapshot(results_dir: Path, snapshot_id: str) -> None:
    """Atomically remove a snapshot from the visible namespace."""
    with _LOCK.held(results_dir):
        snapshot_dir = resolve_snapshot_dir(results_dir, snapshot_id)
        root = _root(results_dir, create=True)
        assert root is not None
        trash = root / TRASH_DIR_NAME
        conversation_core.make_directory_durable(
            trash, exist_ok=True
        )
        target = _trash_destination(trash, snapshot_id)
        conversation_core.replace_durable(snapshot_dir, target)
    shutil.rmtree(target)


def resolve_snapshot_dir(
    results_dir: Path, snapshot_id: str
) -> Path:
    """Resolve one direct published snapshot child."""
    _validate_snapshot_id(snapshot_id)
    root = _root(results_dir, create=False)
    if root is None:
        raise SnapshotNotFoundError(
            f"Saved conversation not found: {snapshot_id}"
        )
    snapshot_dir = (root / snapshot_id).resolve()
    if snapshot_dir.parent != root.resolve():
        raise InvalidSnapshotIdError(
            f"invalid saved conversation id: {snapshot_id}"
        )
    if not (snapshot_dir / METADATA_NAME).is_file():
        raise SnapshotNotFoundError(
            f"Saved conversation not found: {snapshot_id}"
        )
    return snapshot_dir


def resolve_pinned_run_dir(
    results_dir: Path,
    snapshot_id: str,
    assistant_turn_id: str,
) -> Path:
    """Resolve the run revision pinned for one assistant turn."""
    snapshot_dir = resolve_snapshot_dir(results_dir, snapshot_id)
    metadata = read_metadata(results_dir, snapshot_id)
    count = _stored_int(
        metadata.get("turn_count"), "turn_count", minimum=1
    )
    links = _read_links(snapshot_dir, count)
    link = links.get(assistant_turn_id)
    if not isinstance(link, dict):
        raise SnapshotNotFoundError(
            "This saved response has no pinned XAI run."
        )
    if link.get("status") != STATUS_PINNED:
        raise SnapshotNotFoundError(
            "This saved response's XAI run was unavailable."
        )
    relative = link.get("path")
    if not isinstance(relative, str):
        raise SnapshotCorruptError("pinned run path is missing")
    run_dir = (snapshot_dir / relative).resolve()
    if run_dir.parent != (snapshot_dir / RUNS_DIR_NAME).resolve():
        raise SnapshotCorruptError("pinned run leaves its snapshot")
    if not (run_dir / run_store.METADATA_NAME).is_file():
        raise SnapshotCorruptError("pinned run metadata is missing")
    return run_dir


def _new_staging(results_dir: Path, label: str) -> Path:
    root = _root(results_dir, create=True)
    assert root is not None
    staging_root = root / STAGING_DIR_NAME
    conversation_core.make_directory_durable(
        staging_root, exist_ok=True
    )
    staging = staging_root / f"{label}.{uuid4().hex}"
    conversation_core.make_directory_durable(staging)
    return staging


def _root(results_dir: Path, *, create: bool) -> Optional[Path]:
    if (
        not isinstance(results_dir, Path)
        or not results_dir.is_absolute()
    ):
        raise ValueError("results_dir must be an absolute Path")
    root = results_dir / ROOT_NAME
    if create:
        conversation_core.make_directory_durable(
            results_dir, exist_ok=True
        )
        conversation_core.make_directory_durable(root, exist_ok=True)
    elif not root.exists():
        return None
    if not root.is_dir() or root.is_symlink():
        raise SnapshotCorruptError(
            "saved conversations root is not a safe directory"
        )
    if root.resolve().parent != results_dir.resolve():
        raise SnapshotCorruptError(
            "saved conversations root leaves the data root"
        )
    return root


def _default_title(source: conversation_store.SnapshotSource) -> str:
    if source.title != conversation_store.DEFAULT_TITLE:
        return source.title
    for item in source.turns:
        if item.turn.role != "user":
            continue
        clean = item.turn.text.strip()
        if not clean:
            continue
        first_line = clean.splitlines()[0]
        if first_line:
            return first_line[: conversation_store.TITLE_CHARS_MAX]
    return conversation_store.DEFAULT_TITLE


def _inspect_run_links(
    results_dir: Path,
    source: conversation_store.SnapshotSource,
) -> Dict[str, Dict[str, object]]:
    links: Dict[str, Dict[str, object]] = {}
    with run_store.publication_lock(results_dir):
        for item in source.turns:
            turn = item.turn
            if turn.role != "assistant":
                continue
            links[turn.turn_id] = _inspect_one_link(
                results_dir, turn
            )
    return links


def _inspect_one_link(
    results_dir: Path,
    turn: conversation_store.TurnRecord,
) -> Dict[str, object]:
    link = turn.run_link
    if link is None:
        return {"status": STATUS_TEXT_ONLY}
    base: Dict[str, object] = {
        "source_run_id": link.run_id,
        "source_revision": link.revision,
    }
    try:
        actual = run_store.read_revision(results_dir, link.run_id)
    except (run_store.InvalidRunIdError, run_store.RunNotFoundError):
        base["status"] = STATUS_UNAVAILABLE
        base["reason"] = "source run is missing"
        return base
    if actual != link.revision:
        base["status"] = STATUS_UNAVAILABLE
        base["reason"] = (
            f"source run moved from revision {link.revision}"
            f" to {actual}"
        )
        return base
    metadata_path = (
        run_store.resolve_run_dir(results_dir, link.run_id)
        / run_store.METADATA_NAME
    )
    metadata = _read_object(metadata_path, "source run metadata")
    for name in ("backend", "model", "model_type", "processor"):
        value = metadata.get(name)
        if isinstance(value, str):
            base[name] = value
    base["status"] = STATUS_PINNED
    return base


def _pin_run_links(
    results_dir: Path,
    source: conversation_store.SnapshotSource,
    runs_dir: Path,
) -> Dict[str, Dict[str, object]]:
    runs_dir.mkdir(parents=True)
    links: Dict[str, Dict[str, object]] = {}
    copied_bytes = 0
    with run_store.publication_lock(results_dir):
        for item in source.turns:
            turn = item.turn
            if turn.role != "assistant":
                continue
            inspected = _inspect_one_link(results_dir, turn)
            if inspected["status"] != STATUS_PINNED:
                links[turn.turn_id] = inspected
                continue
            target = runs_dir / turn.turn_id
            target.mkdir()
            copied = _pin_one_run(
                results_dir=results_dir,
                turn=turn,
                target=target,
                copied_bytes=copied_bytes,
            )
            copied_bytes += copied
            inspected["path"] = f"{RUNS_DIR_NAME}/{turn.turn_id}"
            inspected["storage"] = (
                "copy" if copied > 0 else "link"
            )
            links[turn.turn_id] = inspected
    return links


def _pin_one_run(
    *,
    results_dir: Path,
    turn: conversation_store.TurnRecord,
    target: Path,
    copied_bytes: int,
) -> int:
    link = turn.run_link
    assert link is not None
    source = run_store.resolve_run_dir(results_dir, link.run_id)
    actual = run_store.read_revision(results_dir, link.run_id)
    if actual != link.revision:
        raise SnapshotOperationConflictError(
            "run changed while its snapshot was being pinned"
        )
    files = sorted(source.iterdir(), key=lambda path: path.name)
    if len(files) > PIN_FILE_COUNT_MAX:
        raise SnapshotCorruptError(
            "saved run exceeds the pin file-count limit"
        )
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise SnapshotCorruptError(
            "saved run contains an unsafe pin entry"
        )
    copied = 0
    for path in files:
        copied += _pin_file(
            source=path,
            target=target / path.name,
            copied_bytes=copied_bytes + copied,
        )
    assert (target / run_store.METADATA_NAME).is_file()
    return copied


def _pin_file(
    *,
    source: Path,
    target: Path,
    copied_bytes: int,
) -> int:
    try:
        os.link(source, target, follow_symlinks=False)
        return 0
    except OSError as exc:
        fallback = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EOPNOTSUPP,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            errno.EMLINK,
        }
        if exc.errno not in fallback:
            raise
    size = source.stat().st_size
    if copied_bytes + size > PIN_COPY_BYTES_MAX:
        raise conversation_store.ConversationLimitError(
            "saved-run copy fallback exceeds its byte limit"
        )
    free = shutil.disk_usage(target.parent).free
    if free < size + PIN_FREE_MARGIN_BYTES:
        raise OSError("not enough disk space to copy a pinned run")
    shutil.copy2(source, target, follow_symlinks=False)
    return size


def _link_counts(
    links: Dict[str, Dict[str, object]],
) -> Dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for link in links.values():
        status = link.get("status")
        if status not in counts:
            raise SnapshotCorruptError("unknown XAI link status")
        counts[str(status)] += 1
    return counts


def _write_staged_snapshot(
    *,
    staging: Path,
    snapshot_id: str,
    title: str,
    digest: str,
    source: conversation_store.SnapshotSource,
    links: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    now = conversation_core.timestamp()
    counts = _link_counts(links)
    source_payload: Dict[str, object] = {
        "conversation_id": source.conversation_id,
        "conversation_title": source.title,
        "branch_id": source.branch_id,
        "branch_revision": source.branch_revision,
        "catalog_revision": source.catalog_revision,
        "turn_count": source.turn_count,
        "tail_turn_id": source.tail_turn_id,
        "tail_version": source.tail_version,
    }
    metadata: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "request_digest": digest,
        "title": title,
        "title_revision": 1,
        "created_at": now,
        "updated_at": now,
        "turn_count": source.turn_count,
        "exchange_count": source.turn_count // 2,
        "xai_count": counts[STATUS_PINNED],
        "text_only_count": counts[STATUS_TEXT_ONLY],
        "unavailable_count": counts[STATUS_UNAVAILABLE],
        "source": source_payload,
    }
    conversation_core.write_json_atomic(
        staging / PATH_NAME, source_payload
    )
    conversation_core.write_json_atomic(staging / LINKS_NAME, links)
    conversation_core.write_json_atomic(
        staging / METADATA_NAME, metadata
    )
    return metadata


def _publish_snapshot(
    results_dir: Path,
    *,
    operation_id: str,
    digest: str,
    staging: Path,
    metadata: Dict[str, object],
) -> SnapshotResult:
    with _LOCK.held(results_dir):
        replay = _replay_result_locked(
            results_dir, operation_id, digest
        )
        if replay is not None:
            shutil.rmtree(staging, ignore_errors=True)
            return replay
        root = _root(results_dir, create=True)
        assert root is not None
        target = root / operation_id
        if target.exists():
            shutil.rmtree(target)
        conversation_core.make_directory_durable(target)
        try:
            _move_staged_snapshot(staging, target)
            _write_receipt(
                root,
                operation_id=operation_id,
                digest=digest,
                title_revision=int(metadata["title_revision"]),
            )
        except Exception:
            if not (target / METADATA_NAME).exists():
                shutil.rmtree(target, ignore_errors=True)
            raise
    return SnapshotResult(operation_id, 1, False)


def _move_staged_snapshot(staging: Path, target: Path) -> None:
    """Publish all content before the visibility marker."""
    for child in sorted(staging.iterdir()):
        if child.name == METADATA_NAME:
            continue
        conversation_core.replace_durable(
            child, target / child.name
        )
    conversation_core.replace_durable(
        staging / METADATA_NAME,
        target / METADATA_NAME,
    )
    staging.rmdir()


def _replay_result(
    results_dir: Path, operation_id: str, digest: str
) -> Optional[SnapshotResult]:
    with _LOCK.held(results_dir):
        return _replay_result_locked(
            results_dir, operation_id, digest
        )


def _replay_result_locked(
    results_dir: Path, operation_id: str, digest: str
) -> Optional[SnapshotResult]:
    root = _root(results_dir, create=False)
    if root is None:
        return None
    target = root / operation_id
    metadata_path = target / METADATA_NAME
    if not metadata_path.is_file():
        return _replay_deleted_receipt(root, operation_id, digest)
    metadata = _read_object(metadata_path, "metadata")
    _validate_metadata(metadata, expected_id=operation_id)
    if metadata.get("request_digest") != digest:
        raise SnapshotOperationConflictError(
            "operation_id was already used for another snapshot"
        )
    revision = _stored_int(
        metadata.get("title_revision"),
        "title_revision",
        minimum=1,
    )
    return SnapshotResult(operation_id, revision, True)


def _replay_deleted_receipt(
    root: Path, operation_id: str, digest: str
) -> Optional[SnapshotResult]:
    receipt_path = (
        root / OPERATIONS_DIR_NAME / f"{operation_id}.json"
    )
    if not receipt_path.is_file():
        return None
    receipt = _read_object(receipt_path, "operation receipt")
    if receipt.get("request_digest") != digest:
        raise SnapshotOperationConflictError(
            "operation_id was already used for another snapshot"
        )
    raise SnapshotOperationConflictError(
        "this save operation already published a deleted snapshot"
    )


def _write_receipt(
    root: Path,
    *,
    operation_id: str,
    digest: str,
    title_revision: int,
) -> None:
    operations = root / OPERATIONS_DIR_NAME
    conversation_core.make_directory_durable(
        operations, exist_ok=True
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation_id,
        "request_digest": digest,
        "snapshot_id": operation_id,
        "title_revision": title_revision,
    }
    conversation_core.write_json_atomic(
        operations / f"{operation_id}.json", receipt
    )


def _request_digest(
    *,
    title: str,
    conversation_id: str,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
) -> str:
    payload = {
        "title": title,
        "conversation_id": conversation_id,
        "branch_id": branch_id,
        "branch_revision": branch_revision,
        "turn_count": turn_count,
        "tail_turn_id": tail_turn_id,
        "tail_version": tail_version,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary(metadata: Dict[str, object]) -> Dict[str, object]:
    fields = (
        "snapshot_id",
        "title",
        "title_revision",
        "created_at",
        "updated_at",
        "turn_count",
        "exchange_count",
        "xai_count",
        "text_only_count",
        "unavailable_count",
        "source",
    )
    return {key: metadata[key] for key in fields}


def _invalid_summary(
    snapshot_id: str, reason: str
) -> Dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "invalid": True,
        "error": reason,
        "title": "",
        "created_at": "",
        "turn_count": 0,
        "exchange_count": 0,
    }


def _validate_metadata(
    raw: Dict[str, object], *, expected_id: str
) -> None:
    required = frozenset({
        "schema_version",
        "snapshot_id",
        "request_digest",
        "title",
        "title_revision",
        "created_at",
        "updated_at",
        "turn_count",
        "exchange_count",
        "xai_count",
        "text_only_count",
        "unavailable_count",
        "source",
    })
    if set(raw) != required:
        raise SnapshotCorruptError(
            "metadata fields do not match schema"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SnapshotCorruptError(
            "unsupported saved conversation schema"
        )
    if raw["snapshot_id"] != expected_id:
        raise SnapshotCorruptError(
            "metadata id does not match directory"
        )
    _validate_snapshot_id(expected_id)
    conversation_store.validate_title(raw["title"])
    _stored_int(raw["title_revision"], "title_revision", minimum=1)
    turn_count = _validate_metadata_counts(raw)
    _validate_metadata_source(raw["source"], turn_count)
    _validate_metadata_digest(raw["request_digest"])
    _validate_metadata_timestamps(raw)


def _validate_metadata_counts(raw: Dict[str, object]) -> int:
    """Validate path and assistant summary counts."""
    turn_count = _stored_int(
        raw["turn_count"], "turn_count", minimum=1
    )
    if turn_count > conversation_store.TURN_COUNT_MAX:
        raise SnapshotCorruptError("turn_count exceeds its limit")
    if turn_count % 2 != 0:
        raise SnapshotCorruptError("turn_count must be even")
    if raw["exchange_count"] != turn_count // 2:
        raise SnapshotCorruptError("exchange_count disagrees")
    total = 0
    for key in ("xai_count", "text_only_count", "unavailable_count"):
        total += _stored_int(raw[key], key, minimum=0)
    if total != turn_count // 2:
        raise SnapshotCorruptError("assistant status counts disagree")
    return turn_count


def _validate_metadata_source(
    source: object, turn_count: int
) -> None:
    """Validate immutable source identity at summary depth."""
    if not isinstance(source, dict):
        raise SnapshotCorruptError(
            "snapshot source must be an object"
        )
    source_count = _stored_int(
        source.get("turn_count"), "source turn_count", minimum=1
    )
    if source_count != turn_count:
        raise SnapshotCorruptError("source turn_count disagrees")
    for key in (
        "conversation_id",
        "branch_id",
        "tail_turn_id",
    ):
        if not isinstance(source.get(key), str) or not source[key]:
            raise SnapshotCorruptError(f"source {key} is invalid")


def _validate_metadata_digest(digest: object) -> None:
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise SnapshotCorruptError("request digest is invalid")


def _validate_metadata_timestamps(raw: Dict[str, object]) -> None:
    for key in ("created_at", "updated_at"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise SnapshotCorruptError(f"{key} is invalid")


def _read_links(
    snapshot_dir: Path, turn_count: int
) -> Dict[str, Dict[str, object]]:
    raw = _read_object(snapshot_dir / LINKS_NAME, "links")
    if len(raw) != turn_count // 2:
        raise SnapshotCorruptError("XAI link count disagrees")
    links: Dict[str, Dict[str, object]] = {}
    for turn_id, value in raw.items():
        if not isinstance(value, dict):
            raise SnapshotCorruptError("XAI link must be an object")
        status = value.get("status")
        if status not in STATUSES:
            raise SnapshotCorruptError("XAI link status is invalid")
        links[turn_id] = value
    return links


def _read_snapshot_turn(
    snapshot_dir: Path, index: int
) -> conversation_store.TurnRecord:
    if not 1 <= index <= conversation_store.TURN_COUNT_MAX:
        raise SnapshotCorruptError("turn index is outside its limit")
    raw = _read_object(
        snapshot_dir / TURNS_DIR_NAME / f"{index:08d}.json",
        "snapshot turn",
    )
    turn = conversation_store.parse_turn(raw)
    if turn.index != index:
        raise SnapshotCorruptError("snapshot turn index disagrees")
    return turn


def _before_index(before: Optional[str], turn_count: int) -> int:
    if before is None:
        return turn_count + 1
    if not isinstance(before, str) or not before.isdigit():
        raise ValueError(
            "before cursor must be a positive turn index"
        )
    value = int(before)
    if not 1 <= value <= turn_count + 1:
        raise ValueError("before cursor is outside this snapshot")
    return value


def _read_object(path: Path, label: str) -> Dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotCorruptError(
            f"could not read {label}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SnapshotCorruptError(f"{label} must be an object")
    return raw


def _stored_int(
    value: object, name: str, *, minimum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotCorruptError(f"{name} must be an integer")
    if value < minimum:
        raise SnapshotCorruptError(f"{name} is below its minimum")
    return value


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(
        snapshot_id
    ):
        raise InvalidSnapshotIdError(
            f"invalid saved conversation id: {snapshot_id}"
        )


def _trash_destination(root: Path, snapshot_id: str) -> Path:
    for _attempt in range(TRASH_ATTEMPTS_MAX):
        suffix = uuid4().hex
        target = root / f"{snapshot_id}.{suffix}"
        if not target.exists():
            return target
    raise RuntimeError("could not allocate saved conversation trash")
