"""Append-oriented durable conversations under the shared data root.

The conversation store is deliberately a standard-library boundary.
It knows nothing about FastAPI, model workers, or saved-run schemas.
The HTTP layer validates model and run identities, then supplies
plain typed values here.

Each conversation has one small manifest and one directory per turn.
Turn versions are immutable. A mutation writes its new version files
first and atomically replaces the manifest last, so readers either
see the old committed state or the complete new one. A shared
``DataRootLock`` keeps the read-check-write decision serial across
browser and desktop supervisors.

Only the tail assistant may gain versions. Appending another user
turn writes a frozen-version pointer for that assistant, then creates
the new user turn and reserved assistant placeholder. This makes old
history append-only without putting an ever-growing turn index in the
manifest.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)
from uuid import uuid4

from src.web.data_root_lock import DataRootLock


SCHEMA_VERSION = 1
CONVERSATIONS_DIR_NAME = "conversations"
MANIFEST_NAME = "manifest.json"
TURNS_DIR_NAME = "turns"
FROZEN_NAME = "frozen.json"
TRASH_DIR_NAME = ".trash"

DEFAULT_TITLE = "New conversation"
TITLE_CHARS_MAX = 200
TEXT_CHARS_MAX = 1_000_000
IDENTIFIER_CHARS_MAX = 128
METADATA_JSON_CHARS_MAX = 64 * 1024

TURN_ID_WIDTH = 8
TURN_COUNT_MAX = 1_000_000
TAIL_VERSIONS_MAX = 64
PAGE_SIZE_DEFAULT = 50
PAGE_SIZE_MAX = 100
LIST_SIZE_DEFAULT = 50
LIST_SIZE_MAX = 100
CONVERSATION_SCAN_MAX = 10_000
ALLOCATION_ATTEMPTS_MAX = 32
TRASH_ATTEMPTS_MAX = 16

JSON_DEPTH_MAX = 8
JSON_NODES_MAX = 4096
JSON_CONTAINER_ITEMS_MAX = 1024
JSON_KEY_CHARS_MAX = 256
JSON_STRING_CHARS_MAX = METADATA_JSON_CHARS_MAX
JSON_INTEGER_BITS_MAX = 4096

Role: TypeAlias = Literal["user", "assistant"]
InputMode: TypeAlias = Literal["chat", "completion"]
JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[
    JsonScalar,
    List["JsonValue"],
    Dict[str, "JsonValue"],
]
JsonObject: TypeAlias = Dict[str, JsonValue]

_CONVERSATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STORE_LOCK = DataRootLock("conversations.lock")

assert TURN_COUNT_MAX < 10**TURN_ID_WIDTH
assert PAGE_SIZE_DEFAULT <= PAGE_SIZE_MAX
assert LIST_SIZE_DEFAULT <= LIST_SIZE_MAX
assert TAIL_VERSIONS_MAX > 1
assert METADATA_JSON_CHARS_MAX < TEXT_CHARS_MAX


class ConversationNotFoundError(FileNotFoundError):
    """No conversation directory exists for a valid identifier."""


class InvalidConversationIdError(ValueError):
    """An identifier is malformed or leaves the conversations root."""


class ConversationCorruptError(RuntimeError):
    """Committed conversation data is missing or invalid."""


class ConversationStateError(RuntimeError):
    """A valid request is not allowed in the current tail state."""


class ConversationLimitError(ConversationStateError):
    """A bounded conversation resource has reached its hard limit."""


class ConversationRevisionConflictError(Exception):
    """The manifest changed after the caller's last read."""

    def __init__(
        self,
        conversation_id: str,
        expected: int,
        actual: int,
    ) -> None:
        super().__init__(
            f"conversation {conversation_id} has moved on: expected"
            f" revision {expected}, found {actual}"
        )
        self.conversation_id = conversation_id
        self.expected = expected
        self.actual = actual


class RunLinkPayload(TypedDict):
    run_id: str
    revision: int


class ManifestPayload(TypedDict):
    schema_version: int
    id: str
    title: str
    revision: int
    created_at: str
    updated_at: str
    turn_count: int
    tail_role: Optional[Role]
    tail_turn_id: Optional[str]
    tail_version: Optional[int]
    pending_assistant_id: Optional[str]


class TurnPayload(TypedDict):
    schema_version: int
    conversation_id: str
    conversation_revision: int
    turn_id: str
    index: int
    version: int
    role: Role
    created_at: str
    updated_at: str
    text: str
    partial: bool
    model_id: Optional[str]
    input_mode: Optional[InputMode]
    context_pack: JsonObject
    metadata: JsonObject
    run_link: Optional[RunLinkPayload]


class FrozenPayload(TypedDict):
    schema_version: int
    turn_id: str
    version: int


@dataclass(frozen=True)
class RunLink:
    """One assistant turn's optional saved-XAI-run reference."""

    run_id: str
    revision: int


@dataclass(frozen=True)
class ConversationManifest:
    """The O(1) committed state for one conversation."""

    id: str
    title: str
    revision: int
    created_at: str
    updated_at: str
    turn_count: int
    tail_role: Optional[Role]
    tail_turn_id: Optional[str]
    tail_version: Optional[int]
    pending_assistant_id: Optional[str]


@dataclass(frozen=True)
class TurnRecord:
    """One committed immutable version of one logical turn."""

    conversation_id: str
    conversation_revision: int
    turn_id: str
    index: int
    version: int
    role: Role
    created_at: str
    updated_at: str
    text: str
    partial: bool
    model_id: Optional[str]
    input_mode: Optional[InputMode]
    context_pack: JsonObject
    metadata: JsonObject
    run_link: Optional[RunLink]


@dataclass(frozen=True)
class ConversationMutation:
    """The manifest and tail produced by one assistant mutation."""

    manifest: ConversationManifest
    turn: TurnRecord


@dataclass(frozen=True)
class AppendResult:
    """The atomic user/assistant pair created by an append."""

    manifest: ConversationManifest
    user_turn: TurnRecord
    assistant_turn: TurnRecord


@dataclass(frozen=True)
class TurnPage:
    """A chronological bounded page from newest history backwards."""

    conversation_id: str
    revision: int
    turns: Tuple[TurnRecord, ...]
    next_before: Optional[str]
    has_more: bool


def create(
    results_dir: Path,
    *,
    title: str = DEFAULT_TITLE,
) -> ConversationManifest:
    """Create one empty conversation and publish its manifest."""
    _require_results_dir(results_dir)
    clean_title = _validate_title(title)
    with _STORE_LOCK.held(results_dir):
        root = _conversations_root(results_dir, create=True)
        conversation_id, conversation_dir = _allocate(root)
        now = _timestamp()
        manifest = ConversationManifest(
            id=conversation_id,
            title=clean_title,
            revision=1,
            created_at=now,
            updated_at=now,
            turn_count=0,
            tail_role=None,
            tail_turn_id=None,
            tail_version=None,
            pending_assistant_id=None,
        )
        try:
            (conversation_dir / TURNS_DIR_NAME).mkdir()
            _write_manifest(conversation_dir, manifest)
        except Exception:
            with contextlib.suppress(OSError):
                shutil.rmtree(conversation_dir)
            raise
    return manifest


def list_conversations(
    results_dir: Path,
    *,
    limit: int = LIST_SIZE_DEFAULT,
) -> List[ConversationManifest]:
    """Return deterministic lightweight summaries, newest first."""
    _require_results_dir(results_dir)
    _validate_limit(limit, maximum=LIST_SIZE_MAX, name="list limit")
    root = _conversations_root(results_dir, create=False)
    if root is None:
        return []
    children = _bounded_children(
        root,
        maximum=CONVERSATION_SCAN_MAX,
        label="conversation directories",
    )
    manifests: List[ConversationManifest] = []
    for child in children:
        manifest = _list_manifest(child)
        if manifest is not None:
            manifests.append(manifest)
    manifests.sort(
        key=lambda item: (item.updated_at, item.id),
        reverse=True,
    )
    return manifests[:limit]


def get_manifest(
    results_dir: Path,
    conversation_id: str,
) -> ConversationManifest:
    """Read one committed conversation manifest."""
    _require_results_dir(results_dir)
    conversation_dir = resolve_conversation_dir(
        results_dir, conversation_id
    )
    return _read_manifest(conversation_dir)


def delete(
    results_dir: Path,
    conversation_id: str,
) -> None:
    """Atomically remove a conversation from the visible namespace."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    with _STORE_LOCK.held(results_dir):
        conversation_dir = resolve_conversation_dir(
            results_dir, conversation_id
        )
        root = _conversations_root(results_dir, create=True)
        assert root is not None
        trash_root = root / TRASH_DIR_NAME
        trash_root.mkdir(parents=False, exist_ok=True)
        condemned = _trash_destination(trash_root, conversation_id)
        conversation_dir.replace(condemned)
    shutil.rmtree(condemned)


def append_user(
    results_dir: Path,
    conversation_id: str,
    *,
    expected_revision: int,
    text: str,
    model_id: str,
    input_mode: InputMode,
    metadata: Optional[Mapping[str, object]] = None,
) -> AppendResult:
    """Append a user turn and reserve its assistant in one commit."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    _validate_expected_revision(expected_revision)
    clean_text = _validate_text(text, role="user")
    clean_model_id = _validate_model_id(model_id)
    clean_input_mode = _validate_input_mode(input_mode)
    clean_metadata = _copy_json_object(metadata or {}, "metadata")

    with _STORE_LOCK.held(results_dir):
        conversation_dir = resolve_conversation_dir(
            results_dir, conversation_id
        )
        manifest = _read_manifest(conversation_dir)
        _require_revision(manifest, expected_revision)
        _require_appendable(manifest)
        result = _append_publish(
            conversation_dir=conversation_dir,
            manifest=manifest,
            text=clean_text,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            metadata=clean_metadata,
        )
    return result


def update_assistant(
    results_dir: Path,
    conversation_id: str,
    assistant_turn_id: str,
    *,
    expected_revision: int,
    text: str,
    partial: bool,
    context_pack: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> ConversationMutation:
    """Complete or revise the reserved tail assistant."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    _validate_turn_id(assistant_turn_id)
    _validate_expected_revision(expected_revision)
    clean_text = _validate_text(text, role="assistant")
    if not isinstance(partial, bool):
        raise ValueError("partial must be a boolean")
    clean_context = _copy_json_object(
        context_pack or {}, "context_pack"
    )
    clean_metadata = _copy_json_object(metadata or {}, "metadata")

    with _STORE_LOCK.held(results_dir):
        conversation_dir = resolve_conversation_dir(
            results_dir, conversation_id
        )
        manifest = _read_manifest(conversation_dir)
        _require_revision(manifest, expected_revision)
        current = _require_tail_assistant(
            conversation_dir, manifest, assistant_turn_id
        )
        changed = _turn_replacement(
            current,
            text=clean_text,
            partial=partial,
            context_pack=clean_context,
            metadata=clean_metadata,
            run_link=current.run_link,
        )
        result = _publish_tail_revision(
            conversation_dir=conversation_dir,
            manifest=manifest,
            changed=changed,
            pending_assistant_id=None,
        )
    return result


def set_run_link(
    results_dir: Path,
    conversation_id: str,
    assistant_turn_id: str,
    *,
    expected_revision: int,
    run_link: Optional[RunLink],
) -> ConversationMutation:
    """Link or unlink a saved run on the completed tail assistant."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    _validate_turn_id(assistant_turn_id)
    _validate_expected_revision(expected_revision)
    clean_link = _validate_run_link(run_link)

    with _STORE_LOCK.held(results_dir):
        conversation_dir = resolve_conversation_dir(
            results_dir, conversation_id
        )
        manifest = _read_manifest(conversation_dir)
        _require_revision(manifest, expected_revision)
        if manifest.pending_assistant_id is not None:
            raise ConversationStateError(
                "a pending assistant cannot link a saved run"
            )
        current = _require_tail_assistant(
            conversation_dir, manifest, assistant_turn_id
        )
        if current.run_link == clean_link:
            return ConversationMutation(manifest, current)
        changed = _turn_replacement(
            current,
            text=current.text,
            partial=current.partial,
            context_pack=current.context_pack,
            metadata=current.metadata,
            run_link=clean_link,
        )
        result = _publish_tail_revision(
            conversation_dir=conversation_dir,
            manifest=manifest,
            changed=changed,
            pending_assistant_id=None,
        )
    return result


def get_turns(
    results_dir: Path,
    conversation_id: str,
    *,
    before: Optional[str] = None,
    limit: int = PAGE_SIZE_DEFAULT,
) -> TurnPage:
    """Read a chronological page before an exclusive cursor."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    _validate_limit(limit, maximum=PAGE_SIZE_MAX, name="page limit")

    with _STORE_LOCK.held(results_dir):
        conversation_dir = resolve_conversation_dir(
            results_dir, conversation_id
        )
        manifest = _read_manifest(conversation_dir)
        before_index = _before_index(before, manifest.turn_count)
        finish = before_index - 1
        start = max(1, finish - limit + 1)
        turns = _read_turn_range(
            conversation_dir=conversation_dir,
            manifest=manifest,
            start=start,
            finish=finish,
        )
    has_more = start > 1
    next_before = _turn_id(start) if has_more else None
    return TurnPage(
        conversation_id=conversation_id,
        revision=manifest.revision,
        turns=tuple(turns),
        next_before=next_before,
        has_more=has_more,
    )


def resolve_conversation_dir(
    results_dir: Path,
    conversation_id: str,
) -> Path:
    """Resolve one direct conversation child, refusing traversal."""
    _require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    root = _conversations_root(results_dir, create=False)
    if root is None:
        raise ConversationNotFoundError(
            f"conversation not found: {conversation_id}"
        )
    candidate = root / conversation_id
    resolved = candidate.resolve()
    if resolved.parent != root.resolve():
        raise InvalidConversationIdError(
            f"invalid conversation id: {conversation_id}"
        )
    if candidate.is_symlink():
        raise InvalidConversationIdError(
            f"invalid conversation id: {conversation_id}"
        )
    if not candidate.is_dir():
        raise ConversationNotFoundError(
            f"conversation not found: {conversation_id}"
        )
    return candidate


def validate_conversation_id(conversation_id: str) -> None:
    """Require the generated lowercase UUID form used on disk."""
    if not isinstance(conversation_id, str):
        raise InvalidConversationIdError(
            "conversation id must be a string"
        )
    if _CONVERSATION_ID_RE.fullmatch(conversation_id) is None:
        raise InvalidConversationIdError(
            f"invalid conversation id: {conversation_id}"
        )


def manifest_to_payload(
    manifest: ConversationManifest,
) -> ManifestPayload:
    """Serialize one manifest without adding transcript content."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": manifest.id,
        "title": manifest.title,
        "revision": manifest.revision,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "turn_count": manifest.turn_count,
        "tail_role": manifest.tail_role,
        "tail_turn_id": manifest.tail_turn_id,
        "tail_version": manifest.tail_version,
        "pending_assistant_id": manifest.pending_assistant_id,
    }


def turn_to_payload(turn: TurnRecord) -> TurnPayload:
    """Serialize one current turn version for disk or HTTP."""
    link: Optional[RunLinkPayload] = None
    if turn.run_link is not None:
        link = {
            "run_id": turn.run_link.run_id,
            "revision": turn.run_link.revision,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": turn.conversation_id,
        "conversation_revision": turn.conversation_revision,
        "turn_id": turn.turn_id,
        "index": turn.index,
        "version": turn.version,
        "role": turn.role,
        "created_at": turn.created_at,
        "updated_at": turn.updated_at,
        "text": turn.text,
        "partial": turn.partial,
        "model_id": turn.model_id,
        "input_mode": turn.input_mode,
        "context_pack": _copy_json_object(
            turn.context_pack, "context_pack"
        ),
        "metadata": _copy_json_object(turn.metadata, "metadata"),
        "run_link": link,
    }


def _append_publish(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    text: str,
    model_id: str,
    input_mode: InputMode,
    metadata: JsonObject,
) -> AppendResult:
    """Write the next pair, freeze the old tail, then commit."""
    if manifest.turn_count + 2 > TURN_COUNT_MAX:
        raise ConversationLimitError(
            f"a conversation holds at most {TURN_COUNT_MAX} turns"
        )
    revision = manifest.revision + 1
    now = _timestamp()
    user_index = manifest.turn_count + 1
    assistant_index = user_index + 1
    user_turn = _new_user_turn(
        manifest=manifest,
        revision=revision,
        index=user_index,
        timestamp=now,
        text=text,
        metadata=metadata,
    )
    assistant_turn = _new_assistant_placeholder(
        manifest=manifest,
        revision=revision,
        index=assistant_index,
        timestamp=now,
        model_id=model_id,
        input_mode=input_mode,
    )
    turns_root = _require_turns_root(conversation_dir)
    user_dir = _prepare_unpublished_turn(turns_root, user_index)
    assistant_dir = _prepare_unpublished_turn(
        turns_root, assistant_index
    )
    _write_version(user_dir, user_turn)
    _write_version(assistant_dir, assistant_turn)
    _freeze_previous_tail(conversation_dir, manifest)

    updated = ConversationManifest(
        id=manifest.id,
        title=manifest.title,
        revision=revision,
        created_at=manifest.created_at,
        updated_at=now,
        turn_count=assistant_index,
        tail_role="assistant",
        tail_turn_id=assistant_turn.turn_id,
        tail_version=assistant_turn.version,
        pending_assistant_id=assistant_turn.turn_id,
    )
    _write_manifest(conversation_dir, updated)
    return AppendResult(updated, user_turn, assistant_turn)


def _new_user_turn(
    *,
    manifest: ConversationManifest,
    revision: int,
    index: int,
    timestamp: str,
    text: str,
    metadata: JsonObject,
) -> TurnRecord:
    return TurnRecord(
        conversation_id=manifest.id,
        conversation_revision=revision,
        turn_id=_turn_id(index),
        index=index,
        version=1,
        role="user",
        created_at=timestamp,
        updated_at=timestamp,
        text=text,
        partial=False,
        model_id=None,
        input_mode=None,
        context_pack={},
        metadata=_copy_json_object(metadata, "metadata"),
        run_link=None,
    )


def _new_assistant_placeholder(
    *,
    manifest: ConversationManifest,
    revision: int,
    index: int,
    timestamp: str,
    model_id: str,
    input_mode: InputMode,
) -> TurnRecord:
    return TurnRecord(
        conversation_id=manifest.id,
        conversation_revision=revision,
        turn_id=_turn_id(index),
        index=index,
        version=1,
        role="assistant",
        created_at=timestamp,
        updated_at=timestamp,
        text="",
        partial=True,
        model_id=model_id,
        input_mode=input_mode,
        context_pack={},
        metadata={},
        run_link=None,
    )


def _turn_replacement(
    current: TurnRecord,
    *,
    text: str,
    partial: bool,
    context_pack: Mapping[str, object],
    metadata: Mapping[str, object],
    run_link: Optional[RunLink],
) -> TurnRecord:
    """Build the next tail value; publication assigns its version."""
    assert current.role == "assistant"
    assert current.model_id is not None
    assert current.input_mode is not None
    return TurnRecord(
        conversation_id=current.conversation_id,
        conversation_revision=current.conversation_revision + 1,
        turn_id=current.turn_id,
        index=current.index,
        version=current.version + 1,
        role="assistant",
        created_at=current.created_at,
        updated_at=_timestamp(),
        text=text,
        partial=partial,
        model_id=current.model_id,
        input_mode=current.input_mode,
        context_pack=_copy_json_object(context_pack, "context_pack"),
        metadata=_copy_json_object(metadata, "metadata"),
        run_link=run_link,
    )


def _publish_tail_revision(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    changed: TurnRecord,
    pending_assistant_id: Optional[str],
) -> ConversationMutation:
    """Publish one immutable tail version and then its manifest."""
    if changed.version > TAIL_VERSIONS_MAX:
        raise ConversationLimitError(
            "the tail assistant has reached its version limit of"
            f" {TAIL_VERSIONS_MAX}"
        )
    assert manifest.tail_turn_id == changed.turn_id
    assert manifest.tail_version is not None
    assert changed.version == manifest.tail_version + 1
    assert changed.conversation_revision == manifest.revision + 1
    turn_dir = _turn_dir(
        _require_turns_root(conversation_dir), changed.index
    )
    version_path = _version_path(turn_dir, changed.version)
    _remove_unpublished_version(version_path)
    _write_version(turn_dir, changed)

    updated = ConversationManifest(
        id=manifest.id,
        title=manifest.title,
        revision=changed.conversation_revision,
        created_at=manifest.created_at,
        updated_at=changed.updated_at,
        turn_count=manifest.turn_count,
        tail_role="assistant",
        tail_turn_id=changed.turn_id,
        tail_version=changed.version,
        pending_assistant_id=pending_assistant_id,
    )
    _write_manifest(conversation_dir, updated)
    return ConversationMutation(updated, changed)


def _require_tail_assistant(
    conversation_dir: Path,
    manifest: ConversationManifest,
    assistant_turn_id: str,
) -> TurnRecord:
    if manifest.tail_role != "assistant":
        raise ConversationStateError(
            "the conversation tail is not an assistant"
        )
    if manifest.tail_turn_id != assistant_turn_id:
        raise ConversationStateError(
            "only the tail assistant may be changed"
        )
    if manifest.tail_version is None:
        raise ConversationCorruptError(
            "the assistant tail has no version"
        )
    index = _turn_index(assistant_turn_id)
    current = _read_turn_version(
        conversation_dir=conversation_dir,
        manifest=manifest,
        index=index,
        version=manifest.tail_version,
    )
    if current.role != "assistant":
        raise ConversationCorruptError(
            "the manifest tail does not name an assistant"
        )
    return current


def _require_appendable(manifest: ConversationManifest) -> None:
    if manifest.pending_assistant_id is not None:
        raise ConversationStateError(
            "complete or stop the pending assistant before appending"
        )
    if manifest.turn_count == 0:
        return
    if manifest.tail_role != "assistant":
        raise ConversationCorruptError(
            "a non-empty conversation must end with an assistant"
        )


def _freeze_previous_tail(
    conversation_dir: Path,
    manifest: ConversationManifest,
) -> None:
    """Publish the old tail's final-version pointer before moving."""
    if manifest.turn_count == 0:
        return
    if manifest.tail_turn_id is None:
        raise ConversationCorruptError(
            "a non-empty conversation has no tail id"
        )
    if manifest.tail_version is None:
        raise ConversationCorruptError(
            "a non-empty conversation has no tail version"
        )
    index = _turn_index(manifest.tail_turn_id)
    turn_dir = _turn_dir(_require_turns_root(conversation_dir), index)
    payload: FrozenPayload = {
        "schema_version": SCHEMA_VERSION,
        "turn_id": manifest.tail_turn_id,
        "version": manifest.tail_version,
    }
    _write_json_atomic(turn_dir / FROZEN_NAME, payload)


def _read_turn_range(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    start: int,
    finish: int,
) -> List[TurnRecord]:
    if finish < start:
        return []
    count = finish - start + 1
    assert count <= PAGE_SIZE_MAX
    turns: List[TurnRecord] = []
    for index in range(start, finish + 1):
        version = _current_turn_version(
            conversation_dir, manifest, index
        )
        turns.append(
            _read_turn_version(
                conversation_dir=conversation_dir,
                manifest=manifest,
                index=index,
                version=version,
            )
        )
    return turns


def _current_turn_version(
    conversation_dir: Path,
    manifest: ConversationManifest,
    index: int,
) -> int:
    if index % 2 == 1:
        return 1
    turn_id = _turn_id(index)
    if turn_id == manifest.tail_turn_id:
        if manifest.tail_version is None:
            raise ConversationCorruptError(
                "the tail turn has no current version"
            )
        return manifest.tail_version
    turn_dir = _turn_dir(_require_turns_root(conversation_dir), index)
    raw = _read_json_object(turn_dir / FROZEN_NAME, "frozen pointer")
    expected = frozenset({"schema_version", "turn_id", "version"})
    _require_keys(raw, expected, "frozen pointer")
    if _require_int(raw["schema_version"], "schema_version") != 1:
        raise ConversationCorruptError(
            "unsupported frozen-pointer schema"
        )
    if raw["turn_id"] != turn_id:
        raise ConversationCorruptError(
            f"frozen pointer does not name turn {turn_id}"
        )
    version = _require_int(raw["version"], "version")
    if not 1 <= version <= TAIL_VERSIONS_MAX:
        raise ConversationCorruptError(
            f"frozen turn {turn_id} has invalid version {version}"
        )
    return version


def _read_turn_version(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    index: int,
    version: int,
) -> TurnRecord:
    turns_root = _require_turns_root(conversation_dir)
    path = _version_path(_turn_dir(turns_root, index), version)
    raw = _read_json_object(path, "turn version")
    turn = _parse_turn(raw)
    expected_id = _turn_id(index)
    if turn.conversation_id != manifest.id:
        raise ConversationCorruptError(
            f"turn {expected_id} names another conversation"
        )
    if turn.turn_id != expected_id or turn.index != index:
        raise ConversationCorruptError(
            f"turn directory {expected_id} contains another turn"
        )
    if turn.version != version:
        raise ConversationCorruptError(
            f"turn {expected_id} has the wrong version"
        )
    expected_role: Role = "user" if index % 2 == 1 else "assistant"
    if turn.role != expected_role:
        raise ConversationCorruptError(
            f"turn {expected_id} breaks role order"
        )
    if turn.conversation_revision > manifest.revision:
        raise ConversationCorruptError(
            f"turn {expected_id} is newer than its manifest"
        )
    if (
        turn.turn_id == manifest.tail_turn_id
        and turn.conversation_revision != manifest.revision
    ):
        raise ConversationCorruptError(
            f"tail turn {expected_id} has a stale revision"
        )
    if turn.turn_id == manifest.pending_assistant_id:
        _validate_pending_turn(turn)
    return turn


def _parse_turn(raw: Dict[str, object]) -> TurnRecord:
    expected = frozenset(TurnPayload.__required_keys__)
    _require_keys(raw, expected, "turn version")
    schema = _require_int(raw["schema_version"], "schema_version")
    if schema != SCHEMA_VERSION:
        raise ConversationCorruptError(
            f"unsupported turn schema version {schema}"
        )
    role = _stored_role(raw["role"])
    model_id = _stored_optional_string(raw["model_id"], "model_id")
    input_mode = _stored_input_mode(raw["input_mode"])
    text = _stored_string(raw["text"], "text", TEXT_CHARS_MAX)
    partial = _stored_bool(raw["partial"], "partial")
    _validate_stored_role_fields(
        role=role,
        text=text,
        partial=partial,
        model_id=model_id,
        input_mode=input_mode,
    )
    turn = TurnRecord(
        conversation_id=_stored_string(
            raw["conversation_id"],
            "conversation_id",
            IDENTIFIER_CHARS_MAX,
        ),
        conversation_revision=_require_positive_int(
            raw["conversation_revision"], "conversation_revision"
        ),
        turn_id=_stored_string(
            raw["turn_id"], "turn_id", TURN_ID_WIDTH
        ),
        index=_require_positive_int(raw["index"], "index"),
        version=_require_positive_int(raw["version"], "version"),
        role=role,
        created_at=_stored_timestamp(raw["created_at"], "created_at"),
        updated_at=_stored_timestamp(raw["updated_at"], "updated_at"),
        text=text,
        partial=partial,
        model_id=model_id,
        input_mode=input_mode,
        context_pack=_stored_json_object(
            raw["context_pack"], "context_pack"
        ),
        metadata=_stored_json_object(raw["metadata"], "metadata"),
        run_link=_stored_run_link(raw["run_link"]),
    )
    _validate_turn_record(turn)
    return turn


def _validate_stored_role_fields(
    *,
    role: Role,
    text: str,
    partial: bool,
    model_id: Optional[str],
    input_mode: Optional[InputMode],
) -> None:
    if role == "user":
        if text.strip() == "":
            raise ConversationCorruptError("a user turn is empty")
        if partial:
            raise ConversationCorruptError("a user turn is partial")
        if model_id is not None or input_mode is not None:
            raise ConversationCorruptError(
                "a user turn carries assistant model fields"
            )
        return
    if model_id is None or input_mode is None:
        raise ConversationCorruptError(
            "an assistant turn is missing model fields"
        )
    try:
        _validate_model_id(model_id)
        _validate_input_mode(input_mode)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def _validate_turn_record(turn: TurnRecord) -> None:
    if turn.created_at > turn.updated_at:
        raise ConversationCorruptError(
            f"turn {turn.turn_id} timestamps run backwards"
        )
    if turn.role == "assistant":
        return
    if turn.context_pack:
        raise ConversationCorruptError(
            "a user turn cannot carry a context pack"
        )
    if turn.run_link is not None:
        raise ConversationCorruptError(
            "a user turn cannot link a saved run"
        )


def _validate_pending_turn(turn: TurnRecord) -> None:
    if turn.role != "assistant":
        raise ConversationCorruptError(
            "the pending turn is not an assistant"
        )
    if turn.version != 1:
        raise ConversationCorruptError(
            "the pending assistant is not its reserved version"
        )
    if turn.text != "" or not turn.partial:
        raise ConversationCorruptError(
            "the pending assistant is not an empty"
            " partial placeholder"
        )
    if (
        turn.context_pack
        or turn.metadata
        or turn.run_link is not None
    ):
        raise ConversationCorruptError(
            "the pending assistant carries terminal data"
        )


def _parse_manifest(
    raw: Dict[str, object],
    *,
    expected_id: str,
) -> ConversationManifest:
    expected = frozenset(ManifestPayload.__required_keys__)
    _require_keys(raw, expected, "conversation manifest")
    schema = _require_int(raw["schema_version"], "schema_version")
    if schema != SCHEMA_VERSION:
        raise ConversationCorruptError(
            f"unsupported conversation schema version {schema}"
        )
    conversation_id = _stored_string(
        raw["id"], "id", IDENTIFIER_CHARS_MAX
    )
    if conversation_id != expected_id:
        raise ConversationCorruptError(
            "manifest id does not match its directory"
        )
    try:
        validate_conversation_id(conversation_id)
        title = _validate_title(
            _stored_string(raw["title"], "title", TITLE_CHARS_MAX)
        )
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc
    manifest = ConversationManifest(
        id=conversation_id,
        title=title,
        revision=_require_positive_int(raw["revision"], "revision"),
        created_at=_stored_timestamp(raw["created_at"], "created_at"),
        updated_at=_stored_timestamp(raw["updated_at"], "updated_at"),
        turn_count=_require_nonnegative_int(
            raw["turn_count"], "turn_count"
        ),
        tail_role=_stored_optional_role(raw["tail_role"]),
        tail_turn_id=_stored_optional_string(
            raw["tail_turn_id"], "tail_turn_id"
        ),
        tail_version=_stored_optional_int(
            raw["tail_version"], "tail_version"
        ),
        pending_assistant_id=_stored_optional_string(
            raw["pending_assistant_id"],
            "pending_assistant_id",
        ),
    )
    _validate_manifest_state(manifest)
    return manifest


def _validate_manifest_state(
    manifest: ConversationManifest,
) -> None:
    if manifest.turn_count > TURN_COUNT_MAX:
        raise ConversationCorruptError("turn count exceeds its limit")
    if manifest.turn_count % 2 != 0:
        raise ConversationCorruptError("turn count must be even")
    minimum_revision = 1 + manifest.turn_count // 2
    if manifest.revision < minimum_revision:
        raise ConversationCorruptError(
            "manifest revision cannot account for its turns"
        )
    if manifest.created_at > manifest.updated_at:
        raise ConversationCorruptError(
            "manifest timestamps run backwards"
        )
    if manifest.turn_count == 0:
        _validate_empty_manifest(manifest)
        return
    _validate_nonempty_manifest(manifest)


def _validate_empty_manifest(
    manifest: ConversationManifest,
) -> None:
    fields = (
        manifest.tail_role,
        manifest.tail_turn_id,
        manifest.tail_version,
        manifest.pending_assistant_id,
    )
    if any(value is not None for value in fields):
        raise ConversationCorruptError(
            "an empty conversation cannot have a tail"
        )


def _validate_nonempty_manifest(
    manifest: ConversationManifest,
) -> None:
    if manifest.tail_role != "assistant":
        raise ConversationCorruptError(
            "a non-empty conversation must have an assistant tail"
        )
    expected_tail = _turn_id(manifest.turn_count)
    if manifest.tail_turn_id != expected_tail:
        raise ConversationCorruptError(
            "manifest tail id does not match its turn count"
        )
    version = manifest.tail_version
    if version is None or not 1 <= version <= TAIL_VERSIONS_MAX:
        raise ConversationCorruptError(
            "manifest tail version is outside its limit"
        )
    pending = manifest.pending_assistant_id
    if pending is not None and pending != expected_tail:
        raise ConversationCorruptError(
            "pending assistant does not name the tail"
        )
    if (version == 1) != (pending is not None):
        raise ConversationCorruptError(
            "only a reserved assistant may remain at version one"
        )


def _read_manifest(conversation_dir: Path) -> ConversationManifest:
    raw = _read_json_object(
        conversation_dir / MANIFEST_NAME, "conversation manifest"
    )
    return _parse_manifest(raw, expected_id=conversation_dir.name)


def _list_manifest(path: Path) -> Optional[ConversationManifest]:
    if path.name.startswith("."):
        return None
    if _CONVERSATION_ID_RE.fullmatch(path.name) is None:
        return None
    if path.is_symlink() or not path.is_dir():
        return None
    if not (path / MANIFEST_NAME).is_file():
        return None
    try:
        return _read_manifest(path)
    except (ConversationCorruptError, OSError, ValueError):
        return None


def _write_manifest(
    conversation_dir: Path,
    manifest: ConversationManifest,
) -> None:
    """Publish the commit marker after all version files exist."""
    _validate_manifest_state(manifest)
    _write_json_atomic(
        conversation_dir / MANIFEST_NAME,
        manifest_to_payload(manifest),
    )


def _write_version(turn_dir: Path, turn: TurnRecord) -> None:
    path = _version_path(turn_dir, turn.version)
    if path.exists():
        raise ConversationCorruptError(
            f"immutable turn version already exists: {path.name}"
        )
    _write_json_atomic(path, turn_to_payload(turn))


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write complete JSON beside its target, then replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _read_json_object(path: Path, label: str) -> Dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
        raw: object = json.loads(text)
    except (OSError, ValueError) as exc:
        raise ConversationCorruptError(
            f"{label} is unreadable: {path}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConversationCorruptError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise ConversationCorruptError(
            f"{label} contains a non-string key"
        )
    return raw


def _require_keys(
    raw: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConversationCorruptError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _conversations_root(
    results_dir: Path,
    *,
    create: bool,
) -> Optional[Path]:
    root = results_dir / CONVERSATIONS_DIR_NAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise ConversationCorruptError(
            "the conversations root is not a real directory"
        )
    if root.resolve().parent != results_dir.resolve():
        raise ConversationCorruptError(
            "the conversations root leaves the data root"
        )
    return root


def _require_turns_root(conversation_dir: Path) -> Path:
    turns_root = conversation_dir / TURNS_DIR_NAME
    if turns_root.is_symlink() or not turns_root.is_dir():
        raise ConversationCorruptError(
            "the conversation turns directory is missing or unsafe"
        )
    if turns_root.resolve().parent != conversation_dir.resolve():
        raise ConversationCorruptError(
            "the turns directory leaves its conversation"
        )
    return turns_root


def _allocate(root: Path) -> Tuple[str, Path]:
    for _attempt in range(ALLOCATION_ATTEMPTS_MAX):
        conversation_id = uuid4().hex
        path = root / conversation_id
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return conversation_id, path
    raise ConversationLimitError(
        "could not allocate a unique conversation id after"
        f" {ALLOCATION_ATTEMPTS_MAX} attempts"
    )


def _trash_destination(root: Path, conversation_id: str) -> Path:
    for _attempt in range(TRASH_ATTEMPTS_MAX):
        candidate = root / f"{conversation_id}.{uuid4().hex}"
        if not candidate.exists():
            return candidate
    raise ConversationLimitError(
        "could not allocate conversation trash after"
        f" {TRASH_ATTEMPTS_MAX} attempts"
    )


def _prepare_unpublished_turn(turns_root: Path, index: int) -> Path:
    path = _turn_dir(turns_root, index)
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir()
    return path


def _remove_unpublished_version(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ConversationCorruptError(
            f"unpublished version path is unsafe: {path}"
        )
    path.unlink()


def _turn_dir(turns_root: Path, index: int) -> Path:
    return turns_root / _turn_id(index)


def _version_path(turn_dir: Path, version: int) -> Path:
    if not 1 <= version <= TAIL_VERSIONS_MAX:
        raise ConversationCorruptError(
            f"turn version outside 1..{TAIL_VERSIONS_MAX}: {version}"
        )
    return turn_dir / f"{version:08d}.json"


def _turn_id(index: int) -> str:
    if not 1 <= index <= TURN_COUNT_MAX:
        raise ConversationCorruptError(
            f"turn index outside 1..{TURN_COUNT_MAX}: {index}"
        )
    return f"{index:0{TURN_ID_WIDTH}d}"


def _turn_index(turn_id: str) -> int:
    _validate_turn_id(turn_id)
    return int(turn_id)


def _validate_turn_id(turn_id: str) -> None:
    if not isinstance(turn_id, str):
        raise ValueError("turn id must be a string")
    if len(turn_id) != TURN_ID_WIDTH or not turn_id.isascii():
        raise ValueError(f"invalid turn id: {turn_id}")
    if not turn_id.isdigit():
        raise ValueError(f"invalid turn id: {turn_id}")
    index = int(turn_id)
    if not 1 <= index <= TURN_COUNT_MAX:
        raise ValueError(f"invalid turn id: {turn_id}")
    if _turn_id(index) != turn_id:
        raise ValueError(f"invalid turn id: {turn_id}")


def _before_index(before: Optional[str], turn_count: int) -> int:
    if before is None:
        return turn_count + 1
    _validate_turn_id(before)
    before_index = int(before)
    if before_index > turn_count + 1:
        raise ValueError(
            f"before cursor {before} is beyond this conversation"
        )
    return before_index


def _validate_title(title: str) -> str:
    if not isinstance(title, str):
        raise ValueError("conversation title must be a string")
    clean = title.strip()
    if clean == "":
        raise ValueError("conversation title must not be blank")
    if len(clean) > TITLE_CHARS_MAX:
        raise ValueError(
            f"conversation title exceeds {TITLE_CHARS_MAX} characters"
        )
    return clean


def _validate_text(text: str, *, role: Role) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{role} text must be a string")
    if len(text) > TEXT_CHARS_MAX:
        raise ValueError(
            f"{role} text exceeds {TEXT_CHARS_MAX} characters"
        )
    if role == "user" and text.strip() == "":
        raise ValueError("user text must not be blank")
    return text


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str):
        raise ValueError("model id must be a string")
    if _MODEL_ID_RE.fullmatch(model_id) is None:
        raise ValueError(f"invalid model id: {model_id}")
    return model_id


def _validate_input_mode(input_mode: str) -> InputMode:
    if input_mode == "chat":
        return "chat"
    if input_mode == "completion":
        return "completion"
    raise ValueError(f"invalid input mode: {input_mode}")


def _validate_run_link(
    run_link: Optional[RunLink],
) -> Optional[RunLink]:
    if run_link is None:
        return None
    if not isinstance(run_link, RunLink):
        raise ValueError("run link must be a RunLink")
    if _RUN_ID_RE.fullmatch(run_link.run_id) is None:
        raise ValueError(f"invalid run id: {run_link.run_id}")
    if isinstance(run_link.revision, bool):
        raise ValueError("run revision must be an integer")
    if not isinstance(run_link.revision, int):
        raise ValueError("run revision must be an integer")
    if run_link.revision < 0:
        raise ValueError("run revision must not be negative")
    return run_link


def _validate_expected_revision(revision: int) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("expected revision must be an integer")
    if revision < 1:
        raise ValueError("expected revision must be positive")


def _require_revision(
    manifest: ConversationManifest,
    expected_revision: int,
) -> None:
    if manifest.revision != expected_revision:
        raise ConversationRevisionConflictError(
            manifest.id,
            expected_revision,
            manifest.revision,
        )


def _validate_limit(limit: int, *, maximum: int, name: str) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= limit <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _require_results_dir(results_dir: Path) -> None:
    if not isinstance(results_dir, Path):
        raise TypeError("results_dir must be a Path")


def _copy_json_object(
    value: Mapping[str, object],
    label: str,
) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    budget = [0]
    copied = _copy_json_value(
        value,
        label=label,
        depth=0,
        budget=budget,
    )
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be an object")
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if len(encoded) > METADATA_JSON_CHARS_MAX:
        raise ValueError(
            f"{label} exceeds {METADATA_JSON_CHARS_MAX} JSON"
            " characters"
        )
    return copied


def _copy_json_value(
    value: object,
    *,
    label: str,
    depth: int,
    budget: List[int],
) -> JsonValue:
    budget[0] += 1
    if budget[0] > JSON_NODES_MAX:
        raise ValueError(f"{label} has too many JSON values")
    if depth > JSON_DEPTH_MAX:
        raise ValueError(f"{label} is nested too deeply")
    if isinstance(value, list):
        return _copy_json_list(
            value,
            label=label,
            depth=depth,
            budget=budget,
        )
    if isinstance(value, Mapping):
        return _copy_json_mapping(
            value,
            label=label,
            depth=depth,
            budget=budget,
        )
    return _copy_json_scalar(value, label)


def _copy_json_scalar(value: object, label: str) -> JsonScalar:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > JSON_STRING_CHARS_MAX:
            raise ValueError(f"{label} contains an oversized string")
        return value
    if isinstance(value, int):
        if value.bit_length() > JSON_INTEGER_BITS_MAX:
            raise ValueError(f"{label} contains an oversized integer")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    raise ValueError(
        f"{label} contains unsupported {type(value).__name__}"
    )


def _copy_json_list(
    value: List[object],
    *,
    label: str,
    depth: int,
    budget: List[int],
) -> List[JsonValue]:
    if len(value) > JSON_CONTAINER_ITEMS_MAX:
        raise ValueError(f"{label} contains an oversized list")
    copied: List[JsonValue] = []
    for item in value:
        copied.append(
            _copy_json_value(
                item,
                label=label,
                depth=depth + 1,
                budget=budget,
            )
        )
    return copied


def _copy_json_mapping(
    value: Mapping[object, object],
    *,
    label: str,
    depth: int,
    budget: List[int],
) -> JsonObject:
    if len(value) > JSON_CONTAINER_ITEMS_MAX:
        raise ValueError(f"{label} contains an oversized object")
    copied: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} contains a non-string key")
        if len(key) > JSON_KEY_CHARS_MAX:
            raise ValueError(f"{label} contains an oversized key")
        copied[key] = _copy_json_value(
            item,
            label=label,
            depth=depth + 1,
            budget=budget,
        )
    return copied


def _stored_json_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ConversationCorruptError(f"{label} must be an object")
    try:
        return _copy_json_object(value, label)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def _stored_run_link(value: object) -> Optional[RunLink]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConversationCorruptError("run_link must be an object")
    _require_keys(
        value,
        frozenset({"run_id", "revision"}),
        "run_link",
    )
    link = RunLink(
        run_id=_stored_string(
            value["run_id"], "run_id", IDENTIFIER_CHARS_MAX
        ),
        revision=_require_nonnegative_int(
            value["revision"], "run revision"
        ),
    )
    try:
        return _validate_run_link(link)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def _stored_role(value: object) -> Role:
    if value == "user":
        return "user"
    if value == "assistant":
        return "assistant"
    raise ConversationCorruptError(f"invalid turn role: {value!r}")


def _stored_optional_role(value: object) -> Optional[Role]:
    if value is None:
        return None
    return _stored_role(value)


def _stored_input_mode(value: object) -> Optional[InputMode]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConversationCorruptError("input_mode must be a string")
    try:
        return _validate_input_mode(value)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def _stored_optional_string(
    value: object,
    name: str,
) -> Optional[str]:
    if value is None:
        return None
    return _stored_string(value, name, IDENTIFIER_CHARS_MAX)


def _stored_string(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ConversationCorruptError(f"{name} must be a string")
    if len(value) > maximum:
        raise ConversationCorruptError(
            f"{name} exceeds {maximum} characters"
        )
    return value


def _stored_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConversationCorruptError(f"{name} must be a boolean")
    return value


def _stored_optional_int(value: object, name: str) -> Optional[int]:
    if value is None:
        return None
    return _require_int(value, name)


def _require_positive_int(value: object, name: str) -> int:
    result = _require_int(value, name)
    if result < 1:
        raise ConversationCorruptError(f"{name} must be positive")
    return result


def _require_nonnegative_int(value: object, name: str) -> int:
    result = _require_int(value, name)
    if result < 0:
        raise ConversationCorruptError(f"{name} must not be negative")
    return result


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConversationCorruptError(f"{name} must be an integer")
    return value


def _stored_timestamp(value: object, name: str) -> str:
    text = _stored_string(value, name, 40)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConversationCorruptError(
            f"{name} is not an ISO timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ConversationCorruptError(
            f"{name} must include a timezone"
        )
    return text


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _bounded_children(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> List[Path]:
    children = list(islice(path.iterdir(), maximum + 1))
    if len(children) > maximum:
        raise ConversationLimitError(
            f"{label} exceed the scan limit of {maximum}"
        )
    return children
