"""Shared durable conversation records, codecs, and v1 persistence.

This module is the bottom of the conversation-store dependency graph.
It imports no branch or public facade module. Schema-v2 branch logic
builds on these strict records, validators, JSON helpers, and the
byte-compatible schema-v1 linear codec.
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

from src import conversation_identity
from src.web import conversation_generation
from src.web.data_root_lock import DataRootLock


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
CONVERSATIONS_DIR_NAME = "conversations"
MANIFEST_NAME = "manifest.json"
TURNS_DIR_NAME = "turns"
BRANCHES_DIR_NAME = "branches"
OPERATIONS_DIR_NAME = "operations"
FROZEN_NAME = "frozen.json"
TRASH_DIR_NAME = ".trash"

DEFAULT_TITLE = "New conversation"
TITLE_CHARS_MAX = 200
TEXT_CHARS_MAX = 1_000_000
IDENTIFIER_CHARS_MAX = 128
METADATA_JSON_CHARS_MAX = 64 * 1024

TURN_ID_WIDTH = conversation_identity.TURN_ID_WIDTH
TURN_COUNT_MAX = conversation_identity.TURN_INDEX_MAX
TAIL_VERSIONS_MAX = 64
BRANCH_COUNT_MAX = 256
BRANCH_SIBLINGS_MAX = 16
BRANCH_DEPTH_MAX = 32
BRANCH_LOCAL_TURNS_MAX = TURN_COUNT_MAX
BRANCH_REVISION_MAX = 10_000_000
CATALOG_REVISION_MAX = 1_000_000
BRANCH_DIRECTORY_SCAN_MAX = BRANCH_COUNT_MAX * 2
OPERATION_RECEIPT_SCAN_MAX = BRANCH_COUNT_MAX
PAGE_SIZE_DEFAULT = 50
PAGE_SIZE_MAX = 100
LIST_SIZE_DEFAULT = 50
LIST_SIZE_MAX = 100
CONVERSATION_SCAN_MAX = 10_000
ALLOCATION_ATTEMPTS_MAX = 32
OPERATION_TEMP_SCAN_MAX = ALLOCATION_ATTEMPTS_MAX
TRASH_ATTEMPTS_MAX = 16

JSON_DEPTH_MAX = 8
JSON_NODES_MAX = 4096
JSON_CONTAINER_ITEMS_MAX = 1024
JSON_KEY_CHARS_MAX = 256
JSON_STRING_CHARS_MAX = METADATA_JSON_CHARS_MAX
JSON_INTEGER_BITS_MAX = 4096
JSON_TEXT_CHAR_BYTES_MAX = 6
JSON_SERIALIZED_CHAR_BYTES_MAX = 4


def _turn_json_envelope_bytes_max() -> int:
    """Bound every non-text, non-metadata byte in a turn file."""
    integer = (1 << JSON_INTEGER_BITS_MAX) - 1
    escaped_identifier = "\0" * IDENTIFIER_CHARS_MAX
    escaped_timestamp = "\0" * 40
    payload = {
        "schema_version": integer,
        "conversation_id": escaped_identifier,
        "conversation_revision": integer,
        "turn_id": escaped_identifier,
        "index": integer,
        "version": integer,
        "role": "assistant",
        "created_at": escaped_timestamp,
        "updated_at": escaped_timestamp,
        "text": "",
        "partial": False,
        "model_id": escaped_identifier,
        "input_mode": "completion",
        "context_pack": {},
        "metadata": {},
        "run_link": {
            "run_id": escaped_identifier,
            "revision": integer,
        },
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return len(encoded)


# ``ensure_ascii=False`` emits at most six UTF-8 bytes per source text
# character: a control character becomes ``\uXXXX``. Metadata is
# bounded after JSON serialization, whose remaining Unicode characters
# take at most four UTF-8 bytes. The measured envelope uses every
# other string and integer field at its validator's maximum.
TURN_JSON_ENVELOPE_BYTES_MAX = _turn_json_envelope_bytes_max()
CURRENT_JSON_FILE_BYTES_MAX = (
    TEXT_CHARS_MAX * JSON_TEXT_CHAR_BYTES_MAX
    + 2
    * METADATA_JSON_CHARS_MAX
    * JSON_SERIALIZED_CHAR_BYTES_MAX
    + TURN_JSON_ENVELOPE_BYTES_MAX
)
# Schema-v1 writers used json.dump's ASCII escaping, where one astral
# code point occupies twelve bytes as a surrogate pair. Keep those
# already-published maximum-size turns readable under the new codec.
LEGACY_JSON_FILE_BYTES_MAX = 16 * 1024 * 1024
JSON_FILE_BYTES_MAX = max(
    CURRENT_JSON_FILE_BYTES_MAX,
    LEGACY_JSON_FILE_BYTES_MAX,
)

Role: TypeAlias = Literal["user", "assistant"]
InputMode: TypeAlias = Literal["chat", "completion"]
BranchStorage: TypeAlias = Literal["branch", "legacy"]
ForkOperationKind: TypeAlias = Literal[
    "edit_user",
    "delete_path",
    "retry_assistant",
]
JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[
    JsonScalar,
    List["JsonValue"],
    Dict[str, "JsonValue"],
]
JsonObject: TypeAlias = Dict[str, JsonValue]
GenerationConfigurationPayload = (
    conversation_generation.GenerationConfigurationPayload
)
PENDING_GENERATION_KEY = (
    conversation_generation.PENDING_GENERATION_KEY
)
GENERATION_CONFIGURATION_CODEC_VERSION = (
    conversation_generation.GENERATION_CONFIGURATION_CODEC_VERSION
)

CONVERSATION_ID_RE = conversation_identity.CONVERSATION_ID_RE
BRANCH_ID_RE = conversation_identity.BRANCH_ID_RE
OPERATION_ID_PATTERN = conversation_identity.OPERATION_ID_PATTERN
OPERATION_ID_RE = conversation_identity.OPERATION_ID_RE
OPAQUE_TURN_ID_RE = conversation_identity.OPAQUE_TURN_ID_RE
SHA256_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STORE_LOCK = DataRootLock("conversations.lock")

assert TURN_COUNT_MAX < 10**TURN_ID_WIDTH
assert PAGE_SIZE_DEFAULT <= PAGE_SIZE_MAX
assert LIST_SIZE_DEFAULT <= LIST_SIZE_MAX
assert TAIL_VERSIONS_MAX > 1
assert METADATA_JSON_CHARS_MAX < TEXT_CHARS_MAX
assert BRANCH_SIBLINGS_MAX < BRANCH_COUNT_MAX
assert BRANCH_DEPTH_MAX < BRANCH_COUNT_MAX
assert BRANCH_LOCAL_TURNS_MAX <= TURN_COUNT_MAX
assert BRANCH_COUNT_MAX < BRANCH_DIRECTORY_SCAN_MAX
assert OPERATION_RECEIPT_SCAN_MAX == BRANCH_COUNT_MAX
assert OPERATION_TEMP_SCAN_MAX < OPERATION_RECEIPT_SCAN_MAX
assert SCHEMA_VERSION > LEGACY_SCHEMA_VERSION
assert TURN_JSON_ENVELOPE_BYTES_MAX < 16 * 1024
assert JSON_FILE_BYTES_MAX > TEXT_CHARS_MAX * 4


class ConversationNotFoundError(FileNotFoundError):
    """No conversation directory exists for a valid identifier."""


class InvalidConversationIdError(ValueError):
    """A conversation identifier is malformed or unsafe."""


class InvalidBranchIdError(ValueError):
    """A branch identifier is malformed or unsafe."""


class InvalidOperationIdError(ValueError):
    """A fork operation identifier is malformed or unsafe."""


class BranchNotFoundError(FileNotFoundError):
    """No catalog branch exists for a valid identifier."""

    def __init__(
        self,
        conversation_id: str,
        branch_id: str,
    ) -> None:
        super().__init__(f"branch not found: {branch_id}")
        self.conversation_id = conversation_id
        self.branch_id = branch_id


class ConversationCorruptError(RuntimeError):
    """Committed conversation data is missing or invalid."""


class ConversationStateError(RuntimeError):
    """A valid request is not allowed in the current state."""


class ConversationLimitError(ConversationStateError):
    """A bounded conversation resource reached its hard limit."""


class ConversationRevisionConflictError(Exception):
    """A branch or v1 manifest changed after the caller read it."""

    def __init__(
        self,
        conversation_id: str,
        expected: int,
        actual: int,
        branch_id: Optional[str] = None,
    ) -> None:
        target = f"conversation {conversation_id}"
        if branch_id is not None:
            target = f"branch {branch_id}"
        super().__init__(
            f"{target} has moved on: expected revision {expected},"
            f" found {actual}"
        )
        self.conversation_id = conversation_id
        self.expected = expected
        self.actual = actual
        self.branch_id = branch_id


class ConversationCatalogRevisionConflictError(Exception):
    """The branch catalog changed after the caller read it."""

    def __init__(
        self,
        conversation_id: str,
        expected: int,
        actual: int,
    ) -> None:
        super().__init__(
            f"conversation {conversation_id} catalog has moved on:"
            f" expected revision {expected}, found {actual}"
        )
        self.conversation_id = conversation_id
        self.expected = expected
        self.actual = actual


class ConversationOperationConflictError(Exception):
    """One operation id was already committed with other semantics."""

    def __init__(self, operation_id: str) -> None:
        super().__init__(
            "operation id was already committed with a different"
            f" request: {operation_id}"
        )
        self.operation_id = operation_id


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


class CatalogPayload(TypedDict):
    schema_version: int
    id: str
    title: str
    catalog_revision: int
    created_at: str
    updated_at: str
    default_branch_id: str
    branch_ids: List[str]


class BranchManifestPayload(TypedDict):
    schema_version: int
    conversation_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    prefix_turn_count: int
    revision: int
    created_at: str
    updated_at: str
    turn_count: int
    tail_role: Optional[Role]
    tail_turn_id: Optional[str]
    tail_version: Optional[int]
    pending_assistant_id: Optional[str]
    depth: int
    storage: BranchStorage
    legacy_turn_count: int


class OperationReceiptPayload(TypedDict):
    schema_version: int
    operation_id: str
    request_digest: str
    kind: ForkOperationKind
    source_branch_id: str
    target_turn_id: str
    result_branch_id: str
    catalog_revision: int
    removed_turn_count: Optional[int]


@dataclass(frozen=True)
class RunLink:
    """One assistant turn's optional saved-XAI-run reference."""

    run_id: str
    revision: int


@dataclass(frozen=True)
class ConversationManifest:
    """A v1 manifest or schema-v2 branch projection."""

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
    schema_version: int = LEGACY_SCHEMA_VERSION
    catalog_revision: int = 0
    default_branch_id: Optional[str] = None
    branch_id: Optional[str] = None


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
    schema_version: int = LEGACY_SCHEMA_VERSION
    branch_id: Optional[str] = None


@dataclass(frozen=True)
class ConversationCatalog:
    """The authoritative ordered schema-v2 branch catalog."""

    conversation_id: str
    title: str
    revision: int
    created_at: str
    updated_at: str
    default_branch_id: str
    branch_ids: Tuple[str, ...]
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class BranchRecord:
    """One branch's bounded shared-prefix state."""

    conversation_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    prefix_turn_count: int
    revision: int
    created_at: str
    updated_at: str
    turn_count: int
    tail_role: Optional[Role]
    tail_turn_id: Optional[str]
    tail_version: Optional[int]
    pending_assistant_id: Optional[str]
    depth: int


@dataclass(frozen=True)
class ForkOperation:
    """One validated fork request's stable semantic identity."""

    operation_id: str
    request_digest: str
    kind: ForkOperationKind
    source_branch_id: str
    target_turn_id: str
    legacy_request_digest: Optional[str] = None


@dataclass(frozen=True)
class OperationReceipt:
    """One immutable fork publication receipt."""

    operation_id: str
    request_digest: str
    kind: ForkOperationKind
    source_branch_id: str
    target_turn_id: str
    result_branch_id: str
    catalog_revision: int
    removed_turn_count: Optional[int]
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ConversationMutation:
    """The branch projection and tail from one mutation."""

    manifest: ConversationManifest
    turn: TurnRecord
    branch: Optional[BranchRecord] = None


@dataclass(frozen=True)
class AppendResult:
    """The atomic user and assistant pair from one append."""

    manifest: ConversationManifest
    user_turn: TurnRecord
    assistant_turn: TurnRecord
    branch: Optional[BranchRecord] = None


@dataclass(frozen=True)
class TurnPage:
    """A chronological bounded page from newest history backwards."""

    conversation_id: str
    revision: int
    turns: Tuple[TurnRecord, ...]
    next_before: Optional[str]
    has_more: bool
    branch_id: Optional[str] = None
    catalog_revision: int = 0
    schema_version: int = LEGACY_SCHEMA_VERSION
    default_branch_id: Optional[str] = None
    branch_points: Tuple["BranchPoint", ...] = ()


@dataclass(frozen=True)
class BranchPoint:
    """Bounded navigation among alternatives at one path index."""

    turn_index: int
    source_branch_id: str
    selected_branch_id: str
    branch_ids: Tuple[str, ...]
    deleted_branch_ids: Tuple[str, ...]


@dataclass(frozen=True)
class BranchListResult:
    """The catalog and all catalog-referenced branches."""

    catalog: ConversationCatalog
    branches: Tuple[BranchRecord, ...]


@dataclass(frozen=True)
class EditUserForkResult:
    """A new path with a replacement user and reserved assistant."""

    catalog: ConversationCatalog
    manifest: ConversationManifest
    branch: BranchRecord
    source_branch_id: str
    replaced_user_turn_id: str
    user_turn: TurnRecord
    assistant_turn: TurnRecord
    generation_configuration: Optional[
        GenerationConfigurationPayload
    ]


@dataclass(frozen=True)
class DeletePathForkResult:
    """A new path ending immediately before one user exchange."""

    catalog: ConversationCatalog
    manifest: ConversationManifest
    branch: BranchRecord
    source_branch_id: str
    deleted_user_turn_id: str
    removed_turn_count: int


@dataclass(frozen=True)
class RetryAssistantForkResult:
    """A new path with one fresh reserved assistant node."""

    catalog: ConversationCatalog
    manifest: ConversationManifest
    branch: BranchRecord
    source_branch_id: str
    retried_assistant_turn_id: str
    assistant_turn: TurnRecord
    generation_configuration: Optional[
        GenerationConfigurationPayload
    ]


def require_results_dir(results_dir: Path) -> None:
    if not isinstance(results_dir, Path):
        raise TypeError("results_dir must be a Path")


def validate_conversation_id(conversation_id: str) -> None:
    try:
        conversation_identity.validate_conversation_id(
            conversation_id
        )
    except ValueError as exc:
        raise InvalidConversationIdError(str(exc)) from exc


def validate_branch_id(branch_id: str) -> None:
    try:
        conversation_identity.validate_branch_id(branch_id)
    except ValueError as exc:
        raise InvalidBranchIdError(str(exc)) from exc


def validate_operation_id(operation_id: str) -> None:
    try:
        conversation_identity.validate_operation_id(operation_id)
    except ValueError as exc:
        raise InvalidOperationIdError(str(exc)) from exc


def validate_optional_branch_id(
    branch_id: Optional[str],
) -> None:
    if branch_id is not None:
        validate_branch_id(branch_id)


def validate_title(title: str) -> str:
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


def validate_text(text: str, *, role: Role) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{role} text must be a string")
    if len(text) > TEXT_CHARS_MAX:
        raise ValueError(
            f"{role} text exceeds {TEXT_CHARS_MAX} characters"
        )
    if role == "user" and text.strip() == "":
        raise ValueError("user text must not be blank")
    return text


def validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str):
        raise ValueError("model id must be a string")
    if MODEL_ID_RE.fullmatch(model_id) is None:
        raise ValueError(f"invalid model id: {model_id}")
    return model_id


def validate_input_mode(input_mode: str) -> InputMode:
    if input_mode == "chat":
        return "chat"
    if input_mode == "completion":
        return "completion"
    raise ValueError(f"invalid input mode: {input_mode}")


def validate_run_link(
    run_link: Optional[RunLink],
) -> Optional[RunLink]:
    if run_link is None:
        return None
    if not isinstance(run_link, RunLink):
        raise ValueError("run link must be a RunLink")
    if RUN_ID_RE.fullmatch(run_link.run_id) is None:
        raise ValueError(f"invalid run id: {run_link.run_id}")
    if isinstance(run_link.revision, bool):
        raise ValueError("run revision must be an integer")
    if not isinstance(run_link.revision, int):
        raise ValueError("run revision must be an integer")
    if run_link.revision < 0:
        raise ValueError("run revision must not be negative")
    if run_link.revision.bit_length() > JSON_INTEGER_BITS_MAX:
        raise ValueError("run revision exceeds the integer limit")
    return run_link


def validate_expected_revision(revision: int) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("expected revision must be an integer")
    if revision < 1:
        raise ValueError("expected revision must be positive")


def require_run_link_identity(
    turn: TurnRecord,
    *,
    expected_turn_index: Optional[int],
    expected_turn_version: Optional[int],
) -> None:
    """Require the exact assistant version a saved run names."""
    if (
        expected_turn_index is None
        and expected_turn_version is None
    ):
        return
    if (
        expected_turn_index is None
        or expected_turn_version is None
    ):
        raise ValueError(
            "run link turn index and version must be supplied"
            " together"
        )
    conversation_identity.validate_turn_index(
        expected_turn_index,
        name="assistant turn index",
    )
    if (
        isinstance(expected_turn_version, bool)
        or not isinstance(expected_turn_version, int)
    ):
        raise ValueError("assistant turn version must be an integer")
    if not 1 <= expected_turn_version <= TAIL_VERSIONS_MAX:
        raise ValueError(
            "assistant turn version is outside its bounds"
        )
    if (
        turn.index != expected_turn_index
        or turn.version != expected_turn_version
    ):
        raise ConversationStateError(
            "saved run does not match the current assistant version"
        )


def validate_limit(
    limit: int,
    *,
    maximum: int,
    name: str,
) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= limit <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def copy_json_object(
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


def copy_user_metadata(
    value: Mapping[str, object],
) -> JsonObject:
    """Copy user metadata while protecting the reserved turn field."""
    copied = copy_json_object(value, "metadata")
    conversation_generation.reject_pending_generation_metadata(
        copied,
        label="user metadata",
    )
    return copied


def validate_generation_configuration(
    value: object,
    *,
    expected_model_id: str,
    expected_input_mode: str,
) -> GenerationConfigurationPayload:
    """Validate one action snapshot at the public store boundary."""
    return conversation_generation.validate_generation_configuration(
        value,
        expected_model_id=expected_model_id,
        expected_input_mode=expected_input_mode,
    )


def parse_generation_configuration(
    value: object,
    *,
    expected_model_id: str,
    expected_input_mode: str,
) -> GenerationConfigurationPayload:
    """Parse durable/replay data without consulting the registry."""
    return conversation_generation.parse_generation_configuration(
        value,
        expected_model_id=expected_model_id,
        expected_input_mode=expected_input_mode,
    )


def default_generation_configuration(
    *,
    model_id: str,
    input_mode: str,
) -> GenerationConfigurationPayload:
    """Preserve direct store callers predating action snapshots."""
    return conversation_generation.default_generation_configuration(
        model_id=model_id,
        input_mode=input_mode,
    )


def generation_configuration_metadata(
    configuration: GenerationConfigurationPayload,
) -> JsonObject:
    """Encode a validated snapshot under its reserved metadata key."""
    raw = conversation_generation.generation_configuration_metadata(
        configuration
    )
    return copy_json_object(raw, "pending generation metadata")


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


def fsync_directory(path: Path) -> None:
    """Persist directory entries, surfacing unsupported barriers."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory_durable(
    path: Path,
    *,
    exist_ok: bool = False,
) -> bool:
    """Create one directory and durably publish its parent entry."""
    try:
        path.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        if not exist_ok:
            raise
        path.mkdir(parents=False, exist_ok=True)
        return False
    fsync_directory(path.parent)
    return True


def replace_durable(source: Path, target: Path) -> None:
    """Replace one entry and persist every changed parent."""
    source_parent = source.parent
    target_parent = target.parent
    source.replace(target)
    fsync_directory(target_parent)
    if source_parent != target_parent:
        fsync_directory(source_parent)


def write_json_atomic(path: Path, payload: object) -> None:
    """Write complete JSON beside its target, then replace it."""
    encoded = _encode_json_file(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        replace_durable(temporary, path)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _encode_json_file(payload: object) -> bytes:
    """Serialize and enforce the same byte bound used by readers."""
    try:
        text = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        encoded = text.encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload is not valid UTF-8 JSON") from exc
    if len(encoded) > JSON_FILE_BYTES_MAX:
        raise ValueError(
            "serialized JSON exceeds the"
            f" {JSON_FILE_BYTES_MAX}-byte file limit"
        )
    return encoded


def read_json_object(path: Path, label: str) -> Dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        with path.open("rb") as handle:
            encoded = handle.read(JSON_FILE_BYTES_MAX + 1)
        if len(encoded) > JSON_FILE_BYTES_MAX:
            raise OSError("file exceeds the JSON size limit")
        text = encoded.decode("utf-8")
        raw: object = json.loads(
            text,
            object_pairs_hook=_json_object_strict,
            parse_constant=_json_constant_refused,
        )
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


def _json_object_strict(
    pairs: List[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_constant_refused(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def require_keys(
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


def stored_string(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ConversationCorruptError(f"{name} must be a string")
    if len(value) > maximum:
        raise ConversationCorruptError(
            f"{name} exceeds {maximum} characters"
        )
    return value


def stored_optional_string(
    value: object,
    name: str,
) -> Optional[str]:
    if value is None:
        return None
    return stored_string(value, name, IDENTIFIER_CHARS_MAX)


def require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConversationCorruptError(f"{name} must be an integer")
    if value.bit_length() > JSON_INTEGER_BITS_MAX:
        raise ConversationCorruptError(
            f"{name} exceeds the integer limit"
        )
    return value


def require_positive_int(value: object, name: str) -> int:
    result = require_int(value, name)
    if result < 1:
        raise ConversationCorruptError(f"{name} must be positive")
    return result


def require_nonnegative_int(value: object, name: str) -> int:
    result = require_int(value, name)
    if result < 0:
        raise ConversationCorruptError(f"{name} must not be negative")
    return result


def stored_optional_int(
    value: object,
    name: str,
) -> Optional[int]:
    if value is None:
        return None
    return require_int(value, name)


def stored_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConversationCorruptError(f"{name} must be a boolean")
    return value


def stored_role(value: object) -> Role:
    if value == "user":
        return "user"
    if value == "assistant":
        return "assistant"
    raise ConversationCorruptError(f"invalid turn role: {value!r}")


def stored_optional_role(value: object) -> Optional[Role]:
    if value is None:
        return None
    return stored_role(value)


def stored_input_mode(value: object) -> Optional[InputMode]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConversationCorruptError("input_mode must be a string")
    try:
        return validate_input_mode(value)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def stored_timestamp(value: object, name: str) -> str:
    text = stored_string(value, name, 40)
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


def stored_json_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ConversationCorruptError(f"{label} must be an object")
    try:
        return copy_json_object(value, label)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def stored_run_link(value: object) -> Optional[RunLink]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConversationCorruptError("run_link must be an object")
    require_keys(
        value,
        frozenset({"run_id", "revision"}),
        "run_link",
    )
    link = RunLink(
        run_id=stored_string(
            value["run_id"], "run_id", IDENTIFIER_CHARS_MAX
        ),
        revision=require_nonnegative_int(
            value["revision"], "run revision"
        ),
    )
    try:
        return validate_run_link(link)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def conversations_root(
    results_dir: Path,
    *,
    create: bool,
) -> Optional[Path]:
    root = results_dir / CONVERSATIONS_DIR_NAME
    if create:
        existed = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        if not existed:
            fsync_directory(root.parent)
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


def resolve_conversation_dir(
    results_dir: Path,
    conversation_id: str,
) -> Path:
    require_results_dir(results_dir)
    validate_conversation_id(conversation_id)
    root = conversations_root(results_dir, create=False)
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


def require_turns_root(conversation_dir: Path) -> Path:
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


def allocate_conversation(root: Path) -> Tuple[str, Path]:
    for _attempt in range(ALLOCATION_ATTEMPTS_MAX):
        conversation_id = uuid4().hex
        path = root / conversation_id
        try:
            make_directory_durable(path)
        except FileExistsError:
            continue
        return conversation_id, path
    raise ConversationLimitError(
        "could not allocate a unique conversation id after"
        f" {ALLOCATION_ATTEMPTS_MAX} attempts"
    )


def trash_destination(root: Path, conversation_id: str) -> Path:
    for _attempt in range(TRASH_ATTEMPTS_MAX):
        candidate = root / f"{conversation_id}.{uuid4().hex}"
        if not candidate.exists():
            return candidate
    raise ConversationLimitError(
        "could not allocate conversation trash after"
        f" {TRASH_ATTEMPTS_MAX} attempts"
    )


def bounded_children(
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


def legacy_turn_id(index: int) -> str:
    try:
        return conversation_identity.legacy_turn_id(index)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def validate_legacy_turn_id(turn_id: str) -> None:
    conversation_identity.legacy_turn_index(turn_id)


def legacy_turn_index(turn_id: str) -> int:
    return conversation_identity.legacy_turn_index(turn_id)


def before_index(before: Optional[str], turn_count: int) -> int:
    if before is None:
        return turn_count + 1
    validate_legacy_turn_id(before)
    index = int(before)
    if index > turn_count + 1:
        raise ValueError(
            f"before cursor {before} is beyond this conversation"
        )
    return index


def version_path(turn_dir: Path, version: int) -> Path:
    if not 1 <= version <= TAIL_VERSIONS_MAX:
        raise ConversationCorruptError(
            f"turn version outside 1..{TAIL_VERSIONS_MAX}: {version}"
        )
    return turn_dir / f"{version:08d}.json"


def remove_unpublished_version(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ConversationCorruptError(
            f"unpublished version path is unsafe: {path}"
        )
    path.unlink()


def manifest_to_payload(
    manifest: ConversationManifest,
) -> ManifestPayload:
    return {
        "schema_version": manifest.schema_version,
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
    link: Optional[RunLinkPayload] = None
    if turn.run_link is not None:
        link = {
            "run_id": turn.run_link.run_id,
            "revision": turn.run_link.revision,
        }
    return {
        "schema_version": turn.schema_version,
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
        "context_pack": copy_json_object(
            turn.context_pack, "context_pack"
        ),
        "metadata": copy_json_object(turn.metadata, "metadata"),
        "run_link": link,
    }


def parse_turn(raw: Dict[str, object]) -> TurnRecord:
    expected = frozenset(TurnPayload.__required_keys__)
    require_keys(raw, expected, "turn version")
    schema = require_int(raw["schema_version"], "schema_version")
    if schema not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
        raise ConversationCorruptError(
            f"unsupported turn schema version {schema}"
        )
    role = stored_role(raw["role"])
    model_id = stored_optional_string(raw["model_id"], "model_id")
    input_mode = stored_input_mode(raw["input_mode"])
    text = stored_string(raw["text"], "text", TEXT_CHARS_MAX)
    partial = stored_bool(raw["partial"], "partial")
    _validate_stored_role_fields(
        role=role,
        text=text,
        partial=partial,
        model_id=model_id,
        input_mode=input_mode,
    )
    turn = TurnRecord(
        conversation_id=stored_string(
            raw["conversation_id"],
            "conversation_id",
            IDENTIFIER_CHARS_MAX,
        ),
        conversation_revision=require_positive_int(
            raw["conversation_revision"], "conversation_revision"
        ),
        turn_id=stored_string(
            raw["turn_id"], "turn_id", IDENTIFIER_CHARS_MAX
        ),
        index=require_positive_int(raw["index"], "index"),
        version=require_positive_int(raw["version"], "version"),
        role=role,
        created_at=stored_timestamp(raw["created_at"], "created_at"),
        updated_at=stored_timestamp(raw["updated_at"], "updated_at"),
        text=text,
        partial=partial,
        model_id=model_id,
        input_mode=input_mode,
        context_pack=stored_json_object(
            raw["context_pack"], "context_pack"
        ),
        metadata=stored_json_object(raw["metadata"], "metadata"),
        run_link=stored_run_link(raw["run_link"]),
        schema_version=schema,
    )
    validate_turn_record(turn)
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
        validate_model_id(model_id)
        validate_input_mode(input_mode)
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc


def validate_turn_record(turn: TurnRecord) -> None:
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


def pending_generation_configuration(
    turn: TurnRecord,
    *,
    required: bool = False,
) -> Optional[GenerationConfigurationPayload]:
    """Read a pending snapshot, translating invalid disk data."""
    if turn.model_id is None or turn.input_mode is None:
        if required:
            raise ConversationCorruptError(
                "pending generation configuration has no model"
            )
        return None
    try:
        configuration = (
            conversation_generation.pending_generation_configuration(
                turn.metadata,
                expected_model_id=turn.model_id,
                expected_input_mode=turn.input_mode,
            )
        )
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc
    if required and configuration is None:
        raise ConversationCorruptError(
            "forked pending assistant has no generation"
            " configuration"
        )
    return configuration


def validate_pending_turn(turn: TurnRecord) -> None:
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
            "the pending assistant is not an empty partial"
            " placeholder"
        )
    if turn.context_pack or turn.run_link is not None:
        raise ConversationCorruptError(
            "the pending assistant carries terminal data"
        )
    if turn.schema_version == LEGACY_SCHEMA_VERSION:
        if turn.metadata and set(turn.metadata) != {
            PENDING_GENERATION_KEY
        }:
            raise ConversationCorruptError(
                "the legacy pending assistant carries terminal"
                " metadata"
            )
        return
    configuration = pending_generation_configuration(turn)
    if turn.metadata and configuration is None:
        raise ConversationCorruptError(
            "the pending assistant carries terminal metadata"
        )


def write_version(turn_dir: Path, turn: TurnRecord) -> None:
    validate_turn_record(turn)
    path = version_path(turn_dir, turn.version)
    if path.exists():
        raise ConversationCorruptError(
            f"immutable turn version already exists: {path.name}"
        )
    write_json_atomic(path, turn_to_payload(turn))


def turn_replacement(
    current: TurnRecord,
    *,
    text: str,
    partial: bool,
    context_pack: Mapping[str, object],
    metadata: Mapping[str, object],
    run_link: Optional[RunLink],
) -> TurnRecord:
    assert current.role == "assistant"
    assert current.model_id is not None
    assert current.input_mode is not None
    clean_metadata = copy_json_object(metadata, "metadata")
    conversation_generation.reject_pending_generation_metadata(
        clean_metadata,
        label="completed assistant metadata",
    )
    return TurnRecord(
        conversation_id=current.conversation_id,
        conversation_revision=current.conversation_revision + 1,
        turn_id=current.turn_id,
        index=current.index,
        version=current.version + 1,
        role="assistant",
        created_at=current.created_at,
        updated_at=timestamp(),
        text=text,
        partial=partial,
        model_id=current.model_id,
        input_mode=current.input_mode,
        context_pack=copy_json_object(context_pack, "context_pack"),
        metadata=clean_metadata,
        run_link=run_link,
        schema_version=current.schema_version,
        branch_id=current.branch_id,
    )


def parse_legacy_manifest(
    raw: Dict[str, object],
    *,
    expected_id: str,
) -> ConversationManifest:
    expected = frozenset(ManifestPayload.__required_keys__)
    require_keys(raw, expected, "conversation manifest")
    schema = require_int(raw["schema_version"], "schema_version")
    if schema != LEGACY_SCHEMA_VERSION:
        raise ConversationCorruptError(
            f"unsupported conversation schema version {schema}"
        )
    conversation_id = stored_string(
        raw["id"], "id", IDENTIFIER_CHARS_MAX
    )
    if conversation_id != expected_id:
        raise ConversationCorruptError(
            "manifest id does not match its directory"
        )
    try:
        validate_conversation_id(conversation_id)
        title = validate_title(
            stored_string(raw["title"], "title", TITLE_CHARS_MAX)
        )
    except ValueError as exc:
        raise ConversationCorruptError(str(exc)) from exc
    manifest = ConversationManifest(
        id=conversation_id,
        title=title,
        revision=require_positive_int(raw["revision"], "revision"),
        created_at=stored_timestamp(raw["created_at"], "created_at"),
        updated_at=stored_timestamp(raw["updated_at"], "updated_at"),
        turn_count=require_nonnegative_int(
            raw["turn_count"], "turn_count"
        ),
        tail_role=stored_optional_role(raw["tail_role"]),
        tail_turn_id=stored_optional_string(
            raw["tail_turn_id"], "tail_turn_id"
        ),
        tail_version=stored_optional_int(
            raw["tail_version"], "tail_version"
        ),
        pending_assistant_id=stored_optional_string(
            raw["pending_assistant_id"],
            "pending_assistant_id",
        ),
    )
    validate_legacy_manifest(manifest)
    return manifest


def validate_legacy_manifest(
    manifest: ConversationManifest,
) -> None:
    if manifest.schema_version != LEGACY_SCHEMA_VERSION:
        raise ConversationCorruptError(
            "legacy manifest schema is not v1"
        )
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
        _validate_empty_legacy_manifest(manifest)
        return
    _validate_nonempty_legacy_manifest(manifest)


def _validate_empty_legacy_manifest(
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


def _validate_nonempty_legacy_manifest(
    manifest: ConversationManifest,
) -> None:
    if manifest.tail_role != "assistant":
        raise ConversationCorruptError(
            "a non-empty conversation must have an assistant tail"
        )
    expected_tail = legacy_turn_id(manifest.turn_count)
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


def read_legacy_manifest(
    conversation_dir: Path,
) -> ConversationManifest:
    raw = read_json_object(
        conversation_dir / MANIFEST_NAME,
        "conversation manifest",
    )
    return parse_legacy_manifest(
        raw, expected_id=conversation_dir.name
    )


def write_legacy_manifest(
    conversation_dir: Path,
    manifest: ConversationManifest,
) -> None:
    validate_legacy_manifest(manifest)
    write_json_atomic(
        conversation_dir / MANIFEST_NAME,
        manifest_to_payload(manifest),
    )


def require_legacy_revision(
    manifest: ConversationManifest,
    expected_revision: int,
) -> None:
    if manifest.revision != expected_revision:
        raise ConversationRevisionConflictError(
            manifest.id,
            expected_revision,
            manifest.revision,
            conversation_identity.legacy_branch_id(manifest.id),
        )


def append_legacy_locked(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    expected_revision: int,
    text: str,
    model_id: str,
    input_mode: InputMode,
    metadata: JsonObject,
) -> AppendResult:
    require_legacy_revision(manifest, expected_revision)
    _require_legacy_appendable(manifest)
    if manifest.turn_count + 2 > TURN_COUNT_MAX:
        raise ConversationLimitError(
            f"a conversation holds at most {TURN_COUNT_MAX} turns"
        )
    revision = manifest.revision + 1
    now = timestamp()
    user_index = manifest.turn_count + 1
    assistant_index = user_index + 1
    user_turn = _new_legacy_user_turn(
        manifest=manifest,
        revision=revision,
        index=user_index,
        now=now,
        text=text,
        metadata=metadata,
    )
    assistant_turn = _new_legacy_assistant_turn(
        manifest=manifest,
        revision=revision,
        index=assistant_index,
        now=now,
        model_id=model_id,
        input_mode=input_mode,
    )
    turns_root = require_turns_root(conversation_dir)
    user_dir = _prepare_legacy_turn(turns_root, user_index)
    assistant_dir = _prepare_legacy_turn(turns_root, assistant_index)
    write_version(user_dir, user_turn)
    write_version(assistant_dir, assistant_turn)
    _freeze_legacy_tail(conversation_dir, manifest)
    updated = ConversationManifest(
        id=manifest.id,
        title=manifest.title,
        revision=revision,
        created_at=manifest.created_at,
        updated_at=now,
        turn_count=assistant_index,
        tail_role="assistant",
        tail_turn_id=assistant_turn.turn_id,
        tail_version=1,
        pending_assistant_id=assistant_turn.turn_id,
    )
    write_legacy_manifest(conversation_dir, updated)
    return AppendResult(updated, user_turn, assistant_turn)


def _new_legacy_user_turn(
    *,
    manifest: ConversationManifest,
    revision: int,
    index: int,
    now: str,
    text: str,
    metadata: JsonObject,
) -> TurnRecord:
    return TurnRecord(
        conversation_id=manifest.id,
        conversation_revision=revision,
        turn_id=legacy_turn_id(index),
        index=index,
        version=1,
        role="user",
        created_at=now,
        updated_at=now,
        text=text,
        partial=False,
        model_id=None,
        input_mode=None,
        context_pack={},
        metadata=copy_json_object(metadata, "metadata"),
        run_link=None,
    )


def _new_legacy_assistant_turn(
    *,
    manifest: ConversationManifest,
    revision: int,
    index: int,
    now: str,
    model_id: str,
    input_mode: InputMode,
) -> TurnRecord:
    return TurnRecord(
        conversation_id=manifest.id,
        conversation_revision=revision,
        turn_id=legacy_turn_id(index),
        index=index,
        version=1,
        role="assistant",
        created_at=now,
        updated_at=now,
        text="",
        partial=True,
        model_id=model_id,
        input_mode=input_mode,
        context_pack={},
        metadata={},
        run_link=None,
    )


def _prepare_legacy_turn(turns_root: Path, index: int) -> Path:
    path = turns_root / legacy_turn_id(index)
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    make_directory_durable(path)
    return path


def _require_legacy_appendable(
    manifest: ConversationManifest,
) -> None:
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


def _freeze_legacy_tail(
    conversation_dir: Path,
    manifest: ConversationManifest,
) -> None:
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
    turn_dir = (
        require_turns_root(conversation_dir) / manifest.tail_turn_id
    )
    payload: FrozenPayload = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "turn_id": manifest.tail_turn_id,
        "version": manifest.tail_version,
    }
    write_json_atomic(turn_dir / FROZEN_NAME, payload)


def update_legacy_locked(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    assistant_turn_id: str,
    expected_revision: int,
    text: str,
    partial: bool,
    context_pack: JsonObject,
    metadata: JsonObject,
) -> ConversationMutation:
    require_legacy_revision(manifest, expected_revision)
    current = _require_legacy_tail_assistant(
        conversation_dir, manifest, assistant_turn_id
    )
    changed = turn_replacement(
        current,
        text=text,
        partial=partial,
        context_pack=context_pack,
        metadata=metadata,
        run_link=None,
    )
    return _publish_legacy_tail_revision(
        conversation_dir=conversation_dir,
        manifest=manifest,
        changed=changed,
        pending_assistant_id=None,
    )


def set_legacy_run_link_locked(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    assistant_turn_id: str,
    expected_revision: int,
    run_link: Optional[RunLink],
    expected_turn_index: Optional[int] = None,
    expected_turn_version: Optional[int] = None,
) -> ConversationMutation:
    require_legacy_revision(manifest, expected_revision)
    if manifest.pending_assistant_id is not None:
        raise ConversationStateError(
            "a pending assistant cannot link a saved run"
        )
    current = _require_legacy_tail_assistant(
        conversation_dir, manifest, assistant_turn_id
    )
    require_run_link_identity(
        current,
        expected_turn_index=expected_turn_index,
        expected_turn_version=expected_turn_version,
    )
    if current.run_link == run_link:
        return ConversationMutation(manifest, current)
    changed = turn_replacement(
        current,
        text=current.text,
        partial=current.partial,
        context_pack=current.context_pack,
        metadata=current.metadata,
        run_link=run_link,
    )
    return _publish_legacy_tail_revision(
        conversation_dir=conversation_dir,
        manifest=manifest,
        changed=changed,
        pending_assistant_id=None,
    )


def _require_legacy_tail_assistant(
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
    current = read_legacy_turn(
        conversation_dir=conversation_dir,
        manifest=manifest,
        index=legacy_turn_index(assistant_turn_id),
        version=manifest.tail_version,
    )
    if current.role != "assistant":
        raise ConversationCorruptError(
            "the manifest tail does not name an assistant"
        )
    return current


def _publish_legacy_tail_revision(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    changed: TurnRecord,
    pending_assistant_id: Optional[str],
) -> ConversationMutation:
    if changed.version > TAIL_VERSIONS_MAX:
        raise ConversationLimitError(
            "the tail assistant has reached its version limit of"
            f" {TAIL_VERSIONS_MAX}"
        )
    assert manifest.tail_turn_id == changed.turn_id
    assert manifest.tail_version is not None
    assert changed.version == manifest.tail_version + 1
    assert changed.conversation_revision == manifest.revision + 1
    turn_dir = require_turns_root(conversation_dir) / changed.turn_id
    path = version_path(turn_dir, changed.version)
    remove_unpublished_version(path)
    write_version(turn_dir, changed)
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
    write_legacy_manifest(conversation_dir, updated)
    return ConversationMutation(updated, changed)


def legacy_page_locked(
    conversation_dir: Path,
    manifest: ConversationManifest,
    *,
    branch_id: str,
    before: Optional[str],
    limit: int,
) -> TurnPage:
    finish_before = before_index(before, manifest.turn_count)
    finish = finish_before - 1
    start = max(1, finish - limit + 1)
    turns = _read_legacy_range(
        conversation_dir=conversation_dir,
        manifest=manifest,
        start=start,
        finish=finish,
    )
    has_more = start > 1
    next_before = legacy_turn_id(start) if has_more else None
    return TurnPage(
        conversation_id=manifest.id,
        revision=manifest.revision,
        turns=tuple(turns),
        next_before=next_before,
        has_more=has_more,
        branch_id=branch_id,
        catalog_revision=0,
        schema_version=LEGACY_SCHEMA_VERSION,
        default_branch_id=branch_id,
    )


def _read_legacy_range(
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
        version = _legacy_current_version(
            conversation_dir, manifest, index
        )
        turns.append(
            read_legacy_turn(
                conversation_dir=conversation_dir,
                manifest=manifest,
                index=index,
                version=version,
            )
        )
    return turns


def _legacy_current_version(
    conversation_dir: Path,
    manifest: ConversationManifest,
    index: int,
) -> int:
    if index % 2 == 1:
        return 1
    turn_id = legacy_turn_id(index)
    if turn_id == manifest.tail_turn_id:
        if manifest.tail_version is None:
            raise ConversationCorruptError(
                "the tail turn has no current version"
            )
        return manifest.tail_version
    turn_dir = require_turns_root(conversation_dir) / turn_id
    raw = read_json_object(turn_dir / FROZEN_NAME, "frozen pointer")
    expected = frozenset(FrozenPayload.__required_keys__)
    require_keys(raw, expected, "frozen pointer")
    schema = require_int(raw["schema_version"], "schema_version")
    if schema != LEGACY_SCHEMA_VERSION:
        raise ConversationCorruptError(
            "unsupported frozen-pointer schema"
        )
    if raw["turn_id"] != turn_id:
        raise ConversationCorruptError(
            f"frozen pointer does not name turn {turn_id}"
        )
    version = require_int(raw["version"], "version")
    if not 1 <= version <= TAIL_VERSIONS_MAX:
        raise ConversationCorruptError(
            f"frozen turn {turn_id} has invalid version {version}"
        )
    return version


def read_legacy_turn(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    index: int,
    version: int,
) -> TurnRecord:
    """Read one immutable turn version from a legacy conversation."""
    turn, _path = read_legacy_turn_source(
        conversation_dir=conversation_dir,
        manifest=manifest,
        index=index,
        version=version,
    )
    return turn


def read_legacy_turn_source(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    index: int,
    version: int,
) -> Tuple[TurnRecord, Path]:
    """Read a legacy turn and name its immutable version file."""
    turn_id = legacy_turn_id(index)
    turn_dir = require_turns_root(conversation_dir) / turn_id
    path = version_path(turn_dir, version)
    raw = read_json_object(
        path,
        "turn version",
    )
    turn = parse_turn(raw)
    if turn.schema_version != LEGACY_SCHEMA_VERSION:
        raise ConversationCorruptError(
            f"legacy turn {turn_id} has the wrong schema"
        )
    if turn.conversation_id != manifest.id:
        raise ConversationCorruptError(
            f"turn {turn_id} names another conversation"
        )
    if turn.turn_id != turn_id or turn.index != index:
        raise ConversationCorruptError(
            f"turn directory {turn_id} contains another turn"
        )
    if turn.version != version:
        raise ConversationCorruptError(
            f"turn {turn_id} has the wrong version"
        )
    expected_role: Role = "user" if index % 2 == 1 else "assistant"
    if turn.role != expected_role:
        raise ConversationCorruptError(
            f"turn {turn_id} breaks role order"
        )
    if turn.conversation_revision > manifest.revision:
        raise ConversationCorruptError(
            f"turn {turn_id} is newer than its manifest"
        )
    if (
        turn.turn_id == manifest.tail_turn_id
        and turn.conversation_revision != manifest.revision
    ):
        raise ConversationCorruptError(
            f"tail turn {turn_id} has a stale revision"
        )
    if turn.turn_id == manifest.pending_assistant_id:
        validate_pending_turn(turn)
    return turn, path
