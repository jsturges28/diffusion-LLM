"""Schema-v2 shared-prefix branch persistence.

The catalog is the authoritative membership and fork commit point.
Branch manifests commit existing-branch mutations independently.
This module depends only on ``_conversation_store_core``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

from src import conversation_identity
from src.web import _conversation_store_core as c


@dataclass(frozen=True)
class StoredBranch:
    """A branch record plus its private segment location."""

    record: c.BranchRecord
    storage: c.BranchStorage
    legacy_turn_count: int


@dataclass(frozen=True)
class BranchSegment:
    """One contiguous owner range in a resolved branch path."""

    branch: StoredBranch
    start: int
    finish: int


@dataclass(frozen=True)
class BranchAlternativeGroup:
    """All reachable alternatives replacing one logical path slot."""

    source_branch_id: str
    prefix_turn_count: int
    branch_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ForkContext:
    """Validated source state held under the data-root lock."""

    catalog: c.ConversationCatalog
    source: StoredBranch
    target: c.TurnRecord
    legacy_manifest: Optional[c.ConversationManifest]


RootState = Union[c.ConversationManifest, c.ConversationCatalog]
ForkResult = Union[
    c.EditUserForkResult,
    c.DeletePathForkResult,
    c.RetryAssistantForkResult,
]

OPERATION_TEMP_NAME_RE = re.compile(
    r"^\.[0-9a-f]{32}\.json\.[a-z0-9_]{8}\.tmp$"
)


def edit_fork_operation(
    *,
    operation_id: str,
    source_branch_id: str,
    target_turn_id: str,
    text: str,
    model_id: str,
    input_mode: c.InputMode,
    generation_configuration: Optional[
        c.GenerationConfigurationPayload
    ],
    metadata: c.JsonObject,
) -> c.ForkOperation:
    """Build the stable semantic identity for one edit fork."""
    legacy_payload: Dict[str, object] = {
        "kind": "edit_user",
        "source_branch_id": source_branch_id,
        "target_turn_id": target_turn_id,
        "text": text,
        "model_id": model_id,
        "input_mode": input_mode,
        "metadata": metadata,
    }
    payload = dict(legacy_payload)
    legacy_payload_for_replay: Optional[Dict[str, object]] = None
    if generation_configuration is not None:
        payload["generation_configuration"] = generation_configuration
        legacy_payload_for_replay = legacy_payload
    return _fork_operation(
        operation_id=operation_id,
        kind="edit_user",
        source_branch_id=source_branch_id,
        target_turn_id=target_turn_id,
        payload=payload,
        legacy_payload=legacy_payload_for_replay,
    )


def delete_fork_operation(
    *,
    operation_id: str,
    source_branch_id: str,
    target_turn_id: str,
) -> c.ForkOperation:
    """Build the stable semantic identity for one delete fork."""
    payload: Dict[str, object] = {
        "kind": "delete_path",
        "source_branch_id": source_branch_id,
        "target_turn_id": target_turn_id,
    }
    return _fork_operation(
        operation_id=operation_id,
        kind="delete_path",
        source_branch_id=source_branch_id,
        target_turn_id=target_turn_id,
        payload=payload,
    )


def retry_fork_operation(
    *,
    operation_id: str,
    source_branch_id: str,
    target_turn_id: str,
    model_id: str,
    input_mode: c.InputMode,
    generation_configuration: Optional[
        c.GenerationConfigurationPayload
    ],
) -> c.ForkOperation:
    """Build the stable semantic identity for one retry fork."""
    legacy_payload: Dict[str, object] = {
        "kind": "retry_assistant",
        "source_branch_id": source_branch_id,
        "target_turn_id": target_turn_id,
        "model_id": model_id,
        "input_mode": input_mode,
    }
    payload = dict(legacy_payload)
    legacy_payload_for_replay: Optional[Dict[str, object]] = None
    if generation_configuration is not None:
        payload["generation_configuration"] = generation_configuration
        legacy_payload_for_replay = legacy_payload
    return _fork_operation(
        operation_id=operation_id,
        kind="retry_assistant",
        source_branch_id=source_branch_id,
        target_turn_id=target_turn_id,
        payload=payload,
        legacy_payload=legacy_payload_for_replay,
    )


def _fork_operation(
    *,
    operation_id: str,
    kind: c.ForkOperationKind,
    source_branch_id: str,
    target_turn_id: str,
    payload: Dict[str, object],
    legacy_payload: Optional[Dict[str, object]] = None,
) -> c.ForkOperation:
    c.validate_operation_id(operation_id)
    c.validate_branch_id(source_branch_id)
    validate_any_turn_id(target_turn_id)
    digest = _fork_request_digest(payload)
    legacy_digest = (
        None
        if legacy_payload is None
        else _fork_request_digest(legacy_payload)
    )
    return c.ForkOperation(
        operation_id=operation_id,
        request_digest=digest,
        kind=kind,
        source_branch_id=source_branch_id,
        target_turn_id=target_turn_id,
        legacy_request_digest=legacy_digest,
    )


def _fork_request_digest(payload: Dict[str, object]) -> str:
    canonical = dict(payload)
    configuration = canonical.get("generation_configuration")
    if configuration is not None:
        canonical["generation_configuration"] = (
            _canonical_generation_digest_value(configuration)
        )
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    assert c.SHA256_DIGEST_RE.fullmatch(digest) is not None
    return digest


def _canonical_generation_digest_value(value: object) -> object:
    """Make equivalent JSON numbers hash alike across round trips."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [
            _canonical_generation_digest_value(item)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _canonical_generation_digest_value(item)
            for key, item in value.items()
        }
    raise TypeError(
        "generation configuration digest contains a non-JSON value"
    )


def read_root_state(conversation_dir: Path) -> RootState:
    raw = c.read_json_object(
        conversation_dir / c.MANIFEST_NAME,
        "conversation manifest",
    )
    schema = c.require_int(
        raw.get("schema_version"), "schema_version"
    )
    if schema == c.LEGACY_SCHEMA_VERSION:
        return c.parse_legacy_manifest(
            raw,
            expected_id=conversation_dir.name,
        )
    if schema == c.SCHEMA_VERSION:
        return parse_catalog(
            raw,
            expected_id=conversation_dir.name,
        )
    raise c.ConversationCorruptError(
        f"unsupported conversation schema version {schema}"
    )


def parse_catalog(
    raw: Dict[str, object],
    *,
    expected_id: str,
) -> c.ConversationCatalog:
    expected = frozenset(c.CatalogPayload.__required_keys__)
    c.require_keys(raw, expected, "conversation catalog")
    schema = c.require_int(raw["schema_version"], "schema_version")
    if schema != c.SCHEMA_VERSION:
        raise c.ConversationCorruptError(
            f"unsupported catalog schema version {schema}"
        )
    conversation_id = c.stored_string(
        raw["id"], "id", c.IDENTIFIER_CHARS_MAX
    )
    if conversation_id != expected_id:
        raise c.ConversationCorruptError(
            "catalog id does not match its directory"
        )
    try:
        c.validate_conversation_id(conversation_id)
        title = c.validate_title(
            c.stored_string(raw["title"], "title", c.TITLE_CHARS_MAX)
        )
        default_branch_id = c.stored_string(
            raw["default_branch_id"],
            "default_branch_id",
            c.IDENTIFIER_CHARS_MAX,
        )
        c.validate_branch_id(default_branch_id)
    except ValueError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc
    branch_ids = _stored_branch_ids(raw["branch_ids"])
    catalog = c.ConversationCatalog(
        conversation_id=conversation_id,
        title=title,
        revision=c.require_positive_int(
            raw["catalog_revision"], "catalog_revision"
        ),
        created_at=c.stored_timestamp(
            raw["created_at"], "created_at"
        ),
        updated_at=c.stored_timestamp(
            raw["updated_at"], "updated_at"
        ),
        default_branch_id=default_branch_id,
        branch_ids=branch_ids,
    )
    validate_catalog(catalog)
    return catalog


def _stored_branch_ids(value: object) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise c.ConversationCorruptError(
            "branch_ids must be an array"
        )
    if not 1 <= len(value) <= c.BRANCH_COUNT_MAX:
        raise c.ConversationCorruptError(
            "branch_ids count is outside its limit"
        )
    branch_ids: List[str] = []
    seen: set[str] = set()
    for item in value:
        branch_id = c.stored_string(
            item, "branch_id", c.IDENTIFIER_CHARS_MAX
        )
        try:
            c.validate_branch_id(branch_id)
        except ValueError as exc:
            raise c.ConversationCorruptError(str(exc)) from exc
        if branch_id in seen:
            raise c.ConversationCorruptError(
                "branch_ids contains a duplicate"
            )
        seen.add(branch_id)
        branch_ids.append(branch_id)
    return tuple(branch_ids)


def validate_catalog(catalog: c.ConversationCatalog) -> None:
    if catalog.schema_version != c.SCHEMA_VERSION:
        raise c.ConversationCorruptError("catalog schema is not v2")
    if not 1 <= catalog.revision <= c.CATALOG_REVISION_MAX:
        raise c.ConversationCorruptError(
            "catalog revision is outside its limit"
        )
    if not 1 <= len(catalog.branch_ids) <= c.BRANCH_COUNT_MAX:
        raise c.ConversationCorruptError(
            "catalog branch count is outside its limit"
        )
    if len(set(catalog.branch_ids)) != len(catalog.branch_ids):
        raise c.ConversationCorruptError(
            "catalog branch ids are not unique"
        )
    if catalog.default_branch_id not in catalog.branch_ids:
        raise c.ConversationCorruptError(
            "catalog default is not a catalog branch"
        )
    if catalog.created_at > catalog.updated_at:
        raise c.ConversationCorruptError(
            "catalog timestamps run backwards"
        )
    try:
        c.validate_conversation_id(catalog.conversation_id)
        c.validate_title(catalog.title)
        for branch_id in catalog.branch_ids:
            c.validate_branch_id(branch_id)
    except ValueError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc


def catalog_payload(
    catalog: c.ConversationCatalog,
) -> c.CatalogPayload:
    return {
        "schema_version": c.SCHEMA_VERSION,
        "id": catalog.conversation_id,
        "title": catalog.title,
        "catalog_revision": catalog.revision,
        "created_at": catalog.created_at,
        "updated_at": catalog.updated_at,
        "default_branch_id": catalog.default_branch_id,
        "branch_ids": list(catalog.branch_ids),
    }


def write_catalog(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> None:
    validate_catalog(catalog)
    c.write_json_atomic(
        conversation_dir / c.MANIFEST_NAME,
        catalog_payload(catalog),
    )


def legacy_branch_id(conversation_id: str) -> str:
    c.validate_conversation_id(conversation_id)
    return conversation_identity.legacy_branch_id(conversation_id)


def virtual_catalog(
    manifest: c.ConversationManifest,
) -> c.ConversationCatalog:
    branch_id = legacy_branch_id(manifest.id)
    return c.ConversationCatalog(
        conversation_id=manifest.id,
        title=manifest.title,
        revision=0,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        default_branch_id=branch_id,
        branch_ids=(branch_id,),
        schema_version=c.LEGACY_SCHEMA_VERSION,
    )


def virtual_legacy_branch(
    manifest: c.ConversationManifest,
) -> StoredBranch:
    record = c.BranchRecord(
        conversation_id=manifest.id,
        branch_id=legacy_branch_id(manifest.id),
        parent_branch_id=None,
        prefix_turn_count=0,
        revision=manifest.revision,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        turn_count=manifest.turn_count,
        tail_role=manifest.tail_role,
        tail_turn_id=manifest.tail_turn_id,
        tail_version=manifest.tail_version,
        pending_assistant_id=manifest.pending_assistant_id,
        depth=0,
    )
    return StoredBranch(record, "legacy", manifest.turn_count)


def require_legacy_branch_selection(
    manifest: c.ConversationManifest,
    branch_id: Optional[str],
) -> None:
    if branch_id is None:
        return
    if branch_id != legacy_branch_id(manifest.id):
        raise c.BranchNotFoundError(
            manifest.id,
            branch_id,
        )


def require_v2_branch_id(
    branch_id: Optional[str],
) -> str:
    if branch_id is None:
        raise ValueError(
            "branch_id is required for schema v2 mutation"
        )
    c.validate_branch_id(branch_id)
    return branch_id


def fork_source_branch_id(
    state: RootState,
    branch_id: Optional[str],
) -> str:
    """Resolve the source identity without applying either CAS."""
    if isinstance(state, c.ConversationManifest):
        require_legacy_branch_selection(state, branch_id)
        return legacy_branch_id(state.id)
    return require_v2_branch_id(branch_id)


def fork_operation_source_branch_id(
    *,
    conversation_dir: Path,
    state: RootState,
    operation_id: str,
    branch_id: Optional[str],
) -> str:
    """Resolve a committed v1 replay before requiring a v2 branch."""
    if isinstance(state, c.ConversationManifest):
        return fork_source_branch_id(state, branch_id)
    if branch_id is not None:
        return fork_source_branch_id(state, branch_id)
    receipt = read_operation_receipt(
        conversation_dir, operation_id
    )
    if (
        receipt is None
        or receipt.result_branch_id not in state.branch_ids
    ):
        return require_v2_branch_id(branch_id)
    _validate_operation_receipts(conversation_dir, state)
    _receipt_catalog_position(state, receipt)
    return receipt.source_branch_id


def branches_root(
    conversation_dir: Path,
    *,
    create: bool,
) -> Path:
    root = conversation_dir / c.BRANCHES_DIR_NAME
    if create:
        c.make_directory_durable(root, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise c.ConversationCorruptError(
            "the branches root is missing or unsafe"
        )
    if root.resolve().parent != conversation_dir.resolve():
        raise c.ConversationCorruptError(
            "the branches root leaves its conversation"
        )
    return root


def operations_root(
    conversation_dir: Path,
    *,
    create: bool,
) -> Optional[Path]:
    """Resolve the fixed receipt root without following links."""
    root = conversation_dir / c.OPERATIONS_DIR_NAME
    if root.is_symlink():
        raise c.ConversationCorruptError(
            "the operations root is missing or unsafe"
        )
    if not root.exists():
        if not create:
            return None
        c.make_directory_durable(root)
    if not root.is_dir():
        raise c.ConversationCorruptError(
            "the operations root is missing or unsafe"
        )
    if root.resolve().parent != conversation_dir.resolve():
        raise c.ConversationCorruptError(
            "the operations root leaves its conversation"
        )
    return root


def operation_receipt_payload(
    receipt: c.OperationReceipt,
) -> c.OperationReceiptPayload:
    return {
        "schema_version": receipt.schema_version,
        "operation_id": receipt.operation_id,
        "request_digest": receipt.request_digest,
        "kind": receipt.kind,
        "source_branch_id": receipt.source_branch_id,
        "target_turn_id": receipt.target_turn_id,
        "result_branch_id": receipt.result_branch_id,
        "catalog_revision": receipt.catalog_revision,
        "removed_turn_count": receipt.removed_turn_count,
    }


def parse_operation_receipt(
    raw: Dict[str, object],
    *,
    expected_operation_id: str,
) -> c.OperationReceipt:
    expected = frozenset(c.OperationReceiptPayload.__required_keys__)
    c.require_keys(raw, expected, "operation receipt")
    receipt = c.OperationReceipt(
        operation_id=c.stored_string(
            raw["operation_id"],
            "operation_id",
            c.IDENTIFIER_CHARS_MAX,
        ),
        request_digest=c.stored_string(
            raw["request_digest"],
            "request_digest",
            64,
        ),
        kind=_stored_operation_kind(raw["kind"]),
        source_branch_id=c.stored_string(
            raw["source_branch_id"],
            "source_branch_id",
            c.IDENTIFIER_CHARS_MAX,
        ),
        target_turn_id=c.stored_string(
            raw["target_turn_id"],
            "target_turn_id",
            c.IDENTIFIER_CHARS_MAX,
        ),
        result_branch_id=c.stored_string(
            raw["result_branch_id"],
            "result_branch_id",
            c.IDENTIFIER_CHARS_MAX,
        ),
        catalog_revision=c.require_positive_int(
            raw["catalog_revision"],
            "catalog_revision",
        ),
        removed_turn_count=c.stored_optional_int(
            raw["removed_turn_count"],
            "removed_turn_count",
        ),
        schema_version=c.require_int(
            raw["schema_version"], "schema_version"
        ),
    )
    if receipt.operation_id != expected_operation_id:
        raise c.ConversationCorruptError(
            "operation receipt id does not match its filename"
        )
    validate_operation_receipt(receipt)
    return receipt


def _stored_operation_kind(value: object) -> c.ForkOperationKind:
    if value == "edit_user":
        return "edit_user"
    if value == "delete_path":
        return "delete_path"
    if value == "retry_assistant":
        return "retry_assistant"
    raise c.ConversationCorruptError(
        f"invalid operation receipt kind: {value!r}"
    )


def validate_operation_receipt(
    receipt: c.OperationReceipt,
) -> None:
    if receipt.schema_version != c.SCHEMA_VERSION:
        raise c.ConversationCorruptError(
            "operation receipt schema is not v2"
        )
    try:
        c.validate_operation_id(receipt.operation_id)
        c.validate_branch_id(receipt.source_branch_id)
        c.validate_branch_id(receipt.result_branch_id)
        validate_any_turn_id(receipt.target_turn_id)
    except ValueError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc
    if c.SHA256_DIGEST_RE.fullmatch(receipt.request_digest) is None:
        raise c.ConversationCorruptError(
            "operation request digest is not canonical SHA-256"
        )
    if receipt.catalog_revision > c.CATALOG_REVISION_MAX:
        raise c.ConversationCorruptError(
            "operation catalog revision exceeds its limit"
        )
    removed = receipt.removed_turn_count
    if receipt.kind == "delete_path":
        if removed is None or not 1 <= removed <= c.TURN_COUNT_MAX:
            raise c.ConversationCorruptError(
                "delete receipt removed count is outside its limit"
            )
        if removed % 2 != 0:
            raise c.ConversationCorruptError(
                "delete receipt removed count must be even"
            )
    elif removed is not None:
        raise c.ConversationCorruptError(
            "non-delete receipt carries a removed count"
        )


def read_operation_receipt(
    conversation_dir: Path,
    operation_id: str,
) -> Optional[c.OperationReceipt]:
    c.validate_operation_id(operation_id)
    root = operations_root(conversation_dir, create=False)
    if root is None:
        return None
    path = root / f"{operation_id}.json"
    if path.is_symlink():
        raise c.ConversationCorruptError(
            "operation receipt is not a safe regular file"
        )
    if not path.exists():
        return None
    if not path.is_file():
        raise c.ConversationCorruptError(
            "operation receipt is not a safe regular file"
        )
    if path.resolve().parent != root.resolve():
        raise c.ConversationCorruptError(
            "operation receipt leaves its operations root"
        )
    raw = c.read_json_object(path, "operation receipt")
    return parse_operation_receipt(
        raw,
        expected_operation_id=operation_id,
    )


def write_operation_receipt(
    conversation_dir: Path,
    receipt: c.OperationReceipt,
) -> None:
    validate_operation_receipt(receipt)
    root = operations_root(conversation_dir, create=True)
    assert root is not None
    path = root / f"{receipt.operation_id}.json"
    if path.exists() or path.is_symlink():
        raise c.ConversationCorruptError(
            "immutable operation receipt already exists"
        )
    c.write_json_atomic(path, operation_receipt_payload(receipt))


def resolve_branch_dir(
    conversation_dir: Path,
    branch_id: str,
) -> Path:
    c.validate_branch_id(branch_id)
    root = branches_root(conversation_dir, create=False)
    candidate = root / branch_id
    if candidate.is_symlink() or not candidate.is_dir():
        raise c.ConversationCorruptError(
            f"catalog branch is missing or unsafe: {branch_id}"
        )
    if candidate.resolve().parent != root.resolve():
        raise c.ConversationCorruptError(
            f"catalog branch leaves its conversation: {branch_id}"
        )
    return candidate


def require_catalog_member(
    catalog: c.ConversationCatalog,
    branch_id: str,
) -> None:
    if branch_id not in catalog.branch_ids:
        raise c.BranchNotFoundError(
            catalog.conversation_id,
            branch_id,
        )


def read_catalog_branch(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch_id: str,
) -> StoredBranch:
    require_catalog_member(catalog, branch_id)
    branch_dir = resolve_branch_dir(conversation_dir, branch_id)
    raw = c.read_json_object(
        branch_dir / c.MANIFEST_NAME,
        "branch manifest",
    )
    return parse_stored_branch(
        raw,
        conversation_id=conversation_dir.name,
        expected_branch_id=branch_id,
    )


def parse_stored_branch(
    raw: Dict[str, object],
    *,
    conversation_id: str,
    expected_branch_id: str,
) -> StoredBranch:
    expected = frozenset(c.BranchManifestPayload.__required_keys__)
    c.require_keys(raw, expected, "branch manifest")
    schema = c.require_int(raw["schema_version"], "schema_version")
    if schema != c.SCHEMA_VERSION:
        raise c.ConversationCorruptError(
            f"unsupported branch schema version {schema}"
        )
    stored_conversation_id = c.stored_string(
        raw["conversation_id"],
        "conversation_id",
        c.IDENTIFIER_CHARS_MAX,
    )
    stored_branch_id = c.stored_string(
        raw["branch_id"],
        "branch_id",
        c.IDENTIFIER_CHARS_MAX,
    )
    if stored_conversation_id != conversation_id:
        raise c.ConversationCorruptError(
            "branch names another conversation"
        )
    if stored_branch_id != expected_branch_id:
        raise c.ConversationCorruptError(
            "branch id does not match its directory"
        )
    branch = StoredBranch(
        record=c.BranchRecord(
            conversation_id=stored_conversation_id,
            branch_id=stored_branch_id,
            parent_branch_id=c.stored_optional_string(
                raw["parent_branch_id"], "parent_branch_id"
            ),
            prefix_turn_count=c.require_nonnegative_int(
                raw["prefix_turn_count"], "prefix_turn_count"
            ),
            revision=c.require_positive_int(
                raw["revision"], "revision"
            ),
            created_at=c.stored_timestamp(
                raw["created_at"], "created_at"
            ),
            updated_at=c.stored_timestamp(
                raw["updated_at"], "updated_at"
            ),
            turn_count=c.require_nonnegative_int(
                raw["turn_count"], "turn_count"
            ),
            tail_role=c.stored_optional_role(raw["tail_role"]),
            tail_turn_id=c.stored_optional_string(
                raw["tail_turn_id"], "tail_turn_id"
            ),
            tail_version=c.stored_optional_int(
                raw["tail_version"], "tail_version"
            ),
            pending_assistant_id=c.stored_optional_string(
                raw["pending_assistant_id"],
                "pending_assistant_id",
            ),
            depth=c.require_nonnegative_int(raw["depth"], "depth"),
        ),
        storage=_stored_branch_storage(raw["storage"]),
        legacy_turn_count=c.require_nonnegative_int(
            raw["legacy_turn_count"], "legacy_turn_count"
        ),
    )
    validate_stored_branch(branch)
    return branch


def _stored_branch_storage(value: object) -> c.BranchStorage:
    if value == "branch":
        return "branch"
    if value == "legacy":
        return "legacy"
    raise c.ConversationCorruptError(
        f"invalid branch storage: {value!r}"
    )


def validate_stored_branch(branch: StoredBranch) -> None:
    record = branch.record
    _validate_branch_identity(record)
    _validate_branch_counts(branch)
    _validate_branch_tail(record)
    if record.created_at > record.updated_at:
        raise c.ConversationCorruptError(
            "branch timestamps run backwards"
        )
    if record.revision > c.BRANCH_REVISION_MAX:
        raise c.ConversationCorruptError(
            "branch revision exceeds its limit"
        )


def _validate_branch_identity(record: c.BranchRecord) -> None:
    try:
        c.validate_conversation_id(record.conversation_id)
        c.validate_branch_id(record.branch_id)
        if record.parent_branch_id is not None:
            c.validate_branch_id(record.parent_branch_id)
    except ValueError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc
    if record.parent_branch_id == record.branch_id:
        raise c.ConversationCorruptError(
            "a branch cannot parent itself"
        )
    if record.depth > c.BRANCH_DEPTH_MAX:
        raise c.ConversationCorruptError(
            "branch depth exceeds its limit"
        )
    if record.parent_branch_id is None:
        if record.depth != 0:
            raise c.ConversationCorruptError(
                "a root branch must have depth zero"
            )
        if record.prefix_turn_count != 0:
            raise c.ConversationCorruptError(
                "a root branch cannot inherit a prefix"
            )
    elif record.depth == 0:
        raise c.ConversationCorruptError(
            "a child branch must have positive depth"
        )


def _validate_branch_counts(branch: StoredBranch) -> None:
    record = branch.record
    if record.turn_count > c.TURN_COUNT_MAX:
        raise c.ConversationCorruptError(
            "turn count exceeds its limit"
        )
    if record.turn_count % 2 != 0:
        raise c.ConversationCorruptError("turn count must be even")
    if record.prefix_turn_count > record.turn_count:
        raise c.ConversationCorruptError(
            "branch prefix exceeds its turn count"
        )
    local_count = record.turn_count - record.prefix_turn_count
    if local_count > c.BRANCH_LOCAL_TURNS_MAX:
        raise c.ConversationCorruptError(
            "branch local turn count exceeds its limit"
        )
    if branch.storage == "legacy":
        _validate_legacy_storage(branch)
    elif branch.legacy_turn_count != 0:
        raise c.ConversationCorruptError(
            "a native branch cannot claim legacy turns"
        )


def _validate_legacy_storage(branch: StoredBranch) -> None:
    record = branch.record
    if record.parent_branch_id is not None:
        raise c.ConversationCorruptError(
            "legacy storage is only valid on a root branch"
        )
    if record.prefix_turn_count != 0 or record.depth != 0:
        raise c.ConversationCorruptError(
            "legacy storage cannot inherit a prefix"
        )
    if branch.legacy_turn_count > record.turn_count:
        raise c.ConversationCorruptError(
            "legacy turn count exceeds branch turns"
        )
    if branch.legacy_turn_count % 2 != 0:
        raise c.ConversationCorruptError(
            "legacy turn count must be even"
        )


def _validate_branch_tail(record: c.BranchRecord) -> None:
    if record.turn_count == 0:
        values = (
            record.tail_role,
            record.tail_turn_id,
            record.tail_version,
            record.pending_assistant_id,
        )
        if any(value is not None for value in values):
            raise c.ConversationCorruptError(
                "an empty branch cannot have a tail"
            )
        return
    if record.tail_role != "assistant":
        raise c.ConversationCorruptError(
            "a non-empty branch must end with an assistant"
        )
    if record.tail_turn_id is None:
        raise c.ConversationCorruptError(
            "a non-empty branch has no tail id"
        )
    validate_any_turn_id(record.tail_turn_id)
    version = record.tail_version
    if version is None or not 1 <= version <= c.TAIL_VERSIONS_MAX:
        raise c.ConversationCorruptError(
            "branch tail version is outside its limit"
        )
    pending = record.pending_assistant_id
    if pending is not None and pending != record.tail_turn_id:
        raise c.ConversationCorruptError(
            "pending assistant does not name the branch tail"
        )
    if (version == 1) != (pending is not None):
        raise c.ConversationCorruptError(
            "only a reserved assistant may remain at version one"
        )


def branch_payload(
    branch: StoredBranch,
) -> c.BranchManifestPayload:
    record = branch.record
    return {
        "schema_version": c.SCHEMA_VERSION,
        "conversation_id": record.conversation_id,
        "branch_id": record.branch_id,
        "parent_branch_id": record.parent_branch_id,
        "prefix_turn_count": record.prefix_turn_count,
        "revision": record.revision,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "turn_count": record.turn_count,
        "tail_role": record.tail_role,
        "tail_turn_id": record.tail_turn_id,
        "tail_version": record.tail_version,
        "pending_assistant_id": record.pending_assistant_id,
        "depth": record.depth,
        "storage": branch.storage,
        "legacy_turn_count": branch.legacy_turn_count,
    }


def write_branch_manifest(
    branch_dir: Path,
    branch: StoredBranch,
) -> None:
    validate_stored_branch(branch)
    c.write_json_atomic(
        branch_dir / c.MANIFEST_NAME,
        branch_payload(branch),
    )


def manifest_from_branch(
    catalog: c.ConversationCatalog,
    branch: c.BranchRecord,
) -> c.ConversationManifest:
    if catalog.conversation_id != branch.conversation_id:
        raise c.ConversationCorruptError(
            "branch belongs to another catalog"
        )
    return c.ConversationManifest(
        id=catalog.conversation_id,
        title=catalog.title,
        revision=branch.revision,
        created_at=catalog.created_at,
        updated_at=branch.updated_at,
        turn_count=branch.turn_count,
        tail_role=branch.tail_role,
        tail_turn_id=branch.tail_turn_id,
        tail_version=branch.tail_version,
        pending_assistant_id=branch.pending_assistant_id,
        schema_version=c.SCHEMA_VERSION,
        catalog_revision=catalog.revision,
        default_branch_id=catalog.default_branch_id,
        branch_id=branch.branch_id,
    )


def create_v2_locked(
    conversation_dir: Path,
    *,
    conversation_id: str,
    title: str,
) -> c.ConversationManifest:
    now = c.timestamp()
    c.make_directory_durable(
        conversation_dir / c.OPERATIONS_DIR_NAME
    )
    root = conversation_dir / c.BRANCHES_DIR_NAME
    c.make_directory_durable(root)
    branch_id, branch_dir = _allocate_branch_dir(root)
    c.make_directory_durable(branch_dir / c.TURNS_DIR_NAME)
    record = c.BranchRecord(
        conversation_id=conversation_id,
        branch_id=branch_id,
        parent_branch_id=None,
        prefix_turn_count=0,
        revision=1,
        created_at=now,
        updated_at=now,
        turn_count=0,
        tail_role=None,
        tail_turn_id=None,
        tail_version=None,
        pending_assistant_id=None,
        depth=0,
    )
    branch = StoredBranch(record, "branch", 0)
    catalog = c.ConversationCatalog(
        conversation_id=conversation_id,
        title=title,
        revision=1,
        created_at=now,
        updated_at=now,
        default_branch_id=branch_id,
        branch_ids=(branch_id,),
    )
    write_branch_manifest(branch_dir, branch)
    write_catalog(conversation_dir, catalog)
    return manifest_from_branch(catalog, record)


def list_conversation_manifest(
    path: Path,
) -> Optional[c.ConversationManifest]:
    if path.name.startswith("."):
        return None
    if c.CONVERSATION_ID_RE.fullmatch(path.name) is None:
        return None
    if path.is_symlink() or not path.is_dir():
        return None
    if not (path / c.MANIFEST_NAME).is_file():
        return None
    try:
        state = read_root_state(path)
        if isinstance(state, c.ConversationManifest):
            return state
        branch = read_catalog_branch(
            path, state, state.default_branch_id
        )
        validate_selected_tail(path, state, branch)
        return manifest_from_branch(state, branch.record)
    except (
        c.BranchNotFoundError,
        c.ConversationCorruptError,
        c.InvalidBranchIdError,
        OSError,
        ValueError,
    ):
        return None


def selected_branch(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch_id: Optional[str],
) -> StoredBranch:
    selected = catalog.default_branch_id
    if branch_id is not None:
        selected = branch_id
    return read_catalog_branch(conversation_dir, catalog, selected)


def branch_manifest_locked(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch_id: Optional[str],
) -> c.ConversationManifest:
    branch = selected_branch(conversation_dir, catalog, branch_id)
    validate_selected_tail(conversation_dir, catalog, branch)
    return manifest_from_branch(catalog, branch.record)


def branch_record_locked(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch_id: str,
) -> c.BranchRecord:
    branch = read_catalog_branch(conversation_dir, catalog, branch_id)
    validate_selected_tail(conversation_dir, catalog, branch)
    return branch.record


def list_branches_locked(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> c.BranchListResult:
    records: List[c.BranchRecord] = []
    for branch_id in catalog.branch_ids:
        branch = read_catalog_branch(
            conversation_dir, catalog, branch_id
        )
        validate_selected_tail(conversation_dir, catalog, branch)
        records.append(branch.record)
    return c.BranchListResult(catalog, tuple(records))


def _recover_unpublished_branch_dirs(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> None:
    root = branches_root(conversation_dir, create=False)
    children = c.bounded_children(
        root,
        maximum=c.BRANCH_DIRECTORY_SCAN_MAX,
        label="branch directories",
    )
    members = frozenset(catalog.branch_ids)
    for child in children:
        if child.name in members:
            continue
        if child.name.startswith("."):
            raise c.ConversationCorruptError(
                "the branches root contains an unsafe entry"
            )
        try:
            c.validate_branch_id(child.name)
        except c.InvalidBranchIdError as exc:
            raise c.ConversationCorruptError(str(exc)) from exc
        if child.is_symlink() or not child.is_dir():
            raise c.ConversationCorruptError(
                "unpublished branch entry is unsafe"
            )
        shutil.rmtree(child)


def _recover_unpublished_operation_receipts(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> None:
    root = operations_root(conversation_dir, create=True)
    assert root is not None
    entries = _bounded_operation_receipts(root)
    members = frozenset(catalog.branch_ids)
    removed = False
    for path, receipt in entries:
        if receipt.result_branch_id not in members:
            path.unlink()
            removed = True
    if removed:
        c.fsync_directory(root)
    _validate_committed_receipts(catalog, entries)


def _validate_operation_receipts(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> None:
    root = operations_root(conversation_dir, create=False)
    if root is None:
        raise c.ConversationCorruptError(
            "a committed operation has no receipt root"
        )
    entries = _bounded_operation_receipts(root)
    _validate_committed_receipts(catalog, entries)


def _bounded_operation_receipts(
    root: Path,
) -> List[Tuple[Path, c.OperationReceipt]]:
    try:
        children = c.bounded_children(
            root,
            maximum=(
                c.OPERATION_RECEIPT_SCAN_MAX
                + c.OPERATION_TEMP_SCAN_MAX
            ),
            label="operation receipt entries",
        )
    except c.ConversationLimitError as exc:
        raise c.ConversationCorruptError(
            "operation entries exceed their bounded recovery limit"
        ) from exc
    receipt_paths: List[Path] = []
    temporary_paths: List[Path] = []
    for child in children:
        if OPERATION_TEMP_NAME_RE.fullmatch(child.name) is not None:
            temporary_paths.append(child)
        else:
            receipt_paths.append(child)
    if len(temporary_paths) > c.OPERATION_TEMP_SCAN_MAX:
        raise c.ConversationCorruptError(
            "operation temporary count exceeds its bound"
        )
    if len(receipt_paths) > c.OPERATION_RECEIPT_SCAN_MAX:
        raise c.ConversationCorruptError(
            "operation receipt count exceeds the branch bound"
        )
    removed_temporary = False
    for temporary in temporary_paths:
        _remove_operation_temporary(root, temporary)
        removed_temporary = True
    if removed_temporary:
        c.fsync_directory(root)
    return [
        (path, _read_scanned_operation_receipt(root, path))
        for path in receipt_paths
    ]


def _remove_operation_temporary(root: Path, path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise c.ConversationCorruptError(
            "operation temporary entry is unsafe"
        )
    if path.resolve().parent != root.resolve():
        raise c.ConversationCorruptError(
            "operation temporary leaves its root"
        )
    path.unlink()


def _validate_committed_receipts(
    catalog: c.ConversationCatalog,
    entries: List[Tuple[Path, c.OperationReceipt]],
) -> None:
    members = frozenset(catalog.branch_ids)
    committed_results: set[str] = set()
    for _path, receipt in entries:
        if receipt.result_branch_id not in members:
            continue
        _receipt_catalog_position(catalog, receipt)
        if receipt.result_branch_id in committed_results:
            raise c.ConversationCorruptError(
                "multiple receipts name one committed branch"
            )
        committed_results.add(receipt.result_branch_id)
    if len(committed_results) >= len(catalog.branch_ids):
        raise c.ConversationCorruptError(
            "committed receipts exceed non-root branches"
        )


def _read_scanned_operation_receipt(
    root: Path,
    path: Path,
) -> c.OperationReceipt:
    if path.is_symlink() or not path.is_file():
        raise c.ConversationCorruptError(
            "the operations root contains an unsafe entry"
        )
    if path.resolve().parent != root.resolve():
        raise c.ConversationCorruptError(
            "an operation receipt leaves its root"
        )
    if path.suffix != ".json":
        raise c.ConversationCorruptError(
            "an operation receipt has an invalid filename"
        )
    operation_id = path.stem
    try:
        c.validate_operation_id(operation_id)
    except c.InvalidOperationIdError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc
    raw = c.read_json_object(path, "operation receipt")
    return parse_operation_receipt(
        raw,
        expected_operation_id=operation_id,
    )


def _recover_unpublished_fork_artifacts(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
) -> None:
    _recover_unpublished_operation_receipts(
        conversation_dir, catalog
    )
    _recover_unpublished_branch_dirs(
        conversation_dir, catalog
    )


def _receipt_catalog_position(
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
) -> int:
    try:
        result_index = catalog.branch_ids.index(
            receipt.result_branch_id
        )
        source_index = catalog.branch_ids.index(
            receipt.source_branch_id
        )
    except ValueError as exc:
        raise c.ConversationCorruptError(
            "committed receipt names a non-catalog branch"
        ) from exc
    if source_index >= result_index:
        raise c.ConversationCorruptError(
            "operation result does not follow its source branch"
        )
    later_count = len(catalog.branch_ids) - result_index - 1
    if receipt.catalog_revision + later_count != catalog.revision:
        raise c.ConversationCorruptError(
            "operation receipt revision disagrees with catalog order"
        )
    return result_index


def _allocate_branch_dir(root: Path) -> Tuple[str, Path]:
    for _attempt in range(c.ALLOCATION_ATTEMPTS_MAX):
        branch_id = f"b_{uuid4().hex}"
        path = root / branch_id
        try:
            c.make_directory_durable(path)
        except FileExistsError:
            continue
        return branch_id, path
    raise c.ConversationLimitError(
        "could not allocate a unique branch id after"
        f" {c.ALLOCATION_ATTEMPTS_MAX} attempts"
    )


def resolve_branch_segments(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    selected: StoredBranch,
) -> List[BranchSegment]:
    reverse_segments: List[BranchSegment] = []
    visited: set[str] = set()
    current = selected
    ceiling = selected.record.turn_count
    ended = False
    for _depth in range(c.BRANCH_DEPTH_MAX + 1):
        record = current.record
        if record.branch_id in visited:
            raise c.ConversationCorruptError(
                "branch ancestry contains a cycle"
            )
        visited.add(record.branch_id)
        if ceiling > record.turn_count:
            raise c.ConversationCorruptError(
                "branch ancestry exceeds a parent path"
            )
        start = record.prefix_turn_count + 1
        if start <= ceiling:
            reverse_segments.append(
                BranchSegment(current, start, ceiling)
            )
        parent_id = record.parent_branch_id
        if parent_id is None:
            if record.prefix_turn_count != 0:
                raise c.ConversationCorruptError(
                    "root branch inherits a prefix"
                )
            ended = True
            break
        if parent_id in visited:
            raise c.ConversationCorruptError(
                "branch ancestry contains a cycle"
            )
        next_ceiling = min(ceiling, record.prefix_turn_count)
        parent = read_catalog_branch(
            conversation_dir, catalog, parent_id
        )
        _validate_parent_relationship(
            child=current,
            parent=parent,
        )
        ceiling = next_ceiling
        current = parent
    if not ended:
        raise c.ConversationCorruptError(
            "branch ancestry exceeds its depth bound"
        )
    segments = list(reversed(reverse_segments))
    _validate_segments(segments, selected.record.turn_count)
    return segments


def _validate_parent_relationship(
    *,
    child: StoredBranch,
    parent: StoredBranch,
) -> None:
    child_record = child.record
    parent_record = parent.record
    if child_record.conversation_id != parent_record.conversation_id:
        raise c.ConversationCorruptError(
            "branch ancestry crosses conversations"
        )
    if child_record.depth != parent_record.depth + 1:
        raise c.ConversationCorruptError(
            "branch depth does not match its parent"
        )
    if child_record.prefix_turn_count >= parent_record.turn_count:
        raise c.ConversationCorruptError(
            "branch prefix does not end inside its parent path"
        )


def _validate_segments(
    segments: List[BranchSegment],
    turn_count: int,
) -> None:
    expected_start = 1
    for segment in segments:
        if segment.start != expected_start:
            raise c.ConversationCorruptError(
                "branch ancestry leaves a turn gap"
            )
        if segment.finish < segment.start:
            raise c.ConversationCorruptError(
                "branch ancestry has an empty segment"
            )
        expected_start = segment.finish + 1
    if expected_start != turn_count + 1:
        raise c.ConversationCorruptError(
            "branch ancestry does not reach its tail"
        )


def turn_owner_for_index(
    segments: List[BranchSegment],
    index: int,
) -> StoredBranch:
    for segment in segments:
        if segment.start <= index <= segment.finish:
            return segment.branch
    raise c.ConversationCorruptError(
        f"branch path does not contain turn index {index}"
    )


def opaque_turn_id(branch_id: str, local_slot: int) -> str:
    try:
        return conversation_identity.opaque_turn_id(
            branch_id, local_slot
        )
    except ValueError as exc:
        raise c.ConversationCorruptError(str(exc)) from exc


def validate_opaque_turn_id(turn_id: str) -> None:
    conversation_identity.opaque_turn_parts(turn_id)


def validate_any_turn_id(turn_id: str) -> None:
    conversation_identity.validate_any_turn_id(turn_id)


def _owned_turn_identity(
    branch: StoredBranch,
    index: int,
) -> Tuple[str, int]:
    record = branch.record
    if not record.prefix_turn_count < index <= record.turn_count:
        raise c.ConversationCorruptError(
            f"turn index {index} is outside its owner segment"
        )
    if (
        branch.storage == "legacy"
        and index <= branch.legacy_turn_count
    ):
        return c.legacy_turn_id(index), c.LEGACY_SCHEMA_VERSION
    local_slot = index - record.prefix_turn_count
    return opaque_turn_id(
        record.branch_id, local_slot
    ), c.SCHEMA_VERSION


def _owned_turn_identity_for_new(
    branch: StoredBranch,
    index: int,
) -> Tuple[str, int]:
    record = branch.record
    if not record.prefix_turn_count < index <= c.TURN_COUNT_MAX:
        raise c.ConversationCorruptError(
            "new turn index is outside its branch segment"
        )
    local_slot = index - record.prefix_turn_count
    return opaque_turn_id(
        record.branch_id, local_slot
    ), c.SCHEMA_VERSION


def _branch_turns_root(
    conversation_dir: Path,
    branch: StoredBranch,
) -> Path:
    if branch.storage == "legacy":
        return c.require_turns_root(conversation_dir)
    branch_dir = resolve_branch_dir(
        conversation_dir, branch.record.branch_id
    )
    turns_root = branch_dir / c.TURNS_DIR_NAME
    if turns_root.is_symlink() or not turns_root.is_dir():
        raise c.ConversationCorruptError(
            "the branch turns directory is missing or unsafe"
        )
    if turns_root.resolve().parent != branch_dir.resolve():
        raise c.ConversationCorruptError(
            "the branch turns directory leaves its branch"
        )
    return turns_root


def _owned_turn_dir(
    conversation_dir: Path,
    branch: StoredBranch,
    *,
    index: int,
    turn_id: str,
) -> Path:
    expected_id, _schema = _owned_turn_identity(branch, index)
    if turn_id != expected_id:
        raise c.ConversationCorruptError(
            "turn id does not match its branch-local slot"
        )
    turns_root = _branch_turns_root(conversation_dir, branch)
    path = turns_root / turn_id
    if path.is_symlink() or not path.is_dir():
        raise c.ConversationCorruptError(
            f"turn directory is missing or unsafe: {turn_id}"
        )
    if path.resolve().parent != turns_root.resolve():
        raise c.ConversationCorruptError(
            f"turn directory leaves its branch: {turn_id}"
        )
    return path


def _read_frozen_version(
    turn_dir: Path,
    *,
    expected_turn_id: str,
    expected_schema: int,
) -> int:
    raw = c.read_json_object(
        turn_dir / c.FROZEN_NAME,
        "frozen pointer",
    )
    expected = frozenset(c.FrozenPayload.__required_keys__)
    c.require_keys(raw, expected, "frozen pointer")
    schema = c.require_int(raw["schema_version"], "schema_version")
    if schema != expected_schema:
        raise c.ConversationCorruptError(
            "frozen pointer has the wrong schema"
        )
    if raw["turn_id"] != expected_turn_id:
        raise c.ConversationCorruptError(
            f"frozen pointer does not name turn {expected_turn_id}"
        )
    version = c.require_int(raw["version"], "version")
    if not 1 <= version <= c.TAIL_VERSIONS_MAX:
        raise c.ConversationCorruptError(
            "frozen pointer version is outside its limit"
        )
    return version


def _owned_turn_version(
    *,
    turn_dir: Path,
    branch: StoredBranch,
    index: int,
    turn_id: str,
    schema: int,
) -> int:
    record = branch.record
    if index % 2 == 1:
        return 1
    if index == record.turn_count:
        if record.tail_turn_id != turn_id:
            raise c.ConversationCorruptError(
                "branch tail id does not match its local slot"
            )
        if record.tail_version is None:
            raise c.ConversationCorruptError(
                "branch tail has no current version"
            )
        return record.tail_version
    return _read_frozen_version(
        turn_dir,
        expected_turn_id=turn_id,
        expected_schema=schema,
    )


def read_owned_turn(
    *,
    conversation_dir: Path,
    branch: StoredBranch,
    index: int,
) -> c.TurnRecord:
    turn_id, schema = _owned_turn_identity(branch, index)
    turn_dir = _owned_turn_dir(
        conversation_dir,
        branch,
        index=index,
        turn_id=turn_id,
    )
    version = _owned_turn_version(
        turn_dir=turn_dir,
        branch=branch,
        index=index,
        turn_id=turn_id,
        schema=schema,
    )
    raw = c.read_json_object(
        c.version_path(turn_dir, version),
        "turn version",
    )
    turn = replace(
        c.parse_turn(raw),
        branch_id=branch.record.branch_id,
    )
    _validate_owned_turn(
        turn=turn,
        branch=branch,
        index=index,
        turn_id=turn_id,
        version=version,
        schema=schema,
    )
    return turn


def _validate_owned_turn(
    *,
    turn: c.TurnRecord,
    branch: StoredBranch,
    index: int,
    turn_id: str,
    version: int,
    schema: int,
) -> None:
    record = branch.record
    if turn.schema_version != schema:
        raise c.ConversationCorruptError(
            f"turn {turn_id} has the wrong schema"
        )
    if turn.conversation_id != record.conversation_id:
        raise c.ConversationCorruptError(
            f"turn {turn_id} names another conversation"
        )
    if turn.turn_id != turn_id or turn.index != index:
        raise c.ConversationCorruptError(
            f"turn directory {turn_id} contains another turn"
        )
    if turn.version != version:
        raise c.ConversationCorruptError(
            f"turn {turn_id} has the wrong version"
        )
    expected_role: c.Role = "user" if index % 2 == 1 else "assistant"
    if turn.role != expected_role:
        raise c.ConversationCorruptError(
            f"turn {turn_id} breaks role order"
        )
    if turn.conversation_revision > record.revision:
        raise c.ConversationCorruptError(
            f"turn {turn_id} is newer than its branch"
        )
    if (
        index == record.turn_count
        and turn.conversation_revision != record.revision
    ):
        raise c.ConversationCorruptError(
            f"tail turn {turn_id} has a stale revision"
        )
    if turn.turn_id == record.pending_assistant_id:
        c.validate_pending_turn(turn)


def validate_selected_tail(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
) -> List[BranchSegment]:
    segments = resolve_branch_segments(
        conversation_dir, catalog, branch
    )
    record = branch.record
    if record.turn_count == 0:
        return segments
    owner = turn_owner_for_index(segments, record.turn_count)
    actual = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=owner,
        index=record.turn_count,
    )
    if actual.turn_id != record.tail_turn_id:
        raise c.ConversationCorruptError(
            "resolved tail id disagrees with branch manifest"
        )
    if actual.version != record.tail_version:
        raise c.ConversationCorruptError(
            "resolved tail version disagrees with branch manifest"
        )
    if actual.role != record.tail_role:
        raise c.ConversationCorruptError(
            "resolved tail role disagrees with branch manifest"
        )
    pending = record.pending_assistant_id
    if pending is None:
        if actual.version == 1:
            raise c.ConversationCorruptError(
                "completed branch tail is still reserved"
            )
    else:
        if actual.turn_id != pending:
            raise c.ConversationCorruptError(
                "resolved tail disagrees with pending assistant"
            )
        c.validate_pending_turn(actual)
    return segments


def branch_page_locked(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    *,
    before: Optional[str],
    limit: int,
) -> c.TurnPage:
    record = branch.record
    segments = validate_selected_tail(
        conversation_dir, catalog, branch
    )
    finish_before = c.before_index(before, record.turn_count)
    finish = finish_before - 1
    start = max(1, finish - limit + 1)
    turns = _read_branch_range(
        conversation_dir=conversation_dir,
        segments=segments,
        start=start,
        finish=finish,
    )
    branch_points = _branch_points_for_page(
        conversation_dir=conversation_dir,
        catalog=catalog,
        selected=branch,
        segments=segments,
        start=start,
        finish=finish,
    )
    has_more = start > 1
    next_before = c.legacy_turn_id(start) if has_more else None
    return c.TurnPage(
        conversation_id=catalog.conversation_id,
        revision=record.revision,
        turns=tuple(turns),
        next_before=next_before,
        has_more=has_more,
        branch_id=record.branch_id,
        catalog_revision=catalog.revision,
        schema_version=c.SCHEMA_VERSION,
        default_branch_id=catalog.default_branch_id,
        branch_points=branch_points,
    )


def _branch_points_for_page(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    selected: StoredBranch,
    segments: List[BranchSegment],
    start: int,
    finish: int,
) -> Tuple[c.BranchPoint, ...]:
    """Describe only fork points intersecting one bounded page."""
    records = _catalog_branch_records(
        conversation_dir, catalog, selected
    )
    groups = _branch_alternative_groups(catalog, records)
    points: List[c.BranchPoint] = []
    for group in groups:
        selected_id = _selected_group_branch(
            group=group,
            selected=selected.record,
            segments=segments,
        )
        if selected_id is None:
            continue
        prefix = group.prefix_turn_count
        deleted_ids = tuple(
            branch_id
            for branch_id in group.branch_ids
            if branch_id != group.source_branch_id
            and records[branch_id].turn_count == prefix
        )
        turn_index = prefix + 1
        if not _branch_point_is_loaded(
            turn_index=turn_index,
            selected_branch_id=selected_id,
            deleted_branch_ids=deleted_ids,
            selected_turn_count=selected.record.turn_count,
            start=start,
            finish=finish,
        ):
            continue
        points.append(
            c.BranchPoint(
                turn_index=turn_index,
                source_branch_id=group.source_branch_id,
                selected_branch_id=selected_id,
                branch_ids=group.branch_ids,
                deleted_branch_ids=deleted_ids,
            )
        )
    order = {
        branch_id: index
        for index, branch_id in enumerate(catalog.branch_ids)
    }
    points.sort(
        key=lambda point: (
            point.turn_index,
            order[point.source_branch_id],
        )
    )
    if len(points) > c.BRANCH_COUNT_MAX:
        raise c.ConversationCorruptError(
            "page branch points exceed the catalog bound"
        )
    return tuple(points)


def _catalog_branch_records(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    selected: StoredBranch,
) -> Dict[str, c.BranchRecord]:
    """Read each bounded manifest, never any branch history."""
    records: Dict[str, c.BranchRecord] = {}
    for branch_id in catalog.branch_ids:
        if branch_id == selected.record.branch_id:
            record = selected.record
        else:
            record = read_catalog_branch(
                conversation_dir, catalog, branch_id
            ).record
        records[branch_id] = record
    if len(records) != len(catalog.branch_ids):
        raise c.ConversationCorruptError(
            "catalog branch records are not unique"
        )
    return records


def _branch_alternative_groups(
    catalog: c.ConversationCatalog,
    records: Dict[str, c.BranchRecord],
) -> Tuple[BranchAlternativeGroup, ...]:
    """Coalesce nested forks that replace the same absolute slot."""
    members_by_key: Dict[Tuple[str, int], set[str]] = {}
    sibling_counts: Dict[Tuple[str, int], int] = {}
    for branch_id in catalog.branch_ids:
        child = records[branch_id]
        parent_id = child.parent_branch_id
        if parent_id is None:
            continue
        prefix = child.prefix_turn_count
        sibling_key = (parent_id, prefix)
        count = sibling_counts.get(sibling_key, 0) + 1
        if count > c.BRANCH_SIBLINGS_MAX:
            raise c.ConversationCorruptError(
                "fork point exceeds its sibling bound"
            )
        sibling_counts[sibling_key] = count
        source_id = _branch_alternative_root(
            child=child,
            prefix=prefix,
            records=records,
        )
        key = (source_id, prefix)
        members = members_by_key.setdefault(key, {source_id})
        members.add(child.branch_id)
    groups: List[BranchAlternativeGroup] = []
    for (source_id, prefix), members in members_by_key.items():
        ordered = tuple(
            branch_id
            for branch_id in catalog.branch_ids
            if branch_id in members and branch_id != source_id
        )
        groups.append(
            BranchAlternativeGroup(
                source_branch_id=source_id,
                prefix_turn_count=prefix,
                branch_ids=(source_id, *ordered),
            )
        )
    return tuple(groups)


def _branch_alternative_root(
    *,
    child: c.BranchRecord,
    prefix: int,
    records: Dict[str, c.BranchRecord],
) -> str:
    """Find the first source in one same-slot fork chain."""
    current = child
    visited: set[str] = set()
    for _depth in range(c.BRANCH_DEPTH_MAX + 1):
        if current.branch_id in visited:
            raise c.ConversationCorruptError(
                "branch ancestry contains a cycle"
            )
        visited.add(current.branch_id)
        parent_id = current.parent_branch_id
        if parent_id is None or current.prefix_turn_count != prefix:
            return current.branch_id
        parent = records.get(parent_id)
        if parent is None:
            raise c.ConversationCorruptError(
                "branch parent is absent from the catalog"
            )
        _validate_record_parent(child=current, parent=parent)
        current = parent
    raise c.ConversationCorruptError(
        "branch ancestry exceeds its depth bound"
    )


def _validate_record_parent(
    *,
    child: c.BranchRecord,
    parent: c.BranchRecord,
) -> None:
    if child.conversation_id != parent.conversation_id:
        raise c.ConversationCorruptError(
            "branch ancestry crosses conversations"
        )
    if child.depth != parent.depth + 1:
        raise c.ConversationCorruptError(
            "branch depth does not match its parent"
        )
    if child.prefix_turn_count >= parent.turn_count:
        raise c.ConversationCorruptError(
            "branch prefix does not end inside its parent path"
        )


def _selected_group_branch(
    *,
    group: BranchAlternativeGroup,
    selected: c.BranchRecord,
    segments: List[BranchSegment],
) -> Optional[str]:
    turn_index = group.prefix_turn_count + 1
    if turn_index <= selected.turn_count:
        owner = turn_owner_for_index(segments, turn_index)
        selected_id = owner.record.branch_id
    elif (
        turn_index == selected.turn_count + 1
        and selected.branch_id in group.branch_ids
    ):
        selected_id = selected.branch_id
    else:
        return None
    if selected_id not in group.branch_ids:
        return None
    return selected_id


def _branch_point_is_loaded(
    *,
    turn_index: int,
    selected_branch_id: str,
    deleted_branch_ids: Tuple[str, ...],
    selected_turn_count: int,
    start: int,
    finish: int,
) -> bool:
    if start <= turn_index <= finish:
        return True
    if turn_index != finish + 1:
        return False
    if finish != selected_turn_count:
        return False
    return selected_branch_id in deleted_branch_ids


def _read_branch_range(
    *,
    conversation_dir: Path,
    segments: List[BranchSegment],
    start: int,
    finish: int,
) -> List[c.TurnRecord]:
    if finish < start:
        return []
    count = finish - start + 1
    assert count <= c.PAGE_SIZE_MAX
    turns: List[c.TurnRecord] = []
    for segment in segments:
        read_start = max(start, segment.start)
        read_finish = min(finish, segment.finish)
        if read_start <= read_finish:
            for index in range(read_start, read_finish + 1):
                turns.append(
                    read_owned_turn(
                        conversation_dir=conversation_dir,
                        branch=segment.branch,
                        index=index,
                    )
                )
    if len(turns) != count:
        raise c.ConversationCorruptError(
            "branch page did not resolve every turn"
        )
    return turns


def resolve_turn_by_id(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    turn_id: str,
) -> c.TurnRecord:
    segments = validate_selected_tail(
        conversation_dir, catalog, branch
    )
    index = _turn_id_index_on_path(
        segments=segments,
        turn_id=turn_id,
    )
    owner = turn_owner_for_index(segments, index)
    expected_id, _schema = _owned_turn_identity(owner, index)
    if expected_id != turn_id:
        raise c.ConversationStateError(
            "turn id is not present on the selected branch"
        )
    return read_owned_turn(
        conversation_dir=conversation_dir,
        branch=owner,
        index=index,
    )


def _turn_id_index_on_path(
    *,
    segments: List[BranchSegment],
    turn_id: str,
) -> int:
    if conversation_identity.is_legacy_turn_id(turn_id):
        index = c.legacy_turn_index(turn_id)
        turn_owner_for_index(segments, index)
        return index
    owner_id, local_slot = (
        conversation_identity.opaque_turn_parts(turn_id)
    )
    for segment in segments:
        record = segment.branch.record
        if record.branch_id == owner_id:
            index = record.prefix_turn_count + local_slot
            if segment.start <= index <= segment.finish:
                return index
    raise c.ConversationStateError(
        "turn id is not present on the selected branch"
    )


def _prepare_owned_turn_dir(
    conversation_dir: Path,
    branch: StoredBranch,
    *,
    index: int,
    turn_id: str,
) -> Path:
    expected_id, _schema = _owned_turn_identity_for_new(branch, index)
    if turn_id != expected_id:
        raise c.ConversationCorruptError(
            "new turn id does not match its local slot"
        )
    turns_root = _branch_turns_root(conversation_dir, branch)
    path = turns_root / turn_id
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    c.make_directory_durable(path)
    return path


def _write_v2_version(
    turn_dir: Path,
    turn: c.TurnRecord,
) -> None:
    if turn.schema_version != c.SCHEMA_VERSION:
        raise c.ConversationCorruptError(
            "native branch turns must use schema v2"
        )
    c.write_version(turn_dir, turn)


def _write_frozen_pointer(
    turn_dir: Path,
    *,
    turn_id: str,
    version: int,
    schema: int,
) -> None:
    payload: c.FrozenPayload = {
        "schema_version": schema,
        "turn_id": turn_id,
        "version": version,
    }
    c.write_json_atomic(turn_dir / c.FROZEN_NAME, payload)


def append_v2_locked(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    expected_revision: int,
    text: str,
    model_id: str,
    input_mode: c.InputMode,
    metadata: c.JsonObject,
) -> c.AppendResult:
    segments = validate_selected_tail(
        conversation_dir, catalog, branch
    )
    record = branch.record
    _require_branch_revision(record, expected_revision)
    _require_branch_appendable(record)
    _require_branch_growth(branch, added_turn_count=2)
    revision = _next_branch_revision(record)
    now = c.timestamp()
    user_index = record.turn_count + 1
    assistant_index = user_index + 1
    user_turn = _new_v2_user_turn(
        branch=branch,
        revision=revision,
        index=user_index,
        now=now,
        text=text,
        metadata=metadata,
    )
    assistant_turn = _new_v2_assistant_turn(
        branch=branch,
        revision=revision,
        index=assistant_index,
        now=now,
        model_id=model_id,
        input_mode=input_mode,
        metadata={},
    )
    _write_new_turn_pair(
        conversation_dir=conversation_dir,
        branch=branch,
        user_turn=user_turn,
        assistant_turn=assistant_turn,
    )
    _freeze_branch_tail(
        conversation_dir=conversation_dir,
        branch=branch,
        segments=segments,
    )
    updated_record = replace(
        record,
        revision=revision,
        updated_at=now,
        turn_count=assistant_index,
        tail_role="assistant",
        tail_turn_id=assistant_turn.turn_id,
        tail_version=1,
        pending_assistant_id=assistant_turn.turn_id,
    )
    updated_branch = replace(branch, record=updated_record)
    publish_existing_branch(conversation_dir, updated_branch)
    manifest = manifest_from_branch(catalog, updated_record)
    return c.AppendResult(
        manifest,
        user_turn,
        assistant_turn,
        updated_record,
    )


def _new_v2_user_turn(
    *,
    branch: StoredBranch,
    revision: int,
    index: int,
    now: str,
    text: str,
    metadata: c.JsonObject,
) -> c.TurnRecord:
    turn_id, schema = _owned_turn_identity_for_new(branch, index)
    assert schema == c.SCHEMA_VERSION
    return c.TurnRecord(
        conversation_id=branch.record.conversation_id,
        conversation_revision=revision,
        turn_id=turn_id,
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
        metadata=c.copy_json_object(metadata, "metadata"),
        run_link=None,
        schema_version=c.SCHEMA_VERSION,
        branch_id=branch.record.branch_id,
    )


def _new_v2_assistant_turn(
    *,
    branch: StoredBranch,
    revision: int,
    index: int,
    now: str,
    model_id: str,
    input_mode: c.InputMode,
    metadata: c.JsonObject,
) -> c.TurnRecord:
    turn_id, schema = _owned_turn_identity_for_new(branch, index)
    assert schema == c.SCHEMA_VERSION
    return c.TurnRecord(
        conversation_id=branch.record.conversation_id,
        conversation_revision=revision,
        turn_id=turn_id,
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
        metadata=c.copy_json_object(metadata, "metadata"),
        run_link=None,
        schema_version=c.SCHEMA_VERSION,
        branch_id=branch.record.branch_id,
    )


def _write_new_turn_pair(
    *,
    conversation_dir: Path,
    branch: StoredBranch,
    user_turn: c.TurnRecord,
    assistant_turn: c.TurnRecord,
) -> None:
    user_dir = _prepare_owned_turn_dir(
        conversation_dir,
        branch,
        index=user_turn.index,
        turn_id=user_turn.turn_id,
    )
    assistant_dir = _prepare_owned_turn_dir(
        conversation_dir,
        branch,
        index=assistant_turn.index,
        turn_id=assistant_turn.turn_id,
    )
    _write_v2_version(user_dir, user_turn)
    _write_v2_version(assistant_dir, assistant_turn)


def _freeze_branch_tail(
    *,
    conversation_dir: Path,
    branch: StoredBranch,
    segments: List[BranchSegment],
) -> None:
    record = branch.record
    if record.turn_count == 0:
        return
    owner = turn_owner_for_index(segments, record.turn_count)
    current = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=owner,
        index=record.turn_count,
    )
    if owner.record.branch_id != record.branch_id:
        return
    turn_dir = _owned_turn_dir(
        conversation_dir,
        owner,
        index=current.index,
        turn_id=current.turn_id,
    )
    _write_frozen_pointer(
        turn_dir,
        turn_id=current.turn_id,
        version=current.version,
        schema=current.schema_version,
    )


def _require_branch_growth(
    branch: StoredBranch,
    *,
    added_turn_count: int,
) -> None:
    if added_turn_count < 1:
        raise ValueError("added turn count must be positive")
    record = branch.record
    if record.turn_count + added_turn_count > c.TURN_COUNT_MAX:
        raise c.ConversationLimitError(
            f"a conversation holds at most {c.TURN_COUNT_MAX} turns"
        )
    local_count = record.turn_count - record.prefix_turn_count
    if local_count + added_turn_count > c.BRANCH_LOCAL_TURNS_MAX:
        raise c.ConversationLimitError(
            "a branch has reached its local turn limit of"
            f" {c.BRANCH_LOCAL_TURNS_MAX}"
        )


def _require_branch_appendable(record: c.BranchRecord) -> None:
    if record.pending_assistant_id is not None:
        raise c.ConversationStateError(
            "complete or stop the pending assistant before appending"
        )
    if record.turn_count == 0:
        return
    if record.tail_role != "assistant":
        raise c.ConversationCorruptError(
            "a non-empty branch must end with an assistant"
        )


def _next_branch_revision(record: c.BranchRecord) -> int:
    if record.revision >= c.BRANCH_REVISION_MAX:
        raise c.ConversationLimitError(
            "branch revision has reached its limit of"
            f" {c.BRANCH_REVISION_MAX}"
        )
    return record.revision + 1


def _require_branch_revision(
    record: c.BranchRecord,
    expected_revision: int,
) -> None:
    if record.revision != expected_revision:
        raise c.ConversationRevisionConflictError(
            record.conversation_id,
            expected_revision,
            record.revision,
            record.branch_id,
        )


def publish_existing_branch(
    conversation_dir: Path,
    branch: StoredBranch,
) -> None:
    branch_dir = resolve_branch_dir(
        conversation_dir, branch.record.branch_id
    )
    write_branch_manifest(branch_dir, branch)


def update_v2_locked(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    assistant_turn_id: str,
    expected_revision: int,
    text: str,
    partial: bool,
    context_pack: c.JsonObject,
    metadata: c.JsonObject,
) -> c.ConversationMutation:
    current = _require_v2_tail_assistant(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=branch,
        assistant_turn_id=assistant_turn_id,
        expected_revision=expected_revision,
    )
    changed = c.turn_replacement(
        current,
        text=text,
        partial=partial,
        context_pack=context_pack,
        metadata=metadata,
        run_link=None,
    )
    return _publish_v2_tail_revision(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=branch,
        changed=changed,
        pending_assistant_id=None,
    )


def set_v2_run_link_locked(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    assistant_turn_id: str,
    expected_revision: int,
    run_link: Optional[c.RunLink],
    expected_turn_index: Optional[int] = None,
    expected_turn_version: Optional[int] = None,
) -> c.ConversationMutation:
    record = branch.record
    _require_branch_revision(record, expected_revision)
    if record.pending_assistant_id is not None:
        raise c.ConversationStateError(
            "a pending assistant cannot link a saved run"
        )
    current = _require_v2_tail_assistant(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=branch,
        assistant_turn_id=assistant_turn_id,
        expected_revision=expected_revision,
    )
    c.require_run_link_identity(
        current,
        expected_turn_index=expected_turn_index,
        expected_turn_version=expected_turn_version,
    )
    if current.run_link == run_link:
        manifest = manifest_from_branch(catalog, record)
        return c.ConversationMutation(manifest, current, record)
    changed = c.turn_replacement(
        current,
        text=current.text,
        partial=current.partial,
        context_pack=current.context_pack,
        metadata=current.metadata,
        run_link=run_link,
    )
    return _publish_v2_tail_revision(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=branch,
        changed=changed,
        pending_assistant_id=None,
    )


def _require_v2_tail_assistant(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    assistant_turn_id: str,
    expected_revision: int,
) -> c.TurnRecord:
    record = branch.record
    _require_branch_revision(record, expected_revision)
    segments = validate_selected_tail(
        conversation_dir, catalog, branch
    )
    if record.tail_role != "assistant":
        raise c.ConversationStateError(
            "the branch tail is not an assistant"
        )
    if record.tail_turn_id != assistant_turn_id:
        raise c.ConversationStateError(
            "only the branch tail assistant may be changed"
        )
    owner = turn_owner_for_index(segments, record.turn_count)
    if owner.record.branch_id != record.branch_id:
        raise c.ConversationStateError(
            "an inherited tail cannot be changed on this branch"
        )
    current = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=owner,
        index=record.turn_count,
    )
    if current.role != "assistant":
        raise c.ConversationCorruptError(
            "the branch tail does not name an assistant"
        )
    return current


def _publish_v2_tail_revision(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    changed: c.TurnRecord,
    pending_assistant_id: Optional[str],
) -> c.ConversationMutation:
    record = branch.record
    if changed.version > c.TAIL_VERSIONS_MAX:
        raise c.ConversationLimitError(
            "the tail assistant has reached its version limit of"
            f" {c.TAIL_VERSIONS_MAX}"
        )
    revision = _next_branch_revision(record)
    assert record.tail_turn_id == changed.turn_id
    assert record.tail_version is not None
    assert changed.version == record.tail_version + 1
    assert changed.conversation_revision == revision
    turn_dir = _owned_turn_dir(
        conversation_dir,
        branch,
        index=changed.index,
        turn_id=changed.turn_id,
    )
    path = c.version_path(turn_dir, changed.version)
    c.remove_unpublished_version(path)
    c.write_version(turn_dir, changed)
    updated_record = replace(
        record,
        revision=revision,
        updated_at=changed.updated_at,
        tail_version=changed.version,
        pending_assistant_id=pending_assistant_id,
    )
    updated_branch = replace(branch, record=updated_record)
    publish_existing_branch(conversation_dir, updated_branch)
    manifest = manifest_from_branch(catalog, updated_record)
    return c.ConversationMutation(
        manifest,
        changed,
        updated_record,
    )


def validate_expected_catalog_revision(
    revision: Optional[int],
) -> None:
    if revision is None:
        return
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError(
            "expected catalog revision must be an integer"
        )
    if not 0 <= revision <= c.CATALOG_REVISION_MAX:
        raise ValueError(
            "expected catalog revision is outside its limit"
        )


def _require_catalog_revision(
    catalog: c.ConversationCatalog,
    expected_revision: Optional[int],
) -> None:
    if expected_revision is None:
        return
    if catalog.revision != expected_revision:
        raise c.ConversationCatalogRevisionConflictError(
            catalog.conversation_id,
            expected_revision,
            catalog.revision,
        )


def prepare_fork_locked(
    *,
    conversation_dir: Path,
    state: RootState,
    branch_id: Optional[str],
    target_turn_id: str,
    expected_revision: int,
    expected_catalog_revision: Optional[int],
) -> ForkContext:
    if isinstance(state, c.ConversationManifest):
        require_legacy_branch_selection(state, branch_id)
        catalog = virtual_catalog(state)
        source = virtual_legacy_branch(state)
        legacy_manifest: Optional[c.ConversationManifest] = state
    else:
        selected_id = require_v2_branch_id(branch_id)
        catalog = state
        source = read_catalog_branch(
            conversation_dir, catalog, selected_id
        )
        legacy_manifest = None
    _require_branch_revision(source.record, expected_revision)
    _require_catalog_revision(catalog, expected_catalog_revision)
    target = resolve_turn_by_id(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=source,
        turn_id=target_turn_id,
    )
    return ForkContext(
        catalog=catalog,
        source=source,
        target=target,
        legacy_manifest=legacy_manifest,
    )


def replay_fork_operation_locked(
    *,
    conversation_dir: Path,
    state: RootState,
    operation: c.ForkOperation,
) -> Optional[ForkResult]:
    """Replay one catalog-authoritative receipt before either CAS."""
    receipt = read_operation_receipt(
        conversation_dir, operation.operation_id
    )
    if receipt is None:
        return None
    if isinstance(state, c.ConversationManifest):
        _clear_unpublished_legacy_artifacts(conversation_dir)
        return None
    if receipt.result_branch_id not in state.branch_ids:
        _recover_unpublished_fork_artifacts(conversation_dir, state)
        return None
    _validate_operation_receipts(conversation_dir, state)
    _require_matching_operation(receipt, operation)
    return _reconstruct_fork_result(
        conversation_dir=conversation_dir,
        catalog=state,
        receipt=receipt,
    )


def _clear_unpublished_legacy_artifacts(
    conversation_dir: Path,
) -> None:
    _clear_unpublished_legacy_root(
        conversation_dir / c.BRANCHES_DIR_NAME,
        label="branches",
        maximum=c.BRANCH_DIRECTORY_SCAN_MAX,
    )
    _clear_unpublished_legacy_root(
        conversation_dir / c.OPERATIONS_DIR_NAME,
        label="operations",
        maximum=c.OPERATION_RECEIPT_SCAN_MAX,
    )


def _require_matching_operation(
    receipt: c.OperationReceipt,
    operation: c.ForkOperation,
) -> None:
    digest_matches = (
        receipt.request_digest == operation.request_digest
    )
    if operation.legacy_request_digest is not None:
        digest_matches = digest_matches or (
            receipt.request_digest
            == operation.legacy_request_digest
        )
    matches = (
        digest_matches
        and receipt.kind == operation.kind
        and receipt.source_branch_id == operation.source_branch_id
        and receipt.target_turn_id == operation.target_turn_id
    )
    if not matches:
        raise c.ConversationOperationConflictError(
            operation.operation_id
        )


def _reconstruct_fork_result(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
) -> ForkResult:
    _receipt_catalog_position(catalog, receipt)
    source = read_catalog_branch(
        conversation_dir, catalog, receipt.source_branch_id
    )
    current_result = read_catalog_branch(
        conversation_dir, catalog, receipt.result_branch_id
    )
    validate_selected_tail(conversation_dir, catalog, source)
    validate_selected_tail(conversation_dir, catalog, current_result)
    target = resolve_turn_by_id(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=source,
        turn_id=receipt.target_turn_id,
    )
    initial_result = _initial_receipt_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        receipt=receipt,
        source=source,
        target=target,
        current_result=current_result,
    )
    committed_catalog = _catalog_at_receipt(
        catalog, receipt, initial_result.record
    )
    return _typed_receipt_result(
        conversation_dir=conversation_dir,
        catalog=committed_catalog,
        receipt=receipt,
        source=source,
        target=target,
        result=initial_result,
    )


def _initial_receipt_branch(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    current_result: StoredBranch,
) -> StoredBranch:
    prefix = target.index - 1
    _validate_receipt_branch_coordinates(
        receipt=receipt,
        source=source,
        target=target,
        current_result=current_result,
    )
    tail = _initial_receipt_tail(
        conversation_dir=conversation_dir,
        catalog=catalog,
        receipt=receipt,
        source=source,
        prefix=prefix,
    )
    turn_count, tail_role, tail_id, tail_version, pending = tail
    record = c.BranchRecord(
        conversation_id=catalog.conversation_id,
        branch_id=receipt.result_branch_id,
        parent_branch_id=receipt.source_branch_id,
        prefix_turn_count=prefix,
        revision=1,
        created_at=current_result.record.created_at,
        updated_at=current_result.record.created_at,
        turn_count=turn_count,
        tail_role=tail_role,
        tail_turn_id=tail_id,
        tail_version=tail_version,
        pending_assistant_id=pending,
        depth=source.record.depth + 1,
    )
    initial = StoredBranch(record, "branch", 0)
    validate_stored_branch(initial)
    _validate_receipt_branch_growth(current_result, initial)
    return initial


def _validate_receipt_branch_coordinates(
    *,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    current_result: StoredBranch,
) -> None:
    result = current_result.record
    if result.parent_branch_id != receipt.source_branch_id:
        raise c.ConversationCorruptError(
            "receipt result has the wrong parent branch"
        )
    if result.prefix_turn_count != target.index - 1:
        raise c.ConversationCorruptError(
            "receipt result has the wrong fork prefix"
        )
    if result.depth != source.record.depth + 1:
        raise c.ConversationCorruptError(
            "receipt result has the wrong branch depth"
        )
    if current_result.storage != "branch":
        raise c.ConversationCorruptError(
            "receipt result is not a native branch"
        )
    if current_result.legacy_turn_count != 0:
        raise c.ConversationCorruptError(
            "receipt result claims legacy turns"
        )


def _initial_receipt_tail(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    prefix: int,
) -> Tuple[
    int,
    Optional[c.Role],
    Optional[str],
    Optional[int],
    Optional[str],
]:
    if receipt.kind == "edit_user":
        tail_id = opaque_turn_id(receipt.result_branch_id, 2)
        return prefix + 2, "assistant", tail_id, 1, tail_id
    if receipt.kind == "retry_assistant":
        tail_id = opaque_turn_id(receipt.result_branch_id, 1)
        return prefix + 1, "assistant", tail_id, 1, tail_id
    tail = _optional_turn_at_index(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=source,
        index=prefix,
    )
    if tail is None:
        return 0, None, None, None, None
    if tail.role != "assistant":
        raise c.ConversationCorruptError(
            "delete receipt prefix does not end with an assistant"
        )
    return prefix, "assistant", tail.turn_id, tail.version, None


def _validate_receipt_branch_growth(
    current: StoredBranch,
    initial: StoredBranch,
) -> None:
    if current.record.turn_count < initial.record.turn_count:
        raise c.ConversationCorruptError(
            "receipt result branch is shorter than its commit"
        )
    if current.record.revision == 1 and current != initial:
        raise c.ConversationCorruptError(
            "unmodified receipt branch differs from its commit"
        )


def _catalog_at_receipt(
    current: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    branch: c.BranchRecord,
) -> c.ConversationCatalog:
    result_index = _receipt_catalog_position(current, receipt)
    committed = replace(
        current,
        revision=receipt.catalog_revision,
        updated_at=branch.updated_at,
        default_branch_id=branch.branch_id,
        branch_ids=current.branch_ids[: result_index + 1],
    )
    validate_catalog(committed)
    return committed


def _typed_receipt_result(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    result: StoredBranch,
) -> ForkResult:
    if receipt.kind == "edit_user":
        return _replayed_edit_result(
            conversation_dir, catalog, receipt, source, target, result
        )
    if receipt.kind == "delete_path":
        return _replayed_delete_result(
            catalog, receipt, source, target, result
        )
    return _replayed_retry_result(
        conversation_dir, catalog, receipt, source, target, result
    )


def _replayed_edit_result(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    result: StoredBranch,
) -> c.EditUserForkResult:
    if target.role != "user":
        raise c.ConversationCorruptError(
            "edit receipt target is not a user turn"
        )
    user_turn = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=result,
        index=target.index,
    )
    assistant_turn = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=result,
        index=target.index + 1,
    )
    if user_turn.role != "user" or assistant_turn.role != "assistant":
        raise c.ConversationCorruptError(
            "edit receipt result has invalid turn roles"
        )
    manifest = manifest_from_branch(catalog, result.record)
    return c.EditUserForkResult(
        catalog=catalog,
        manifest=manifest,
        branch=result.record,
        source_branch_id=source.record.branch_id,
        replaced_user_turn_id=target.turn_id,
        user_turn=user_turn,
        assistant_turn=assistant_turn,
        generation_configuration=_fork_generation_configuration(
            assistant_turn
        ),
    )


def _replayed_delete_result(
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    result: StoredBranch,
) -> c.DeletePathForkResult:
    removed = receipt.removed_turn_count
    if target.role != "user" or removed is None:
        raise c.ConversationCorruptError(
            "delete receipt does not describe a user deletion"
        )
    if removed % 2 != 0:
        raise c.ConversationCorruptError(
            "delete receipt removed count must be even"
        )
    if result.record.turn_count + removed > source.record.turn_count:
        raise c.ConversationCorruptError(
            "delete receipt removes beyond the source path"
        )
    manifest = manifest_from_branch(catalog, result.record)
    return c.DeletePathForkResult(
        catalog=catalog,
        manifest=manifest,
        branch=result.record,
        source_branch_id=source.record.branch_id,
        deleted_user_turn_id=target.turn_id,
        removed_turn_count=removed,
    )


def _replayed_retry_result(
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    receipt: c.OperationReceipt,
    source: StoredBranch,
    target: c.TurnRecord,
    result: StoredBranch,
) -> c.RetryAssistantForkResult:
    if target.role != "assistant":
        raise c.ConversationCorruptError(
            "retry receipt target is not an assistant turn"
        )
    assistant_turn = read_owned_turn(
        conversation_dir=conversation_dir,
        branch=result,
        index=target.index,
    )
    if assistant_turn.role != "assistant":
        raise c.ConversationCorruptError(
            "retry receipt result is not an assistant turn"
        )
    manifest = manifest_from_branch(catalog, result.record)
    return c.RetryAssistantForkResult(
        catalog=catalog,
        manifest=manifest,
        branch=result.record,
        source_branch_id=source.record.branch_id,
        retried_assistant_turn_id=target.turn_id,
        assistant_turn=assistant_turn,
        generation_configuration=_fork_generation_configuration(
            assistant_turn
        ),
    )


def _fork_generation_configuration(
    assistant_turn: c.TurnRecord,
) -> Optional[c.GenerationConfigurationPayload]:
    configuration = c.pending_generation_configuration(
        assistant_turn,
        required=False,
    )
    if configuration is not None:
        return configuration
    if assistant_turn.metadata:
        raise c.ConversationCorruptError(
            "legacy pending assistant carries unexpected metadata"
        )
    return None


def materialize_fork_source_locked(
    conversation_dir: Path,
    context: ForkContext,
) -> Tuple[c.ConversationCatalog, StoredBranch]:
    legacy = context.legacy_manifest
    if legacy is None:
        return context.catalog, context.source
    branch = virtual_legacy_branch(legacy)
    branch_root = conversation_dir / c.BRANCHES_DIR_NAME
    operation_root = conversation_dir / c.OPERATIONS_DIR_NAME
    _clear_unpublished_legacy_root(
        branch_root,
        label="branches",
        maximum=c.BRANCH_DIRECTORY_SCAN_MAX,
    )
    _clear_unpublished_legacy_root(
        operation_root,
        label="operations",
        maximum=c.OPERATION_RECEIPT_SCAN_MAX,
    )
    c.make_directory_durable(branch_root)
    c.make_directory_durable(operation_root)
    branch_dir = branch_root / branch.record.branch_id
    c.make_directory_durable(branch_dir)
    write_branch_manifest(branch_dir, branch)
    return context.catalog, branch


def _clear_unpublished_legacy_root(
    root: Path,
    *,
    label: str,
    maximum: int,
) -> None:
    if root.is_symlink():
        raise c.ConversationCorruptError(
            f"unpublished {label} root is unsafe"
        )
    if not root.exists():
        return
    if not root.is_dir():
        raise c.ConversationCorruptError(
            f"unpublished {label} root is unsafe"
        )
    try:
        c.bounded_children(
            root,
            maximum=maximum,
            label=f"unpublished {label}",
        )
    except c.ConversationLimitError as exc:
        raise c.ConversationCorruptError(
            f"unpublished {label} exceed their scan bound"
        ) from exc
    shutil.rmtree(root)


def _assert_fork_operation(
    *,
    context: ForkContext,
    operation: c.ForkOperation,
    kind: c.ForkOperationKind,
) -> None:
    assert operation.kind == kind
    assert (
        operation.source_branch_id
        == context.source.record.branch_id
    )
    assert operation.target_turn_id == context.target.turn_id


def fork_edit_user_locked(
    *,
    conversation_dir: Path,
    context: ForkContext,
    operation: c.ForkOperation,
    text: str,
    model_id: str,
    input_mode: c.InputMode,
    generation_configuration: c.GenerationConfigurationPayload,
    metadata: c.JsonObject,
) -> c.EditUserForkResult:
    _assert_fork_operation(
        context=context,
        operation=operation,
        kind="edit_user",
    )
    if context.target.role != "user":
        raise c.ConversationStateError(
            "edit-user fork requires a user turn"
        )
    catalog, source = materialize_fork_source_locked(
        conversation_dir, context
    )
    _require_fork_local_turns(2)
    prefix = context.target.index - 1
    branch, branch_dir = _allocate_child_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        source=source,
        prefix_turn_count=prefix,
    )
    now = branch.record.created_at
    user_turn = _new_v2_user_turn(
        branch=branch,
        revision=1,
        index=context.target.index,
        now=now,
        text=text,
        metadata=metadata,
    )
    assistant_turn = _new_v2_assistant_turn(
        branch=branch,
        revision=1,
        index=context.target.index + 1,
        now=now,
        model_id=model_id,
        input_mode=input_mode,
        metadata=c.generation_configuration_metadata(
            generation_configuration
        ),
    )
    updated_record = replace(
        branch.record,
        turn_count=context.target.index + 1,
        tail_role="assistant",
        tail_turn_id=assistant_turn.turn_id,
        tail_version=1,
        pending_assistant_id=assistant_turn.turn_id,
    )
    updated_branch = replace(branch, record=updated_record)
    _write_new_turn_pair(
        conversation_dir=conversation_dir,
        branch=updated_branch,
        user_turn=user_turn,
        assistant_turn=assistant_turn,
    )
    updated_catalog = _commit_new_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=updated_branch,
        branch_dir=branch_dir,
        operation=operation,
        removed_turn_count=None,
    )
    manifest = manifest_from_branch(updated_catalog, updated_record)
    return c.EditUserForkResult(
        catalog=updated_catalog,
        manifest=manifest,
        branch=updated_record,
        source_branch_id=source.record.branch_id,
        replaced_user_turn_id=context.target.turn_id,
        user_turn=user_turn,
        assistant_turn=assistant_turn,
        generation_configuration=_fork_generation_configuration(
            assistant_turn
        ),
    )


def fork_delete_path_locked(
    *,
    conversation_dir: Path,
    context: ForkContext,
    operation: c.ForkOperation,
) -> c.DeletePathForkResult:
    _assert_fork_operation(
        context=context,
        operation=operation,
        kind="delete_path",
    )
    if context.target.role != "user":
        raise c.ConversationStateError(
            "delete-from-path fork requires a user turn"
        )
    catalog, source = materialize_fork_source_locked(
        conversation_dir, context
    )
    prefix = context.target.index - 1
    tail = _optional_turn_at_index(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=source,
        index=prefix,
    )
    tail_id: Optional[str] = None
    tail_version: Optional[int] = None
    tail_role: Optional[c.Role] = None
    if tail is not None:
        if tail.role != "assistant":
            raise c.ConversationCorruptError(
                "delete prefix does not end with an assistant"
            )
        tail_id = tail.turn_id
        tail_version = tail.version
        tail_role = "assistant"
    branch, branch_dir = _allocate_child_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        source=source,
        prefix_turn_count=prefix,
    )
    updated_record = replace(
        branch.record,
        turn_count=prefix,
        tail_role=tail_role,
        tail_turn_id=tail_id,
        tail_version=tail_version,
        pending_assistant_id=None,
    )
    updated_branch = replace(branch, record=updated_record)
    removed_turn_count = source.record.turn_count - prefix
    updated_catalog = _commit_new_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=updated_branch,
        branch_dir=branch_dir,
        operation=operation,
        removed_turn_count=removed_turn_count,
    )
    manifest = manifest_from_branch(updated_catalog, updated_record)
    return c.DeletePathForkResult(
        catalog=updated_catalog,
        manifest=manifest,
        branch=updated_record,
        source_branch_id=source.record.branch_id,
        deleted_user_turn_id=context.target.turn_id,
        removed_turn_count=removed_turn_count,
    )


def fork_retry_assistant_locked(
    *,
    conversation_dir: Path,
    context: ForkContext,
    operation: c.ForkOperation,
    model_id: str,
    input_mode: c.InputMode,
    generation_configuration: c.GenerationConfigurationPayload,
) -> c.RetryAssistantForkResult:
    _assert_fork_operation(
        context=context,
        operation=operation,
        kind="retry_assistant",
    )
    if context.target.role != "assistant":
        raise c.ConversationStateError(
            "retry fork requires an assistant turn"
        )
    if context.target.turn_id == (
        context.source.record.pending_assistant_id
    ):
        raise c.ConversationStateError(
            "a pending assistant cannot be retried"
        )
    catalog, source = materialize_fork_source_locked(
        conversation_dir, context
    )
    _require_fork_local_turns(1)
    prefix = context.target.index - 1
    paired_user = _optional_turn_at_index(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=source,
        index=prefix,
    )
    if paired_user is None or paired_user.role != "user":
        raise c.ConversationCorruptError(
            "retry assistant has no paired user"
        )
    branch, branch_dir = _allocate_child_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        source=source,
        prefix_turn_count=prefix,
    )
    assistant_turn = _new_v2_assistant_turn(
        branch=branch,
        revision=1,
        index=context.target.index,
        now=branch.record.created_at,
        model_id=model_id,
        input_mode=input_mode,
        metadata=c.generation_configuration_metadata(
            generation_configuration
        ),
    )
    updated_record = replace(
        branch.record,
        turn_count=context.target.index,
        tail_role="assistant",
        tail_turn_id=assistant_turn.turn_id,
        tail_version=1,
        pending_assistant_id=assistant_turn.turn_id,
    )
    updated_branch = replace(branch, record=updated_record)
    turn_dir = _prepare_owned_turn_dir(
        conversation_dir,
        updated_branch,
        index=assistant_turn.index,
        turn_id=assistant_turn.turn_id,
    )
    _write_v2_version(turn_dir, assistant_turn)
    updated_catalog = _commit_new_branch(
        conversation_dir=conversation_dir,
        catalog=catalog,
        branch=updated_branch,
        branch_dir=branch_dir,
        operation=operation,
        removed_turn_count=None,
    )
    manifest = manifest_from_branch(updated_catalog, updated_record)
    return c.RetryAssistantForkResult(
        catalog=updated_catalog,
        manifest=manifest,
        branch=updated_record,
        source_branch_id=source.record.branch_id,
        retried_assistant_turn_id=context.target.turn_id,
        assistant_turn=assistant_turn,
        generation_configuration=_fork_generation_configuration(
            assistant_turn
        ),
    )


def _allocate_child_branch(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    source: StoredBranch,
    prefix_turn_count: int,
) -> Tuple[StoredBranch, Path]:
    _require_child_capacity(
        conversation_dir=conversation_dir,
        catalog=catalog,
        source=source,
        prefix_turn_count=prefix_turn_count,
    )
    _recover_unpublished_fork_artifacts(
        conversation_dir, catalog
    )
    root = branches_root(conversation_dir, create=False)
    branch_id, branch_dir = _allocate_branch_dir(root)
    c.make_directory_durable(branch_dir / c.TURNS_DIR_NAME)
    now = c.timestamp()
    record = c.BranchRecord(
        conversation_id=catalog.conversation_id,
        branch_id=branch_id,
        parent_branch_id=source.record.branch_id,
        prefix_turn_count=prefix_turn_count,
        revision=1,
        created_at=now,
        updated_at=now,
        turn_count=prefix_turn_count,
        tail_role=None,
        tail_turn_id=None,
        tail_version=None,
        pending_assistant_id=None,
        depth=source.record.depth + 1,
    )
    return StoredBranch(record, "branch", 0), branch_dir


def _require_child_capacity(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    source: StoredBranch,
    prefix_turn_count: int,
) -> None:
    if source.record.depth >= c.BRANCH_DEPTH_MAX:
        raise c.ConversationLimitError(
            f"branch depth is limited to {c.BRANCH_DEPTH_MAX}"
        )
    if not 0 <= prefix_turn_count < source.record.turn_count:
        raise c.ConversationStateError(
            "fork prefix must end inside the source path"
        )
    if len(catalog.branch_ids) >= c.BRANCH_COUNT_MAX:
        raise c.ConversationLimitError(
            "a conversation holds at most"
            f" {c.BRANCH_COUNT_MAX} branches"
        )
    if catalog.revision >= c.CATALOG_REVISION_MAX:
        raise c.ConversationLimitError(
            "catalog revision has reached its limit of"
            f" {c.CATALOG_REVISION_MAX}"
        )
    sibling_count = 0
    for branch_id in catalog.branch_ids:
        branch = read_catalog_branch(
            conversation_dir, catalog, branch_id
        )
        if (
            branch.record.parent_branch_id == source.record.branch_id
            and branch.record.prefix_turn_count == prefix_turn_count
        ):
            sibling_count += 1
    if sibling_count >= c.BRANCH_SIBLINGS_MAX:
        raise c.ConversationLimitError(
            "a fork point holds at most"
            f" {c.BRANCH_SIBLINGS_MAX} sibling branches"
        )


def _require_fork_local_turns(turn_count: int) -> None:
    if not 0 <= turn_count <= c.BRANCH_LOCAL_TURNS_MAX:
        raise c.ConversationLimitError(
            "a fork would exceed the local turn limit of"
            f" {c.BRANCH_LOCAL_TURNS_MAX}"
        )


def _optional_turn_at_index(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    index: int,
) -> Optional[c.TurnRecord]:
    if index == 0:
        return None
    segments = resolve_branch_segments(
        conversation_dir, catalog, branch
    )
    owner = turn_owner_for_index(segments, index)
    return read_owned_turn(
        conversation_dir=conversation_dir,
        branch=owner,
        index=index,
    )


def _next_catalog_with_branch(
    catalog: c.ConversationCatalog,
    branch: c.BranchRecord,
) -> c.ConversationCatalog:
    if branch.branch_id in catalog.branch_ids:
        raise c.ConversationCorruptError(
            "new branch is already in the catalog"
        )
    if len(catalog.branch_ids) >= c.BRANCH_COUNT_MAX:
        raise c.ConversationLimitError(
            "catalog branch count has reached its limit"
        )
    if catalog.revision >= c.CATALOG_REVISION_MAX:
        raise c.ConversationLimitError(
            "catalog revision has reached its limit of"
            f" {c.CATALOG_REVISION_MAX}"
        )
    return replace(
        catalog,
        revision=catalog.revision + 1,
        updated_at=branch.updated_at,
        default_branch_id=branch.branch_id,
        branch_ids=(*catalog.branch_ids, branch.branch_id),
        schema_version=c.SCHEMA_VERSION,
    )


def _commit_new_branch(
    *,
    conversation_dir: Path,
    catalog: c.ConversationCatalog,
    branch: StoredBranch,
    branch_dir: Path,
    operation: c.ForkOperation,
    removed_turn_count: Optional[int],
) -> c.ConversationCatalog:
    updated_catalog = _next_catalog_with_branch(
        catalog, branch.record
    )
    receipt = c.OperationReceipt(
        operation_id=operation.operation_id,
        request_digest=operation.request_digest,
        kind=operation.kind,
        source_branch_id=operation.source_branch_id,
        target_turn_id=operation.target_turn_id,
        result_branch_id=branch.record.branch_id,
        catalog_revision=updated_catalog.revision,
        removed_turn_count=removed_turn_count,
    )
    write_branch_manifest(branch_dir, branch)
    write_operation_receipt(conversation_dir, receipt)
    write_catalog(conversation_dir, updated_catalog)
    return updated_catalog
