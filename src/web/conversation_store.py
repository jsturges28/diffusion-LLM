"""Stable public facade for durable branch conversations.

Dependency direction is deliberately one-way:
API -> conversation_store -> _conversation_branch_store
-> _conversation_store_core.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.web import _conversation_branch_store as branches
from src.web import _conversation_store_core as core


SCHEMA_VERSION = core.SCHEMA_VERSION
LEGACY_SCHEMA_VERSION = core.LEGACY_SCHEMA_VERSION
CONVERSATIONS_DIR_NAME = core.CONVERSATIONS_DIR_NAME
MANIFEST_NAME = core.MANIFEST_NAME
TURNS_DIR_NAME = core.TURNS_DIR_NAME
BRANCHES_DIR_NAME = core.BRANCHES_DIR_NAME
OPERATIONS_DIR_NAME = core.OPERATIONS_DIR_NAME
FROZEN_NAME = core.FROZEN_NAME
TRASH_DIR_NAME = core.TRASH_DIR_NAME

DEFAULT_TITLE = core.DEFAULT_TITLE
TITLE_CHARS_MAX = core.TITLE_CHARS_MAX
TEXT_CHARS_MAX = core.TEXT_CHARS_MAX
IDENTIFIER_CHARS_MAX = core.IDENTIFIER_CHARS_MAX
METADATA_JSON_CHARS_MAX = core.METADATA_JSON_CHARS_MAX
PENDING_GENERATION_KEY = core.PENDING_GENERATION_KEY
GENERATION_CONFIGURATION_CODEC_VERSION = (
    core.GENERATION_CONFIGURATION_CODEC_VERSION
)
JSON_FILE_BYTES_MAX = core.JSON_FILE_BYTES_MAX
TURN_JSON_ENVELOPE_BYTES_MAX = core.TURN_JSON_ENVELOPE_BYTES_MAX
OPERATION_ID_PATTERN = core.OPERATION_ID_PATTERN

TURN_ID_WIDTH = core.TURN_ID_WIDTH
TURN_COUNT_MAX = core.TURN_COUNT_MAX
TAIL_VERSIONS_MAX = core.TAIL_VERSIONS_MAX
BRANCH_COUNT_MAX = core.BRANCH_COUNT_MAX
BRANCH_SIBLINGS_MAX = core.BRANCH_SIBLINGS_MAX
BRANCH_DEPTH_MAX = core.BRANCH_DEPTH_MAX
BRANCH_LOCAL_TURNS_MAX = core.BRANCH_LOCAL_TURNS_MAX
BRANCH_REVISION_MAX = core.BRANCH_REVISION_MAX
CATALOG_REVISION_MAX = core.CATALOG_REVISION_MAX
PAGE_SIZE_DEFAULT = core.PAGE_SIZE_DEFAULT
PAGE_SIZE_MAX = core.PAGE_SIZE_MAX
LIST_SIZE_DEFAULT = core.LIST_SIZE_DEFAULT
LIST_SIZE_MAX = core.LIST_SIZE_MAX
CONVERSATION_SCAN_MAX = core.CONVERSATION_SCAN_MAX
SNAPSHOT_TURN_COPY_BYTES_MAX = 256 * 1024 * 1024

Role = core.Role
InputMode = core.InputMode
JsonScalar = core.JsonScalar
JsonValue = core.JsonValue
JsonObject = core.JsonObject
GenerationConfigurationPayload = core.GenerationConfigurationPayload

ConversationNotFoundError = core.ConversationNotFoundError
InvalidConversationIdError = core.InvalidConversationIdError
InvalidBranchIdError = core.InvalidBranchIdError
InvalidOperationIdError = core.InvalidOperationIdError
BranchNotFoundError = core.BranchNotFoundError
ConversationCorruptError = core.ConversationCorruptError
ConversationStateError = core.ConversationStateError
ConversationLimitError = core.ConversationLimitError
ConversationRevisionConflictError = (
    core.ConversationRevisionConflictError
)
ConversationCatalogRevisionConflictError = (
    core.ConversationCatalogRevisionConflictError
)
ConversationOperationConflictError = (
    core.ConversationOperationConflictError
)

RunLinkPayload = core.RunLinkPayload
ManifestPayload = core.ManifestPayload
TurnPayload = core.TurnPayload
FrozenPayload = core.FrozenPayload
CatalogPayload = core.CatalogPayload
BranchManifestPayload = core.BranchManifestPayload
OperationReceiptPayload = core.OperationReceiptPayload

RunLink = core.RunLink
ConversationManifest = core.ConversationManifest
TurnRecord = core.TurnRecord
ConversationCatalog = core.ConversationCatalog
BranchRecord = core.BranchRecord
OperationReceipt = core.OperationReceipt
BranchPoint = core.BranchPoint
ConversationMutation = core.ConversationMutation
AppendResult = core.AppendResult
TurnPage = core.TurnPage
BranchListResult = core.BranchListResult
EditUserForkResult = core.EditUserForkResult
DeletePathForkResult = core.DeletePathForkResult
RetryAssistantForkResult = core.RetryAssistantForkResult


@dataclass(frozen=True)
class SnapshotTurnSource:
    """One exact path turn copied into private snapshot staging."""

    turn: TurnRecord
    path: Path
    linked: bool


@dataclass(frozen=True)
class SnapshotSource:
    """One stable selected path and its private staged turn files."""

    conversation_id: str
    title: str
    branch_id: str
    branch_revision: int
    catalog_revision: int
    turn_count: int
    tail_turn_id: str
    tail_version: int
    turns: Tuple[SnapshotTurnSource, ...]

resolve_conversation_dir = core.resolve_conversation_dir
validate_conversation_id = core.validate_conversation_id
validate_branch_id = core.validate_branch_id
validate_operation_id = core.validate_operation_id
validate_title = core.validate_title
manifest_to_payload = core.manifest_to_payload
turn_to_payload = core.turn_to_payload
parse_turn = core.parse_turn
legacy_branch_id = branches.legacy_branch_id


def create(
    results_dir: Path,
    *,
    title: str = DEFAULT_TITLE,
) -> ConversationManifest:
    """Create one empty schema-v2 conversation."""
    core.require_results_dir(results_dir)
    clean_title = core.validate_title(title)
    with core.STORE_LOCK.held(results_dir):
        root = core.conversations_root(results_dir, create=True)
        assert root is not None
        conversation_id, conversation_dir = (
            core.allocate_conversation(root)
        )
        try:
            manifest = branches.create_v2_locked(
                conversation_dir,
                conversation_id=conversation_id,
                title=clean_title,
            )
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
    """Return default-branch summaries, newest first."""
    core.require_results_dir(results_dir)
    core.validate_limit(
        limit,
        maximum=LIST_SIZE_MAX,
        name="list limit",
    )
    with core.STORE_LOCK.held(results_dir):
        root = core.conversations_root(results_dir, create=False)
        if root is None:
            return []
        children = core.bounded_children(
            root,
            maximum=CONVERSATION_SCAN_MAX,
            label="conversation directories",
        )
        manifests = _list_manifests(children)
    manifests.sort(
        key=lambda item: (item.updated_at, item.id),
        reverse=True,
    )
    return manifests[:limit]


def _list_manifests(
    children: List[Path],
) -> List[ConversationManifest]:
    manifests: List[ConversationManifest] = []
    for child in children:
        manifest = branches.list_conversation_manifest(child)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def get_manifest(
    results_dir: Path,
    conversation_id: str,
    *,
    branch_id: Optional[str] = None,
) -> ConversationManifest:
    """Read one branch, defaulting only for read-only browsing."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    core.validate_optional_branch_id(branch_id)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return state
        return branches.branch_manifest_locked(
            conversation_dir, state, branch_id
        )


def get_catalog(
    results_dir: Path,
    conversation_id: str,
) -> ConversationCatalog:
    """Read the authoritative catalog without changing it."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            return branches.virtual_catalog(state)
        return state


def get_branch(
    results_dir: Path,
    conversation_id: str,
    branch_id: str,
) -> BranchRecord:
    """Read one catalog-referenced branch record."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    core.validate_branch_id(branch_id)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return branches.virtual_legacy_branch(state).record
        return branches.branch_record_locked(
            conversation_dir, state, branch_id
        )


def list_branches(
    results_dir: Path,
    conversation_id: str,
) -> BranchListResult:
    """Read catalog branches in durable publication order."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            catalog = branches.virtual_catalog(state)
            record = branches.virtual_legacy_branch(state).record
            return BranchListResult(catalog, (record,))
        return branches.list_branches_locked(conversation_dir, state)


def delete(results_dir: Path, conversation_id: str) -> None:
    """Atomically remove a conversation from the visible namespace."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        root = core.conversations_root(results_dir, create=True)
        assert root is not None
        trash_root = root / TRASH_DIR_NAME
        core.make_directory_durable(trash_root, exist_ok=True)
        condemned = core.trash_destination(
            trash_root, conversation_id
        )
        core.replace_durable(conversation_dir, condemned)
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
    branch_id: Optional[str] = None,
) -> AppendResult:
    """Append on one explicitly identified schema-v2 branch."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    clean_text = core.validate_text(text, role="user")
    clean_model_id = core.validate_model_id(model_id)
    clean_input_mode = core.validate_input_mode(input_mode)
    clean_metadata = core.copy_user_metadata(metadata or {})
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return core.append_legacy_locked(
                conversation_dir=conversation_dir,
                manifest=state,
                expected_revision=expected_revision,
                text=clean_text,
                model_id=clean_model_id,
                input_mode=clean_input_mode,
                metadata=clean_metadata,
            )
        selected_id = branches.require_v2_branch_id(branch_id)
        selected = branches.read_catalog_branch(
            conversation_dir, state, selected_id
        )
        return branches.append_v2_locked(
            conversation_dir=conversation_dir,
            catalog=state,
            branch=selected,
            expected_revision=expected_revision,
            text=clean_text,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            metadata=clean_metadata,
        )


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
    branch_id: Optional[str] = None,
) -> ConversationMutation:
    """Complete or revise an explicitly identified branch tail."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    branches.validate_any_turn_id(assistant_turn_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    clean_text = core.validate_text(text, role="assistant")
    if not isinstance(partial, bool):
        raise ValueError("partial must be a boolean")
    clean_context = core.copy_json_object(
        context_pack or {}, "context_pack"
    )
    clean_metadata = core.copy_json_object(metadata or {}, "metadata")
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return core.update_legacy_locked(
                conversation_dir=conversation_dir,
                manifest=state,
                assistant_turn_id=assistant_turn_id,
                expected_revision=expected_revision,
                text=clean_text,
                partial=partial,
                context_pack=clean_context,
                metadata=clean_metadata,
            )
        selected_id = branches.require_v2_branch_id(branch_id)
        selected = branches.read_catalog_branch(
            conversation_dir, state, selected_id
        )
        return branches.update_v2_locked(
            conversation_dir=conversation_dir,
            catalog=state,
            branch=selected,
            assistant_turn_id=assistant_turn_id,
            expected_revision=expected_revision,
            text=clean_text,
            partial=partial,
            context_pack=clean_context,
            metadata=clean_metadata,
        )


def set_run_link(
    results_dir: Path,
    conversation_id: str,
    assistant_turn_id: str,
    *,
    expected_revision: int,
    run_link: Optional[RunLink],
    branch_id: Optional[str] = None,
    expected_turn_index: Optional[int] = None,
    expected_turn_version: Optional[int] = None,
) -> ConversationMutation:
    """Set one explicitly identified branch tail's run link."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    branches.validate_any_turn_id(assistant_turn_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    clean_link = core.validate_run_link(run_link)
    if clean_link is not None and (
        expected_turn_index is None
        or expected_turn_version is None
    ):
        raise ValueError(
            "linking a run requires the assistant turn index"
            " and version"
        )
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return core.set_legacy_run_link_locked(
                conversation_dir=conversation_dir,
                manifest=state,
                assistant_turn_id=assistant_turn_id,
                expected_revision=expected_revision,
                run_link=clean_link,
                expected_turn_index=expected_turn_index,
                expected_turn_version=expected_turn_version,
            )
        selected_id = branches.require_v2_branch_id(branch_id)
        selected = branches.read_catalog_branch(
            conversation_dir, state, selected_id
        )
        return branches.set_v2_run_link_locked(
            conversation_dir=conversation_dir,
            catalog=state,
            branch=selected,
            assistant_turn_id=assistant_turn_id,
            expected_revision=expected_revision,
            run_link=clean_link,
            expected_turn_index=expected_turn_index,
            expected_turn_version=expected_turn_version,
        )


def get_turns(
    results_dir: Path,
    conversation_id: str,
    *,
    before: Optional[str] = None,
    limit: int = PAGE_SIZE_DEFAULT,
    branch_id: Optional[str] = None,
) -> TurnPage:
    """Read a chronological branch page with an index cursor."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_limit(
        limit,
        maximum=PAGE_SIZE_MAX,
        name="page limit",
    )
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            branches.require_legacy_branch_selection(state, branch_id)
            return core.legacy_page_locked(
                conversation_dir,
                state,
                branch_id=branches.legacy_branch_id(state.id),
                before=before,
                limit=limit,
            )
        selected = branches.selected_branch(
            conversation_dir, state, branch_id
        )
        return branches.branch_page_locked(
            conversation_dir,
            state,
            selected,
            before=before,
            limit=limit,
        )


def capture_snapshot_source(
    results_dir: Path,
    conversation_id: str,
    *,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
    turns_dir: Path,
    durable: bool = True,
) -> SnapshotSource:
    """Stage one exact path while its source stays locked."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    core.validate_branch_id(branch_id)
    core.validate_expected_revision(branch_revision)
    if not 1 <= turn_count <= TURN_COUNT_MAX:
        raise ValueError("snapshot turn_count is outside its limit")
    if turn_count % 2 != 0:
        raise ValueError("snapshot path must end on an assistant")
    branches.validate_any_turn_id(tail_turn_id)
    if not 1 <= tail_version <= TAIL_VERSIONS_MAX:
        raise ValueError("snapshot tail_version is outside its limit")
    if not isinstance(durable, bool):
        raise TypeError("durable must be a boolean")
    if turns_dir.exists():
        raise ValueError("snapshot turns staging already exists")
    if durable:
        core.make_directory_durable(turns_dir)
    else:
        turns_dir.mkdir(parents=True)
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        if isinstance(state, core.ConversationManifest):
            return _capture_legacy_snapshot(
                conversation_dir=conversation_dir,
                manifest=state,
                branch_id=branch_id,
                branch_revision=branch_revision,
                turn_count=turn_count,
                tail_turn_id=tail_turn_id,
                tail_version=tail_version,
                turns_dir=turns_dir,
                durable=durable,
            )
        return _capture_v2_snapshot(
            conversation_dir=conversation_dir,
            catalog=state,
            branch_id=branch_id,
            branch_revision=branch_revision,
            turn_count=turn_count,
            tail_turn_id=tail_turn_id,
            tail_version=tail_version,
            turns_dir=turns_dir,
            durable=durable,
        )


def _capture_legacy_snapshot(
    *,
    conversation_dir: Path,
    manifest: ConversationManifest,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
    turns_dir: Path,
    durable: bool,
) -> SnapshotSource:
    branches.require_legacy_branch_selection(manifest, branch_id)
    _require_snapshot_head(
        conversation_id=manifest.id,
        actual_revision=manifest.revision,
        expected_revision=branch_revision,
        actual_count=manifest.turn_count,
        expected_count=turn_count,
        actual_tail_id=manifest.tail_turn_id,
        expected_tail_id=tail_turn_id,
        actual_tail_version=manifest.tail_version,
        expected_tail_version=tail_version,
        pending_id=manifest.pending_assistant_id,
        branch_id=branch_id,
    )
    sources: List[Tuple[TurnRecord, Path]] = []
    for index in range(1, turn_count + 1):
        version = core._legacy_current_version(  # noqa: SLF001
            conversation_dir, manifest, index
        )
        sources.append(
            core.read_legacy_turn_source(
                conversation_dir=conversation_dir,
                manifest=manifest,
                index=index,
                version=version,
            )
        )
    staged = _stage_snapshot_turns(
        sources, turns_dir, durable=durable
    )
    return SnapshotSource(
        conversation_id=manifest.id,
        title=manifest.title,
        branch_id=branch_id,
        branch_revision=manifest.revision,
        catalog_revision=0,
        turn_count=manifest.turn_count,
        tail_turn_id=tail_turn_id,
        tail_version=tail_version,
        turns=staged,
    )


def _capture_v2_snapshot(
    *,
    conversation_dir: Path,
    catalog: ConversationCatalog,
    branch_id: str,
    branch_revision: int,
    turn_count: int,
    tail_turn_id: str,
    tail_version: int,
    turns_dir: Path,
    durable: bool,
) -> SnapshotSource:
    selected = branches.read_catalog_branch(
        conversation_dir, catalog, branch_id
    )
    record = selected.record
    _require_snapshot_head(
        conversation_id=record.conversation_id,
        actual_revision=record.revision,
        expected_revision=branch_revision,
        actual_count=record.turn_count,
        expected_count=turn_count,
        actual_tail_id=record.tail_turn_id,
        expected_tail_id=tail_turn_id,
        actual_tail_version=record.tail_version,
        expected_tail_version=tail_version,
        pending_id=record.pending_assistant_id,
        branch_id=branch_id,
    )
    segments = branches.validate_selected_tail(
        conversation_dir, catalog, selected
    )
    sources: List[Tuple[TurnRecord, Path]] = []
    for index in range(1, turn_count + 1):
        owner = branches.turn_owner_for_index(segments, index)
        sources.append(
            branches.read_owned_turn_source(
                conversation_dir=conversation_dir,
                branch=owner,
                index=index,
            )
        )
    staged = _stage_snapshot_turns(
        sources, turns_dir, durable=durable
    )
    return SnapshotSource(
        conversation_id=record.conversation_id,
        title=catalog.title,
        branch_id=branch_id,
        branch_revision=record.revision,
        catalog_revision=catalog.revision,
        turn_count=record.turn_count,
        tail_turn_id=tail_turn_id,
        tail_version=tail_version,
        turns=staged,
    )


def _require_snapshot_head(
    *,
    conversation_id: str,
    actual_revision: int,
    expected_revision: int,
    actual_count: int,
    expected_count: int,
    actual_tail_id: Optional[str],
    expected_tail_id: str,
    actual_tail_version: Optional[int],
    expected_tail_version: int,
    pending_id: Optional[str],
    branch_id: str,
) -> None:
    if actual_revision != expected_revision:
        raise ConversationRevisionConflictError(
            conversation_id,
            expected_revision,
            actual_revision,
            branch_id,
        )
    if pending_id is not None:
        raise ConversationStateError(
            "finish the pending assistant before saving"
            " the conversation"
        )
    if (
        actual_count != expected_count
        or actual_tail_id != expected_tail_id
        or actual_tail_version != expected_tail_version
    ):
        raise ConversationStateError(
            "the selected conversation head changed before"
            " it was saved"
        )


def _stage_snapshot_turns(
    sources: List[Tuple[TurnRecord, Path]],
    turns_dir: Path,
    *,
    durable: bool,
) -> Tuple[SnapshotTurnSource, ...]:
    staged: List[SnapshotTurnSource] = []
    copied_bytes = 0
    for turn, source in sources:
        target = turns_dir / f"{turn.index:08d}.json"
        linked, copied = _clone_snapshot_turn(
            source=source,
            target=target,
            copied_bytes=copied_bytes,
            durable=durable,
        )
        copied_bytes += copied
        staged.append(SnapshotTurnSource(turn, target, linked))
    assert len(staged) == len(sources)
    if durable:
        core.fsync_directory(turns_dir)
    return tuple(staged)


def _clone_snapshot_turn(
    *,
    source: Path,
    target: Path,
    copied_bytes: int,
    durable: bool,
) -> Tuple[bool, int]:
    if not source.is_file() or source.is_symlink():
        raise ConversationCorruptError(
            f"snapshot turn source is unsafe: {source}"
        )
    try:
        os.link(source, target, follow_symlinks=False)
        return True, 0
    except OSError as exc:
        fallback_errors = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EOPNOTSUPP,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        }
        if exc.errno not in fallback_errors:
            raise
    size = source.stat().st_size
    if copied_bytes + size > SNAPSHOT_TURN_COPY_BYTES_MAX:
        raise ConversationLimitError(
            "snapshot turn-copy fallback exceeds its byte limit"
        )
    shutil.copy2(source, target, follow_symlinks=False)
    if durable:
        with target.open("rb") as copied:
            os.fsync(copied.fileno())
    return False, size


def fork_edit_user(
    results_dir: Path,
    conversation_id: str,
    user_turn_id: str,
    *,
    operation_id: str,
    expected_revision: int,
    text: str,
    model_id: str,
    input_mode: InputMode,
    generation_configuration: Optional[Mapping[str, object]] = None,
    allow_compatibility_default: bool = True,
    metadata: Optional[Mapping[str, object]] = None,
    branch_id: Optional[str] = None,
    expected_catalog_revision: Optional[int] = None,
) -> EditUserForkResult:
    """Fork before one user and reserve its replacement response."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    branches.validate_any_turn_id(user_turn_id)
    core.validate_operation_id(operation_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    branches.validate_expected_catalog_revision(
        expected_catalog_revision
    )
    clean_text = core.validate_text(text, role="user")
    clean_model_id = core.validate_model_id(model_id)
    clean_input_mode = core.validate_input_mode(input_mode)
    assert isinstance(allow_compatibility_default, bool)
    parsed_configuration = None
    if generation_configuration is not None:
        parsed_configuration = core.parse_generation_configuration(
            generation_configuration,
            expected_model_id=clean_model_id,
            expected_input_mode=clean_input_mode,
        )
    clean_metadata = core.copy_user_metadata(metadata or {})
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        source_branch_id = branches.fork_operation_source_branch_id(
            conversation_dir=conversation_dir,
            state=state,
            operation_id=operation_id,
            branch_id=branch_id,
        )
        operation = branches.edit_fork_operation(
            operation_id=operation_id,
            source_branch_id=source_branch_id,
            target_turn_id=user_turn_id,
            text=clean_text,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            generation_configuration=parsed_configuration,
            metadata=clean_metadata,
        )
        try:
            replayed = branches.replay_fork_operation_locked(
                conversation_dir=conversation_dir,
                state=state,
                operation=operation,
            )
        except core.ConversationOperationConflictError:
            if parsed_configuration is not None:
                raise
            replayed = None
        if replayed is not None:
            assert isinstance(replayed, core.EditUserForkResult)
            return replayed
        if parsed_configuration is None:
            if not allow_compatibility_default:
                raise ValueError(
                    "generation_configuration is required for a new"
                    " edit fork"
                )
            clean_configuration = (
                core.default_generation_configuration(
                    model_id=clean_model_id,
                    input_mode=clean_input_mode,
                )
            )
            operation = branches.edit_fork_operation(
                operation_id=operation_id,
                source_branch_id=source_branch_id,
                target_turn_id=user_turn_id,
                text=clean_text,
                model_id=clean_model_id,
                input_mode=clean_input_mode,
                generation_configuration=clean_configuration,
                metadata=clean_metadata,
            )
            replayed = branches.replay_fork_operation_locked(
                conversation_dir=conversation_dir,
                state=state,
                operation=operation,
            )
            if replayed is not None:
                assert isinstance(replayed, core.EditUserForkResult)
                return replayed
        else:
            clean_configuration = (
                core.validate_generation_configuration(
                    parsed_configuration,
                    expected_model_id=clean_model_id,
                    expected_input_mode=clean_input_mode,
                )
            )
        context = branches.prepare_fork_locked(
            conversation_dir=conversation_dir,
            state=state,
            branch_id=branch_id,
            target_turn_id=user_turn_id,
            expected_revision=expected_revision,
            expected_catalog_revision=expected_catalog_revision,
        )
        return branches.fork_edit_user_locked(
            conversation_dir=conversation_dir,
            context=context,
            operation=operation,
            text=clean_text,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            generation_configuration=clean_configuration,
            metadata=clean_metadata,
        )


def fork_delete_from_path(
    results_dir: Path,
    conversation_id: str,
    user_turn_id: str,
    *,
    operation_id: str,
    expected_revision: int,
    branch_id: Optional[str] = None,
    expected_catalog_revision: Optional[int] = None,
) -> DeletePathForkResult:
    """Fork a path that ends immediately before one user turn."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    branches.validate_any_turn_id(user_turn_id)
    core.validate_operation_id(operation_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    branches.validate_expected_catalog_revision(
        expected_catalog_revision
    )
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        source_branch_id = branches.fork_operation_source_branch_id(
            conversation_dir=conversation_dir,
            state=state,
            operation_id=operation_id,
            branch_id=branch_id,
        )
        operation = branches.delete_fork_operation(
            operation_id=operation_id,
            source_branch_id=source_branch_id,
            target_turn_id=user_turn_id,
        )
        replayed = branches.replay_fork_operation_locked(
            conversation_dir=conversation_dir,
            state=state,
            operation=operation,
        )
        if replayed is not None:
            assert isinstance(replayed, core.DeletePathForkResult)
            return replayed
        context = branches.prepare_fork_locked(
            conversation_dir=conversation_dir,
            state=state,
            branch_id=branch_id,
            target_turn_id=user_turn_id,
            expected_revision=expected_revision,
            expected_catalog_revision=expected_catalog_revision,
        )
        return branches.fork_delete_path_locked(
            conversation_dir=conversation_dir,
            context=context,
            operation=operation,
        )


def fork_retry_assistant(
    results_dir: Path,
    conversation_id: str,
    assistant_turn_id: str,
    *,
    operation_id: str,
    expected_revision: int,
    model_id: str,
    input_mode: InputMode,
    generation_configuration: Optional[Mapping[str, object]] = None,
    allow_compatibility_default: bool = True,
    branch_id: Optional[str] = None,
    expected_catalog_revision: Optional[int] = None,
) -> RetryAssistantForkResult:
    """Fork before one answer and reserve a fresh assistant node."""
    core.require_results_dir(results_dir)
    core.validate_conversation_id(conversation_id)
    branches.validate_any_turn_id(assistant_turn_id)
    core.validate_operation_id(operation_id)
    core.validate_optional_branch_id(branch_id)
    core.validate_expected_revision(expected_revision)
    branches.validate_expected_catalog_revision(
        expected_catalog_revision
    )
    clean_model_id = core.validate_model_id(model_id)
    clean_input_mode = core.validate_input_mode(input_mode)
    assert isinstance(allow_compatibility_default, bool)
    parsed_configuration = None
    if generation_configuration is not None:
        parsed_configuration = core.parse_generation_configuration(
            generation_configuration,
            expected_model_id=clean_model_id,
            expected_input_mode=clean_input_mode,
        )
    with core.STORE_LOCK.held(results_dir):
        conversation_dir = core.resolve_conversation_dir(
            results_dir, conversation_id
        )
        state = branches.read_root_state(conversation_dir)
        source_branch_id = branches.fork_operation_source_branch_id(
            conversation_dir=conversation_dir,
            state=state,
            operation_id=operation_id,
            branch_id=branch_id,
        )
        operation = branches.retry_fork_operation(
            operation_id=operation_id,
            source_branch_id=source_branch_id,
            target_turn_id=assistant_turn_id,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            generation_configuration=parsed_configuration,
        )
        try:
            replayed = branches.replay_fork_operation_locked(
                conversation_dir=conversation_dir,
                state=state,
                operation=operation,
            )
        except core.ConversationOperationConflictError:
            if parsed_configuration is not None:
                raise
            replayed = None
        if replayed is not None:
            assert isinstance(
                replayed, core.RetryAssistantForkResult
            )
            return replayed
        if parsed_configuration is None:
            if not allow_compatibility_default:
                raise ValueError(
                    "generation_configuration is required for a new"
                    " retry fork"
                )
            clean_configuration = (
                core.default_generation_configuration(
                    model_id=clean_model_id,
                    input_mode=clean_input_mode,
                )
            )
            operation = branches.retry_fork_operation(
                operation_id=operation_id,
                source_branch_id=source_branch_id,
                target_turn_id=assistant_turn_id,
                model_id=clean_model_id,
                input_mode=clean_input_mode,
                generation_configuration=clean_configuration,
            )
            replayed = branches.replay_fork_operation_locked(
                conversation_dir=conversation_dir,
                state=state,
                operation=operation,
            )
            if replayed is not None:
                assert isinstance(
                    replayed, core.RetryAssistantForkResult
                )
                return replayed
        else:
            clean_configuration = (
                core.validate_generation_configuration(
                    parsed_configuration,
                    expected_model_id=clean_model_id,
                    expected_input_mode=clean_input_mode,
                )
            )
        context = branches.prepare_fork_locked(
            conversation_dir=conversation_dir,
            state=state,
            branch_id=branch_id,
            target_turn_id=assistant_turn_id,
            expected_revision=expected_revision,
            expected_catalog_revision=expected_catalog_revision,
        )
        return branches.fork_retry_assistant_locked(
            conversation_dir=conversation_dir,
            context=context,
            operation=operation,
            model_id=clean_model_id,
            input_mode=clean_input_mode,
            generation_configuration=clean_configuration,
        )


__all__ = (
    "AppendResult",
    "BRANCHES_DIR_NAME",
    "BRANCH_COUNT_MAX",
    "BRANCH_DEPTH_MAX",
    "BRANCH_LOCAL_TURNS_MAX",
    "BRANCH_REVISION_MAX",
    "BRANCH_SIBLINGS_MAX",
    "BranchNotFoundError",
    "BranchPoint",
    "BranchListResult",
    "BranchManifestPayload",
    "BranchRecord",
    "CATALOG_REVISION_MAX",
    "CONVERSATIONS_DIR_NAME",
    "CatalogPayload",
    "ConversationCatalog",
    "ConversationCatalogRevisionConflictError",
    "ConversationCorruptError",
    "ConversationLimitError",
    "ConversationManifest",
    "ConversationMutation",
    "ConversationNotFoundError",
    "ConversationOperationConflictError",
    "ConversationRevisionConflictError",
    "ConversationStateError",
    "DEFAULT_TITLE",
    "DeletePathForkResult",
    "EditUserForkResult",
    "FROZEN_NAME",
    "FrozenPayload",
    "GENERATION_CONFIGURATION_CODEC_VERSION",
    "GenerationConfigurationPayload",
    "IDENTIFIER_CHARS_MAX",
    "InputMode",
    "InvalidOperationIdError",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "JSON_FILE_BYTES_MAX",
    "LEGACY_SCHEMA_VERSION",
    "LIST_SIZE_DEFAULT",
    "LIST_SIZE_MAX",
    "MANIFEST_NAME",
    "METADATA_JSON_CHARS_MAX",
    "ManifestPayload",
    "OPERATIONS_DIR_NAME",
    "OperationReceipt",
    "OperationReceiptPayload",
    "OPERATION_ID_PATTERN",
    "PAGE_SIZE_DEFAULT",
    "PAGE_SIZE_MAX",
    "PENDING_GENERATION_KEY",
    "RetryAssistantForkResult",
    "Role",
    "RunLink",
    "RunLinkPayload",
    "SCHEMA_VERSION",
    "TAIL_VERSIONS_MAX",
    "TEXT_CHARS_MAX",
    "TITLE_CHARS_MAX",
    "TRASH_DIR_NAME",
    "TURNS_DIR_NAME",
    "TURN_COUNT_MAX",
    "TURN_ID_WIDTH",
    "TURN_JSON_ENVELOPE_BYTES_MAX",
    "TurnPage",
    "TurnPayload",
    "TurnRecord",
    "append_user",
    "create",
    "delete",
    "fork_delete_from_path",
    "fork_edit_user",
    "fork_retry_assistant",
    "get_branch",
    "get_catalog",
    "get_manifest",
    "get_turns",
    "legacy_branch_id",
    "list_branches",
    "list_conversations",
    "manifest_to_payload",
    "resolve_conversation_dir",
    "set_run_link",
    "turn_to_payload",
    "update_assistant",
    "validate_branch_id",
    "validate_conversation_id",
)
