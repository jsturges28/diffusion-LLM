"""FastAPI boundary for durable, pageable conversations.

The store owns filesystem shape and state transitions. This module
owns HTTP request models, response shapes, live supervisor
dependencies, model checks, saved-run checks, and error translation.
It never imports ``server``; the supervisor includes the router with
a callable that reads its current results root.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.backends.registry import REGISTRY
from src.web import (
    conversation_generation,
    conversation_store,
    run_store,
)


logger = logging.getLogger("diffusion_supervisor")

ResultsDirReader = Callable[[], Path]
GenerationParameterValue = Union[bool, int, float, str]
STRICT = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class ConversationApiDependencies:
    """Supervisor-owned values read by conversation handlers."""

    results_dir: ResultsDirReader

    def __post_init__(self) -> None:
        assert callable(self.results_dir)


class CreateConversationRequest(BaseModel):
    """Metadata for a new empty conversation."""

    model_config = STRICT

    title: str = Field(
        default=conversation_store.DEFAULT_TITLE,
        max_length=conversation_store.TITLE_CHARS_MAX,
    )


class BranchMutationRequest(BaseModel):
    """The selected branch and the revision the caller read."""

    model_config = STRICT

    branch_id: str = Field(
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    branch_revision: int = Field(ge=1)


class CatalogBranchMutationRequest(BaseModel):
    """A fork mutation, including legacy lost-response replay."""

    model_config = STRICT

    branch_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    branch_revision: int = Field(ge=1)
    catalog_revision: int = Field(ge=0)
    operation_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=conversation_store.OPERATION_ID_PATTERN,
    )


class AppendUserRequest(BranchMutationRequest):
    """One user turn plus the model reserved for its assistant."""

    text: str = Field(max_length=conversation_store.TEXT_CHARS_MAX)
    model_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    input_mode: Literal["chat", "completion"]
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


class UpdateAssistantRequest(BranchMutationRequest):
    """The terminal or revised value of a reserved assistant."""

    text: str = Field(max_length=conversation_store.TEXT_CHARS_MAX)
    partial: bool
    context_pack: Dict[str, JsonValue] = Field(default_factory=dict)
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


class LinkRunRequest(BranchMutationRequest):
    """A saved run revision to attach to the tail assistant."""

    assistant_turn_index: int = Field(
        ge=1,
        le=conversation_store.TURN_COUNT_MAX,
        strict=True,
    )
    assistant_turn_version: int = Field(
        ge=1,
        le=conversation_store.TAIL_VERSIONS_MAX,
        strict=True,
    )
    run_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    run_revision: int = Field(ge=0)


class UnlinkRunRequest(BranchMutationRequest):
    """The conversation revision an unlink is based on."""


class GenerationConfigurationRequest(BaseModel):
    """One exact action-local Run settings snapshot."""

    model_config = STRICT

    codec_version: Literal[1]
    model_id: str = Field(
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    input_mode: Literal["chat", "completion"]
    device: str = Field(
        min_length=1,
        max_length=(
            conversation_generation.GENERATION_DEVICE_CHARS_MAX
        ),
    )
    schema_id: str = Field(
        min_length=(
            conversation_generation.GENERATION_SCHEMA_ID_CHARS
        ),
        max_length=(
            conversation_generation.GENERATION_SCHEMA_ID_CHARS
        ),
        pattern=r"^[0-9a-f]{64}$",
    )
    experimental: bool
    parameters: Dict[str, GenerationParameterValue] = Field(
        max_length=(
            conversation_generation.GENERATION_PARAMETER_COUNT_MAX
        )
    )


class EditUserForkRequest(CatalogBranchMutationRequest):
    """Replacement text and current model for a user fork."""

    text: str = Field(max_length=conversation_store.TEXT_CHARS_MAX)
    model_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    input_mode: Literal["chat", "completion"]
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)
    generation_configuration: Optional[
        GenerationConfigurationRequest
    ] = None


class DeletePathForkRequest(CatalogBranchMutationRequest):
    """The exact source state for a delete-from-path fork."""


class RetryAssistantForkRequest(CatalogBranchMutationRequest):
    """The current model reserved for a retried response."""

    model_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    input_mode: Literal["chat", "completion"]
    generation_configuration: Optional[
        GenerationConfigurationRequest
    ] = None


class RunRevisionConflictError(Exception):
    """A requested run revision is no longer current."""

    def __init__(
        self,
        run_id: str,
        expected: int,
        actual: int,
    ) -> None:
        super().__init__(
            f"run {run_id} has moved on: expected revision"
            f" {expected}, found {actual}"
        )
        self.run_id = run_id
        self.expected = expected
        self.actual = actual


class RunOwnershipConflictError(Exception):
    """A saved run does not name the requested assistant version."""


@dataclass(frozen=True)
class ConversationApi:
    """Conversation handlers registered by the router factory."""

    dependencies: ConversationApiDependencies

    def _results_dir(self) -> Path:
        results_dir = self.dependencies.results_dir()
        assert isinstance(results_dir, Path)
        return results_dir

    async def create_conversation(
        self,
        body: CreateConversationRequest,
    ) -> JSONResponse:
        try:
            manifest = await asyncio.to_thread(
                conversation_store.create,
                self._results_dir(),
                title=body.title,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            status_code=201,
            content={
                "conversation": _conversation_payload(manifest)
            },
        )

    async def list_conversations(
        self,
        limit: int = conversation_store.LIST_SIZE_DEFAULT,
    ) -> JSONResponse:
        try:
            manifests = await asyncio.to_thread(
                conversation_store.list_conversations,
                self._results_dir(),
                limit=limit,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            content={
                "conversations": [
                    _conversation_payload(manifest)
                    for manifest in manifests
                ]
            }
        )

    async def conversation_metadata(
        self,
        conversation_id: str,
        branch_id: Optional[str] = None,
    ) -> JSONResponse:
        try:
            manifest = await asyncio.to_thread(
                conversation_store.get_manifest,
                self._results_dir(),
                conversation_id,
                branch_id=branch_id,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            content={
                "conversation": _conversation_payload(manifest)
            }
        )

    async def conversation_branches(
        self,
        conversation_id: str,
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                conversation_store.list_branches,
                self._results_dir(),
                conversation_id,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_branches_payload(result))

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> JSONResponse:
        try:
            await asyncio.to_thread(
                conversation_store.delete,
                self._results_dir(),
                conversation_id,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            content={
                "success": True,
                "conversation_id": conversation_id,
            }
        )

    async def conversation_turns(
        self,
        conversation_id: str,
        before: Optional[str] = None,
        limit: int = conversation_store.PAGE_SIZE_DEFAULT,
        branch_id: Optional[str] = None,
    ) -> JSONResponse:
        try:
            page = await asyncio.to_thread(
                conversation_store.get_turns,
                self._results_dir(),
                conversation_id,
                before=before,
                limit=limit,
                branch_id=branch_id,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_page_payload(page))

    async def append_user(
        self,
        conversation_id: str,
        body: AppendUserRequest,
    ) -> JSONResponse:
        try:
            _validate_model(body.model_id, body.input_mode)
            result = await asyncio.to_thread(
                conversation_store.append_user,
                self._results_dir(),
                conversation_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                text=body.text,
                model_id=body.model_id,
                input_mode=body.input_mode,
                metadata=body.metadata,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            status_code=201,
            content={
                "conversation": _conversation_payload(
                    result.manifest
                ),
                "user_turn": _turn_payload(
                    result.user_turn,
                    selected_branch_id=body.branch_id,
                ),
                "assistant_turn": _turn_payload(
                    result.assistant_turn,
                    selected_branch_id=body.branch_id,
                ),
            },
        )

    async def update_assistant(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        body: UpdateAssistantRequest,
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                conversation_store.update_assistant,
                self._results_dir(),
                conversation_id,
                assistant_turn_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                text=body.text,
                partial=body.partial,
                context_pack=body.context_pack,
                metadata=body.metadata,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_mutation_payload(result))

    async def link_run(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        body: LinkRunRequest,
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                _set_validated_run_link,
                self._results_dir(),
                conversation_id=conversation_id,
                branch_id=body.branch_id,
                assistant_turn_id=assistant_turn_id,
                branch_revision=body.branch_revision,
                run_id=body.run_id,
                run_revision=body.run_revision,
                assistant_turn_index=body.assistant_turn_index,
                assistant_turn_version=body.assistant_turn_version,
            )
        except _API_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_mutation_payload(result))

    async def unlink_run(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        body: UnlinkRunRequest,
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                conversation_store.set_run_link,
                self._results_dir(),
                conversation_id,
                assistant_turn_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                run_link=None,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_mutation_payload(result))

    async def fork_edit_user(
        self,
        conversation_id: str,
        user_turn_id: str,
        body: EditUserForkRequest,
    ) -> JSONResponse:
        try:
            configuration = (
                None
                if body.generation_configuration is None
                else body.generation_configuration.model_dump(
                    mode="python"
                )
            )
            result = await asyncio.to_thread(
                conversation_store.fork_edit_user,
                self._results_dir(),
                conversation_id,
                user_turn_id,
                operation_id=body.operation_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                expected_catalog_revision=body.catalog_revision,
                text=body.text,
                model_id=body.model_id,
                input_mode=body.input_mode,
                generation_configuration=configuration,
                allow_compatibility_default=False,
                metadata=body.metadata,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            status_code=201,
            content=_edit_fork_payload(result),
        )

    async def fork_delete_from_path(
        self,
        conversation_id: str,
        user_turn_id: str,
        body: DeletePathForkRequest,
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                conversation_store.fork_delete_from_path,
                self._results_dir(),
                conversation_id,
                user_turn_id,
                operation_id=body.operation_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                expected_catalog_revision=body.catalog_revision,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            status_code=201,
            content=_delete_fork_payload(result),
        )

    async def fork_retry_assistant(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        body: RetryAssistantForkRequest,
    ) -> JSONResponse:
        try:
            configuration = (
                None
                if body.generation_configuration is None
                else body.generation_configuration.model_dump(
                    mode="python"
                )
            )
            result = await asyncio.to_thread(
                conversation_store.fork_retry_assistant,
                self._results_dir(),
                conversation_id,
                assistant_turn_id,
                operation_id=body.operation_id,
                branch_id=body.branch_id,
                expected_revision=body.branch_revision,
                expected_catalog_revision=body.catalog_revision,
                model_id=body.model_id,
                input_mode=body.input_mode,
                generation_configuration=configuration,
                allow_compatibility_default=False,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            status_code=201,
            content=_retry_fork_payload(result),
        )


_STORE_FAILURES = (
    conversation_store.InvalidConversationIdError,
    conversation_store.InvalidBranchIdError,
    conversation_store.ConversationNotFoundError,
    conversation_store.BranchNotFoundError,
    conversation_store.ConversationRevisionConflictError,
    conversation_store.ConversationCatalogRevisionConflictError,
    conversation_store.ConversationOperationConflictError,
    conversation_store.ConversationStateError,
    conversation_store.ConversationCorruptError,
    OverflowError,
    ValueError,
    OSError,
)

_API_FAILURES = (
    *_STORE_FAILURES,
    run_store.InvalidRunIdError,
    run_store.RunNotFoundError,
    RunRevisionConflictError,
    RunOwnershipConflictError,
)


def create_conversation_router(
    dependencies: ConversationApiDependencies,
) -> APIRouter:
    """Build the router around one narrow live dependency."""
    assert isinstance(dependencies, ConversationApiDependencies)

    api = ConversationApi(dependencies)
    router = APIRouter()
    router.add_api_route(
        "/api/conversations",
        api.create_conversation,
        methods=["POST"],
    )
    router.add_api_route(
        "/api/conversations",
        api.list_conversations,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/metadata",
        api.conversation_metadata,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/branches",
        api.conversation_branches,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}",
        api.delete_conversation,
        methods=["DELETE"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/turns",
        api.conversation_turns,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/turns",
        api.append_user,
        methods=["POST"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}"
        ),
        api.update_assistant,
        methods=["PUT"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}/run"
        ),
        api.link_run,
        methods=["PUT"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}/run"
        ),
        api.unlink_run,
        methods=["DELETE"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/branches/"
            "edit-user/{user_turn_id}"
        ),
        api.fork_edit_user,
        methods=["POST"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/branches/"
            "delete-from-path/{user_turn_id}"
        ),
        api.fork_delete_from_path,
        methods=["POST"],
    )
    router.add_api_route(
        (
            "/api/conversations/{conversation_id}/branches/"
            "retry-assistant/{assistant_turn_id}"
        ),
        api.fork_retry_assistant,
        methods=["POST"],
    )
    assert len(router.routes) == 13, (
        "all conversation routes registered"
    )
    return router


def _page_payload(
    page: conversation_store.TurnPage,
) -> Dict[str, object]:
    branch_id = page.branch_id
    if branch_id is None:
        branch_id = conversation_store.legacy_branch_id(
            page.conversation_id
        )
    default_branch_id = page.default_branch_id or branch_id
    return {
        "schema_version": page.schema_version,
        "conversation_id": page.conversation_id,
        "branch_id": branch_id,
        "branch_revision": page.revision,
        "revision": page.revision,
        "catalog_revision": page.catalog_revision,
        "default_branch_id": default_branch_id,
        "turns": [
            _turn_payload(
                turn,
                selected_branch_id=branch_id,
            )
            for turn in page.turns
        ],
        "next_before": page.next_before,
        "has_more": page.has_more,
        "branch_points": [
            _branch_point_payload(point)
            for point in page.branch_points
        ],
    }


def _conversation_payload(
    manifest: conversation_store.ConversationManifest,
) -> Dict[str, object]:
    payload: Dict[str, object] = dict(
        conversation_store.manifest_to_payload(manifest)
    )
    branch_id = manifest.branch_id
    if branch_id is None:
        branch_id = conversation_store.legacy_branch_id(manifest.id)
    default_branch_id = manifest.default_branch_id or branch_id
    payload.update(
        {
            "branch_id": branch_id,
            "branch_revision": manifest.revision,
            "catalog_revision": manifest.catalog_revision,
            "default_branch_id": default_branch_id,
        }
    )
    return payload


def _turn_payload(
    turn: conversation_store.TurnRecord,
    *,
    selected_branch_id: str,
) -> Dict[str, object]:
    payload: Dict[str, object] = dict(
        conversation_store.turn_to_payload(turn)
    )
    payload["branch_id"] = turn.branch_id or selected_branch_id
    return payload


def _branch_point_payload(
    point: conversation_store.BranchPoint,
) -> Dict[str, object]:
    assert point.selected_branch_id in point.branch_ids
    assert (
        len(point.branch_ids) <= conversation_store.BRANCH_COUNT_MAX
    )
    return {
        "turn_index": point.turn_index,
        "source_branch_id": point.source_branch_id,
        "selected_branch_id": point.selected_branch_id,
        "branch_ids": list(point.branch_ids),
        "deleted_branch_ids": list(point.deleted_branch_ids),
    }


def _catalog_payload(
    catalog: conversation_store.ConversationCatalog,
) -> Dict[str, object]:
    return {
        "schema_version": catalog.schema_version,
        "conversation_id": catalog.conversation_id,
        "title": catalog.title,
        "catalog_revision": catalog.revision,
        "created_at": catalog.created_at,
        "updated_at": catalog.updated_at,
        "default_branch_id": catalog.default_branch_id,
        "branch_ids": list(catalog.branch_ids),
    }


def _branch_payload(
    branch: conversation_store.BranchRecord,
) -> Dict[str, object]:
    return {
        "conversation_id": branch.conversation_id,
        "branch_id": branch.branch_id,
        "parent_branch_id": branch.parent_branch_id,
        "prefix_turn_count": branch.prefix_turn_count,
        "branch_revision": branch.revision,
        "revision": branch.revision,
        "created_at": branch.created_at,
        "updated_at": branch.updated_at,
        "turn_count": branch.turn_count,
        "tail_role": branch.tail_role,
        "tail_turn_id": branch.tail_turn_id,
        "tail_version": branch.tail_version,
        "pending_assistant_id": branch.pending_assistant_id,
        "depth": branch.depth,
    }


def _branches_payload(
    result: conversation_store.BranchListResult,
) -> Dict[str, object]:
    payload = _catalog_payload(result.catalog)
    payload["branches"] = [
        _branch_payload(branch) for branch in result.branches
    ]
    return payload


def _mutation_payload(
    result: conversation_store.ConversationMutation,
) -> Dict[str, object]:
    conversation = _conversation_payload(result.manifest)
    branch_id = str(conversation["branch_id"])
    return {
        "conversation": conversation,
        "turn": _turn_payload(
            result.turn,
            selected_branch_id=branch_id,
        ),
    }


def _edit_fork_payload(
    result: conversation_store.EditUserForkResult,
) -> Dict[str, object]:
    branch_id = result.branch.branch_id
    return {
        "catalog": _catalog_payload(result.catalog),
        "conversation": _conversation_payload(result.manifest),
        "branch": _branch_payload(result.branch),
        "source_branch_id": result.source_branch_id,
        "replaced_user_turn_id": result.replaced_user_turn_id,
        "user_turn": _turn_payload(
            result.user_turn,
            selected_branch_id=branch_id,
        ),
        "assistant_turn": _turn_payload(
            result.assistant_turn,
            selected_branch_id=branch_id,
        ),
        "generation_configuration": result.generation_configuration,
    }


def _delete_fork_payload(
    result: conversation_store.DeletePathForkResult,
) -> Dict[str, object]:
    return {
        "catalog": _catalog_payload(result.catalog),
        "conversation": _conversation_payload(result.manifest),
        "branch": _branch_payload(result.branch),
        "source_branch_id": result.source_branch_id,
        "deleted_user_turn_id": result.deleted_user_turn_id,
        "removed_turn_count": result.removed_turn_count,
    }


def _retry_fork_payload(
    result: conversation_store.RetryAssistantForkResult,
) -> Dict[str, object]:
    branch_id = result.branch.branch_id
    return {
        "catalog": _catalog_payload(result.catalog),
        "conversation": _conversation_payload(result.manifest),
        "branch": _branch_payload(result.branch),
        "source_branch_id": result.source_branch_id,
        "retried_assistant_turn_id": (
            result.retried_assistant_turn_id
        ),
        "assistant_turn": _turn_payload(
            result.assistant_turn,
            selected_branch_id=branch_id,
        ),
        "generation_configuration": result.generation_configuration,
    }


def _validate_model(model_id: str, input_mode: str) -> None:
    entry = REGISTRY.get(model_id)
    if entry is None:
        raise ValueError(f"unknown model: {model_id}")
    expected = entry.capabilities.input_mode
    if input_mode != expected:
        raise ValueError(
            f"model {model_id} uses {expected} input,"
            f" not {input_mode}"
        )


def _validated_run_link(
    results_dir: Path,
    run_id: str,
    requested_revision: int,
    *,
    conversation_id: str,
    branch_id: str,
    assistant_turn_id: str,
    assistant_turn_index: int,
    assistant_turn_version: int,
) -> conversation_store.RunLink:
    """Resolve one current run and verify its exact durable owner."""
    metadata = run_store.read_metadata(results_dir, run_id)
    actual_revision = _run_metadata_revision(metadata)
    if requested_revision != actual_revision:
        raise RunRevisionConflictError(
            run_id,
            requested_revision,
            actual_revision,
        )
    expected = {
        "conversation_id": conversation_id,
        "branch_id": branch_id,
        "assistant_turn_id": assistant_turn_id,
        "turn_index": assistant_turn_index,
        "assistant_turn_version": assistant_turn_version,
    }
    if not _run_metadata_has_owner(metadata, expected):
        raise RunOwnershipConflictError(
            f"run {run_id} does not belong to the requested"
            " assistant version"
        )
    return conversation_store.RunLink(run_id, actual_revision)


def _set_validated_run_link(
    results_dir: Path,
    *,
    conversation_id: str,
    branch_id: str,
    assistant_turn_id: str,
    branch_revision: int,
    run_id: str,
    run_revision: int,
    assistant_turn_index: int,
    assistant_turn_version: int,
) -> conversation_store.ConversationMutation:
    """Validate and attach one run at a single linearization point.

    Lock order is always ``runs.lock`` then ``conversations.lock``.
    The run store never imports the conversation store, so no inverse
    acquisition exists.
    """
    with run_store.publication_lock(results_dir):
        link = _validated_run_link(
            results_dir,
            run_id,
            run_revision,
            conversation_id=conversation_id,
            branch_id=branch_id,
            assistant_turn_id=assistant_turn_id,
            assistant_turn_index=assistant_turn_index,
            assistant_turn_version=assistant_turn_version,
        )
        return conversation_store.set_run_link(
            results_dir,
            conversation_id,
            assistant_turn_id,
            branch_id=branch_id,
            expected_revision=branch_revision,
            run_link=link,
            expected_turn_index=assistant_turn_index,
            expected_turn_version=assistant_turn_version,
        )


def _run_metadata_revision(metadata: Dict[str, object]) -> int:
    revision = metadata.get(run_store.REVISION_KEY)
    if isinstance(revision, bool) or not isinstance(revision, int):
        return 0
    return revision


def _run_metadata_has_owner(
    metadata: Dict[str, object],
    expected: Dict[str, object],
) -> bool:
    """Whether metadata names one complete exact turn version."""
    for name, expected_value in expected.items():
        actual = metadata.get(name)
        if isinstance(expected_value, int):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
            ):
                return False
        elif not isinstance(actual, str):
            return False
        if actual != expected_value:
            return False
    return True


def _run_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, RunRevisionConflictError):
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "reason": "run_revision_conflict",
                "run_id": exc.run_id,
                "expected_revision": exc.expected,
                "revision": exc.actual,
            },
        )
    assert isinstance(exc, RunOwnershipConflictError)
    return JSONResponse(
        status_code=409,
        content={
            "error": str(exc),
            "reason": "run_ownership_conflict",
        },
    )


def _conversation_conflict_response(
    exc: Exception,
) -> JSONResponse:
    if isinstance(
        exc, conversation_store.ConversationRevisionConflictError
    ):
        return _revision_conflict_response(exc)
    if isinstance(
        exc,
        conversation_store.ConversationCatalogRevisionConflictError,
    ):
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "reason": "catalog_revision_conflict",
                "conversation_id": exc.conversation_id,
                "expected_catalog_revision": exc.expected,
                "catalog_revision": exc.actual,
            },
        )
    assert isinstance(
        exc, conversation_store.ConversationOperationConflictError
    )
    return JSONResponse(
        status_code=409,
        content={
            "error": str(exc),
            "reason": "operation_id_conflict",
            "operation_id": exc.operation_id,
        },
    )


def _revision_conflict_response(
    exc: conversation_store.ConversationRevisionConflictError,
) -> JSONResponse:
    if exc.branch_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "reason": "branch_revision_conflict",
                "conversation_id": exc.conversation_id,
                "branch_id": exc.branch_id,
                "expected_branch_revision": exc.expected,
                "branch_revision": exc.actual,
            },
        )
    return JSONResponse(
        status_code=409,
        content={
            "error": str(exc),
            "reason": "revision_conflict",
            "conversation_id": exc.conversation_id,
            "expected_revision": exc.expected,
            "revision": exc.actual,
        },
    )


def _error_response(exc: Exception) -> JSONResponse:
    """Translate expected store and boundary failures to HTTP."""
    if isinstance(
        exc,
        (
            conversation_store.ConversationRevisionConflictError,
            conversation_store.ConversationCatalogRevisionConflictError,
            conversation_store.ConversationOperationConflictError,
        ),
    ):
        return _conversation_conflict_response(exc)
    if isinstance(
        exc,
        (RunRevisionConflictError, RunOwnershipConflictError),
    ):
        return _run_error_response(exc)
    if isinstance(exc, conversation_store.ConversationStateError):
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "reason": "state_conflict"},
        )
    if isinstance(exc, conversation_store.BranchNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "error": str(exc),
                "reason": "branch_not_found",
                "conversation_id": exc.conversation_id,
                "branch_id": exc.branch_id,
            },
        )
    if isinstance(
        exc,
        (
            conversation_store.ConversationNotFoundError,
            run_store.RunNotFoundError,
        ),
    ):
        return JSONResponse(
            status_code=404,
            content={"error": str(exc), "reason": "not_found"},
        )
    if isinstance(
        exc,
        conversation_store.InvalidBranchIdError,
    ):
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "reason": "invalid_branch"},
        )
    if isinstance(
        exc,
        (
            conversation_store.InvalidConversationIdError,
            run_store.InvalidRunIdError,
            OverflowError,
            ValueError,
        ),
    ):
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "reason": "invalid_request"},
        )
    logger.exception("conversation API failed")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "reason": "store_error"},
    )
