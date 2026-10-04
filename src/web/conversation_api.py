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
from typing import Callable, Dict, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.backends.registry import REGISTRY
from src.web import conversation_store, run_store


logger = logging.getLogger("diffusion_supervisor")

ResultsDirReader = Callable[[], Path]
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


class AppendUserRequest(BaseModel):
    """One user turn plus the model reserved for its assistant."""

    model_config = STRICT

    expected_revision: int = Field(ge=1)
    text: str = Field(max_length=conversation_store.TEXT_CHARS_MAX)
    model_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    input_mode: Literal["chat", "completion"]
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


class UpdateAssistantRequest(BaseModel):
    """The terminal or revised value of a reserved assistant."""

    model_config = STRICT

    expected_revision: int = Field(ge=1)
    text: str = Field(max_length=conversation_store.TEXT_CHARS_MAX)
    partial: bool
    context_pack: Dict[str, JsonValue] = Field(default_factory=dict)
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


class LinkRunRequest(BaseModel):
    """A saved run revision to attach to the tail assistant."""

    model_config = STRICT

    expected_revision: int = Field(ge=1)
    run_id: str = Field(
        max_length=conversation_store.IDENTIFIER_CHARS_MAX
    )
    run_revision: int = Field(ge=0)


class UnlinkRunRequest(BaseModel):
    """The conversation revision an unlink is based on."""

    model_config = STRICT

    expected_revision: int = Field(ge=1)


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
                "conversation": (
                    conversation_store.manifest_to_payload(manifest)
                )
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
                    conversation_store.manifest_to_payload(manifest)
                    for manifest in manifests
                ]
            }
        )

    async def conversation_metadata(
        self,
        conversation_id: str,
    ) -> JSONResponse:
        try:
            manifest = await asyncio.to_thread(
                conversation_store.get_manifest,
                self._results_dir(),
                conversation_id,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(
            content={
                "conversation": (
                    conversation_store.manifest_to_payload(manifest)
                )
            }
        )

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
    ) -> JSONResponse:
        try:
            page = await asyncio.to_thread(
                conversation_store.get_turns,
                self._results_dir(),
                conversation_id,
                before=before,
                limit=limit,
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
                expected_revision=body.expected_revision,
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
                "conversation": (
                    conversation_store.manifest_to_payload(
                        result.manifest
                    )
                ),
                "user_turn": conversation_store.turn_to_payload(
                    result.user_turn
                ),
                "assistant_turn": (
                    conversation_store.turn_to_payload(
                        result.assistant_turn
                    )
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
                expected_revision=body.expected_revision,
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
            link = await asyncio.to_thread(
                _validated_run_link,
                self._results_dir(),
                body.run_id,
                body.run_revision,
            )
            result = await asyncio.to_thread(
                conversation_store.set_run_link,
                self._results_dir(),
                conversation_id,
                assistant_turn_id,
                expected_revision=body.expected_revision,
                run_link=link,
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
                expected_revision=body.expected_revision,
                run_link=None,
            )
        except _STORE_FAILURES as exc:
            return _error_response(exc)
        return JSONResponse(content=_mutation_payload(result))


_STORE_FAILURES = (
    conversation_store.InvalidConversationIdError,
    conversation_store.ConversationNotFoundError,
    conversation_store.ConversationRevisionConflictError,
    conversation_store.ConversationStateError,
    conversation_store.ConversationCorruptError,
    ValueError,
    OSError,
)

_API_FAILURES = (
    *_STORE_FAILURES,
    run_store.InvalidRunIdError,
    run_store.RunNotFoundError,
    RunRevisionConflictError,
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
    assert len(router.routes) == 9, (
        "all conversation routes registered"
    )
    return router


def _page_payload(
    page: conversation_store.TurnPage,
) -> Dict[str, object]:
    return {
        "conversation_id": page.conversation_id,
        "revision": page.revision,
        "turns": [
            conversation_store.turn_to_payload(turn)
            for turn in page.turns
        ],
        "next_before": page.next_before,
        "has_more": page.has_more,
    }


def _mutation_payload(
    result: conversation_store.ConversationMutation,
) -> Dict[str, object]:
    return {
        "conversation": conversation_store.manifest_to_payload(
            result.manifest
        ),
        "turn": conversation_store.turn_to_payload(result.turn),
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
) -> conversation_store.RunLink:
    """Resolve one current run and pin the revision the caller saw."""
    run_store.resolve_run_dir(results_dir, run_id)
    actual_revision = run_store.read_revision(results_dir, run_id)
    if requested_revision != actual_revision:
        raise RunRevisionConflictError(
            run_id,
            requested_revision,
            actual_revision,
        )
    return conversation_store.RunLink(run_id, actual_revision)


def _error_response(exc: Exception) -> JSONResponse:
    """Translate expected store and boundary failures to HTTP."""
    if isinstance(
        exc, conversation_store.ConversationRevisionConflictError
    ):
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
    if isinstance(exc, conversation_store.ConversationStateError):
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "reason": "state_conflict"},
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
        (
            conversation_store.InvalidConversationIdError,
            run_store.InvalidRunIdError,
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
