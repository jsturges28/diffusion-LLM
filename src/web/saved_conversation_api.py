"""HTTP boundary for immutable saved conversation snapshots."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.analytics.metrics import UnsupportedRunVersionError
from src.web import analytics_api
from src.web import conversation_store
from src.web import saved_conversation_store as snapshots


logger = logging.getLogger("diffusion_supervisor")

ResultsDirReader = Callable[[], Path]
STRICT = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class SavedConversationApiDependencies:
    """Supervisor-owned values read by snapshot handlers."""

    results_dir: ResultsDirReader

    def __post_init__(self) -> None:
        assert callable(self.results_dir)


class SnapshotHeadRequest(BaseModel):
    """One exact selected conversation head."""

    model_config = STRICT

    conversation_id: str = Field(
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    branch_id: str = Field(
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    branch_revision: int = Field(ge=1)
    turn_count: int = Field(
        ge=1, le=conversation_store.TURN_COUNT_MAX
    )
    tail_turn_id: str = Field(
        min_length=1,
        max_length=conversation_store.IDENTIFIER_CHARS_MAX,
    )
    tail_version: int = Field(
        ge=1, le=conversation_store.TAIL_VERSIONS_MAX
    )


class CreateSnapshotRequest(SnapshotHeadRequest):
    """Exact head, user title, and idempotency identity."""

    operation_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=conversation_store.OPERATION_ID_PATTERN,
    )
    title: str = Field(
        min_length=1,
        max_length=conversation_store.TITLE_CHARS_MAX,
    )


class RenameSnapshotRequest(BaseModel):
    """Mutable title update against its own revision."""

    model_config = STRICT

    title: str = Field(
        min_length=1,
        max_length=conversation_store.TITLE_CHARS_MAX,
    )
    expected_title_revision: int = Field(ge=1)


@dataclass(frozen=True)
class SavedConversationApi:
    """Bound handlers for snapshot creation and Analytics reads."""

    dependencies: SavedConversationApiDependencies

    def _results_dir(self) -> Path:
        results_dir = self.dependencies.results_dir()
        assert isinstance(results_dir, Path)
        return results_dir

    async def preview(
        self, body: SnapshotHeadRequest
    ) -> JSONResponse:
        try:
            preview = await asyncio.to_thread(
                snapshots.preview_snapshot,
                self._results_dir(),
                **body.model_dump(),
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(
            content={
                "default_title": preview.default_title,
                "turn_count": preview.turn_count,
                "exchange_count": preview.exchange_count,
                "xai_count": preview.xai_count,
                "text_only_count": preview.text_only_count,
                "unavailable_count": preview.unavailable_count,
            }
        )

    async def create(
        self, body: CreateSnapshotRequest
    ) -> JSONResponse:
        try:
            result = await asyncio.to_thread(
                snapshots.create_snapshot,
                self._results_dir(),
                **body.model_dump(),
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(
            content={
                "snapshot_id": result.snapshot_id,
                "title_revision": result.title_revision,
                "replayed": result.replayed,
                "analytics_url": (
                    "/analytics.html?conversation="
                    + result.snapshot_id
                ),
            }
        )

    async def list(self) -> JSONResponse:
        try:
            rows = await asyncio.to_thread(
                snapshots.list_snapshots, self._results_dir()
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content=rows)

    async def metadata(self, snapshot_id: str) -> JSONResponse:
        try:
            metadata = await asyncio.to_thread(
                snapshots.read_metadata,
                self._results_dir(),
                snapshot_id,
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content=metadata)

    async def turns(
        self,
        snapshot_id: str,
        before: str | None = None,
        limit: int = conversation_store.PAGE_SIZE_DEFAULT,
    ) -> JSONResponse:
        try:
            page = await asyncio.to_thread(
                snapshots.page_turns,
                self._results_dir(),
                snapshot_id,
                before=before,
                limit=limit,
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content=page)

    async def rename(
        self,
        snapshot_id: str,
        body: RenameSnapshotRequest,
    ) -> JSONResponse:
        try:
            metadata = await asyncio.to_thread(
                snapshots.rename_snapshot,
                self._results_dir(),
                snapshot_id,
                title=body.title,
                expected_revision=body.expected_title_revision,
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content=metadata)

    async def delete(self, snapshot_id: str) -> JSONResponse:
        try:
            await asyncio.to_thread(
                snapshots.delete_snapshot,
                self._results_dir(),
                snapshot_id,
            )
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content={"deleted": snapshot_id})

    async def pinned_metadata(
        self, snapshot_id: str, turn_id: str
    ) -> JSONResponse:
        return await self._pinned_payload(
            snapshot_id,
            turn_id,
            analytics_api.run_metadata_from_dir,
        )

    async def pinned_metrics(
        self, snapshot_id: str, turn_id: str
    ) -> JSONResponse:
        return await self._pinned_payload(
            snapshot_id,
            turn_id,
            analytics_api.run_metrics_from_dir,
            include_id=True,
        )

    async def pinned_frames(
        self, snapshot_id: str, turn_id: str
    ) -> JSONResponse:
        return await self._pinned_payload(
            snapshot_id,
            turn_id,
            analytics_api.run_frames_from_dir,
            include_id=True,
        )

    async def _pinned_payload(
        self,
        snapshot_id: str,
        turn_id: str,
        reader: Callable[..., dict[str, object]],
        *,
        include_id: bool = False,
    ) -> JSONResponse:
        try:
            run_dir = await asyncio.to_thread(
                snapshots.resolve_pinned_run_dir,
                self._results_dir(),
                snapshot_id,
                turn_id,
            )
            if include_id:
                payload = await asyncio.to_thread(
                    reader, run_dir, turn_id
                )
            else:
                payload = await asyncio.to_thread(reader, run_dir)
        except UnsupportedRunVersionError as exc:
            return analytics_api._unsupported_version_response(exc)
        except Exception as exc:
            return _store_error(exc)
        return JSONResponse(content=payload)


def create_saved_conversation_router(
    dependencies: SavedConversationApiDependencies,
) -> APIRouter:
    """Build the snapshot router around one live data-root reader."""
    assert isinstance(dependencies, SavedConversationApiDependencies)
    api = SavedConversationApi(dependencies)
    router = APIRouter()
    root = "/api/analytics/conversations"
    router.add_api_route(
        root + "/preview", api.preview, methods=["POST"]
    )
    router.add_api_route(root, api.create, methods=["POST"])
    router.add_api_route(root, api.list, methods=["GET"])
    router.add_api_route(
        root + "/{snapshot_id}/metadata",
        api.metadata,
        methods=["GET"],
    )
    router.add_api_route(
        root + "/{snapshot_id}/turns",
        api.turns,
        methods=["GET"],
    )
    router.add_api_route(
        root + "/{snapshot_id}",
        api.rename,
        methods=["PATCH"],
    )
    router.add_api_route(
        root + "/{snapshot_id}",
        api.delete,
        methods=["DELETE"],
    )
    pinned = root + "/{snapshot_id}/turns/{turn_id}/run"
    router.add_api_route(
        pinned + "/metadata",
        api.pinned_metadata,
        methods=["GET"],
    )
    router.add_api_route(
        pinned + "/metrics",
        api.pinned_metrics,
        methods=["GET"],
    )
    router.add_api_route(
        pinned + "/frames",
        api.pinned_frames,
        methods=["GET"],
    )
    assert len(router.routes) == 10
    return router


def _store_error(exc: Exception) -> JSONResponse:
    if isinstance(
        exc,
        (
            snapshots.SnapshotNotFoundError,
            conversation_store.ConversationNotFoundError,
        ),
    ):
        return JSONResponse(
            status_code=404, content={"error": str(exc)}
        )
    if isinstance(
        exc,
        (
            snapshots.SnapshotRevisionConflictError,
            snapshots.SnapshotOperationConflictError,
            conversation_store.ConversationRevisionConflictError,
            conversation_store.ConversationStateError,
        ),
    ):
        return JSONResponse(
            status_code=409, content={"error": str(exc)}
        )
    if isinstance(
        exc,
        (
            snapshots.InvalidSnapshotIdError,
            snapshots.SnapshotCorruptError,
            conversation_store.InvalidConversationIdError,
            conversation_store.InvalidBranchIdError,
            ValueError,
        ),
    ):
        return JSONResponse(
            status_code=400, content={"error": str(exc)}
        )
    logger.exception(
        "saved conversation request failed", exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Saved conversation operation failed."},
    )
