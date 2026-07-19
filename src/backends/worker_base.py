"""Shared FastAPI scaffolding for per-model diffusion workers.

A worker loads exactly one model and exposes a WebSocket
streaming interface identical to the original single-process
server. The supervisor proxies browser traffic to it. All
model-specific logic (loading, validation, sampling, resume)
lives in a ``Backend`` supplied by each worker module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.backends.protocol import (
    MSG_CANCEL,
    MSG_GENERATE,
    MSG_RESUME,
    ModelInfo,
)

logger = logging.getLogger("diffusion_worker")


class FrameStreamer:
    """Forwards frames from an async generator to a WebSocket."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def run(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
        *,
        max_frames: Optional[int] = None,
    ) -> bool:
        """Stream frames; return True if a ``done`` was sent.

        Stops early (returning False) when *max_frames* frame
        messages have been forwarded.
        """
        frame_count = 0
        done_sent = False
        async for frame in generator:
            elapsed = time.monotonic() - start_time
            frame["elapsed"] = round(elapsed, 2)
            await self._ws.send_json(frame)
            if frame.get("type") == "done":
                done_sent = True
            elif frame.get("type") == "frame":
                frame_count += 1
                if (
                    max_frames is not None
                    and frame_count >= max_frames
                ):
                    break
        return done_sent


class Backend(ABC):
    """Model-specific worker logic (loading + streaming)."""

    model_info: ModelInfo

    @abstractmethod
    def load(self) -> None:
        """Blocking model + tokenizer load (runs in a thread)."""

    @abstractmethod
    async def handle_generate(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """Validate, run, and stream a generation request."""

    async def handle_resume(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """Resume from a saved frame (override if supported)."""
        raise NotImplementedError("resume not supported")


async def _send_busy(ws: WebSocket) -> None:
    await ws.send_json(
        {
            "type": "error",
            "message": (
                "A generation is already running."
                " Please wait."
            ),
        }
    )


def create_worker_app(backend: Backend) -> FastAPI:
    """Build the FastAPI app hosting a single model worker."""
    app = FastAPI(title=f"worker:{backend.model_info.id}")
    model_ready = asyncio.Event()
    load_failed = asyncio.Event()
    load_error: Dict[str, str] = {}
    gen_lock = asyncio.Lock()

    @app.on_event("startup")
    async def _startup() -> None:
        async def _load() -> None:
            try:
                await asyncio.to_thread(backend.load)
            except Exception as exc:  # noqa: BLE001
                load_error["message"] = str(exc)
                load_failed.set()
                logger.exception(
                    "model %s failed to load",
                    backend.model_info.id,
                )
                return
            model_ready.set()
            logger.info(
                "model %s ready", backend.model_info.id
            )

        asyncio.create_task(_load())

    @app.get("/health")
    async def _health() -> JSONResponse:
        if load_failed.is_set():
            status = "error"
        elif model_ready.is_set():
            status = "ready"
        else:
            status = "loading"
        versions: Dict[str, str] = {}
        try:
            import torch
            import transformers

            versions = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        except Exception:  # noqa: BLE001
            versions = {}
        payload: Dict[str, Any] = {
            "status": status,
            "id": backend.model_info.id,
            "versions": versions,
        }
        if status == "error":
            payload["message"] = load_error.get(
                "message", "Model failed to load."
            )
        return JSONResponse(payload)

    @app.get("/params")
    async def _params() -> JSONResponse:
        return JSONResponse(backend.model_info.model_dump())

    @app.websocket("/ws")
    async def _ws(ws: WebSocket) -> None:
        await ws.accept()
        cancel_event = asyncio.Event()
        stream = FrameStreamer(ws)
        try:
            if load_failed.is_set():
                await ws.send_json(
                    {
                        "type": "error",
                        "message": load_error.get(
                            "message", "Model failed to load."
                        ),
                    }
                )
                return
            if not model_ready.is_set():
                await ws.send_json(
                    {
                        "type": "model_status",
                        "status": "loading",
                    }
                )
                ready_task = asyncio.ensure_future(
                    model_ready.wait()
                )
                failed_task = asyncio.ensure_future(
                    load_failed.wait()
                )
                _, pending = await asyncio.wait(
                    {ready_task, failed_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if load_failed.is_set():
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": load_error.get(
                                "message",
                                "Model failed to load.",
                            ),
                        }
                    )
                    return
            await ws.send_json(
                {"type": "model_status", "status": "ready"}
            )
            while True:
                data = await ws.receive_json()
                mtype = data.get("type")

                if mtype == MSG_CANCEL:
                    cancel_event.set()
                    continue

                if mtype == MSG_GENERATE:
                    if gen_lock.locked():
                        await _send_busy(ws)
                        continue
                    async with gen_lock:
                        cancel_event.clear()
                        await backend.handle_generate(
                            ws, data, cancel_event, stream
                        )
                    continue

                if mtype == MSG_RESUME:
                    if gen_lock.locked():
                        await _send_busy(ws)
                        continue
                    async with gen_lock:
                        cancel_event.clear()
                        await backend.handle_resume(
                            ws, data, cancel_event, stream
                        )
                    continue

                await ws.send_json(
                    {
                        "type": "error",
                        "message": (
                            f"Unknown message type: {mtype}"
                        ),
                    }
                )
        except WebSocketDisconnect:
            cancel_event.set()
            logger.info("worker client disconnected")

    return app
