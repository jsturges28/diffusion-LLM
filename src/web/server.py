"""FastAPI server for live LLaDA diffusion visualization.

Loads the model once at startup (in a background thread) and exposes
a WebSocket endpoint that streams diffusion frames to the browser.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)

from src.inference.streaming_sampler import streaming_generate

logger = logging.getLogger("llada_web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

PARAM_LIMITS = {
    "steps": (1, 512),
    "gen_length": (1, 1024),
    "block_length": (1, 1024),
    "temperature": (0.0, 2.0),
    "cfg_scale": (0.0, 10.0),
}

app = FastAPI(title="LLaDA Diffusion Visualizer")

model: Any = None
tokenizer: Any = None
model_ready = asyncio.Event()
generation_lock = asyncio.Lock()


def _load_model() -> tuple[Any, Any]:
    """Blocking model + tokenizer load (runs in a thread)."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Loading tokenizer from %s ...", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tok.padding_side != "left":
        tok.padding_side = "left"

    logger.info("Loading model from %s ...", MODEL_NAME)
    mdl = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16 if device.type == "cuda" else None
        ),
        device_map=(
            "auto" if device.type == "cuda" else None
        ),
    ).eval()

    logger.info("Model loaded successfully on %s.", device)
    return mdl, tok


@app.on_event("startup")
async def startup_load_model() -> None:
    """Kick off model loading in a background thread."""

    async def _load() -> None:
        global model, tokenizer  # noqa: PLW0603
        mdl, tok = await asyncio.to_thread(_load_model)
        model = mdl
        tokenizer = tok
        model_ready.set()

    asyncio.create_task(_load())


def _clamp(
    value: float, low: float, high: float
) -> float:
    return max(low, min(high, value))


def _validate_params(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Extract and clamp generation parameters from a message."""
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    steps = int(
        _clamp(
            float(data.get("steps", 128)),
            *PARAM_LIMITS["steps"],
        )
    )
    gen_length = int(
        _clamp(
            float(data.get("gen_length", 128)),
            *PARAM_LIMITS["gen_length"],
        )
    )
    block_length = int(
        _clamp(
            float(data.get("block_length", 32)),
            *PARAM_LIMITS["block_length"],
        )
    )
    temperature = float(
        _clamp(
            float(data.get("temperature", 0.0)),
            *PARAM_LIMITS["temperature"],
        )
    )
    cfg_scale = float(
        _clamp(
            float(data.get("cfg_scale", 0.0)),
            *PARAM_LIMITS["cfg_scale"],
        )
    )

    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible"
            f" by block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by"
            f" num_blocks ({num_blocks})"
        )

    return {
        "prompt": prompt,
        "steps": steps,
        "gen_length": gen_length,
        "block_length": block_length,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    if not model_ready.is_set():
        await ws.send_json(
            {"type": "model_status", "status": "loading"}
        )
        await model_ready.wait()
    await ws.send_json(
        {"type": "model_status", "status": "ready"}
    )

    cancel_event = asyncio.Event()

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "cancel":
                cancel_event.set()
                continue

            if msg_type != "generate":
                await ws.send_json(
                    {
                        "type": "error",
                        "message": (
                            f"Unknown message type: {msg_type}"
                        ),
                    }
                )
                continue

            try:
                params = _validate_params(data)
            except (ValueError, TypeError) as exc:
                await ws.send_json(
                    {"type": "error", "message": str(exc)}
                )
                continue

            if generation_lock.locked():
                await ws.send_json(
                    {
                        "type": "error",
                        "message": (
                            "A generation is already running."
                            " Please wait."
                        ),
                    }
                )
                continue

            cancel_event.clear()
            start_time = time.monotonic()

            async with generation_lock:
                try:
                    generator = streaming_generate(
                        model,
                        tokenizer,
                        params["prompt"],
                        steps=params["steps"],
                        gen_length=params["gen_length"],
                        block_length=params[
                            "block_length"
                        ],
                        temperature=params["temperature"],
                        cfg_scale=params["cfg_scale"],
                        cancel_event=cancel_event,
                    )
                    async for frame in generator:
                        elapsed = (
                            time.monotonic() - start_time
                        )
                        frame["elapsed"] = round(
                            elapsed, 2
                        )
                        await ws.send_json(frame)
                except Exception as exc:
                    logger.exception(
                        "Generation failed: %s", exc
                    )
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": str(exc),
                        }
                    )

    except WebSocketDisconnect:
        cancel_event.set()
        logger.info("Client disconnected.")


app.mount(
    "/",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="static",
)
