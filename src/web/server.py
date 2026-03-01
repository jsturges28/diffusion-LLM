"""FastAPI server for live LLaDA diffusion visualization.

Loads the model once at startup (in a background thread) and exposes
a WebSocket endpoint that streams diffusion frames to the browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers.models.auto.modeling_auto import (
    AutoModel,
)
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)

from src.inference.render_gif import history_to_gif
from src.inference.streaming_sampler import (
    streaming_generate,
    streaming_resume,
)

logger = logging.getLogger("llada_web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

PARAM_LIMITS_RECOMMENDED: Dict[str, tuple[float, float]] = {
    "steps": (8, 150),
    "gen_length": (16, 160),
    "block_length": (8, 160),
    "temperature": (0.0, 1.0),
    "cfg_scale": (0.0, 2.0),
}

PARAM_LIMITS_EXPERIMENTAL: Dict[str, tuple[float, float]] = {
    "steps": (1, 1024),
    "gen_length": (1, 1024),
    "block_length": (1, 1024),
    "temperature": (0.0, 10.0),
    "cfg_scale": (0.0, 20.0),
}

VALID_REMASKING = {"low_confidence", "random"}

RESULTS_DIR = Path("Results")


# -- Pydantic models for save endpoint --

class RemaskEdit(BaseModel):
    frame_index: int
    token_positions: List[int]


class SaveRunParams(BaseModel):
    steps: int
    gen_length: int
    block_length: int
    temperature: float
    cfg_scale: float
    remasking: str


class SaveRunRequest(BaseModel):
    prompt: str
    params: SaveRunParams
    frames: List[str] = Field(min_length=1)
    final_text: str
    remask_edits: Optional[List[RemaskEdit]] = None


# -- App and shared state --

app = FastAPI(title="LLaDA Diffusion Visualizer")

model: Any = None
tokenizer: Any = None
model_ready = asyncio.Event()
generation_lock = asyncio.Lock()

# Stores tensor history and metadata from the most recent
# completed generation so the client can resume from any frame.
last_run_state: Dict[str, Any] | None = None


def _load_model() -> tuple[Any, Any]:
    """Blocking model + tokenizer load (runs in a thread)."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(
        "Loading tokenizer from %s ...", MODEL_NAME
    )
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
            torch.bfloat16
            if device.type == "cuda"
            else None
        ),
        device_map=(
            "auto" if device.type == "cuda" else None
        ),
    ).eval()

    logger.info(
        "Model loaded successfully on %s.", device
    )
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
    """Extract and clamp generation parameters."""
    experimental = bool(
        data.get("experimental", False)
    )
    limits = (
        PARAM_LIMITS_EXPERIMENTAL
        if experimental
        else PARAM_LIMITS_RECOMMENDED
    )

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    remasking = str(
        data.get("remasking", "low_confidence")
    )
    if remasking not in VALID_REMASKING:
        raise ValueError(
            f"remasking must be one of"
            f" {VALID_REMASKING},"
            f" got '{remasking}'"
        )

    steps = int(
        _clamp(
            float(data.get("steps", 128)),
            *limits["steps"],
        )
    )
    gen_length = int(
        _clamp(
            float(data.get("gen_length", 128)),
            *limits["gen_length"],
        )
    )
    block_length = int(
        _clamp(
            float(data.get("block_length", 32)),
            *limits["block_length"],
        )
    )
    temperature = float(
        _clamp(
            float(data.get("temperature", 0.0)),
            *limits["temperature"],
        )
    )
    cfg_scale = float(
        _clamp(
            float(data.get("cfg_scale", 0.0)),
            *limits["cfg_scale"],
        )
    )

    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be"
            f" divisible by block_length"
            f" ({block_length})"
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
        "remasking": remasking,
    }


def _validate_resume(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Validate a resume request against last_run_state."""
    if last_run_state is None:
        raise ValueError(
            "No previous generation to resume from."
        )

    frame_index = int(data.get("frame_index", -1))
    tensor_history: List[torch.Tensor] = (
        last_run_state["tensor_history"]
    )
    if frame_index < 0 or frame_index >= len(
        tensor_history
    ):
        raise ValueError(
            f"frame_index {frame_index} is out of"
            f" range [0, {len(tensor_history) - 1}]."
        )

    raw_positions = data.get("remask_positions", [])
    if not isinstance(raw_positions, list):
        raise ValueError(
            "remask_positions must be a list."
        )
    if len(raw_positions) == 0:
        raise ValueError(
            "remask_positions must not be empty."
        )

    gen_length: int = last_run_state["gen_length"]
    remask_positions: List[int] = []
    for pos in raw_positions:
        pos = int(pos)
        if pos < 0 or pos >= gen_length:
            raise ValueError(
                f"remask position {pos} is out of"
                f" range [0, {gen_length})."
            )
        remask_positions.append(pos)

    total_steps: int = last_run_state["total_steps"]
    remaining_steps = total_steps - frame_index
    if remaining_steps <= 0:
        raise ValueError(
            "Cannot resume from the final frame."
        )

    return {
        "frame_index": frame_index,
        "remask_positions": remask_positions,
        "remaining_steps": remaining_steps,
    }


async def _stream_frames(
    generator: Any,
    ws: WebSocket,
    start_time: float,
) -> None:
    """Iterate an async frame generator and send each
    frame over the WebSocket with elapsed time."""
    async for frame in generator:
        elapsed = time.monotonic() - start_time
        frame["elapsed"] = round(elapsed, 2)
        await ws.send_json(frame)


@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
) -> None:
    global last_run_state  # noqa: PLW0603

    await ws.accept()

    if not model_ready.is_set():
        await ws.send_json(
            {
                "type": "model_status",
                "status": "loading",
            }
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

            if msg_type == "generate":
                await _handle_generate(
                    ws, data, cancel_event
                )
                continue

            if msg_type == "resume":
                await _handle_resume(
                    ws, data, cancel_event
                )
                continue

            await ws.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Unknown message type:"
                        f" {msg_type}"
                    ),
                }
            )

    except WebSocketDisconnect:
        cancel_event.set()
        logger.info("Client disconnected.")


async def _handle_generate(
    ws: WebSocket,
    data: dict[str, Any],
    cancel_event: asyncio.Event,
) -> None:
    """Handle a 'generate' WebSocket message."""
    global last_run_state  # noqa: PLW0603

    try:
        params = _validate_params(data)
    except (ValueError, TypeError) as exc:
        await ws.send_json(
            {"type": "error", "message": str(exc)}
        )
        return

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
        return

    cancel_event.clear()
    start_time = time.monotonic()

    async with generation_lock:
        try:
            tensor_history: List[torch.Tensor] = []

            generator = streaming_generate(
                model,
                tokenizer,
                params["prompt"],
                steps=params["steps"],
                gen_length=params["gen_length"],
                block_length=params["block_length"],
                temperature=params["temperature"],
                cfg_scale=params["cfg_scale"],
                remasking=params["remasking"],
                cancel_event=cancel_event,
                tensor_history=tensor_history,
            )
            await _stream_frames(
                generator, ws, start_time
            )

            # Build prompt_ids and attention_mask from
            # the tokenizer so resume can reconstruct x.
            message = {
                "role": "user",
                "content": params["prompt"],
            }
            chat_text = tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = tokenizer(
                [chat_text],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].cpu()
            prompt_len = prompt_ids.shape[1]
            gen_length = params["gen_length"]

            full_attention = torch.cat(
                [
                    encoded["attention_mask"],
                    torch.ones(
                        (1, gen_length),
                        dtype=encoded[
                            "attention_mask"
                        ].dtype,
                    ),
                ],
                dim=-1,
            ).cpu()

            num_blocks = (
                gen_length // params["block_length"]
            )
            total_steps = params["steps"]

            last_run_state = {
                "tensor_history": tensor_history,
                "prompt_ids": prompt_ids,
                "attention_mask": full_attention,
                "gen_length": gen_length,
                "total_steps": total_steps,
                "temperature": params["temperature"],
                "cfg_scale": params["cfg_scale"],
                "remasking": params["remasking"],
            }

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


async def _handle_resume(
    ws: WebSocket,
    data: dict[str, Any],
    cancel_event: asyncio.Event,
) -> None:
    """Handle a 'resume' WebSocket message."""
    global last_run_state  # noqa: PLW0603

    try:
        resume_params = _validate_resume(data)
    except (ValueError, TypeError) as exc:
        await ws.send_json(
            {"type": "error", "message": str(exc)}
        )
        return

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
        return

    assert last_run_state is not None
    cancel_event.clear()
    start_time = time.monotonic()

    frame_index = resume_params["frame_index"]
    base_tensor = last_run_state["tensor_history"][
        frame_index
    ]

    # Truncate tensor_history to just before the
    # branch point. Resume tensors (including the
    # remasked initial state) will be appended so
    # indices stay in sync with the client.
    last_run_state["tensor_history"] = (
        last_run_state["tensor_history"][
            :frame_index
        ]
    )

    resume_tensor_history: List[torch.Tensor] = []

    async with generation_lock:
        try:
            generator = streaming_resume(
                model,
                tokenizer,
                base_tokens=base_tensor,
                prompt_ids=last_run_state[
                    "prompt_ids"
                ],
                attention_mask=last_run_state[
                    "attention_mask"
                ],
                remask_positions=resume_params[
                    "remask_positions"
                ],
                remaining_steps=resume_params[
                    "remaining_steps"
                ],
                gen_length=last_run_state[
                    "gen_length"
                ],
                temperature=last_run_state[
                    "temperature"
                ],
                cfg_scale=last_run_state[
                    "cfg_scale"
                ],
                remasking=last_run_state[
                    "remasking"
                ],
                cancel_event=cancel_event,
                tensor_history=(
                    resume_tensor_history
                ),
            )
            await _stream_frames(
                generator, ws, start_time
            )

            last_run_state[
                "tensor_history"
            ].extend(resume_tensor_history)
            last_run_state["total_steps"] = len(
                last_run_state["tensor_history"]
            ) - 1

        except Exception as exc:
            logger.exception(
                "Resume failed: %s", exc
            )
            await ws.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )


# -- Save endpoint --


def _make_run_dir(base: Path) -> Path:
    """Create a timestamped subdirectory."""
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    run_dir = base / f"{timestamp}_llada"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_blocking(body: SaveRunRequest) -> str:
    """Write all result files to disk (blocking I/O).

    Returns the run directory path as a string.
    """
    run_dir = _make_run_dir(RESULTS_DIR)

    metadata: Dict[str, Any] = {
        "backend": "llada",
        "model": MODEL_NAME,
        "prompt": body.prompt,
        "final_text": body.final_text,
        "params": body.params.model_dump(),
    }
    if body.remask_edits:
        metadata["remask_edits"] = [
            edit.model_dump()
            for edit in body.remask_edits
        ]

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            metadata, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    final_path = run_dir / "final.txt"
    final_path.write_text(
        body.final_text, encoding="utf-8"
    )

    hist_path = run_dir / "history.txt"
    with hist_path.open("w", encoding="utf-8") as fh:
        for i, frame_text in enumerate(body.frames):
            fh.write(f"\n===== FRAME {i} =====\n")
            fh.write(frame_text)
            fh.write("\n")

    gif_path = run_dir / "diffusion.gif"
    history_to_gif(
        body.frames,
        gif_path,
        header_text=body.prompt,
    )

    return str(run_dir)


@app.post("/api/save")
async def save_run(
    body: SaveRunRequest,
) -> JSONResponse:
    """Persist a completed generation run to Results/."""
    try:
        run_path = await asyncio.to_thread(
            _save_run_blocking, body
        )
    except Exception as exc:
        logger.exception(
            "Failed to save run: %s", exc
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    logger.info("Saved run to %s", run_path)
    return JSONResponse(
        content={"success": True, "path": run_path}
    )


app.mount(
    "/",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="static",
)
