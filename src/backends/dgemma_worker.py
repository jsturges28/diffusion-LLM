"""DiffusionGemma worker: NF4 block-diffusion backend.

Runs in ``.venv-dgemma`` (Transformers v5). Loads the NF4
checkpoint via ``dgemma_nf4.load_quantized`` and streams denoising
frames through the shared worker contract. Text-only (uses the
tokenizer, not the multimodal processor). Resume is not supported
in phase 1.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import WebSocket
from transformers import AutoTokenizer  # type: ignore[attr-defined]

from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_STALE_RUN,
    MSG_GENERATE,
    MSG_RESUME,
    request_error,
    request_id_of,
)
from src.backends.registry import DGEMMA
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
)
from src.inference.dgemma_nf4 import load_quantized
from src.inference.load_progress import (
    HOST_STAGE_CEILING_PICKLED,
    load_target_bytes,
    sample_load_progress,
)
from src.inference.dgemma_sampler import (
    streaming_generate,
    streaming_resume,
)

logger = logging.getLogger("dgemma_worker")


def _clamp_int(value: Any, bounds: Tuple[float, float]) -> int:
    low, high = bounds
    return int(max(low, min(high, float(value))))


def _clamp_float(
    value: Any, bounds: Tuple[float, float]
) -> float:
    low, high = bounds
    return float(max(low, min(high, float(value))))


class DgemmaBackend(Backend):
    def __init__(self) -> None:
        self.model_info = DGEMMA
        self.model: Any = None
        self.tokenizer: Any = None
        self.last_run_state: Optional[Dict[str, Any]] = None
        # No download phase here (the NF4 checkpoint is produced
        # locally by scripts/quantize_diffusiongemma_nf4.py), but the
        # read into memory is the longest of the three models, so it
        # reports through the same attribute the others use.
        self.load_progress: Optional[Dict[str, Any]] = None

    def load(self, *, device: str = "cuda") -> None:
        # The NF4 experts assume a CUDA compute path (bitsandbytes),
        # so DiffusionGemma is GPU-only; a CPU request is refused
        # rather than silently attempting an unsupported placement.
        if device != "cuda":
            raise RuntimeError(
                "DiffusionGemma (NF4) requires a CUDA GPU;"
                f" device={device!r} is not supported."
            )
        path = Path(self.model_info.checkpoint).expanduser()
        if not path.is_dir():
            raise RuntimeError(
                f"NF4 checkpoint not found: {path}."
                " Run scripts/quantize_diffusiongemma_nf4.py"
                " first."
            )
        logger.info("loading tokenizer from %s", path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        logger.info("loading NF4 model from %s", path)
        # The checkpoint is a single pickled state dict already in its
        # packed NF4 form, so its size on disk is the target and there
        # is no dtype conversion to scale by.
        target = load_target_bytes(path)
        # Alone among the three, this one unpickles the entire state
        # dict into RAM before copying any of it across, so the read
        # would otherwise fill the bar and leave the copy nowhere to
        # go. The ceiling reserves it a tail.
        with sample_load_progress(
            target_bytes=target,
            sink=lambda p: setattr(self, "load_progress", p),
            host_stage_ceiling=HOST_STAGE_CEILING_PICKLED,
        ):
            self.model = load_quantized(str(path), device=device)
        # Always "cuda": the guard above refuses anything else, so
        # unlike the other two backends there is no fallback for this
        # to disagree with. Set anyway so every backend attests.
        self.effective_device = device
        logger.info("DiffusionGemma NF4 loaded")

    def _bounds(
        self, name: str, experimental: bool
    ) -> Tuple[float, float]:
        for spec in self.model_info.param_specs:
            if spec.name != name:
                continue
            bounds = (
                spec.experimental
                if experimental
                else spec.recommended
            )
            if bounds is not None:
                return bounds
        return (float("-inf"), float("inf"))

    def _validate_generate(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        experimental = bool(data.get("experimental", False))
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        return {
            "prompt": prompt,
            "max_new_tokens": _clamp_int(
                data.get("max_new_tokens", 256),
                self._bounds("max_new_tokens", experimental),
            ),
            "max_denoising_steps": _clamp_int(
                data.get("max_denoising_steps", 48),
                self._bounds(
                    "max_denoising_steps", experimental
                ),
            ),
            "t_max": _clamp_float(
                data.get("t_max", 0.8),
                self._bounds("t_max", experimental),
            ),
            "t_min": _clamp_float(
                data.get("t_min", 0.4),
                self._bounds("t_min", experimental),
            ),
            "thinking": bool(data.get("thinking", False)),
            "entropy_signal": bool(
                data.get("entropy_signal", False)
            ),
            "seed": int(data.get("seed", -1)),
        }

    async def handle_generate(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: threading.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            params = self._validate_generate(data)
        except (ValueError, TypeError) as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
            )
            return

        self.begin_run()
        start = time.monotonic()
        frame_history: List[Dict[str, Any]] = []
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
                params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                max_denoising_steps=params[
                    "max_denoising_steps"
                ],
                t_max=params["t_max"],
                t_min=params["t_min"],
                thinking=params["thinking"],
                entropy_signal=params["entropy_signal"],
                seed=params["seed"],
                cancel_event=cancel_event,
                frame_history=frame_history,
            )
            await stream.run(generator, start)
            self._store_state(params, frame_history)
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
            )

    def _store_state(
        self,
        params: Dict[str, Any],
        frame_history: List[Dict[str, Any]],
    ) -> None:
        """Retain the just-finished run so it can be resumed.

        ``frame_history`` holds one entry per streamed frame with its
        canvas token ids and canvas index, mirroring LLaDA's tensor
        history but at the token-id level.
        """
        self.last_run_state = {
            "prompt": params["prompt"],
            "t_max": params["t_max"],
            "t_min": params["t_min"],
            "thinking": params["thinking"],
            "entropy_signal": params["entropy_signal"],
            "seed": params["seed"],
            "max_denoising_steps": params[
                "max_denoising_steps"
            ],
            "frame_history": frame_history,
        }

    # -- resume --

    def _validate_resume(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        state = self.last_run_state
        if state is None:
            raise ValueError(
                "No previous generation to resume from."
            )
        history: List[Dict[str, Any]] = state["frame_history"]
        if len(history) == 0:
            raise ValueError(
                "No frames available to resume from."
            )
        # Single-canvas scope: a multi-canvas run cannot be
        # re-entered with this seed-canvas strategy.
        max_canvas = max(
            int(f.get("canvas_index", 0)) for f in history
        )
        if max_canvas > 0:
            raise ValueError(
                "Resume is only supported for single-canvas"
                " runs (max 256 tokens)."
            )
        frame_index = int(data.get("frame_index", -1))
        if frame_index < 0 or frame_index >= len(history):
            raise ValueError(
                f"frame_index {frame_index} is out of range"
                f" [0, {len(history) - 1}]."
            )
        raw = data.get("remask_positions", [])
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(
                "remask_positions must be a non-empty list."
            )
        canvas_length = len(history[frame_index]["ids"])
        positions: List[int] = []
        for pos in raw:
            pos = int(pos)
            if pos < 0 or pos >= canvas_length:
                raise ValueError(
                    f"remask position {pos} out of range"
                    f" [0, {canvas_length})."
                )
            positions.append(pos)
        remaining = max(
            1, state["max_denoising_steps"] - frame_index
        )
        return {
            "frame_index": frame_index,
            "remask_positions": positions,
            "remaining_steps": remaining,
        }

    async def handle_resume(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: threading.Event,
        stream: FrameStreamer,
    ) -> None:
        # Ordinary frames are forwarded directly (see
        # _forward_resume, which has to drain the generator), but the
        # terminal frame still goes through the streamer, because
        # that is what stamps the run's provenance.
        try:
            self.check_run_token(data)
            resume_params = self._validate_resume(data)
        except StaleRunError as exc:
            # Before StaleRunError's base, ValueError, or the stale
            # case would be reported as a malformed request.
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_STALE_RUN,
                    request_type=MSG_RESUME,
                    request_id=request_id_of(data),
                )
            )
            return
        except (ValueError, TypeError) as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_RESUME,
                    request_id=request_id_of(data),
                )
            )
            return

        state = self.last_run_state
        assert state is not None
        start = time.monotonic()
        frame_index = resume_params["frame_index"]
        max_frames: Optional[int] = data.get("max_frames")
        seed_ids = list(
            state["frame_history"][frame_index]["ids"]
        )
        base_history = state["frame_history"][:frame_index]
        resume_frames: List[Dict[str, Any]] = []
        try:
            generator = streaming_resume(
                self.model,
                self.tokenizer,
                prompt=state["prompt"],
                seed_canvas_ids=seed_ids,
                remask_positions=resume_params[
                    "remask_positions"
                ],
                remaining_steps=resume_params[
                    "remaining_steps"
                ],
                t_max=state["t_max"],
                t_min=state["t_min"],
                thinking=state["thinking"],
                entropy_signal=state["entropy_signal"],
                seed=state["seed"],
                cancel_event=cancel_event,
                frame_history=resume_frames,
            )
            await self._forward_resume(
                ws, stream, generator, start, max_frames
            )
            # Splice: keep only the frames the client received so
            # the worker history stays aligned with the browser's
            # for any subsequent resume.
            kept = (
                resume_frames
                if max_frames is None
                else resume_frames[:max_frames]
            )
            state["frame_history"] = base_history + kept
        except Exception as exc:  # noqa: BLE001
            logger.exception("resume failed")
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_RESUME,
                    request_id=request_id_of(data),
                )
            )

    async def _forward_resume(
        self,
        ws: WebSocket,
        stream: FrameStreamer,
        generator: Any,
        start: float,
        max_frames: Optional[int],
    ) -> None:
        """Forward resume frames, always draining the generator.

        DiffusionGemma runs ``generate`` in a background thread, so
        the generator must be consumed to completion before returning
        (otherwise the thread would keep using the GPU after the
        request finishes). When ``max_frames`` is set (guided "run to
        here"), frames past the budget are drained silently and an
        explicit ``done`` is sent so the client stops at the target.

        Both terminal frames go out through the streamer even though
        the ordinary ones do not, because that is what stamps the
        run's provenance. Draining is this method's reason to exist,
        so it cannot hand the generator to ``stream.run``.
        """
        sent = 0
        async for frame in generator:
            ftype = frame.get("type")
            if ftype == "frame":
                if max_frames is None or sent < max_frames:
                    frame["elapsed"] = round(
                        time.monotonic() - start, 2
                    )
                    await ws.send_json(frame)
                    sent += 1
                continue
            if ftype == "done" and max_frames is None:
                await stream.send_done(frame, start)
        if max_frames is not None:
            await stream.send_done(
                {"type": "done", "final_text": ""}, start
            )


def build_backend() -> Backend:
    return DgemmaBackend()
