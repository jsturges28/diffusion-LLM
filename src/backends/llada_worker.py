"""LLaDA worker: masked discrete diffusion backend.

Wraps the existing streaming sampler and preserves the original
generate / resume / remask semantics, now behind the shared
``Backend`` contract. Runs in ``.venv`` (Transformers 4.38.2).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import WebSocket
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)

from src.backends.registry import LLADA
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
)
from src.inference.hf_download import download_with_progress
from src.inference.streaming_sampler import (
    streaming_generate,
    streaming_resume,
)

logger = logging.getLogger("llada_worker")

VALID_REMASKING = {"low_confidence", "random"}


def _clamp_int(value: Any, bounds: Tuple[float, float]) -> int:
    low, high = bounds
    return int(max(low, min(high, float(value))))


def _clamp_float(
    value: Any, bounds: Tuple[float, float]
) -> float:
    low, high = bounds
    return float(max(low, min(high, float(value))))


def _apply_seed(seed: int) -> None:
    """Seed torch/numpy for reproducible sampling (seed >= 0)."""
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LladaBackend(Backend):
    def __init__(self) -> None:
        self.model_info = LLADA
        self.model: Any = None
        self.tokenizer: Any = None
        self.last_run_state: Optional[Dict[str, Any]] = None
        self.load_progress: Optional[Dict[str, Any]] = None

    # -- loading --

    def load(self, *, device: str = "cuda") -> None:
        # Fall back to CPU when CUDA was requested but is unavailable,
        # so a GPU-less host degrades instead of erroring on load.
        resolved = (
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        device = torch.device(resolved)
        name = self.model_info.checkpoint
        # Fetch weights first (progress via /health) so the first
        # activation shows a download bar; a cache hit is a no-op.
        logger.info("ensuring weights for %s", name)
        download_with_progress(
            name, sink=lambda p: setattr(self, "load_progress", p)
        )
        self.load_progress = None
        logger.info("loading tokenizer %s", name)
        tok = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True
        )
        if tok.padding_side != "left":
            tok.padding_side = "left"
        logger.info("loading model %s", name)
        mdl = AutoModel.from_pretrained(
            name,
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
        self.model = mdl
        self.tokenizer = tok
        logger.info("LLaDA loaded on %s", device)

    # -- validation --

    def _limits(
        self, experimental: bool
    ) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for spec in self.model_info.param_specs:
            bounds = (
                spec.experimental
                if experimental
                else spec.recommended
            )
            if bounds is not None:
                out[spec.name] = bounds
        return out

    def _validate_generate(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        experimental = bool(data.get("experimental", False))
        limits = self._limits(experimental)

        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        remasking = str(
            data.get("remasking", "low_confidence")
        )
        if remasking not in VALID_REMASKING:
            raise ValueError(
                f"remasking must be one of {VALID_REMASKING}"
            )

        steps = _clamp_int(
            data.get("steps", 128), limits["steps"]
        )
        gen_length = _clamp_int(
            data.get("gen_length", 128), limits["gen_length"]
        )
        block_length = _clamp_int(
            data.get("block_length", 32),
            limits["block_length"],
        )
        temperature = _clamp_float(
            data.get("temperature", 0.0),
            limits["temperature"],
        )
        cfg_scale = _clamp_float(
            data.get("cfg_scale", 0.0), limits["cfg_scale"]
        )
        seed = int(data.get("seed", -1))

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
            "seed": seed,
        }

    # -- generation --

    async def handle_generate(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            params = self._validate_generate(data)
        except (ValueError, TypeError) as exc:
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )
            return

        _apply_seed(params["seed"])
        start = time.monotonic()
        tensor_history: List[torch.Tensor] = []
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
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
            await stream.run(generator, start)
            self._store_state(params, tensor_history)
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )

    def _store_state(
        self,
        params: Dict[str, Any],
        tensor_history: List[torch.Tensor],
    ) -> None:
        message = {
            "role": "user",
            "content": params["prompt"],
        }
        chat_text = self.tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(
            [chat_text],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = encoded["input_ids"].cpu()
        gen_length = params["gen_length"]
        full_attention = torch.cat(
            [
                encoded["attention_mask"],
                torch.ones(
                    (1, gen_length),
                    dtype=encoded["attention_mask"].dtype,
                ),
            ],
            dim=-1,
        ).cpu()
        self.last_run_state = {
            "tensor_history": tensor_history,
            "prompt_ids": prompt_ids,
            "attention_mask": full_attention,
            "gen_length": gen_length,
            "total_steps": params["steps"],
            "temperature": params["temperature"],
            "cfg_scale": params["cfg_scale"],
            "remasking": params["remasking"],
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
        frame_index = int(data.get("frame_index", -1))
        history: List[torch.Tensor] = state["tensor_history"]
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
        gen_length: int = state["gen_length"]
        positions: List[int] = []
        for pos in raw:
            pos = int(pos)
            if pos < 0 or pos >= gen_length:
                raise ValueError(
                    f"remask position {pos} out of range"
                    f" [0, {gen_length})."
                )
            positions.append(pos)
        remaining = state["total_steps"] - frame_index
        if remaining <= 0:
            raise ValueError(
                "Cannot resume from the final frame."
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
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            resume_params = self._validate_resume(data)
        except (ValueError, TypeError) as exc:
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )
            return

        state = self.last_run_state
        assert state is not None
        start = time.monotonic()
        frame_index = resume_params["frame_index"]
        max_frames: Optional[int] = data.get("max_frames")
        base_tensor = state["tensor_history"][frame_index]
        state["tensor_history"] = state["tensor_history"][
            :frame_index
        ]
        resume_history: List[torch.Tensor] = []
        try:
            generator = streaming_resume(
                self.model,
                self.tokenizer,
                base_tokens=base_tensor,
                prompt_ids=state["prompt_ids"],
                attention_mask=state["attention_mask"],
                remask_positions=resume_params[
                    "remask_positions"
                ],
                remaining_steps=resume_params[
                    "remaining_steps"
                ],
                gen_length=state["gen_length"],
                temperature=state["temperature"],
                cfg_scale=state["cfg_scale"],
                remasking=state["remasking"],
                cancel_event=cancel_event,
                tensor_history=resume_history,
            )
            done = await stream.run(
                generator, start, max_frames=max_frames
            )
            state["tensor_history"].extend(resume_history)
            if done:
                state["total_steps"] = (
                    len(state["tensor_history"]) - 1
                )
            else:
                final_text = ""
                if resume_history:
                    final_text = self.tokenizer.batch_decode(
                        resume_history[-1],
                        skip_special_tokens=True,
                    )[0]
                await ws.send_json(
                    {
                        "type": "done",
                        "final_text": final_text,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("resume failed")
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )


def build_backend() -> Backend:
    return LladaBackend()
