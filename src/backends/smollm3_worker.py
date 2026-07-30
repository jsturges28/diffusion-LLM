"""SmolLM3 worker: autoregressive (left-to-right) backend.

Runs in ``.venv-ar`` (Transformers >= 4.53). Loads SmolLM3-3B via
``AutoModelForCausalLM`` and streams a growing token sequence
through the shared worker contract, one frame per new token, so the
existing scrubber/save/overlay tooling works unchanged (as a
left-to-right replay). Resume is not supported (autoregressive
resume is a later, separate feature), so ``handle_resume`` is left
as the base ``NotImplementedError``.

Runs on GPU when available and on CPU otherwise, chosen per
activation by the supervisor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
from fastapi import WebSocket
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.backends.registry import SMOLLM3
from src.backends.worker_base import Backend, FrameStreamer
from src.inference.ar_sampler import streaming_generate
from src.inference.hf_download import download_with_progress

logger = logging.getLogger("smollm3_worker")


def _clamp_int(value: Any, bounds: Tuple[float, float]) -> int:
    low, high = bounds
    return int(max(low, min(high, float(value))))


def _clamp_float(
    value: Any, bounds: Tuple[float, float]
) -> float:
    low, high = bounds
    return float(max(low, min(high, float(value))))


class Smollm3Backend(Backend):
    def __init__(self) -> None:
        self.model_info = SMOLLM3
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cuda"
        self.load_progress: Optional[Dict[str, Any]] = None

    def load(self, *, device: str = "cuda") -> None:
        # Fall back to CPU when CUDA was requested but is unavailable,
        # so a GPU-less host degrades instead of erroring on load.
        resolved = (
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.device = resolved
        name = self.model_info.checkpoint
        # Fetch weights first (reporting progress via /health) so the
        # first activation shows a download bar; a cache hit is a no-op.
        logger.info("ensuring weights for %s", name)
        download_with_progress(
            name, sink=lambda p: setattr(self, "load_progress", p)
        )
        self.load_progress = None
        logger.info("loading tokenizer %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        logger.info(
            "loading model %s on %s (bfloat16)", name, resolved
        )
        # bfloat16 on both devices halves the ~12 GiB fp32 footprint
        # (to ~6 GiB), which matters most for CPU/RAM-constrained hosts.
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16
        )
        self.model = model.to(resolved).eval()
        logger.info("SmolLM3 loaded on %s", resolved)

    def _bounds(
        self, name: str, experimental: bool
    ) -> Tuple[float, float]:
        """Device-aware (low, high) bounds for a numeric parameter.

        Applies the spec's per-device override for ``self.device``
        when present, so the CPU token cap is enforced identically to
        what the frontend shows (no hidden clamp).
        """
        for spec in self.model_info.param_specs:
            if spec.name != name:
                continue
            override = (
                spec.overrides.get(self.device)
                if spec.overrides
                else None
            )
            if override is not None:
                bounds = (
                    override.experimental
                    if experimental
                    else override.recommended
                )
                if bounds is not None:
                    return bounds
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
        # Device-aware bounds (see _bounds) enforce the CPU token cap
        # transparently, matching what the frontend shows.
        max_new_tokens = _clamp_int(
            data.get("max_new_tokens", 256),
            self._bounds("max_new_tokens", experimental),
        )
        return {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": _clamp_float(
                data.get("temperature", 0.6),
                self._bounds("temperature", experimental),
            ),
            "top_p": _clamp_float(
                data.get("top_p", 0.95),
                self._bounds("top_p", experimental),
            ),
            "thinking": bool(data.get("thinking", False)),
            "seed": int(data.get("seed", -1)),
        }

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

        start = time.monotonic()
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
                params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                thinking=params["thinking"],
                seed=params["seed"],
                cancel_event=cancel_event,
            )
            await stream.run(generator, start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )


def build_backend() -> Backend:
    return Smollm3Backend()
