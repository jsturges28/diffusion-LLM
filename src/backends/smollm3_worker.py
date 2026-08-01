"""SmolLM3 worker: autoregressive (left-to-right) backend.

Runs in ``.venv-ar`` (Transformers >= 4.53). Loads SmolLM3-3B via
``AutoModelForCausalLM`` and streams a growing token sequence
through the shared worker contract, one frame per new token, so the
existing scrubber/save/overlay tooling works unchanged (as a
left-to-right replay).

Diffusion-style remask/resume does not apply to a left-to-right
model, so ``handle_resume`` stays the base ``NotImplementedError``.
The autoregressive counterfactual is ``handle_substitute`` instead:
force one position to a captured alternative and regenerate forward.

Runs on GPU when available and on CPU otherwise, chosen per
activation by the supervisor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import WebSocket
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.backends.registry import SMOLLM3
from src.backends.worker_base import Backend, FrameStreamer
from src.inference.ar_sampler import (
    streaming_generate,
    streaming_substitute,
)
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
        # Prompt, params, and per-position trace of the most recent
        # run, kept so a substitution can re-enter at any position
        # without replaying the whole generation. None until a run
        # completes.
        self.last_run_state: Optional[Dict[str, Any]] = None

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
            "alternatives": bool(
                data.get("alternatives", False)
            ),
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
        # Discard any prior run's trace up front, so a failure here
        # cannot leave a stale state that a substitution would then
        # re-enter against the wrong prompt.
        self.last_run_state = None
        state: Dict[str, Any] = {}
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
                params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                thinking=params["thinking"],
                alternatives=params["alternatives"],
                seed=params["seed"],
                cancel_event=cancel_event,
                state_sink=state,
            )
            await stream.run(generator, start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )
            return
        if state.get("ids"):
            # Copied key by key, not via update(params): the trace's
            # "alternatives" is the per-position candidate list, while
            # the param of that name is the capture flag.
            state["prompt"] = params["prompt"]
            state["max_new_tokens"] = params["max_new_tokens"]
            state["thinking"] = params["thinking"]
            state["seed"] = params["seed"]
            state["alternatives_enabled"] = params[
                "alternatives"
            ]
            self.last_run_state = state

    # -- substitution (the autoregressive counterfactual) --

    def _validate_substitute(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check a substitution request against the last run.

        Only a candidate the model actually considered at that
        position may be forced, so the counterfactual stays a real
        branch of the recorded decision rather than an arbitrary
        edit.
        """
        state = self.last_run_state
        if state is None:
            raise ValueError(
                "No previous generation to substitute into."
            )
        ids: List[int] = state["ids"]
        position = int(data.get("position", -1))
        if position < 0 or position >= len(ids):
            raise ValueError(
                f"position {position} is out of range"
                f" [0, {len(ids) - 1}]."
            )
        if position == 0 and len(ids) == 1:
            raise ValueError(
                "Nothing follows the only token; substituting"
                " it would change nothing."
            )
        captured = state["alternatives"][position]
        if not captured:
            raise ValueError(
                "No alternatives were captured at position"
                f" {position}. Re-run with Alternatives on."
            )
        token_id = int(data.get("token_id", -1))
        chosen = None
        for candidate in captured:
            if int(candidate["id"]) == token_id:
                chosen = candidate
                break
        if chosen is None:
            raise ValueError(
                f"token {token_id} was not among the captured"
                f" candidates at position {position}."
            )
        return {
            "position": position,
            "forced_id": token_id,
            "forced_conf": float(chosen["p"]),
            "forced_alts": captured,
        }

    async def handle_substitute(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            request = self._validate_substitute(data)
        except (ValueError, TypeError, KeyError) as exc:
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )
            return

        state = self.last_run_state
        assert state is not None
        position = request["position"]
        start = time.monotonic()
        branch: Dict[str, Any] = {}
        try:
            generator = streaming_substitute(
                self.model,
                self.tokenizer,
                state["prompt"],
                position=position,
                forced_id=request["forced_id"],
                forced_conf=request["forced_conf"],
                forced_entropy=state["entropies"][position],
                forced_alts=request["forced_alts"],
                prefix_ids=state["ids"][:position],
                prefix_confs=state["confidences"][:position],
                prefix_entropies=state["entropies"][
                    :position
                ],
                prefix_alts=state["alternatives"][:position],
                max_new_tokens=state["max_new_tokens"],
                # Greedy: the divergence after the forced token
                # should be the intervention's effect, not fresh
                # sampling noise in a shifted context.
                temperature=0.0,
                top_p=1.0,
                thinking=state["thinking"],
                alternatives=state["alternatives_enabled"],
                seed=state["seed"],
                cancel_event=cancel_event,
                state_sink=branch,
            )
            await stream.run(generator, start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("substitution failed")
            await ws.send_json(
                {"type": "error", "message": str(exc)}
            )
            return
        # Chain substitutions: the branch becomes the run a further
        # substitution re-enters.
        if branch.get("ids"):
            state.update(branch)


def build_backend() -> Backend:
    return Smollm3Backend()
