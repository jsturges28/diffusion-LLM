"""LLaDA worker: masked discrete diffusion backend.

Wraps the existing streaming sampler and preserves the original
generate / resume / remask semantics, now behind the shared
``Backend`` contract. Runs in ``.venv`` (Transformers 4.38.2).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import WebSocket
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
)

from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    MSG_GENERATE,
    MSG_RESUME,
    request_error,
    request_id_of,
)
from src.backends.registry import LLADA
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
)
from src.inference.hf_download import download_with_progress
from src.inference.load_progress import (
    load_target_bytes,
    sample_load_progress,
)
from src.inference.streaming_sampler import (
    build_llada_inputs,
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


def _commit_resume(
    state: Dict[str, Any],
    base_history: List[torch.Tensor],
    resume_history: List[torch.Tensor],
    *,
    done: bool,
) -> None:
    """Swap the staged candidate in as the retained history.

    Called only once a terminal outcome has reached the client, so
    that every failure before it leaves the original history and
    its step count exactly as they were. That is what keeps the
    worker agreeing with the browser, which rolls back to its own
    pre-edit snapshot when a resume fails.

    An empty ``resume_history`` cannot come from a terminal
    outcome: ``streaming_resume`` appends the remasked canvas
    before yielding its first frame. Committing one anyway would
    cut the run back to the surviving prefix, which is the exact
    loss this staging exists to prevent, so it is refused rather
    than trusted.

    ``total_steps`` moves only on a completed resume. A guided
    "run to here" stops short on purpose and leaves the figure
    describing the run it branched from.
    """
    if len(resume_history) == 0:
        return
    candidate = base_history + resume_history
    assert len(candidate) > len(base_history), (
        "a committed candidate extends the surviving prefix"
    )
    state["tensor_history"] = candidate
    assert len(state["tensor_history"]) == len(candidate), (
        "the retained history is the candidate"
    )
    if done:
        state["total_steps"] = len(candidate) - 1


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
        self.effective_device = resolved
        device = torch.device(resolved)
        name = self.model_info.checkpoint
        # Fetch weights first (progress via /health) so the first
        # activation shows a download bar; a cache hit is a no-op.
        logger.info("ensuring weights for %s", name)
        snapshot = download_with_progress(
            name, sink=lambda p: setattr(self, "load_progress", p)
        )
        self.load_progress = None
        # local_files_only from here down. Returning from
        # download_with_progress means every file is on disk, so any
        # request past this point is transformers revalidating a
        # checkpoint we already have. Offline that is not a slower
        # load, it is a hang: this model carries remote code, and
        # resolving it against an unreachable Hub retries for minutes
        # before giving up.
        logger.info("loading tokenizer %s", name)
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tok.padding_side != "left":
            tok.padding_side = "left"
        logger.info("loading model %s", name)
        load_dtype = (
            torch.bfloat16 if device.type == "cuda" else None
        )
        # torch_dtype=None does not mean "leave it alone", it means
        # torch's default, which is fp32. This checkpoint is BF16 on
        # disk, so the CPU path takes twice its size in RAM and the
        # target has to be told that or the bar would stall at half.
        target = load_target_bytes(
            Path(snapshot),
            target_dtype=(
                load_dtype
                if load_dtype is not None
                else torch.get_default_dtype()
            ),
        )
        with sample_load_progress(
            target_bytes=target,
            sink=lambda p: setattr(self, "load_progress", p),
        ):
            # device_map="auto" streams shards straight to the GPU, so
            # there is no separate .to(device) to account for here.
            mdl = AutoModel.from_pretrained(
                name,
                trust_remote_code=True,
                torch_dtype=load_dtype,
                local_files_only=True,
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
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
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
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
            )

    def prompt_token_count(
        self, prompt: str, *, thinking: bool = False
    ) -> int:
        """Tokens LLaDA's own encode produces for this prompt.

        Overridden because the base class templates and tokenizes in
        one call with ``enable_thinking``, and LLaDA does neither: it
        templates to a string and encodes separately, and it has no
        reasoning channel for the flag to select. Sharing
        ``build_llada_inputs`` with the generator is what makes this
        the run's real count rather than an approximation of it.

        ``thinking`` is accepted and ignored to keep one signature
        across backends; the client sends whatever the active model
        declares and this model declares no such parameter.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        if prompt == "":
            return 0
        encoded = build_llada_inputs(self.tokenizer, prompt)
        count = int(encoded["input_ids"].shape[-1])
        assert count > 0, "a templated prompt has tokens"
        return count

    def _store_state(
        self,
        params: Dict[str, Any],
        tensor_history: List[torch.Tensor],
    ) -> None:
        # The generator's own encode, so a resumed run re-enters the
        # exact prefix it produced rather than a re-derivation of it.
        encoded = build_llada_inputs(
            self.tokenizer, params["prompt"]
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
        base_tensor = state["tensor_history"][frame_index]
        # Staged, never truncated in place. The prefix is a new
        # list of the same tensor references, and streaming_resume
        # builds its own sequence with torch.cat rather than
        # writing through base_tensor, so rolling back costs
        # nothing and duplicates no tensor storage.
        base_history: List[torch.Tensor] = state[
            "tensor_history"
        ][:frame_index]
        assert len(base_history) == frame_index, (
            "the staged prefix stops at the resume frame"
        )
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
            if not done:
                # The worker's own terminal message, for a guided
                # "run to here" that stopped at its budget or a
                # cancelled resume. Sent before the commit so a
                # failed send rolls back too, matching the done
                # path where the sampler's terminal frame has
                # already gone out through the streamer.
                await stream.send_done(
                    {
                        "type": "done",
                        "final_text": self._resume_final_text(
                            resume_history
                        ),
                    },
                    start,
                )
            _commit_resume(
                state,
                base_history,
                resume_history,
                done=done,
            )
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

    def _resume_final_text(
        self, resume_history: List[torch.Tensor]
    ) -> str:
        """Decode the last staged resume frame, or "" if none.

        Only the guided and cancelled paths need this. A resume
        that ran to completion gets its text from the sampler's
        own terminal frame instead.
        """
        if len(resume_history) == 0:
            return ""
        decoded = self.tokenizer.batch_decode(
            resume_history[-1],
            skip_special_tokens=True,
        )
        assert len(decoded) > 0, "batch_decode returned nothing"
        return str(decoded[0])


def build_backend() -> Backend:
    return LladaBackend()
