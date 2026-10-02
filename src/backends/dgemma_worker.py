"""DiffusionGemma worker: NF4 block-diffusion backend.

Runs in ``.venv-dgemma`` (Transformers v5). Loads the NF4
checkpoint via ``dgemma_nf4.load_quantized`` and streams denoising
frames through the shared worker contract. Text-only (uses the
tokenizer, not the multimodal processor). A resume re-enters a
single canvas, so a run that spans more than one is refused.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from transformers import AutoTokenizer  # type: ignore[attr-defined]

from src.backends.params import resolve_params
from src.backends.text_adapter import DGEMMA_TEXT
from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_STALE_RUN,
    MSG_GENERATE,
    MSG_RESUME,
    TERMINAL_CANCELLED,
    request_error,
    request_id_of,
)
from src.backends.registry import DGEMMA
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
)
from src.inference.checkpoint import FrameCheckpoint
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


def _commit_resume(
    state: Dict[str, Any],
    base_history: List[FrameCheckpoint],
    forwarded: List[FrameCheckpoint],
) -> None:
    """Swap the frames the client received in as the retained run.

    Called once the terminal frame has reached the client, with the
    checkpoints of exactly the frames that did. Nothing forwarded is
    an outcome here rather than a broken caller, unlike LLaDA's
    commit: this resume's first frame needs a denoising step, so a
    stop can end it having sent none. No branch reached the client
    then, and the run stays as it was.
    """
    if len(forwarded) == 0:
        return
    candidate = base_history + forwarded
    assert len(candidate) > len(base_history), (
        "a committed branch extends the surviving prefix"
    )
    state["frame_history"] = candidate


def _validate_resume_budget(
    data: Dict[str, Any],
) -> Optional[int]:
    """A guided edit's frame budget, or None to resume to the end.

    The commit keeps as many frames as were sent, so the budget has
    to be a count: a whole number, at least one. A bool is refused
    although Python counts it as an int, because ``true`` off the
    wire is a malformed request rather than a budget of one.
    """
    raw = data.get("max_frames")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            "max_frames must be a whole number of frames."
        )
    if raw < 1:
        raise ValueError(
            f"max_frames must be at least 1, not {raw}."
        )
    return raw


async def _send_guided_terminal(
    stream: FrameStreamer,
    start: float,
    *,
    final_text: str,
    cut_short: bool,
) -> None:
    """End a guided edit with the text of the last frame it sent.

    The sampler's own terminal frame describes drafts past the
    budget, which the client never receives, so the worker builds
    this one from what did arrive. A stop that came before the budget
    was met cut the request short and says so, as every stopped run
    does. A stop during the drain after it, or none at all, leaves a
    completed request: the client has every frame it asked for.
    """
    if cut_short:
        await stream.send_cancelled(final_text, start)
    else:
        await stream.send_done(
            {"type": "done", "final_text": final_text}, start
        )


class DgemmaBackend(Backend):
    # A resume splices the retained history, so an abandoned edit
    # session has to be able to put it back. No step count beside
    # it, unlike LLaDA: the remaining budget is derived from
    # ``max_denoising_steps``, which a resume never writes to.
    REWIND_KEYS = (
        ("frame_history", "generated_frame_history"),
    )

    def __init__(self) -> None:
        self.model_info = DGEMMA
        self.text_adapter = DGEMMA_TEXT
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

    def _validate_generate(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One request as this sampler's arguments.

        Types, defaults and bounds come from the registry through
        ``resolve_params``; only the prompt is handled here, since it
        is not a declared parameter. This model has no relational
        rules between its parameters.
        """
        params = resolve_params(
            self.model_info.param_specs,
            data,
            device=self.effective_device,
            experimental=bool(
                data.get("experimental", False)
            ),
        )
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        params["prompt"] = prompt
        self.check_prompt_fits(
            prompt, thinking=bool(params["thinking"])
        )
        return params

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
        frame_history: List[FrameCheckpoint] = []
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
                self.text_adapter,
                params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                max_denoising_steps=params[
                    "max_denoising_steps"
                ],
                t_max=params["t_max"],
                t_min=params["t_min"],
                confidence_threshold=params[
                    "confidence_threshold"
                ],
                stability_threshold=params[
                    "stability_threshold"
                ],
                thinking=params["thinking"],
                seed=params["seed"],
                alternatives=params["alternatives"],
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
        frame_history: List[FrameCheckpoint],
    ) -> None:
        """Retain the just-finished run so it can be resumed.

        ``frame_history`` holds one checkpoint per streamed frame:
        its canvas ids and canvas index, the stability state its
        confidence was derived from, and the random state the next
        step would have drawn from.
        """
        self.last_run_state = {
            "prompt": params["prompt"],
            "t_max": params["t_max"],
            "t_min": params["t_min"],
            # So an edit stops its canvas by the rule the run it
            # branches from used, which is the rule the page's
            # readout is still drawn against.
            "confidence_threshold": params[
                "confidence_threshold"
            ],
            "stability_threshold": params[
                "stability_threshold"
            ],
            "thinking": params["thinking"],
            "seed": params["seed"],
            # So an edit captures candidates exactly when the run it
            # branches from did.
            "alternatives": params["alternatives"],
            "max_denoising_steps": params[
                "max_denoising_steps"
            ],
            "frame_history": frame_history,
            # The same checkpoints under a second name, as the run
            # was generated. A resume splices the list above; this
            # one it never touches, so an edit session can start
            # from the run the browser is showing rather than from
            # a branch the user already discarded. Two lists over
            # one set of objects, so it costs references and not
            # canvases.
            "generated_frame_history": list(frame_history),
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
        history: List[FrameCheckpoint] = state["frame_history"]
        if len(history) == 0:
            raise ValueError(
                "No frames available to resume from."
            )
        # Single-canvas scope: a multi-canvas run cannot be
        # re-entered with this seed-canvas strategy.
        max_canvas = max(f.canvas_index for f in history)
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
        canvas_length = int(history[frame_index].ids.numel())
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
            "max_frames": _validate_resume_budget(data),
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
        max_frames: Optional[int] = resume_params["max_frames"]
        base = state["frame_history"][frame_index]
        base_history = state["frame_history"][:frame_index]
        assert len(base_history) == frame_index, (
            "the staged prefix stops at the resume frame"
        )
        resume_frames: List[FrameCheckpoint] = []
        try:
            generator = streaming_resume(
                self.model,
                self.tokenizer,
                self.text_adapter,
                prompt=state["prompt"],
                base=base,
                remask_positions=resume_params[
                    "remask_positions"
                ],
                remaining_steps=resume_params[
                    "remaining_steps"
                ],
                t_max=state["t_max"],
                t_min=state["t_min"],
                confidence_threshold=state[
                    "confidence_threshold"
                ],
                stability_threshold=state[
                    "stability_threshold"
                ],
                thinking=state["thinking"],
                seed=state["seed"],
                alternatives=state["alternatives"],
                cancel_event=cancel_event,
                frame_history=resume_frames,
            )
            sent = await self._forward_resume(
                ws, stream, generator, start, max_frames
            )
            # Keep only the frames the client received, so the
            # worker history stays aligned with the browser's for
            # any subsequent resume.
            assert sent <= len(resume_frames), (
                "every forwarded frame left a checkpoint"
            )
            _commit_resume(
                state, base_history, resume_frames[:sent]
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

    async def _forward_resume(
        self,
        ws: WebSocket,
        stream: FrameStreamer,
        generator: Any,
        start: float,
        max_frames: Optional[int],
    ) -> int:
        """Forward resume frames, always draining the generator.

        DiffusionGemma runs ``generate`` in a background thread, so
        the generator must be consumed to completion before returning
        (otherwise the thread would keep using the GPU after the
        request finishes). When ``max_frames`` is set (guided "run to
        here"), frames past the budget are drained silently and an
        explicit ``done`` is sent so the client stops at the target.
        It carries the text of the last frame the client received,
        and says the run stopped only when a stop cut the budget
        short (``_send_guided_terminal``).

        Both terminal frames go out through the streamer even though
        the ordinary ones do not, because that is what stamps the
        run's provenance. Draining is this method's reason to exist,
        so it cannot hand the generator to ``stream.run``.

        Candidates go out only when every frame did. A guided edit's
        would name frames past its budget, which the page never
        receives, so it sends none, as a guided LLaDA edit does.

        Returns how many frames reached the client, which is how many
        of the staged checkpoints the run may keep.
        """
        sent = 0
        last_text = ""
        stopped = False
        async for frame in generator:
            ftype = frame.get("type")
            if ftype == "frame":
                if max_frames is None or sent < max_frames:
                    frame["elapsed"] = round(
                        time.monotonic() - start, 2
                    )
                    await ws.send_json(frame)
                    sent += 1
                    last_text = frame["text"]
                continue
            if ftype == "candidates" and max_frames is None:
                await ws.send_json(frame)
            if ftype == "done":
                stopped = frame.get(TERMINAL_CANCELLED) is True
                if max_frames is None:
                    await stream.send_done(frame, start)
        if max_frames is not None:
            assert sent <= max_frames, (
                "a guided edit sends no more than its budget"
            )
            assert isinstance(last_text, str), (
                "a frame's text is a string"
            )
            await _send_guided_terminal(
                stream,
                start,
                final_text=last_text,
                cut_short=stopped and sent < max_frames,
            )
        return sent


def build_backend() -> Backend:
    return DgemmaBackend()
