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
import functools
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import WebSocket
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.backends.params import resolve_params
from src.backends.text_adapter import SMOLLM3_TEXT
from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_STALE_RUN,
    MSG_GENERATE,
    MSG_PROBE,
    MSG_PROBE_RESULT,
    MSG_SUBSTITUTE,
    request_error,
    request_id_of,
)
from src.backends.registry import SMOLLM3
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
    tokenize_pieces,
)
from src.inference.ar_sampler import (
    probe_token,
    streaming_generate,
    streaming_substitute,
)
from src.inference.hf_download import download_with_progress
from src.inference.load_progress import (
    load_target_bytes,
    sample_load_progress,
)

logger = logging.getLogger("smollm3_worker")


class Smollm3Backend(Backend):
    def __init__(self) -> None:
        self.model_info = SMOLLM3
        self.text_adapter = SMOLLM3_TEXT
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
        self.effective_device = resolved
        name = self.model_info.checkpoint
        # Fetch weights first (reporting progress via /health) so the
        # first activation shows a download bar; a cache hit is a no-op.
        logger.info("ensuring weights for %s", name)
        snapshot = download_with_progress(
            name, sink=lambda p: setattr(self, "load_progress", p)
        )
        self.load_progress = None
        # local_files_only from here down. Returning from
        # download_with_progress means every file is on disk, so any
        # request past this point is transformers revalidating a
        # checkpoint we already have. Offline that fails outright:
        # this environment's transformers asks the Hub API for
        # additional chat templates while building the tokenizer.
        logger.info("loading tokenizer %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, local_files_only=True
        )
        logger.info(
            "loading model %s on %s (bfloat16)", name, resolved
        )
        target = load_target_bytes(
            Path(snapshot), target_dtype=torch.bfloat16
        )
        # The .to() is inside the sampled block because it is half the
        # wait on this model: from_pretrained fills RAM, then the copy
        # fills VRAM, and the sampler follows whichever is climbing.
        with sample_load_progress(
            target_bytes=target,
            sink=lambda p: setattr(self, "load_progress", p),
        ):
            # bfloat16 on both devices halves the ~12 GiB fp32
            # footprint (to ~6 GiB), which matters most for
            # CPU/RAM-constrained hosts.
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            self.model = model.to(resolved).eval()
        logger.info("SmolLM3 loaded on %s", resolved)

    def _validate_generate(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One request as this sampler's arguments.

        This model's two helpers, a device-aware bounds lookup and a
        registry default lookup, were the pattern the report asked the
        other workers to adopt. They moved into ``resolve_params``
        instead of being copied twice, so the device override that
        enforces the lower CPU token cap is now applied by the same
        code every model goes through.

        Only the prompt is handled here, since it is not a declared
        parameter, and this model has no relational rules between its
        parameters.
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

        start = time.monotonic()
        # Discards the prior run's trace and retires its token
        # together, so a failure here cannot leave a stale state that
        # a substitution would re-enter against the wrong prompt.
        # This is also what frees the previous run's KV cache, and it
        # happens before the new one allocates so the two never sit
        # in device memory at once.
        self.begin_run()
        state: Dict[str, Any] = {}
        try:
            generator = streaming_generate(
                self.model,
                self.tokenizer,
                self.text_adapter,
                params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
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
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
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

        Two paths, kept deliberately separate. A captured candidate
        must be one the model actually considered at that position,
        so the counterfactual stays a real branch of a decision the
        model faced. A typed token is the explicit opt-out from
        that, checked against the vocabulary instead. The strict
        path stays strict rather than being loosened to accommodate
        the new one, so an unmarked request still cannot smuggle in
        an arbitrary id.
        """
        state = self.last_run_state
        if state is None:
            raise ValueError(
                "No previous generation to substitute into."
            )
        position = _substitute_position(state, data)
        captured = state["alternatives"][position]
        if not captured:
            raise ValueError(
                "No alternatives were captured at position"
                f" {position}. Re-run with Alternatives on."
            )
        token_id = int(data.get("token_id", -1))
        forced_conf: Optional[float] = None
        if data.get("typed"):
            self._check_typed_token(data, token_id)
            # Left None on purpose. A typed token has no recorded
            # probability, and the sampler is about to compute the
            # distribution it belongs to anyway, so inventing a
            # number here would only get in the way of the real one.
        else:
            forced_conf = _captured_confidence(
                captured, token_id, position
            )
        return {
            "position": position,
            "forced_id": token_id,
            "forced_conf": forced_conf,
            "forced_alts": captured,
        }

    def _check_typed_token(
        self, data: Dict[str, Any], token_id: int
    ) -> None:
        """Re-resolve a typed token against the vocabulary.

        The client disables its confirm button until the preview
        resolves to one token, but that gate is a convenience and
        can be bypassed. This is the contract behind it. Requiring
        the id to match what the text resolves to also stops a
        preview that went stale mid-keystroke from forcing a token
        the user never saw.
        """
        text = data.get("typed_text", "")
        if not isinstance(text, str) or text == "":
            raise ValueError("No text was typed.")
        pieces = tokenize_pieces(self.tokenizer, text)
        if len(pieces) != 1:
            raise ValueError(
                f"{text!r} is {len(pieces)} tokens; exactly"
                " one is required."
            )
        resolved = int(pieces[0]["id"])
        if resolved != token_id:
            raise ValueError(
                f"typed text resolves to token {resolved},"
                f" not {token_id}."
            )

    async def handle_substitute(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: threading.Event,
        stream: FrameStreamer,
    ) -> None:
        try:
            self.check_run_token(data)
            request = self._validate_substitute(data)
        except StaleRunError as exc:
            # Before StaleRunError's base, ValueError, or the stale
            # case would be reported as a malformed request.
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_STALE_RUN,
                    request_type=MSG_SUBSTITUTE,
                    request_id=request_id_of(data),
                )
            )
            return
        except (ValueError, TypeError, KeyError) as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_SUBSTITUTE,
                    request_id=request_id_of(data),
                )
            )
            return

        state = self.last_run_state
        assert state is not None
        assert state.get("ids"), "run state has no token trace"
        position = request["position"]
        start = time.monotonic()
        try:
            generator = streaming_substitute(
                self.model,
                self.tokenizer,
                self.text_adapter,
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
                # sampling noise in a shifted context. Both
                # truncations are therefore inert here (argmax
                # survives either), and are passed off explicitly
                # rather than left to a default.
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                thinking=state["thinking"],
                alternatives=state["alternatives_enabled"],
                seed=state["seed"],
                cancel_event=cancel_event,
                # The branch's trace is deliberately discarded, so
                # last_run_state stays pinned to the recorded run.
                # Retry on the client restores its arrays to the
                # pre-substitution run (restoreEditSnapshot in
                # app.js), so adopting the branch here would leave
                # the two sides validating against different
                # candidate sets and would reject every position at
                # or after the edit. Each substitution therefore
                # re-enters the run the user still sees.
                state_sink=None,
                # The recorded run's attention state, which is what
                # makes re-entering it cheap: without this the whole
                # kept prefix is prefilled again before the first new
                # token appears. Absent on a run that predates it or
                # one whose cache exceeded the ceiling, and the
                # sampler prefills in that case.
                cache=state.get("cache"),
            )
            await stream.run(generator, start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("substitution failed")
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_SUBSTITUTE,
                    request_id=request_id_of(data),
                )
            )
            return

    async def handle_probe(
        self, ws: WebSocket, data: Dict[str, Any]
    ) -> None:
        """Measure a token's probability at a recorded position.

        Backs the figure on the What If typed row, so the user can
        see the odds before deciding to override them. Validated the
        same way a substitution is, and against the same run state,
        because a probe that answered for a position the substitution
        would reject is worse than no answer.

        Run off the event loop: this is a real forward pass, and the
        loop still has a socket to serve while it happens.
        """
        try:
            self.check_run_token(data)
            state = self._probe_state()
            position = _substitute_position(state, data)
            token_id = _probe_token_id(data)
        except StaleRunError as exc:
            # Before StaleRunError's base, ValueError, or the stale
            # case would be reported as a malformed request.
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_STALE_RUN,
                    request_type=MSG_PROBE,
                    request_id=request_id_of(data),
                )
            )
            return
        except (ValueError, TypeError, KeyError) as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_PROBE,
                    request_id=request_id_of(data),
                )
            )
            return

        loop = asyncio.get_running_loop()
        try:
            measured = await loop.run_in_executor(
                None,
                functools.partial(
                    probe_token,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    adapter=self.text_adapter,
                    prompt=state["prompt"],
                    prefix_ids=state["ids"][:position],
                    token_id=token_id,
                    thinking=state["thinking"],
                    # With the run's own cache the measurement is the
                    # run's arithmetic, so a token the run captured
                    # measures to its recorded probability exactly
                    # rather than a bf16 rounding step away from it.
                    cache=state.get("cache"),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("probe failed")
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_GENERATION_FAILED,
                    request_type=MSG_PROBE,
                    request_id=request_id_of(data),
                )
            )
            return

        await ws.send_json(
            {
                "type": MSG_PROBE_RESULT,
                # Echoed so a reply for a token the user has since
                # retried out of is dropped rather than displayed.
                "request_id": int(data.get("request_id", 0)),
                "position": position,
                "token_id": token_id,
                "probability": measured["probability"],
                "rank": measured["rank"],
                "vocab_size": measured["vocab_size"],
            }
        )

    def _probe_state(self) -> Dict[str, Any]:
        """The run a probe reads, or a stated reason it cannot."""
        if self.model is None:
            raise ValueError("No model is loaded.")
        state = self.last_run_state
        if state is None:
            raise ValueError("No previous generation to probe.")
        return state


def _probe_token_id(data: Dict[str, Any]) -> int:
    """Range-check a requested token id.

    Only the lower bound is checkable here; the upper one belongs to
    the model's output width, which ``probe_token`` holds.
    """
    token_id = int(data.get("token_id", -1))
    if token_id < 0:
        raise ValueError(f"token id {token_id} is not valid.")
    return token_id


def _substitute_position(
    state: Dict[str, Any], data: Dict[str, Any]
) -> int:
    """Range-check a requested position against the recorded run."""
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
    return position


def _captured_confidence(
    captured: List[Dict[str, Any]],
    token_id: int,
    position: int,
) -> float:
    """The recorded probability of a candidate the model offered.

    Raising rather than defaulting is the point: a token the run
    never considered is not a candidate substitution, and letting
    one through would quietly turn a recorded counterfactual into
    an invented one.
    """
    for candidate in captured:
        if int(candidate["id"]) == token_id:
            return float(candidate["p"])
    raise ValueError(
        f"token {token_id} was not among the captured"
        f" candidates at position {position}."
    )


def build_backend() -> Backend:
    return Smollm3Backend()
