"""The worker shell every append-only model shares.

SmolLM3 and Mamba-3 both decode left to right through
``src/inference/ar_sampler.py`` and serve the same four requests:
generate, What If substitution, the typed-token probe, and raw-text
KGW detection. Those handlers need only a model, a tokenizer and a
text adapter, so they
live here once instead of in each worker. A worker adds ``load()``,
its registry entry and its adapter.

What If re-enters the last run, never a branch of it, and validates
against that run's own record: a captured candidate must be one the
model actually considered there, and a typed token must resolve to
exactly the id the client sent. Both rules are the reason a
counterfactual here is a real branch of a decision the model faced.

Diffusion-style remask and resume do not apply to a left-to-right
model, so ``handle_resume`` stays the base ``NotImplementedError``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import WebSocket

from src.backends.context_pack import ContextRequestError
from src.backends.params import request_bool, resolve_params
from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_STALE_RUN,
    ERROR_WATERMARK_KEY_MISMATCH,
    ERROR_WATERMARK_KEY_MISSING,
    ERROR_WATERMARK_KEY_STATE,
    MSG_DETECT_WATERMARK,
    MSG_DETECT_WATERMARK_RESULT,
    MSG_GENERATE,
    MSG_PROBE,
    MSG_PROBE_RESULT,
    MSG_SUBSTITUTE,
    ParamSpec,
    request_error,
    request_id_of,
)
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
    describe_output_width,
    tokenize_pieces,
)
from src.inference.ar_sampler import (
    probe_token,
    streaming_generate,
    streaming_substitute,
)
from src.inference.kgw_key import load_key, load_or_create_key
from src.inference.kgw_watermark import (
    KGW_DISPLAY_Z_THRESHOLD_DEFAULT,
    KGW_SCHEME,
    KGW_VERSION,
    KgwConfig,
    KgwWatermark,
    detect_token_ids,
    detection_display_status,
    tokenizer_fingerprint,
)

logger = logging.getLogger("append_only_backend")

WATERMARK_DETECT_TEXT_MAX_CHARS = 100_000
WATERMARK_DETECT_TOKENS_MAX = 4_096

assert WATERMARK_DETECT_TEXT_MAX_CHARS > 0
assert WATERMARK_DETECT_TOKENS_MAX > 0


class _WatermarkKeyMismatch(ValueError):
    """The detector was asked to use another key identity."""


class _WatermarkKeyStateError(OSError):
    """The durable KGW key exists but cannot be used safely."""


def _parameter_default(
    specs: List[ParamSpec], name: str
) -> Union[int, float, str, bool]:
    """Read one registry default without restating its value."""
    for spec in specs:
        if spec.name == name:
            return spec.default
    raise AssertionError(f"missing detector parameter {name}")


def _validate_watermark_detection_text(value: object) -> str:
    """One non-empty pasted string within the worker's bound."""
    if not isinstance(value, str):
        raise TypeError("Detector text must be a string.")
    if value == "":
        raise ValueError("Paste text to detect first.")
    if len(value) > WATERMARK_DETECT_TEXT_MAX_CHARS:
        raise ValueError(
            "Detector text is "
            f"{len(value):,} characters; the limit is "
            f"{WATERMARK_DETECT_TEXT_MAX_CHARS:,}."
        )
    return value


def _validate_watermark_detection_gamma(value: object) -> float:
    """A finite keyed-set fraction strictly inside (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Detector gamma must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Detector gamma must be finite.")
    if not 0.0 < number < 1.0:
        raise ValueError("Detector gamma must be between 0 and 1.")
    return number


def _validate_watermark_detection_threshold(value: object) -> float:
    """A finite non-negative display threshold."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Display z threshold must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Display z threshold must be finite.")
    if number < 0.0:
        raise ValueError("Display z threshold must be non-negative.")
    return number


def _validate_watermark_detection_key(
    value: object,
) -> Optional[str]:
    """An optional lowercase 64-bit key identifier."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Expected key id must be a string.")
    if len(value) != 16:
        raise ValueError(
            "Expected key id must be 16 hexadecimal characters."
        )
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(
            "Expected key id must be lowercase hexadecimal."
        )
    return value


class AppendOnlyBackend(Backend):
    """Generate, substitute and probe for a left-to-right model.

    A subclass sets ``model_info``, ``text_adapter``, ``model``,
    ``tokenizer`` and ``last_run_state`` in its constructor and
    implements ``load()``. ``model`` is anything the sampler can
    call in Hugging Face's shape; ``thinking`` is read only when the
    registry declares it, since a completion model has no reasoning
    channel to select.
    """

    model: Any
    tokenizer: Any

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
        experimental = request_bool(data, "experimental")
        params = resolve_params(
            self.model_info.param_specs,
            data,
            device=self.effective_device,
            experimental=experimental,
        )
        thinking = bool(params.get("thinking", False))
        prompt = self.prepare_generation_prompt(
            data,
            output_reserve=int(params["max_new_tokens"]),
            thinking=thinking,
        )
        params["prompt"] = prompt.value
        params["prompt_text"] = prompt.pending_user_text
        params["context_pack"] = prompt.context_pack
        params["_watermark"] = self._resolve_watermark(
            params,
            experimental=experimental,
        )
        return params

    def _resolve_watermark(
        self,
        params: Dict[str, Any],
        *,
        experimental: bool,
    ) -> Optional[KgwWatermark]:
        """Build keyed state only after the opt-in resolves true."""
        if not bool(params.get("watermark", False)):
            return None
        if not experimental:
            raise ValueError(
                "KGW watermarking requires Experimental mode."
            )
        width = describe_output_width(self.model)
        if width is None:
            raise ValueError(
                "KGW watermarking needs the model vocabulary width."
            )
        try:
            key = load_or_create_key()
        except OSError as exc:
            raise _WatermarkKeyStateError(str(exc)) from exc
        config = KgwConfig(
            secret=key.secret,
            key_id=key.key_id,
            model_id=self.model_info.id,
            tokenizer_fingerprint=tokenizer_fingerprint(
                self.tokenizer
            ),
            vocab_size=width,
            gamma=float(params["watermark_gamma"]),
            delta=float(params["watermark_delta"]),
        )
        return KgwWatermark(config)

    async def handle_detect_watermark(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Tokenize raw pasted text and score it without a forward."""
        request_id = request_id_of(data)
        try:
            request = self._validate_watermark_detection(data)
            result = await asyncio.to_thread(
                self._detect_watermark_text,
                text=request["text"],
                gamma=request["gamma"],
                z_threshold=request["z_threshold"],
                expected_key_id=request["expected_key_id"],
                cancel_event=cancel_event,
            )
        except InterruptedError:
            return
        except FileNotFoundError:
            await ws.send_json(
                request_error(
                    message=(
                        "No local KGW key exists yet. Enable the"
                        " watermark for a run first."
                    ),
                    code=ERROR_WATERMARK_KEY_MISSING,
                    request_type=MSG_DETECT_WATERMARK,
                    request_id=request_id,
                )
            )
            return
        except _WatermarkKeyStateError as exc:
            await ws.send_json(
                request_error(
                    message=(
                        "The local KGW key could not be read safely:"
                        f" {exc}"
                    ),
                    code=ERROR_WATERMARK_KEY_STATE,
                    request_type=MSG_DETECT_WATERMARK,
                    request_id=request_id,
                )
            )
            return
        except _WatermarkKeyMismatch as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_WATERMARK_KEY_MISMATCH,
                    request_type=MSG_DETECT_WATERMARK,
                    request_id=request_id,
                )
            )
            return
        except (ValueError, TypeError) as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_DETECT_WATERMARK,
                    request_id=request_id,
                )
            )
            return
        result["type"] = MSG_DETECT_WATERMARK_RESULT
        result["request_id"] = 0 if request_id is None else request_id
        await ws.send_json(result)

    def _validate_watermark_detection(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate bounded detector inputs before scheduling work."""
        text = _validate_watermark_detection_text(data.get("text"))
        gamma = _validate_watermark_detection_gamma(
            data.get(
                "gamma",
                _parameter_default(
                    self.model_info.param_specs,
                    "watermark_gamma",
                ),
            )
        )
        threshold = _validate_watermark_detection_threshold(
            data.get("z_threshold", KGW_DISPLAY_Z_THRESHOLD_DEFAULT)
        )
        expected = _validate_watermark_detection_key(
            data.get("expected_key_id")
        )
        return {
            "text": text,
            "gamma": gamma,
            "z_threshold": threshold,
            "expected_key_id": expected,
        }

    def _detect_watermark_text(
        self,
        *,
        text: str,
        gamma: float,
        z_threshold: float,
        expected_key_id: Optional[str],
        cancel_event: Optional[threading.Event],
    ) -> Dict[str, Any]:
        """Score ids from the raw tokenizer with the existing key."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("No tokenizer is loaded.")
        width = describe_output_width(self.model)
        if width is None:
            raise ValueError(
                "KGW detection needs the model vocabulary width."
            )
        try:
            key = load_key()
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise _WatermarkKeyStateError(str(exc)) from exc
        if (
            expected_key_id is not None
            and expected_key_id != key.key_id
        ):
            raise _WatermarkKeyMismatch(
                f"The loaded key id is {key.key_id}, not "
                f"{expected_key_id}."
            )
        config = KgwConfig(
            secret=key.secret,
            key_id=key.key_id,
            model_id=self.model_info.id,
            tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
            vocab_size=width,
            gamma=gamma,
            delta=0.0,
        )
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in encoded]
        if len(token_ids) > WATERMARK_DETECT_TOKENS_MAX:
            raise ValueError(
                "Detector text became "
                f"{len(token_ids):,} tokens; the limit is "
                f"{WATERMARK_DETECT_TOKENS_MAX:,}."
            )
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("watermark detection cancelled")
        evidence = [index > 0 for index in range(len(token_ids))]
        detected = detect_token_ids(
            token_ids,
            evidence,
            config=config,
            cancelled=(
                cancel_event.is_set
                if cancel_event is not None
                else None
            ),
        )
        return {
            "scheme": KGW_SCHEME,
            "version": KGW_VERSION,
            "key_id": key.key_id,
            "model_id": self.model_info.id,
            "tokenizer_fingerprint": (config.tokenizer_fingerprint),
            "vocab_size": config.vocab_size,
            "green_list_size": config.green_list_size,
            "gamma": gamma,
            "token_count": len(token_ids),
            "evidence_status": detected.status,
            "status": detection_display_status(
                detected,
                z_threshold=z_threshold,
            ),
            "display_threshold": z_threshold,
            **{
                name: value
                for name, value in detected.as_dict().items()
                if name != "status"
            },
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
        except ContextRequestError as exc:
            await ws.send_json(
                request_error(
                    message=str(exc),
                    code=exc.code,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
            )
            return
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
        except _WatermarkKeyStateError as exc:
            await ws.send_json(
                request_error(
                    message=(
                        f"KGW key state could not be prepared: {exc}"
                    ),
                    code=ERROR_INVALID_REQUEST,
                    request_type=MSG_GENERATE,
                    request_id=request_id_of(data),
                )
            )
            return

        start = time.monotonic()
        thinking = bool(params.get("thinking", False))
        # Discards the prior run's trace and retires its token
        # together, so a failure here cannot leave a stale state that
        # a substitution would re-enter against the wrong prompt.
        # This is also what frees the previous run's KV cache, and it
        # happens before the new one allocates so the two never sit
        # in device memory at once.
        self.begin_run(context_pack=params.get("context_pack"))
        watermark = params.get("_watermark")
        assert watermark is None or isinstance(
            watermark, KgwWatermark
        )
        self.run_watermark = watermark
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
                thinking=thinking,
                alternatives=params["alternatives"],
                seed=params["seed"],
                watermark=watermark,
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
            prompt = params["prompt"]
            state["prompt"] = prompt
            prompt_text = (
                prompt
                if isinstance(prompt, str)
                else prompt[-1].content
            )
            state["prompt_text"] = params.get(
                "prompt_text", prompt_text
            )
            context_pack = params.get("context_pack")
            if context_pack is not None:
                state["context_pack"] = context_pack
            state["max_new_tokens"] = params["max_new_tokens"]
            state["thinking"] = thinking
            state["seed"] = params["seed"]
            state["alternatives_enabled"] = params["alternatives"]
            if watermark is not None:
                state["watermark_run"] = watermark
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
        watermark = _watermark_branch(state, position)
        self.run_watermark = watermark
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
                prefix_entropies=state["entropies"][:position],
                prefix_alts=state["alternatives"][:position],
                prefix_signals=_prefix_signals(state, position),
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
                watermark=watermark,
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
                prefix_watermark_memberships=state.get(
                    "watermark_memberships", []
                )[:position],
                prefix_watermark_evidence=state.get(
                    "watermark_evidence", []
                )[:position],
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


def _prefix_signals(
    state: Dict[str, Any], position: int
) -> Optional[List[Dict[str, float]]]:
    """The recorded run's per-token values before `position`, so a
    branch keeps them; None for a run that recorded none."""
    signals = state.get("signals")
    if signals is None:
        return None
    return list(signals[:position])


def _watermark_branch(
    state: Dict[str, Any], position: int
) -> Optional[KgwWatermark]:
    """Seed a What If score from the original run's kept prefix."""
    retained = state.get("watermark_run")
    if retained is None:
        return None
    if not isinstance(retained, KgwWatermark):
        raise TypeError("retained watermark state has the wrong type")
    memberships = state.get("watermark_memberships")
    evidence = state.get("watermark_evidence")
    if not isinstance(memberships, list):
        raise ValueError("watermarked run lost its memberships")
    if not isinstance(evidence, list):
        raise ValueError("watermarked run lost its evidence flags")
    return retained.fork(
        memberships[:position],
        evidence[:position],
    )


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
