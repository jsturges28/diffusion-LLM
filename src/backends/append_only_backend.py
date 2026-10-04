"""The worker shell every append-only model shares.

SmolLM3 and Mamba-3 both decode left to right through
``src/inference/ar_sampler.py`` and serve the same three requests:
generate, What If substitution, and the typed-token probe. Those
handlers need only a model, a tokenizer and a text adapter, so they
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
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from src.backends.context_pack import ContextRequestError
from src.backends.params import resolve_params
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

logger = logging.getLogger("append_only_backend")


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
        params = resolve_params(
            self.model_info.param_specs,
            data,
            device=self.effective_device,
            experimental=bool(
                data.get("experimental", False)
            ),
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

        start = time.monotonic()
        thinking = bool(params.get("thinking", False))
        # Discards the prior run's trace and retires its token
        # together, so a failure here cannot leave a stale state that
        # a substitution would re-enter against the wrong prompt.
        # This is also what frees the previous run's KV cache, and it
        # happens before the new one allocates so the two never sit
        # in device memory at once.
        self.begin_run(context_pack=params.get("context_pack"))
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


def _prefix_signals(
    state: Dict[str, Any], position: int
) -> Optional[List[Dict[str, float]]]:
    """The recorded run's per-token values before `position`, so a
    branch keeps them; None for a run that recorded none."""
    signals = state.get("signals")
    if signals is None:
        return None
    return list(signals[:position])


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
