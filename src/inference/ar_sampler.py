"""Streaming autoregressive sampler.

Drives a manual token-by-token decoding loop (not HF's text
streamer) so we can capture per-token sampling confidence, and
emits one protocol frame per new token in the shared visualizer
shape. Each frame is a full snapshot of the sequence generated so
far, mapping the diffusion "canvas" onto a growing left-to-right
sequence: every token is resolved (``m`` false) and ``c`` is the
softmax probability of the chosen token.

Generic across autoregressive ``AutoModelForCausalLM`` checkpoints;
the SmolLM3 worker is the first caller.

Two XAI signals ride alongside confidence. Per-token **entropy** is
always captured (one float, off the softmax the sampler already
computes) and answers a different question than confidence: how
undecided the model was over the whole vocabulary, not how likely
the token it chose was. Per-position **top-k alternatives** are
opt-in, mirroring DiffusionGemma's entropy-signal toggle.

Note on payload size: full-snapshot frames make the streamed
payload O(n^2) in the token count. This is fine for the few-hundred
token budgets used here (see the registry's recommended cap) and is
revisited only if long AR runs are added. Alternatives deliberately
do NOT ride every snapshot (see ``_build_frame``), which would
multiply that by k.
"""

from __future__ import annotations

import asyncio
import queue
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import torch

# Competing candidates captured per position when the opt-in
# alternatives signal is on. Fixed rather than user-facing: five is
# enough to read a decision and keeps the payload predictable.
TOP_K_ALTERNATIVES = 5

# Control/structure tokens hidden from the per-token display so the
# chat scaffolding does not clutter the streamed output.
_STRIP_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _StepPick(NamedTuple):
    """One decoding step's sampled token and captured signals.

    ``entropy`` is in nats over the full untempered distribution.
    ``alternatives`` is None unless the opt-in capture is enabled.
    """

    token_id: int
    confidence: float
    entropy: float
    alternatives: Optional[List[Dict[str, Any]]]


def _sanitize(text: str) -> str:
    for token in _STRIP_TOKENS:
        text = text.replace(token, "")
    return text


def _split_thinking(raw: str) -> Tuple[str, str]:
    """Separate the reasoning trace from the final answer.

    SmolLM3 wraps extended reasoning in ``<think> ... </think>``
    before the answer. Returns (thinking, answer), both cleaned of
    control tokens. When no reasoning trace is present (thinking
    disabled), thinking is empty and the whole output is the answer.
    """
    if _THINK_CLOSE in raw:
        head, _, answer = raw.partition(_THINK_CLOSE)
        head = head.replace(_THINK_OPEN, "", 1)
        return _sanitize(head).strip(), _sanitize(answer).strip()
    return "", _sanitize(raw).strip()


def _seed(seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_inputs(
    tokenizer: Any, model: Any, prompt: str, *, thinking: bool
) -> Any:
    """Tokenize a chat prompt into model-ready generate inputs."""
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=thinking,
    ).to(model.device)


def _stop_ids(tokenizer: Any, model: Any) -> Set[int]:
    """Collect token ids that end generation (EOS + chat turn end)."""
    ids: Set[int] = set()

    def _add(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, int):
                    ids.add(int(item))
        elif isinstance(value, int):
            ids.add(int(value))

    _add(tokenizer.eos_token_id)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        _add(getattr(gen_cfg, "eos_token_id", None))
    # The ChatML turn terminator is a distinct token from EOS.
    unk = tokenizer.unk_token_id
    turn_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(turn_end, int) and turn_end >= 0 and turn_end != unk:
        ids.add(int(turn_end))
    return ids


def _top_p_filter(
    probs: torch.Tensor, top_p: float
) -> torch.Tensor:
    """Zero out the tail beyond nucleus mass ``top_p`` and renormalize.

    ``probs`` is a 1-D distribution. The smallest set of highest
    tokens whose cumulative mass reaches ``top_p`` is kept.
    """
    assert probs.dim() == 1, "probs must be 1-D"
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the first token that crosses the threshold; drop the rest.
    beyond = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(beyond, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    total = float(filtered.sum().item())
    if total <= 0.0:
        return probs  # Degenerate top_p: fall back to full distribution.
    return filtered / total


def _entropy_nats(probs: torch.Tensor) -> float:
    """Shannon entropy of a 1-D distribution, in nats.

    Uses ``torch.special.entr`` so zero-probability entries
    contribute 0 rather than a NaN from log(0).

    Reported raw instead of normalized into [0,1] by log(vocab):
    across a ~128k vocabulary every realistic value would collapse
    into the bottom of that scale, so the display side normalizes
    against its own reference maximum instead.
    """
    assert probs.dim() == 1, "probs must be 1-D"
    value = float(torch.special.entr(probs).sum().item())
    assert value >= 0.0, "entropy must be non-negative"
    return value


def _top_alternatives(
    probs: torch.Tensor, tokenizer: Any, k: int
) -> List[Dict[str, Any]]:
    """Capture the k highest-probability candidates at one step.

    Text is the raw decode with control tokens intact, so a
    candidate like an end-of-turn marker stays legible in the UI
    instead of sanitizing away to an empty string. Rendering
    whitespace-only candidates readably is the frontend's job.
    """
    assert probs.dim() == 1, "probs must be 1-D"
    assert k > 0, "k must be positive"
    count = min(k, int(probs.numel()))
    values, indices = torch.topk(probs, count)
    candidates: List[Dict[str, Any]] = []
    for rank in range(count):
        token_id = int(indices[rank].item())
        candidates.append(
            {
                "id": token_id,
                "t": tokenizer.decode(
                    [token_id], skip_special_tokens=False
                ),
                "p": round(float(values[rank].item()), 4),
            }
        )
    return candidates


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    tokenizer: Any,
    alternatives: bool,
) -> _StepPick:
    """Pick the next token and its signals from step logits.

    Confidence is the token's probability under the untempered
    softmax (temperature 1), a faithful "how sure was the model"
    signal independent of the sampling temperature, mirroring
    LLaDA's reveal-time softmax probability. Entropy and the
    candidate set come off that same distribution, so the sampling
    temperature never distorts any of the three.
    """
    logits = logits.float().squeeze(0)  # (vocab,)
    base_probs = torch.softmax(logits, dim=-1)
    if temperature <= 0.0:
        next_id = int(torch.argmax(logits).item())
    else:
        scaled = torch.softmax(logits / temperature, dim=-1)
        scaled = _top_p_filter(scaled, top_p)
        next_id = int(torch.multinomial(scaled, 1).item())
    assert 0 <= next_id < int(base_probs.numel()), (
        "sampled id out of vocabulary range"
    )
    candidates = (
        _top_alternatives(
            base_probs, tokenizer, TOP_K_ALTERNATIVES
        )
        if alternatives
        else None
    )
    return _StepPick(
        token_id=next_id,
        confidence=float(base_probs[next_id].item()),
        entropy=_entropy_nats(base_probs),
        alternatives=candidates,
    )


def _build_frame(
    tokenizer: Any,
    ids: List[int],
    confs: List[float],
    entropies: List[float],
    *,
    frame_index: int,
    total_steps: int,
    newest_alternatives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one full-snapshot protocol frame for the growing sequence."""
    assert len(ids) == len(confs), "ids/confs length mismatch"
    assert len(ids) == len(entropies), "ids/entropy mismatch"
    tokens: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    conf_sum = 0.0
    for i, token_id in enumerate(ids):
        display = _sanitize(
            tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
        )
        confidence = confs[i]
        conf_sum += confidence
        tokens.append(
            {
                "t": display,
                "m": False,
                "id": int(token_id),
                "c": round(confidence, 4),
                "e": round(entropies[i], 4),
            }
        )
        text_parts.append(display)
    count = len(ids)
    frame: Dict[str, Any] = {
        "type": "frame",
        # 1-based to match the diffusion step convention (the frontend
        # prints ``index`` verbatim): the first token reads "Step 1" and
        # a full run of N tokens ends at "Step N/N", not "N-1/N".
        "index": frame_index + 1,
        "total_steps": total_steps,
        "canvas_index": 0,
        "mean_conf": (
            round(conf_sum / count, 4) if count else 0.0
        ),
        "text": "".join(text_parts),
        "tokens": tokens,
    }
    if newest_alternatives is not None:
        # A position's candidate set is fixed the moment it is
        # sampled and never revised, so it rides only the frame that
        # introduces that position (always the last token here) and
        # the client accumulates. Repeating it on every snapshot
        # would multiply an already O(n^2) payload by k.
        frame["alts"] = newest_alternatives
    return frame


def _grow_attention(
    attention_mask: Optional[torch.Tensor], device: Any
) -> Optional[torch.Tensor]:
    """Extend an attention mask by one attended position.

    The KV-cached loop forwards only the newest token each step, so
    the mask must grow to cover prompt plus every token produced so
    far. Passes None through for models that need no mask.
    """
    if attention_mask is None:
        return None
    assert attention_mask.dim() == 2, "mask must be 2-D"
    return torch.cat(
        [
            attention_mask,
            torch.ones(
                (1, 1),
                dtype=attention_mask.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )


class _Trace:
    """Mutable per-position record of one (partial) generation.

    Owned by the calling loop so both a fresh generation and a
    substitution branch can seed it and then extend it in place.
    Every list stays index-aligned with ``ids``.
    """

    def __init__(self) -> None:
        self.ids: List[int] = []
        self.confs: List[float] = []
        self.entropies: List[float] = []
        self.alts: List[Optional[List[Dict[str, Any]]]] = []

    def append(self, pick: _StepPick) -> None:
        self.ids.append(pick.token_id)
        self.confs.append(pick.confidence)
        self.entropies.append(pick.entropy)
        self.alts.append(pick.alternatives)

    def check(self) -> None:
        assert len(self.ids) == len(self.confs), "conf misalign"
        assert len(self.ids) == len(self.entropies), (
            "entropy misalign"
        )
        assert len(self.ids) == len(self.alts), "alts misalign"


def _stream_tokens(
    *,
    model: Any,
    tokenizer: Any,
    step_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    trace: _Trace,
    budget: int,
    total_steps: int,
    temperature: float,
    top_p: float,
    alternatives: bool,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
) -> None:
    """Decode up to ``budget`` tokens, emitting one frame each.

    ``step_ids`` is the prefill for the first forward pass (a prompt,
    or a prompt plus an already-committed prefix); afterwards only the
    newest token is forwarded, against a KV cache. ``trace`` carries
    any committed prefix in and is extended in place. Frame indices
    continue from the trace's current length, so a substitution branch
    numbers its frames as a continuation of the original run.
    """
    assert budget >= 0, "budget must be non-negative"
    trace.check()
    stop_ids = _stop_ids(tokenizer, model)
    device = model.device
    dtype = step_ids.dtype
    past: Any = None

    with torch.no_grad():
        for _ in range(budget):
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                break
            outputs = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            pick = _sample_next(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                tokenizer=tokenizer,
                alternatives=alternatives,
            )
            frame_index = len(trace.ids)
            trace.append(pick)
            out_queue.put(
                _build_frame(
                    tokenizer,
                    trace.ids,
                    trace.confs,
                    trace.entropies,
                    frame_index=frame_index,
                    total_steps=total_steps,
                    newest_alternatives=pick.alternatives,
                )
            )
            if pick.token_id in stop_ids:
                break
            step_ids = torch.tensor(
                [[pick.token_id]], dtype=dtype, device=device
            )
            attention_mask = _grow_attention(
                attention_mask, device
            )
    trace.check()


def _finalize(
    tokenizer: Any, trace: _Trace, result: Dict[str, Any]
) -> None:
    """Record the decoded text and the full trace for the caller.

    The trace is what a later substitution needs: the worker keeps it
    so a counterfactual can re-enter at any position without replaying
    the run.
    """
    raw = tokenizer.decode(
        trace.ids, skip_special_tokens=False
    )
    thinking_text, final_text = _split_thinking(raw)
    result["final_text"] = final_text
    result["thinking"] = thinking_text
    result["ids"] = list(trace.ids)
    result["confidences"] = list(trace.confs)
    result["entropies"] = list(trace.entropies)
    result["alternatives"] = list(trace.alts)


def _decode_loop(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alternatives: bool,
    seed: int,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    result: Dict[str, Any],
) -> None:
    """Blocking token-by-token generation; runs in a worker thread."""
    _seed(seed)
    trace = _Trace()
    _stream_tokens(
        model=model,
        tokenizer=tokenizer,
        step_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        trace=trace,
        budget=max_new_tokens,
        total_steps=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        alternatives=alternatives,
        out_queue=out_queue,
        cancel_event=cancel_event,
    )
    _finalize(tokenizer, trace, result)


def _substitute_loop(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Any,
    prefix_ids: List[int],
    prefix_confs: List[float],
    prefix_entropies: List[float],
    prefix_alts: List[Optional[List[Dict[str, Any]]]],
    position: int,
    forced_id: int,
    forced_conf: float,
    forced_entropy: float,
    forced_alts: Optional[List[Dict[str, Any]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alternatives: bool,
    seed: int,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    result: Dict[str, Any],
) -> None:
    """Force one position, then continue generating from it.

    Emits a seed frame covering positions 0..position (the forced
    token last), then streams the re-generated continuation. The
    forced position keeps its originally captured confidence,
    entropy, and candidate set, since those describe the decision the
    model actually faced there.
    """
    assert position >= 0, "position must be non-negative"
    assert len(prefix_ids) == position, (
        "prefix must end just before the forced position"
    )
    _seed(seed)

    trace = _Trace()
    trace.ids = list(prefix_ids)
    trace.confs = list(prefix_confs)
    trace.entropies = list(prefix_entropies)
    trace.alts = list(prefix_alts)
    trace.ids.append(forced_id)
    trace.confs.append(forced_conf)
    trace.entropies.append(forced_entropy)
    trace.alts.append(forced_alts)
    trace.check()

    # The seed frame carries the substituted token so the client can
    # splice the branch on immediately, before any new decoding.
    out_queue.put(
        _build_frame(
            tokenizer,
            trace.ids,
            trace.confs,
            trace.entropies,
            frame_index=position,
            total_steps=max_new_tokens,
            newest_alternatives=forced_alts,
        )
    )

    # One prefill over prompt + kept prefix + the forced token; its
    # last position's logits predict the next one.
    prompt_ids = inputs["input_ids"]
    device = model.device
    branch = torch.tensor(
        [trace.ids], dtype=prompt_ids.dtype, device=device
    )
    step_ids = torch.cat([prompt_ids, branch], dim=-1)
    attention_mask = _prefill_attention(
        inputs.get("attention_mask"),
        step_ids.shape[-1],
        device,
    )

    remaining = max_new_tokens - (position + 1)
    _stream_tokens(
        model=model,
        tokenizer=tokenizer,
        step_ids=step_ids,
        attention_mask=attention_mask,
        trace=trace,
        budget=max(0, remaining),
        total_steps=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        alternatives=alternatives,
        out_queue=out_queue,
        cancel_event=cancel_event,
    )
    _finalize(tokenizer, trace, result)


def _prefill_attention(
    prompt_mask: Optional[torch.Tensor],
    length: int,
    device: Any,
) -> Optional[torch.Tensor]:
    """All-ones mask covering a prompt plus committed prefix.

    Every prefilled position is real (no padding), so the mask is
    uniform; its dtype follows the prompt's when one exists.
    """
    if prompt_mask is None:
        return None
    assert length > 0, "prefill length must be positive"
    return torch.ones(
        (1, length), dtype=prompt_mask.dtype, device=device
    )


async def streaming_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.95,
    thinking: bool = False,
    alternatives: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
    state_sink: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield one frame per generated token, then a ``done`` frame.

    Runs the blocking decode loop in a thread and forwards frames as
    they arrive, matching the diffusion workers' async-generator
    contract so the shared ``FrameStreamer`` can drive it unchanged.

    ``alternatives`` opts into per-position top-k candidate capture,
    which costs an extra topk plus k decodes per token.
    ``state_sink``, when given, receives the run's per-position trace
    so the worker can serve a later substitution.
    """
    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    out_queue: "queue.Queue[Any]" = queue.Queue()
    result: Dict[str, Any] = {}

    def run() -> None:
        try:
            _decode_loop(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                alternatives=alternatives,
                seed=seed,
                out_queue=out_queue,
                cancel_event=cancel_event,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc
        finally:
            out_queue.put(None)

    async for frame in _drain_frames(
        runner=run,
        out_queue=out_queue,
        result=result,
        state_sink=state_sink,
    ):
        yield frame


async def streaming_substitute(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    position: int,
    forced_id: int,
    forced_conf: float,
    forced_entropy: float,
    forced_alts: Optional[List[Dict[str, Any]]],
    prefix_ids: List[int],
    prefix_confs: List[float],
    prefix_entropies: List[float],
    prefix_alts: List[Optional[List[Dict[str, Any]]]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    thinking: bool = False,
    alternatives: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
    state_sink: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Substitute one position's token, then regenerate forward.

    The autoregressive counterfactual: keep positions before
    ``position``, force ``forced_id`` there, and continue decoding.
    Emits a seed frame for the truncated-plus-forced prefix, then one
    frame per newly generated token, then ``done``, so the client can
    splice the branch onto its history exactly like a diffusion
    resume.

    Defaults to greedy (temperature 0) so downstream divergence is
    attributable to the intervention rather than to fresh sampling
    noise landing in a different context.
    """
    assert position >= 0, "position must be non-negative"
    assert len(prefix_ids) == position, (
        "prefix length must equal the forced position"
    )
    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    out_queue: "queue.Queue[Any]" = queue.Queue()
    result: Dict[str, Any] = {}

    def run() -> None:
        try:
            _substitute_loop(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                prefix_ids=prefix_ids,
                prefix_confs=prefix_confs,
                prefix_entropies=prefix_entropies,
                prefix_alts=prefix_alts,
                position=position,
                forced_id=forced_id,
                forced_conf=forced_conf,
                forced_entropy=forced_entropy,
                forced_alts=forced_alts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                alternatives=alternatives,
                seed=seed,
                out_queue=out_queue,
                cancel_event=cancel_event,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc
        finally:
            out_queue.put(None)

    async for frame in _drain_frames(
        runner=run,
        out_queue=out_queue,
        result=result,
        state_sink=state_sink,
    ):
        yield frame


async def _drain_frames(
    *,
    runner: Any,
    out_queue: "queue.Queue[Any]",
    result: Dict[str, Any],
    state_sink: Optional[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run a blocking decode loop in a thread and forward its frames.

    Yields each queued frame, then a terminal ``done``. Re-raises
    whatever the loop failed with. The per-position trace goes to
    ``state_sink`` rather than onto the wire: the browser does not
    need it, but the worker does to serve a substitution.
    """
    task = asyncio.create_task(asyncio.to_thread(runner))
    try:
        while True:
            item = await asyncio.to_thread(out_queue.get)
            if item is None:
                break
            yield item
    finally:
        await task

    if "err" in result:
        raise result["err"]

    if state_sink is not None:
        state_sink["ids"] = result.get("ids", [])
        state_sink["confidences"] = result.get(
            "confidences", []
        )
        state_sink["entropies"] = result.get("entropies", [])
        state_sink["alternatives"] = result.get(
            "alternatives", []
        )

    yield {
        "type": "done",
        "final_text": result.get("final_text", ""),
        "thinking": result.get("thinking", ""),
    }
