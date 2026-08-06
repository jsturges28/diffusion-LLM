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


def _top_k_filter(
    probs: torch.Tensor, top_k: int
) -> torch.Tensor:
    """Keep only the ``top_k`` highest tokens and renormalize.

    A hard truncation, unlike top-p's adaptive one: it keeps the
    same number of tokens whether the model is certain or torn.
    Applied before the nucleus cut, matching the order Hugging Face
    uses, so the two compose as "the best k, then the nucleus within
    them" rather than competing.

    Anything at or below 0 disables it. The UI's "off" value is -1
    (see the ParamSpec in registry.py), but 0 disables too, so runs
    saved while 0 was the default still mean what they meant.
    """
    assert probs.dim() == 1, "probs must be 1-D"
    assert top_k >= -1, "top_k must be -1 or greater"
    if top_k <= 0 or top_k >= int(probs.numel()):
        return probs
    kept, indices = torch.topk(probs, top_k)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, indices, kept)
    total = float(filtered.sum().item())
    if total <= 0.0:
        return probs  # Degenerate: fall back to the full set.
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


def _candidates_hold(
    candidates: List[Dict[str, Any]], token_id: int
) -> bool:
    """Whether the captured set already lists a token.

    Five dict lookups, which is why the callers gate on this before
    reaching for a rank: that costs a full-vocabulary reduction, and
    the answer here is yes on almost every step.
    """
    for candidate in candidates:
        if candidate["id"] == token_id:
            return True
    return False


def _chosen_candidate(
    *,
    token_id: int,
    probability: float,
    rank: int,
    tokenizer: Any,
) -> Dict[str, Any]:
    """The token a position committed, as an extra candidate row.

    Appended only where the captured top five omit it, which happens
    two ways: a warm temperature samples outside the nucleus's head,
    or the user forces a token of their own. Both are exactly the
    cases where the popover had nothing to say about the token on
    screen, and the row it could not explain was its own chosen mark.

    It appends rather than displacing the fifth. The popover's
    contract is "here is what the model preferred", and quietly
    dropping the fifth to make room would break it to make a point.

    ``rank`` is explicit because this entry's place in the list is
    not its place in the distribution: it goes last and may be the
    forty-thousandth. ``p`` is unrounded for the same reason the rank
    is carried at all, since this is routinely the entry whose
    probability rounds to zero at four places.
    """
    assert token_id >= 0, "token_id must be non-negative"
    assert rank >= 1, "rank is one-based"
    assert 0.0 <= probability <= 1.0, "probability out of range"
    return {
        "id": token_id,
        "t": tokenizer.decode(
            [token_id], skip_special_tokens=False
        ),
        "p": probability,
        "rank": rank,
    }


def _token_rank(probs: torch.Tensor, probability: float) -> int:
    """How many tokens the model preferred, plus one.

    A comparison and a sum rather than a sort: the distribution is
    already in hand, so this is one reduction over it.

    Ties share the better rank, since the count is strictly greater.
    Two tokens at identical probability are genuinely not ordered,
    and inventing an order from index would be a fiction.
    """
    assert probs.dim() == 1, "probs must be 1-D"
    rank = int((probs > probability).sum().item()) + 1
    assert 1 <= rank <= int(probs.numel()), "rank out of range"
    return rank


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    tokenizer: Any,
    alternatives: bool,
) -> _StepPick:
    """Pick the next token and its signals from step logits.

    Confidence is the token's probability under the untempered
    softmax (temperature 1), a faithful "how sure was the model"
    signal independent of the sampling temperature, mirroring
    LLaDA's reveal-time softmax probability. Entropy and the
    candidate set come off that same distribution, so neither the
    sampling temperature nor either truncation distorts any of the
    three: they shape what gets picked, not what gets reported.
    """
    logits = logits.float().squeeze(0)  # (vocab,)
    base_probs = torch.softmax(logits, dim=-1)
    if temperature <= 0.0:
        # Greedy takes the argmax, which no truncation can move:
        # the highest token survives every top-k and every nucleus.
        next_id = int(torch.argmax(logits).item())
    else:
        scaled = torch.softmax(logits / temperature, dim=-1)
        scaled = _top_k_filter(scaled, top_k)
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
    confidence = float(base_probs[next_id].item())
    # A warm temperature reaches past the captured five, and the
    # popover then marks nothing as chosen. The gate is here rather
    # than inside the helper so the common case pays a dict scan and
    # not a reduction over the vocabulary.
    if candidates is not None and not _candidates_hold(
        candidates, next_id
    ):
        candidates.append(
            _chosen_candidate(
                token_id=next_id,
                probability=confidence,
                rank=_token_rank(base_probs, confidence),
                tokenizer=tokenizer,
            )
        )
    return _StepPick(
        token_id=next_id,
        confidence=confidence,
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
        # Decoding is strictly left to right, so the frame that grows
        # the sequence to n tokens is the one that just produced
        # position n-1, and no earlier position can change.
        "revealed": [count - 1] if count else [],
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
    top_k: int,
    alternatives: bool,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    past: Any = None,
) -> Any:
    """Decode up to ``budget`` tokens, emitting one frame each.

    ``step_ids`` is the prefill for the first forward pass (a prompt,
    or a prompt plus an already-committed prefix); afterwards only the
    newest token is forwarded, against a KV cache. ``trace`` carries
    any committed prefix in and is extended in place. Frame indices
    continue from the trace's current length, so a substitution branch
    numbers its frames as a continuation of the original run.

    ``past`` lets a caller hand over a cache it has already built, in
    which case ``step_ids`` is only the part not yet covered by it.
    The substitution path uses this to read the forced position's own
    distribution without paying for a second pass over the prefix.

    Returns the cache as it stands at the end, which is the run's own
    attention state over everything it generated. The caller decides
    whether to keep it (see ``_cache_record``); a later probe against
    it reproduces a sampled probability exactly rather than to within
    a bf16 rounding step.
    """
    assert budget >= 0, "budget must be non-negative"
    trace.check()
    stop_ids = _stop_ids(tokenizer, model)
    device = model.device
    dtype = step_ids.dtype

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
                top_k=top_k,
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
    return past


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
    top_k: int,
    alternatives: bool,
    seed: int,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    result: Dict[str, Any],
) -> None:
    """Blocking token-by-token generation; runs in a worker thread."""
    _seed(seed)
    trace = _Trace()
    past = _stream_tokens(
        model=model,
        tokenizer=tokenizer,
        step_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        trace=trace,
        budget=max_new_tokens,
        total_steps=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alternatives=alternatives,
        out_queue=out_queue,
        cancel_event=cancel_event,
    )
    _finalize(tokenizer, trace, result)
    result["cache"] = _cache_record(
        past, int(inputs["input_ids"].shape[-1]), trace.ids
    )


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
    forced_conf: Optional[float],
    forced_entropy: float,
    forced_alts: Optional[List[Dict[str, Any]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    alternatives: bool,
    seed: int,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    result: Dict[str, Any],
    cache: Optional[Dict[str, Any]] = None,
) -> None:
    """Force one position, then continue generating from it.

    Emits a seed frame covering positions 0..position (the forced
    token last), then streams the re-generated continuation. The
    forced position keeps its originally captured entropy and
    candidate set, since those describe the distribution the model
    faced there, which does not change with the token forced into it.

    ``forced_conf`` is that candidate's recorded probability, or None
    for a token the user typed, which by definition has none. In that
    case it is read from the model instead (see
    ``_probe_forced_position``).
    """
    assert position >= 0, "position must be non-negative"
    assert len(prefix_ids) == position, (
        "prefix must end just before the forced position"
    )
    _seed(seed)

    prompt_ids = inputs["input_ids"]
    probe = _probe_forced_position(
        model=model,
        prompt_ids=prompt_ids,
        prompt_mask=inputs.get("attention_mask"),
        prefix_ids=prefix_ids,
        forced_id=forced_id,
        cache=cache,
    )
    true_conf, forced_rank, past, attention_mask = probe

    trace = _forced_trace(
        prefix_ids=prefix_ids,
        prefix_confs=prefix_confs,
        prefix_entropies=prefix_entropies,
        prefix_alts=prefix_alts,
        forced_id=forced_id,
        forced_conf=forced_conf,
        true_conf=true_conf,
        forced_rank=forced_rank,
        forced_entropy=forced_entropy,
        forced_alts=forced_alts,
        tokenizer=tokenizer,
    )

    _emit_seed_frame(
        tokenizer=tokenizer,
        trace=trace,
        position=position,
        total_steps=max_new_tokens,
        out_queue=out_queue,
    )

    # Only the forced token is left to forward: the probe's cache
    # already covers the prompt and the kept prefix.
    device = model.device
    remaining = max_new_tokens - (position + 1)
    _stream_tokens(
        model=model,
        tokenizer=tokenizer,
        step_ids=torch.tensor(
            [[forced_id]],
            dtype=prompt_ids.dtype,
            device=device,
        ),
        attention_mask=_grow_attention(
            attention_mask, device
        ),
        trace=trace,
        budget=max(0, remaining),
        total_steps=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alternatives=alternatives,
        out_queue=out_queue,
        cancel_event=cancel_event,
        past=past,
    )
    _finalize(tokenizer, trace, result)


def _emit_seed_frame(
    *,
    tokenizer: Any,
    trace: _Trace,
    position: int,
    total_steps: int,
    out_queue: "queue.Queue[Any]",
) -> None:
    """The frame that splices the forced position onto the client.

    Emitted after the probe rather than before it, or it would carry
    no measured confidence for a typed token.

    Its candidates come off the trace rather than from the request's
    ``forced_alts``, so the row the client shows and the row the
    saved run keeps are the same list by construction.
    """
    out_queue.put(
        _build_frame(
            tokenizer,
            trace.ids,
            trace.confs,
            trace.entropies,
            frame_index=position,
            total_steps=total_steps,
            newest_alternatives=trace.alts[position],
        )
    )


def _forced_trace(
    *,
    prefix_ids: List[int],
    prefix_confs: List[float],
    prefix_entropies: List[float],
    prefix_alts: List[Optional[List[Dict[str, Any]]]],
    forced_id: int,
    forced_conf: Optional[float],
    true_conf: float,
    forced_rank: int,
    forced_entropy: float,
    forced_alts: Optional[List[Dict[str, Any]]],
    tokenizer: Any,
) -> _Trace:
    """The branch's trace: the kept prefix plus the forced position.

    Which confidence lands at the forced position is the only real
    decision here. A captured candidate keeps the probability its
    own run recorded, so an existing substitution reads exactly as
    it did before typed tokens existed. A typed token has no such
    record, which is what ``forced_conf`` of None means, and takes
    the value measured off the model instead.

    The forced position also gains a row for the token now sitting in
    it, when the five it kept do not already list one. Membership
    decides that and not ``forced_conf``, because a typed token can
    land on a captured candidate by coincidence.
    """
    trace = _Trace()
    trace.ids = list(prefix_ids)
    trace.confs = list(prefix_confs)
    trace.entropies = list(prefix_entropies)
    trace.alts = list(prefix_alts)
    trace.ids.append(forced_id)
    trace.confs.append(
        true_conf if forced_conf is None else forced_conf
    )
    trace.entropies.append(forced_entropy)
    trace.alts.append(
        _forced_candidates(
            forced_alts,
            forced_id=forced_id,
            true_conf=true_conf,
            forced_rank=forced_rank,
            tokenizer=tokenizer,
        )
    )
    trace.check()
    assert len(trace.ids) == len(prefix_ids) + 1, (
        "the trace must add exactly the forced position"
    )
    return trace


def _forced_candidates(
    forced_alts: Optional[List[Dict[str, Any]]],
    *,
    forced_id: int,
    true_conf: float,
    forced_rank: int,
    tokenizer: Any,
) -> Optional[List[Dict[str, Any]]]:
    """The forced position's candidates, with its own token in them.

    Copied before appending, unlike the generating path: this list
    arrived from the client's record of the original run, and the
    caller holding it is not expecting it to grow underneath them.
    """
    if forced_alts is None:
        return None
    if _candidates_hold(forced_alts, forced_id):
        return forced_alts
    augmented = list(forced_alts)
    augmented.append(
        _chosen_candidate(
            token_id=forced_id,
            probability=true_conf,
            rank=forced_rank,
            tokenizer=tokenizer,
        )
    )
    return augmented


def _probe_forced_position(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    prefix_ids: List[int],
    forced_id: int,
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int, Any, Optional[torch.Tensor]]:
    """Prefill up to the forced position and read its probability.

    Returns the forced token's true probability under the model at
    that position, its rank in that distribution, the KV cache
    covering prompt plus prefix, and the attention mask for it.

    The rank rides along because the distribution is only in hand
    here. It is what the popover's appended row needs to say where a
    forced token stood, and it costs one reduction on the control
    plane rather than one per generated token.

    Splitting the pass at this boundary costs nothing, because the
    caller hands the cache to ``_stream_tokens`` and the forced token
    is forwarded against it instead of being prefilled alongside the
    prefix.
    """
    probs, past, attention_mask = _position_distribution(
        model=model,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        prefix_ids=prefix_ids,
        cache=cache,
    )
    assert 0 <= forced_id < int(probs.numel()), (
        "forced id out of vocabulary range"
    )
    probability = float(probs[forced_id].item())
    return (
        probability,
        _token_rank(probs, probability),
        past,
        attention_mask,
    )


def _position_distribution(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    prefix_ids: List[int],
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
    """The distribution one position past the prefix, plus its cache.

    The pass deliberately stops *before* the position in question, so
    the last logits are the distribution the model would have sampled
    it from. That is the number that makes a typed token interesting:
    it can honestly report that the user forced something the model
    gave 0.003 to.

    Untempered, matching what ``_sample_next`` reports, so a figure
    taken from here is directly comparable to a captured candidate's
    recorded probability rather than living on a different scale.

    ``cache`` is the recorded run's own KV cache, when one was kept.
    With it, the pass reproduces the call the run made (see
    ``_reuse_cache``) and the figure is the run's arithmetic rather
    than a re-derivation of it. Without it, a fresh prefill agrees to
    about three decimals and then parts ways, because bf16 rounds a
    long reduction and an incremental one differently.

    The one choke point both the probe and the substitution reach, so
    the reuse and its fallback live here once.
    """
    assert len(prefix_ids) >= 0, "prefix must be a list"
    device = model.device
    prompt_len = int(prompt_ids.shape[-1])
    attention_mask = _prefill_attention(
        prompt_mask, prompt_len + len(prefix_ids), device
    )
    reused = _reuse_cache(cache, prefix_ids, prompt_len)
    if reused is None:
        outputs = _forward_prefill(
            model, prompt_ids, prefix_ids, attention_mask
        )
    else:
        outputs = _forward_decode(
            model,
            prompt_ids.dtype,
            prefix_ids[-1],
            attention_mask,
            reused,
        )
    probs = torch.softmax(
        outputs.logits[:, -1, :].float().squeeze(0), dim=-1
    )
    assert probs.dim() == 1, "probs must be 1-D"
    return probs, outputs.past_key_values, attention_mask


def _forward_prefill(
    model: Any,
    prompt_ids: torch.Tensor,
    prefix_ids: List[int],
    attention_mask: Optional[torch.Tensor],
) -> Any:
    """One pass over the prompt and the whole kept prefix."""
    step_ids = prompt_ids
    if prefix_ids:
        kept = torch.tensor(
            [prefix_ids],
            dtype=prompt_ids.dtype,
            device=model.device,
        )
        step_ids = torch.cat([prompt_ids, kept], dim=-1)
    with torch.no_grad():
        return model(
            input_ids=step_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )


def _forward_decode(
    model: Any,
    dtype: torch.dtype,
    last_id: int,
    attention_mask: Optional[torch.Tensor],
    past: Any,
) -> Any:
    """One token against a cache, the shape a decode step used."""
    with torch.no_grad():
        return model(
            input_ids=torch.tensor(
                [[last_id]], dtype=dtype, device=model.device
            ),
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )


def _reuse_cache(
    cache: Optional[Dict[str, Any]],
    prefix_ids: List[int],
    prompt_len: int,
) -> Any:
    """A view of the retained cache to forward one token against.

    The call-shape rule, which is the whole content of this: the run
    sampled position 0 from a prompt prefill and every later position
    from a one-token decode against a cache of everything before it.
    So the cache is sliced to ``prompt + n - 1`` and the prefix's last
    token is the token forwarded. Slicing to *n* and forwarding
    nothing, or decoding at position 0, would each reintroduce the
    rounding difference this exists to remove.

    Returns None on every disagreement instead of raising. The cache
    is an accelerator and a precision improvement, not a dependency:
    a stale one costs a forward pass, where trusting it would cost a
    wrong number.
    """
    if cache is None:
        return None
    if not prefix_ids:
        return None
    if cache.get("prompt_len") != prompt_len:
        return None
    ids = cache.get("ids") or []
    if len(ids) < len(prefix_ids):
        return None
    if list(ids[: len(prefix_ids)]) != list(prefix_ids):
        return None
    return _sliced_cache(
        cache.get("past"), prompt_len + len(prefix_ids) - 1
    )


def _sliced_cache(past: Any, length: int) -> Any:
    """A fresh cache over views of the first ``length`` positions.

    Deliberately not ``DynamicCache.crop``, which transformers
    implements in place: the retained cache has to survive for the
    next probe, which may want a different position. Building a new
    cache object over slices leaves the original untouched, and the
    slices are views, so this is pointer work proportional to the
    layer count rather than a copy of tens of megabytes.

    Returns None for a cache that does not speak the legacy tuple
    format, which is the caller's cue to prefill instead.
    """
    assert length >= 0, "length must be non-negative"
    if past is None:
        return None
    to_legacy = getattr(past, "to_legacy_cache", None)
    from_legacy = getattr(
        type(past), "from_legacy_cache", None
    )
    if to_legacy is None or from_legacy is None:
        return None
    layers = to_legacy()
    if not layers:
        return None
    if int(layers[0][0].shape[-2]) < length:
        return None
    sliced = tuple(
        (keys[..., :length, :], values[..., :length, :])
        for keys, values in layers
    )
    return from_legacy(sliced)


# Ceiling on a retained cache, well above what a run produces: 72 KiB
# per token for SmolLM3 (36 layers, 4 KV heads under grouped-query
# attention, head_dim 128) puts a 2048-token run near 148 MiB against
# roughly 6 GiB of weights. Present so an unfamiliar geometry cannot
# quietly pin an unbounded amount of device memory for a session.
AR_CACHE_BYTES_MAX = 512 * 1024 * 1024


def _cache_record(
    past: Any, prompt_len: int, ids: List[int]
) -> Optional[Dict[str, Any]]:
    """The finished run's cache, packaged for later reuse, or None.

    Kept with the ids it covers and the prompt length it starts at,
    because a probe has to prove the cache describes the prefix it is
    asking about before trusting a number that came out of it.

    Dropped rather than trimmed above the ceiling: a run large enough
    to pass it is one whose cache was never worth the residency, and
    a probe against it simply prefills as it did before.
    """
    if past is None:
        return None
    size = _cache_bytes(past)
    if size is None or size > AR_CACHE_BYTES_MAX:
        return None
    return {
        "past": past,
        "prompt_len": prompt_len,
        "ids": list(ids),
    }


def _cache_bytes(past: Any) -> Optional[int]:
    """A cache's tensor footprint, or None when unreadable."""
    to_legacy = getattr(past, "to_legacy_cache", None)
    if to_legacy is None:
        return None
    total = 0
    for keys, values in to_legacy():
        total += keys.numel() * keys.element_size()
        total += values.numel() * values.element_size()
    return total


def probe_token(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    prefix_ids: List[int],
    token_id: int,
    thinking: bool = False,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure what the model gave one token at one position.

    Answers "how wild was that?" for a token the run never picked,
    which is the point of typing your own: the typed row is
    otherwise the only one in the popover with nothing to say, and
    deciding to force a token is more interesting when you can see
    the odds you are overriding first.

    The figure returned is the same one a substitution at this
    position would report, since both read this distribution, so the
    popover cannot promise a number the run then contradicts.

    ``cache`` is the recorded run's KV cache, when one was kept. With
    it, a token the run *did* capture measures to its recorded value
    exactly, because the pass is the one the run made rather than a
    reconstruction of it.

    ``rank`` counts how many tokens the model preferred, and comes
    from a comparison and a sum rather than a sort: the distribution
    is already in hand, so it is free next to the pass that made it.
    It is what keeps a wild choice legible where the probability
    itself has collapsed to a rounded zero.

    ``vocab_size`` is the model's output width, deliberately not the
    tokenizer's ``vocab_size``: the two differ where a checkpoint
    pads its embedding (128,256 against 128,000 for SmolLM3), and
    the rank's denominator is the number of tokens that could have
    been ranked.
    """
    assert token_id >= 0, "token_id must be non-negative"
    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    probs, _past, _mask = _position_distribution(
        model=model,
        prompt_ids=inputs["input_ids"],
        prompt_mask=inputs.get("attention_mask"),
        prefix_ids=prefix_ids,
        cache=cache,
    )
    vocab_size = int(probs.numel())
    if token_id >= vocab_size:
        raise ValueError(
            f"token {token_id} is outside the model's"
            f" {vocab_size}-token output."
        )
    probability = float(probs[token_id].item())
    rank = _token_rank(probs, probability)
    return {
        "probability": probability,
        "rank": rank,
        "vocab_size": vocab_size,
    }


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
    top_k: int = 0,
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
                top_k=top_k,
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
    forced_conf: Optional[float],
    forced_entropy: float,
    forced_alts: Optional[List[Dict[str, Any]]],
    prefix_ids: List[int],
    prefix_confs: List[float],
    prefix_entropies: List[float],
    prefix_alts: List[Optional[List[Dict[str, Any]]]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    thinking: bool = False,
    alternatives: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
    state_sink: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
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

    ``forced_conf`` is None for a token the user typed, which has no
    recorded probability; the true one is measured instead.

    ``cache`` is the recorded run's KV cache, when the worker kept
    one. It removes this operation's dominant cost: without it the
    entire kept prefix is prefilled again before a single new token is
    produced, and every substitution pays that.
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
                top_k=top_k,
                alternatives=alternatives,
                seed=seed,
                out_queue=out_queue,
                cancel_event=cancel_event,
                result=result,
                cache=cache,
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
        # Rides with the trace rather than on its own channel,
        # because it is only meaningful alongside it: the cache and
        # the ids describe the same sequence, and a sink holding one
        # without the other could not check that they agree.
        state_sink["cache"] = result.get("cache")

    yield {
        "type": "done",
        "final_text": result.get("final_text", ""),
        "thinking": result.get("thinking", ""),
    }
