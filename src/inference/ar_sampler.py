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

Note on payload size: full-snapshot frames make the streamed
payload O(n^2) in the token count. This is fine for the few-hundred
token budgets used here (see the registry's recommended cap) and is
revisited only if long AR runs are added.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

import torch

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


def _sample_next(
    logits: torch.Tensor, *, temperature: float, top_p: float
) -> Tuple[int, float]:
    """Pick the next token id and its confidence from step logits.

    Confidence is the token's probability under the untempered
    softmax (temperature 1), a faithful "how sure was the model"
    signal independent of the sampling temperature, mirroring
    LLaDA's reveal-time softmax probability.
    """
    logits = logits.float().squeeze(0)  # (vocab,)
    base_probs = torch.softmax(logits, dim=-1)
    if temperature <= 0.0:
        next_id = int(torch.argmax(logits).item())
    else:
        scaled = torch.softmax(logits / temperature, dim=-1)
        scaled = _top_p_filter(scaled, top_p)
        next_id = int(torch.multinomial(scaled, 1).item())
    confidence = float(base_probs[next_id].item())
    return next_id, confidence


def _build_frame(
    tokenizer: Any,
    ids: List[int],
    confs: List[float],
    *,
    frame_index: int,
    total_steps: int,
) -> Dict[str, Any]:
    """Build one full-snapshot protocol frame for the growing sequence."""
    assert len(ids) == len(confs), "ids/confs length mismatch"
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
            }
        )
        text_parts.append(display)
    count = len(ids)
    return {
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


def _decode_loop(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    out_queue: "queue.Queue[Any]",
    cancel_event: Optional[asyncio.Event],
    result: Dict[str, Any],
) -> None:
    """Blocking token-by-token loop; runs in a worker thread.

    Pushes one frame dict per generated token onto ``out_queue`` and
    records the final decoded text in ``result``. Uses a KV cache so
    each step forwards only the newest token. Checks ``cancel_event``
    between tokens so a cancel stops within one step.
    """
    _seed(seed)
    stop_ids = _stop_ids(tokenizer, model)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    device = model.device

    generated_ids: List[int] = []
    generated_confs: List[float] = []
    past: Any = None
    step_ids = input_ids

    with torch.no_grad():
        for frame_index in range(max_new_tokens):
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
            next_id, confidence = _sample_next(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
            )
            generated_ids.append(next_id)
            generated_confs.append(confidence)
            out_queue.put(
                _build_frame(
                    tokenizer,
                    generated_ids,
                    generated_confs,
                    frame_index=frame_index,
                    total_steps=max_new_tokens,
                )
            )
            if next_id in stop_ids:
                break
            # Next step forwards only the new token; grow the mask to
            # cover prompt + all tokens produced so far.
            step_ids = torch.tensor(
                [[next_id]], dtype=input_ids.dtype, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
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

    raw = tokenizer.decode(
        generated_ids, skip_special_tokens=False
    )
    thinking_text, final_text = _split_thinking(raw)
    result["final_text"] = final_text
    result["thinking"] = thinking_text


async def streaming_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.95,
    thinking: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield one frame per generated token, then a ``done`` frame.

    Runs the blocking decode loop in a thread and forwards frames as
    they arrive, matching the diffusion workers' async-generator
    contract so the shared ``FrameStreamer`` can drive it unchanged.
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
                seed=seed,
                out_queue=out_queue,
                cancel_event=cancel_event,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc
        finally:
            out_queue.put(None)

    task = asyncio.create_task(asyncio.to_thread(run))
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

    yield {
        "type": "done",
        "final_text": result.get("final_text", ""),
        "thinking": result.get("thinking", ""),
    }
