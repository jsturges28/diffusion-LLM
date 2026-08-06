"""Streaming DiffusionGemma sampler.

Wraps ``model.generate`` (which drives a ``BaseStreamer`` via
synchronous ``put`` / ``put_draft`` callbacks) as an async
generator that yields frames in the shared visualizer protocol,
matching the LLaDA worker's frame shape.

Unresolved tokens are detected by stability: DiffusionGemma
renoises non-accepted canvas positions to fresh random tokens, so
a position that changed since the previous denoising step is
treated as unresolved (rendered as the mask glyph); a position
that has stabilized is treated as resolved.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer

from src.inference.reveal import newly_revealed

MASK_CHAR = "\u2591"

# Consecutive stable steps for a proxy-confidence of 1.0.
STABLE_WINDOW = 3

# Control/structure tokens hidden from the per-token display.
_STRIP_TOKENS = (
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<end_of_turn>",
    "<start_of_turn>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|think|>",
)

_CHANNEL_CLOSE = "<channel|>"


def _sanitize(text: str) -> str:
    for token in _STRIP_TOKENS:
        text = text.replace(token, "")
    return text


def _split_thinking(raw: str) -> tuple[str, str]:
    """Separate the reasoning channel from the final answer.

    DiffusionGemma emits ``<|channel>thought\\n ... <channel|>``
    before the answer. Returns (thinking, answer), both cleaned of
    control tokens.
    """
    if _CHANNEL_CLOSE in raw:
        head, _, answer = raw.rpartition(_CHANNEL_CLOSE)
        if "<|channel>" in head:
            head = head.split("<|channel>", 1)[1]
        head = head.replace("thought", "", 1)
        return _sanitize(head).strip(), _sanitize(answer).strip()
    return "", _sanitize(raw).strip()


class FrameQueueStreamer(BaseStreamer):
    """Turns generate's streamer callbacks into protocol frames.

    ``put`` receives the prompt first (skipped) and then each
    committed canvas; ``put_draft`` receives every intermediate
    denoising canvas.
    """

    def __init__(
        self, tokenizer: Any, out_queue: "queue.Queue[Any]"
    ) -> None:
        self.tokenizer = tokenizer
        self._queue = out_queue
        self._prev: Optional[List[int]] = None
        self._stable: Optional[List[int]] = None
        # Positions already reported as born on this canvas. Unlike
        # LLaDA, a draft token here can settle, change again, and
        # settle a second time; without this the same position would
        # be reported born repeatedly and flicker.
        self._seen_revealed: set[int] = set()
        self._index = 0
        self._canvas_index = 0
        self._prompt_seen = False
        # When True, put_draft receives logits (huge on a 262K
        # vocab) instead of tokens, enabling true entropy/conf.
        self._takes_logits = False

    @staticmethod
    def _canvas_ids(value: torch.Tensor) -> List[int]:
        tensor = value
        if hasattr(tensor, "dim") and tensor.dim() > 1:
            tensor = tensor[0]
        return tensor.detach().to("cpu").tolist()

    @staticmethod
    def _from_logits(
        logits: torch.Tensor,
    ) -> tuple[List[int], List[float]]:
        tensor = logits
        if hasattr(tensor, "dim") and tensor.dim() > 2:
            tensor = tensor[0]
        probs = torch.softmax(tensor.float(), dim=-1)
        conf, ids = probs.max(dim=-1)
        return (
            ids.detach().to("cpu").tolist(),
            conf.detach().to("cpu").tolist(),
        )

    def _emit(
        self,
        ids: List[int],
        *,
        committed: bool,
        conf_override: Optional[List[float]] = None,
    ) -> None:
        count = len(ids)
        if self._stable is None or len(self._stable) != count:
            self._stable = [0] * count
        tokens: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        resolved: List[bool] = []
        conf_sum = 0.0
        for i, token_id in enumerate(ids):
            changed = (
                self._prev is None
                or i >= len(self._prev)
                or self._prev[i] != token_id
            )
            unresolved = (not committed) and changed
            if changed:
                self._stable[i] = 0
            else:
                self._stable[i] += 1
            if conf_override is not None:
                conf = float(conf_override[i])
            elif committed:
                conf = 1.0
            else:
                conf = min(
                    self._stable[i] / STABLE_WINDOW, 1.0
                )
            conf_sum += conf
            display = _sanitize(
                self.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
            )
            token: Dict[str, Any] = {
                "t": display,
                "m": unresolved,
                "id": int(token_id),
            }
            if not unresolved:
                token["c"] = round(conf, 4)
            tokens.append(token)
            resolved.append(not unresolved)
            text_parts.append(
                MASK_CHAR if unresolved else display
            )
        born = newly_revealed(resolved, self._seen_revealed)
        self._seen_revealed.update(born)
        self._prev = ids
        self._queue.put(
            {
                "type": "frame",
                "index": self._index,
                "total_steps": None,
                "canvas_index": self._canvas_index,
                "mean_conf": (
                    round(conf_sum / count, 4) if count else 0.0
                ),
                "text": "".join(text_parts),
                "tokens": tokens,
                "revealed": born,
            }
        )
        self._index += 1

    def put(self, value: torch.Tensor) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self._emit(self._canvas_ids(value), committed=True)
        # Next canvas restarts from fresh noise, so its positions are
        # unrelated to this one's and start unborn again.
        self._prev = None
        self._stable = None
        self._seen_revealed = set()
        self._canvas_index += 1

    def put_draft(
        self,
        value: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        if logits is not None:
            ids, conf = self._from_logits(logits)
            self._emit(
                ids, committed=False, conf_override=conf
            )
        elif value is not None:
            self._emit(
                self._canvas_ids(value), committed=False
            )

    def end(self) -> None:
        self._queue.put(None)


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


async def _run_streamed(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Any,
    prompt_len: int,
    streamer: "FrameQueueStreamer",
    out_queue: "queue.Queue[Any]",
    generate_kwargs: Dict[str, Any],
    seed: int,
    cancel_event: Optional[asyncio.Event],
    frame_history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Drive ``model.generate`` in a thread, yielding protocol frames.

    Shared by ``streaming_generate`` and ``streaming_resume``. When
    ``frame_history`` is provided, each streamed frame's canvas token
    ids and canvas index are recorded so a worker can later re-enter a
    chosen frame (resume-from-frame).
    """
    result: Dict[str, Any] = {}

    def run() -> None:
        try:
            _seed(seed)
            result["out"] = model.generate(
                **inputs, streamer=streamer, **generate_kwargs
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
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                break
            if (
                frame_history is not None
                and item.get("type") == "frame"
            ):
                frame_history.append(
                    {
                        "canvas_index": item.get(
                            "canvas_index", 0
                        ),
                        "ids": [
                            tok["id"]
                            for tok in item["tokens"]
                        ],
                    }
                )
            yield item
    finally:
        await task

    if "err" in result:
        raise result["err"]

    output = result.get("out")
    final_text = ""
    thinking_text = ""
    if output is not None:
        sequences = getattr(output, "sequences", output)
        raw = tokenizer.decode(
            sequences[0][prompt_len:],
            skip_special_tokens=False,
        )
        thinking_text, final_text = _split_thinking(raw)
    yield {
        "type": "done",
        "final_text": final_text,
        "thinking": thinking_text,
        # The templated length this run actually built, so the saved
        # run records a measurement rather than the client's estimate.
        "prompt_len": prompt_len,
    }


async def streaming_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    max_denoising_steps: int = 48,
    t_max: float = 0.8,
    t_min: float = 0.4,
    thinking: bool = False,
    entropy_signal: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
    frame_history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield one dict per denoising step, then a ``done`` frame.

    When ``frame_history`` is provided, per-frame canvas ids and
    canvas indices are appended to it so the worker can support
    resume-from-frame.
    """
    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    out_queue: "queue.Queue[Any]" = queue.Queue()
    streamer = FrameQueueStreamer(tokenizer, out_queue)
    streamer._takes_logits = entropy_signal

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "max_denoising_steps": max_denoising_steps,
        "t_max": t_max,
        "t_min": t_min,
    }
    async for item in _run_streamed(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        prompt_len=prompt_len,
        streamer=streamer,
        out_queue=out_queue,
        generate_kwargs=generate_kwargs,
        seed=seed,
        cancel_event=cancel_event,
        frame_history=frame_history,
    ):
        yield item


async def streaming_resume(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    seed_canvas_ids: List[int],
    remask_positions: List[int],
    remaining_steps: int,
    t_max: float = 0.8,
    t_min: float = 0.4,
    thinking: bool = False,
    entropy_signal: bool = False,
    seed: int = -1,
    cancel_event: Optional[asyncio.Event] = None,
    frame_history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Resume denoising from a chosen frame's canvas (single canvas).

    Re-enters the diffusion process by seeding ``generate`` with the
    frame's canvas via ``decoder_input_ids`` and running a reduced
    ``max_denoising_steps`` budget. User-remasked positions are
    *renoised* (set to fresh random tokens) so the denoiser re-decides
    them; unlike LLaDA, non-remasked positions are biased by the seed
    but not frozen. ``max_new_tokens`` is pinned to a single canvas so
    the resume never chains additional canvases.
    """
    assert remaining_steps > 0, "remaining_steps must be positive"
    canvas_length = int(model.config.canvas_length)
    if len(seed_canvas_ids) != canvas_length:
        raise ValueError(
            f"seed canvas has {len(seed_canvas_ids)} ids,"
            f" expected {canvas_length}"
        )

    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    _seed(seed)
    vocab_size = int(model.config.text_config.vocab_size)
    seed_canvas = torch.tensor(
        seed_canvas_ids, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    for pos in remask_positions:
        if not 0 <= pos < canvas_length:
            raise ValueError(
                f"remask position {pos} out of range"
                f" [0, {canvas_length})"
            )
        seed_canvas[0, pos] = int(
            torch.randint(0, vocab_size, (1,)).item()
        )

    out_queue: "queue.Queue[Any]" = queue.Queue()
    streamer = FrameQueueStreamer(tokenizer, out_queue)
    streamer._takes_logits = entropy_signal

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": canvas_length,
        "max_denoising_steps": remaining_steps,
        "t_max": t_max,
        "t_min": t_min,
        "decoder_input_ids": seed_canvas,
    }
    async for item in _run_streamed(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        prompt_len=prompt_len,
        streamer=streamer,
        out_queue=out_queue,
        generate_kwargs=generate_kwargs,
        seed=seed,
        cancel_event=cancel_event,
        frame_history=frame_history,
    ):
        yield item
