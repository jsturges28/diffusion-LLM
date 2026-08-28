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
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer

from src.backends.protocol import TERMINAL_CANCELLED
from src.inference.checkpoint import (
    CheckpointBudget,
    DgemmaFrame,
    FrameCheckpoint,
    RngState,
    rng_capture,
    rng_restore,
)
from src.inference.frame_queue import (
    FrameQueueCancelled,
    frame_queue_close,
    frame_queue_create,
    frame_queue_drain_until_done,
    frame_queue_put,
)
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

    It is also where a cancelled run is stopped. ``generate``
    consults an externally supplied stopping criterion only once
    per canvas, so a criterion cannot interrupt a single-canvas
    run at all, while ``put_draft`` lands on every denoising step.
    Raising from here is therefore the only hook with the
    granularity a user expects from pressing Stop.
    """

    def __init__(
        self,
        tokenizer: Any,
        out_queue: "queue.Queue[Any]",
        stop_event: Optional[threading.Event] = None,
        budget: Optional[CheckpointBudget] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._queue = out_queue
        self._stop_event = stop_event
        # Present only when a caller intends to collect checkpoints.
        # Without it nothing is recorded, so a run whose frames are
        # never claimed cannot accumulate them.
        self._budget = budget
        self._checkpoints: Dict[int, FrameCheckpoint] = {}
        # The most recent canvas as text, kept so a cancelled run
        # can still report what it produced: ``generate`` returns
        # nothing when it is unwound, and the alternative is a
        # terminal frame claiming the run produced no text at all.
        self.last_text = ""
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
            # Settled tokens always carry confidence. Unsettled ones
            # carry it only when it is real: with the Entropy Signal
            # off, an unsettled position is by definition one that
            # just changed, so its stability count was reset to zero
            # a few lines up and the number could only ever be 0.0.
            # The client's mask opacity treats zero and absent alike,
            # so writing it would cost payload on every frame for an
            # identical canvas. With the signal on it is the model's
            # own probability, which is what fades each mask by how
            # sure the model is of the guess underneath it.
            if not unresolved or conf_override is not None:
                token["c"] = round(conf, 4)
            tokens.append(token)
            resolved.append(not unresolved)
            text_parts.append(
                MASK_CHAR if unresolved else display
            )
        born = newly_revealed(resolved, self._seen_revealed)
        self._seen_revealed.update(born)
        self._prev = ids
        text = "".join(text_parts)
        self.last_text = text
        # Recorded before the hand-off, never after. The consumer
        # runs on the event loop and can dequeue a frame the instant
        # it lands, so recording afterwards is a race the producer
        # loses roughly one run in a hundred: the frame arrives, the
        # consumer claims its checkpoint, and the thread has not
        # written it yet.
        self._record_checkpoint(ids)
        delivered = frame_queue_put(
            self._queue,
            {
                "type": "frame",
                "index": self._index,
                "total_steps": None,
                "canvas_index": self._canvas_index,
                "mean_conf": (
                    round(conf_sum / count, 4) if count else 0.0
                ),
                "text": text,
                "tokens": tokens,
                "revealed": born,
            },
            stop_event=self._stop_event,
        )
        if not delivered:
            # Undelivered, so unclaimable. Dropping it here is what
            # keeps the recorded history to the frames the client
            # actually saw, which is the invariant a later resume
            # depends on when it names a frame by index.
            self._checkpoints.pop(self._index, None)
            raise FrameQueueCancelled(
                "denoising stopped: nobody is reading"
            )
        self._index += 1

    def _record_checkpoint(self, ids: List[int]) -> None:
        """Stash what a resume would need to re-enter this frame.

        Keyed by the frame index because the producer runs ahead of
        the consumer by up to the queue's depth, and the consumer is
        the one that decides which frames the run may remember.
        """
        if self._budget is None:
            return
        assert self._stable is not None, "a frame set stability"
        self._checkpoints[self._index] = FrameCheckpoint(
            ids=torch.tensor(ids, dtype=torch.long),
            canvas_index=self._canvas_index,
            rng=self._budget.capture_rng(),
            extra=DgemmaFrame(
                stable=tuple(self._stable),
                seen_revealed=frozenset(self._seen_revealed),
            ),
        )

    def take_checkpoint(self, index: int) -> FrameCheckpoint:
        """Hand over one frame's checkpoint and forget it here.

        Popped rather than read so the streamer holds at most the
        frames still in flight, and so a second claim on the same
        index fails loudly instead of returning a stale record.
        """
        checkpoint = self._checkpoints.pop(index, None)
        assert checkpoint is not None, (
            f"frame {index} was delivered without a checkpoint"
        )
        return checkpoint

    def restore(self, checkpoint: FrameCheckpoint) -> None:
        """Re-enter a frame with the state that produced it.

        Without this a resumed run builds a fresh streamer, whose
        empty ``_prev`` makes every position read as changed: the
        first resumed frame renders as an entirely masked canvas,
        confidence restarts from zero for tokens that had been
        stable for many steps, and the inherited prefix is reported
        born a second time.
        """
        extra = checkpoint.extra
        assert isinstance(extra, DgemmaFrame), (
            "a DiffusionGemma checkpoint carries stability"
        )
        self._prev = checkpoint.ids.tolist()
        self._stable = list(extra.stable)
        self._seen_revealed = set(extra.seen_revealed)
        assert len(self._stable) == len(self._prev), (
            "stability covers the canvas it was taken from"
        )

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
        frame_queue_close(self._queue)


def _seed(seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enter_rng(seed: int, rng: Optional[RngState]) -> None:
    """Put the generator where this run should start drawing.

    A resume enters from the frame's own captured state, because
    re-seeding would reproduce the run's *first* step rather than
    the step the user chose to branch from. The seed remains the
    entry point for a fresh generation, and for a resume whose
    frame outran the checkpoint budget.
    """
    if rng is not None:
        rng_restore(rng)
        return
    _seed(seed)


def _budget_for(
    frame_history: Optional[List[FrameCheckpoint]],
) -> Optional[CheckpointBudget]:
    """A budget only when someone will collect the checkpoints.

    Both the recording and the collecting hang off the same
    condition, so a run cannot record frames nobody claims.
    """
    if frame_history is None:
        return None
    return CheckpointBudget()


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
    cancel_event: Optional[threading.Event],
    frame_history: Optional[List[FrameCheckpoint]] = None,
    rng: Optional[RngState] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Drive ``model.generate`` in a thread, yielding protocol frames.

    Shared by ``streaming_generate`` and ``streaming_resume``. When
    ``frame_history`` is provided, each streamed frame's checkpoint is
    recorded so a worker can later re-enter a chosen frame
    (resume-from-frame).

    ``rng`` re-enters a captured generator state instead of seeding,
    which is what a resume passes.
    """
    result: Dict[str, Any] = {}

    def run() -> None:
        try:
            _enter_rng(seed, rng)
            result["out"] = model.generate(
                **inputs, streamer=streamer, **generate_kwargs
            )
        except FrameQueueCancelled:
            # An outcome, not a failure: the streamer unwound
            # generate because the run was cancelled. Recorded so
            # the terminal frame can say so rather than reporting
            # an error the user did not cause.
            result["cancelled"] = True
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc
        finally:
            frame_queue_close(out_queue)

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
                    streamer.take_checkpoint(item["index"])
                )
            yield item
    finally:
        await frame_queue_drain_until_done(out_queue, task)

    if "err" in result:
        raise result["err"]

    yield _terminal_frame(
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        streamer=streamer,
        result=result,
        cancel_event=cancel_event,
    )


def _terminal_frame(
    *,
    tokenizer: Any,
    prompt_len: int,
    streamer: "FrameQueueStreamer",
    result: Dict[str, Any],
    cancel_event: Optional[threading.Event],
) -> Dict[str, Any]:
    """The one ``done`` a run ends with, finished or stopped."""
    output = result.get("out")
    final_text = ""
    thinking_text = ""
    if output is None:
        # A stopped run never returns sequences, so the text comes
        # off the last canvas the streamer built rather than being
        # reported as nothing produced.
        final_text = streamer.last_text
    else:
        sequences = getattr(output, "sequences", output)
        raw = tokenizer.decode(
            sequences[0][prompt_len:],
            skip_special_tokens=False,
        )
        thinking_text, final_text = _split_thinking(raw)
    done: Dict[str, Any] = {
        "type": "done",
        "final_text": final_text,
        "thinking": thinking_text,
        # The templated length this run actually built, so the saved
        # run records a measurement rather than the client's estimate.
        "prompt_len": prompt_len,
    }
    # Either the streamer unwound generate, or the consumer stopped
    # forwarding first; both mean the user stopped this run.
    stopped = result.get("cancelled", False) or (
        cancel_event is not None and cancel_event.is_set()
    )
    if stopped:
        done[TERMINAL_CANCELLED] = True
    return done


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
    cancel_event: Optional[threading.Event] = None,
    frame_history: Optional[List[FrameCheckpoint]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield one dict per denoising step, then a ``done`` frame.

    When ``frame_history`` is provided, each frame's checkpoint is
    appended to it so the worker can support resume-from-frame.
    """
    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    out_queue: "queue.Queue[Any]" = frame_queue_create()
    streamer = FrameQueueStreamer(
        tokenizer,
        out_queue,
        stop_event=cancel_event,
        budget=_budget_for(frame_history),
    )
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
    base: FrameCheckpoint,
    remask_positions: List[int],
    remaining_steps: int,
    t_max: float = 0.8,
    t_min: float = 0.4,
    thinking: bool = False,
    entropy_signal: bool = False,
    seed: int = -1,
    cancel_event: Optional[threading.Event] = None,
    frame_history: Optional[List[FrameCheckpoint]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Resume denoising from a chosen frame's canvas (single canvas).

    Re-enters the diffusion process by seeding ``generate`` with the
    frame's canvas via ``decoder_input_ids`` and running a reduced
    ``max_denoising_steps`` budget. User-remasked positions are
    *renoised* (set to fresh random tokens) so the denoiser re-decides
    them; unlike LLaDA, non-remasked positions are biased by the seed
    but not frozen. ``max_new_tokens`` is pinned to a single canvas so
    the resume never chains additional canvases.

    ``base`` is the chosen frame's checkpoint. Its canvas is the seed,
    its stability state is handed to the new streamer so the resumed
    frames continue the confidence the user was looking at, and its
    random state is what the renoise and the denoiser draw from.
    """
    assert remaining_steps > 0, "remaining_steps must be positive"
    canvas_length = int(model.config.canvas_length)
    seed_canvas_ids: List[int] = base.ids.tolist()
    if len(seed_canvas_ids) != canvas_length:
        raise ValueError(
            f"seed canvas has {len(seed_canvas_ids)} ids,"
            f" expected {canvas_length}"
        )

    inputs = _build_inputs(
        tokenizer, model, prompt, thinking=thinking
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    # The renoise below draws from the frame's own state, so the same
    # edit repeated on the same frame renoises to the same tokens and
    # the denoiser then continues from a matching offset.
    _enter_rng(seed, base.rng)
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

    # Pinned after the renoise rather than left implicit, so the
    # denoiser enters from a known point even if something else draws
    # from the generator before the worker thread starts.
    entry_rng = rng_capture()

    out_queue: "queue.Queue[Any]" = frame_queue_create()
    streamer = FrameQueueStreamer(
        tokenizer,
        out_queue,
        stop_event=cancel_event,
        budget=_budget_for(frame_history),
    )
    streamer._takes_logits = entropy_signal
    streamer.restore(base)

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
        rng=entry_rng,
    ):
        yield item
