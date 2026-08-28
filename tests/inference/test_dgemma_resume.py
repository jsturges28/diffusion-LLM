"""Tests that a resumed DiffusionGemma canvas keeps its state.

Strategy: the same stub ``generate`` the cancel tests use, driving
the real streamer and the real ``_run_streamed`` on CPU with no
checkpoint. The assertions read protocol frames, because that is
where the defects showed.

DiffusionGemma has no mask token. A position reads as unresolved
when it changed since the previous step, and its confidence is how
many consecutive steps it has held. A resume built a fresh
streamer, so all of that state started empty and three things went
wrong at once: with no previous canvas every position counted as
changed, so the first resumed frame rendered as an entirely masked
canvas; confidence restarted from zero for tokens that had been
stable for many steps; and the inherited prefix was reported born a
second time, which is the same rebirth LLaDA's resume fixes
deliberately.

Passing proves the streamer can re-enter a recorded frame with the
state that produced it, that checkpoints are recorded only for
frames the consumer actually received, and that an unsettled token
carries confidence exactly when that confidence is real.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Dict, List, Optional

import torch

from src.inference.checkpoint import (
    CheckpointBudget,
    DgemmaFrame,
    FrameCheckpoint,
)
from src.inference.dgemma_sampler import (
    STABLE_WINDOW,
    FrameQueueStreamer,
    _run_streamed,
)
from src.inference.frame_queue import (
    FrameQueueCancelled,
    frame_queue_create,
)

CANVAS_LENGTH = 4

# The canvas a resume inherits, and a stability state saying every
# position has been settled long enough to read as fully confident.
SETTLED_IDS = [11, 12, 13, 14]
SETTLED_STABLE = (STABLE_WINDOW,) * CANVAS_LENGTH

assert len(SETTLED_IDS) == CANVAS_LENGTH, "the canvas is full"
assert min(SETTLED_STABLE) >= STABLE_WINDOW, (
    "the inherited canvas is fully settled, so any confidence"
    " below 1.0 after a resume comes from losing that state"
)


class _StubTokenizer:
    def decode(
        self, ids: Any, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, torch.Tensor):
            values = ids.tolist()
        else:
            values = list(ids)
        return "".join(
            chr(ord("a") + (int(i) % 26)) for i in values
        )


def _canvas(values: List[int]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long)


class _StubModel:
    """Redraws the inherited canvas unchanged for a few steps."""

    def __init__(self, steps: int = 3) -> None:
        self.steps = steps
        self.device = "cpu"

    def generate(self, *, streamer: Any, **kwargs: Any) -> Any:
        streamer.put(_canvas([0] * CANVAS_LENGTH))
        for _ in range(self.steps):
            streamer.put_draft(value=_canvas(SETTLED_IDS))
        streamer.end()
        return _canvas(SETTLED_IDS)


def _checkpoint() -> FrameCheckpoint:
    return FrameCheckpoint(
        ids=torch.tensor(SETTLED_IDS, dtype=torch.long),
        canvas_index=0,
        rng=None,
        extra=DgemmaFrame(
            stable=SETTLED_STABLE,
            seen_revealed=frozenset(range(CANVAS_LENGTH)),
        ),
    )


def _drive(
    *,
    restore: bool,
    entropy_signal: bool = False,
    history: Optional[List[FrameCheckpoint]] = None,
) -> List[Dict[str, Any]]:
    """Stream a short run, optionally re-entering a recorded frame."""
    out_queue = frame_queue_create()
    stop = threading.Event()
    budget = None if history is None else CheckpointBudget()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        out_queue,
        stop_event=stop,
        budget=budget,
    )
    streamer._takes_logits = entropy_signal
    if restore:
        streamer.restore(_checkpoint())
    frames: List[Dict[str, Any]] = []

    async def drive() -> None:
        generator = _run_streamed(
            model=_StubModel(),
            tokenizer=_StubTokenizer(),
            inputs={},
            prompt_len=0,
            streamer=streamer,
            out_queue=out_queue,
            generate_kwargs={},
            seed=-1,
            cancel_event=stop,
            frame_history=history,
        )
        async for frame in generator:
            frames.append(frame)

    asyncio.run(drive())
    return [f for f in frames if f.get("type") == "frame"]


# -- re-entering a recorded frame --


def test_a_fresh_streamer_masks_the_whole_inherited_canvas(
) -> None:
    """The defect, stated as the baseline the fix moves off."""
    first = _drive(restore=False)[0]

    assert all(tok["m"] for tok in first["tokens"])


def test_a_restored_streamer_shows_the_canvas_it_inherited(
) -> None:
    first = _drive(restore=True)[0]

    assert not any(tok["m"] for tok in first["tokens"])


def test_a_restored_canvas_keeps_the_confidence_it_earned(
) -> None:
    """Without the stability state these read 0.0 and climb, which
    reports a settled canvas as freshly uncertain."""
    first = _drive(restore=True)[0]

    assert all(tok["c"] == 1.0 for tok in first["tokens"])
    assert first["mean_conf"] == 1.0


def test_a_fresh_canvas_starts_from_no_confidence_at_all() -> None:
    """The other half of the same baseline: the numbers a resumed
    run reported before the state was carried over."""
    second = _drive(restore=False)[1]

    assert all(tok["c"] < 1.0 for tok in second["tokens"])


def test_a_restored_prefix_is_not_born_a_second_time() -> None:
    frames = _drive(restore=True)

    assert all(f["revealed"] == [] for f in frames)


def test_a_fresh_prefix_is_born_and_proves_the_contrast() -> None:
    """Negative space: without the seen-born set the same canvas
    does report births, so the assertion above is measuring the
    restore rather than a stub that never births anything."""
    born = [f["revealed"] for f in _drive(restore=False)]

    assert any(len(positions) > 0 for positions in born)


# -- what gets recorded --


def test_every_delivered_frame_leaves_a_checkpoint() -> None:
    history: List[FrameCheckpoint] = []
    frames = _drive(restore=True, history=history)

    assert len(history) == len(frames)
    assert history[0].ids.tolist() == SETTLED_IDS


def test_a_checkpoint_carries_the_state_of_its_frame() -> None:
    history: List[FrameCheckpoint] = []
    _drive(restore=True, history=history)
    last = history[-1].extra

    assert isinstance(last, DgemmaFrame)
    assert len(last.stable) == CANVAS_LENGTH
    assert max(last.stable) >= STABLE_WINDOW


def test_a_checkpoint_exists_before_its_frame_is_handed_over(
) -> None:
    """The ordering the consumer's claim depends on.

    Recording after the hand-off is a race rather than a style
    choice: the consumer runs on the event loop and can dequeue the
    instant a frame lands, so it can ask for a checkpoint the
    producer thread has not written yet. It fails about one run in
    a hundred, which is exactly often enough to be dismissed as
    flakiness. Asserting inside ``put`` pins the ordering without
    depending on a thread interleaving.
    """
    recorded: List[bool] = []

    class _CheckingQueue(queue.Queue):  # type: ignore[type-arg]
        def put(
            self, item: Any, block: bool = True, timeout: Any = None
        ) -> None:
            index = item.get("index")
            recorded.append(index in streamer._checkpoints)
            super().put(item, block, timeout)

    out_queue = _CheckingQueue(maxsize=8)
    streamer = FrameQueueStreamer(
        _StubTokenizer(), out_queue, budget=CheckpointBudget()
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    streamer.put_draft(value=_canvas(SETTLED_IDS))
    streamer.put_draft(value=_canvas(SETTLED_IDS))

    assert recorded == [True, True]


def test_an_undelivered_frame_leaves_no_checkpoint() -> None:
    """The other half of moving the record earlier: a frame the
    consumer never receives must not stay claimable, or a resume
    could name a frame index the browser does not have."""
    stop = threading.Event()
    stop.set()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        frame_queue_create(),
        stop_event=stop,
        budget=CheckpointBudget(),
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))

    try:
        streamer.put_draft(value=_canvas(SETTLED_IDS))
    except FrameQueueCancelled:
        pass
    else:
        raise AssertionError("a stopped run must unwind")

    assert len(streamer._checkpoints) == 0


def test_nothing_is_recorded_when_nobody_is_collecting() -> None:
    """A generation with no history sink must not accumulate
    checkpoints nobody will ever claim."""
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(_StubTokenizer(), out_queue)

    streamer.put(_canvas([0] * CANVAS_LENGTH))
    streamer.put_draft(value=_canvas(SETTLED_IDS))

    assert len(streamer._checkpoints) == 0


def test_a_claimed_checkpoint_is_released() -> None:
    """Popped rather than read, so the streamer holds only the
    frames still in flight and cannot grow across a long run."""
    history: List[FrameCheckpoint] = []
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(
        _StubTokenizer(), out_queue, budget=CheckpointBudget()
    )

    async def drive() -> None:
        generator = _run_streamed(
            model=_StubModel(),
            tokenizer=_StubTokenizer(),
            inputs={},
            prompt_len=0,
            streamer=streamer,
            out_queue=out_queue,
            generate_kwargs={},
            seed=-1,
            cancel_event=None,
            frame_history=history,
        )
        async for _ in generator:
            pass

    asyncio.run(drive())

    assert len(history) > 0
    assert len(streamer._checkpoints) == 0


# -- confidence on unsettled positions --


def test_an_unsettled_token_carries_confidence_from_logits(
) -> None:
    """With the Entropy Signal on, the number is the model's own
    probability, which is what fades a mask by how sure the model
    is of the guess underneath it."""
    streamer, out_queue = _bare_streamer(entropy_signal=True)
    streamer.put_draft(logits=_logits())

    tokens = out_queue.get()["tokens"]
    assert all(tok["m"] for tok in tokens)
    assert all("c" in tok for tok in tokens)


def test_an_unsettled_token_carries_nothing_without_logits(
) -> None:
    """Off, an unsettled position just changed, so its stability
    count was reset and the value could only ever be 0.0. The
    client treats zero and absent alike, so it is not written."""
    streamer, out_queue = _bare_streamer(entropy_signal=False)
    streamer.put_draft(value=_canvas(SETTLED_IDS))

    tokens = out_queue.get()["tokens"]
    assert all(tok["m"] for tok in tokens)
    assert not any("c" in tok for tok in tokens)


def test_a_settled_token_carries_confidence_either_way() -> None:
    """The pre-existing behaviour, pinned so the gate above cannot
    be widened into one that drops it."""
    streamer, out_queue = _bare_streamer(entropy_signal=False)
    streamer.put(_canvas(SETTLED_IDS))

    tokens = out_queue.get()["tokens"]
    assert all("c" in tok for tok in tokens)


def _bare_streamer(
    *, entropy_signal: bool
) -> tuple[FrameQueueStreamer, Any]:
    """A streamer past its prompt, ready to emit.

    ``put`` swallows the first canvas it sees, which is the prompt
    echo transformers hands every streamer. A test that skipped
    this would block forever on an empty queue.
    """
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(_StubTokenizer(), out_queue)
    streamer._takes_logits = entropy_signal
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    assert out_queue.qsize() == 0, "the prompt echo is not a frame"
    return streamer, out_queue


def _logits() -> torch.Tensor:
    """Per-position logits favouring one id, so conf is real."""
    values = torch.zeros(1, CANVAS_LENGTH, 32)
    for position, token_id in enumerate(SETTLED_IDS):
        values[0, position, token_id] = 4.0
    return values
