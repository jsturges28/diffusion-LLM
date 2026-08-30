"""Tests that a resumed DiffusionGemma canvas keeps its state.

Strategy: the same stub ``generate`` the cancel tests use, driving
the real streamer and the real ``_run_streamed`` on CPU with no
checkpoint. The assertions read protocol frames, because that is
where the defects showed.

DiffusionGemma has no mask token. A position reads as unresolved
when it changed since the previous step. A resume built a fresh
streamer, so that state started empty and two things went wrong at
once: with no previous canvas every position counted as changed, so
the first resumed frame rendered as an entirely masked canvas, and
the inherited prefix was reported born a second time, which is the
same rebirth LLaDA's resume fixes deliberately.

A third failure used to live here and no longer can. Confidence was
once derived from how many consecutive steps a position had held,
so a resume that lost the counter reported a settled canvas as
freshly uncertain. That derivation is gone: confidence is the
model's own probability, measured fresh from the logits of each
frame, so it does not depend on carried state at all. The test
below that says so is the replacement for two that used to pin the
counter across a resume.

Passing proves the streamer can re-enter a recorded frame with the
state that produced it, that checkpoints are recorded only for
frames the consumer actually received, and that a token carries
confidence exactly when that confidence was measured.
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
    FrameQueueStreamer,
    _run_streamed,
)
from src.inference.frame_queue import (
    FrameQueueCancelled,
    frame_queue_create,
)

CANVAS_LENGTH = 4

# The canvas a resume inherits.
SETTLED_IDS = [11, 12, 13, 14]

assert len(SETTLED_IDS) == CANVAS_LENGTH, "the canvas is full"


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
    """Redraws the inherited canvas unchanged for a few steps.

    ``logits`` picks which half of the streamer's contract to
    exercise. Real generation hands over logits, so that is the path
    confidence comes from; the token path is what a caller that does
    not ask for logits would produce, and it must record no
    confidence at all.
    """

    def __init__(self, steps: int = 3, logits: bool = False) -> None:
        self.steps = steps
        self.logits = logits
        self.device = "cpu"

    def generate(self, *, streamer: Any, **kwargs: Any) -> Any:
        streamer.put(_canvas([0] * CANVAS_LENGTH))
        for _ in range(self.steps):
            if self.logits:
                streamer.put_draft(logits=_logits())
            else:
                streamer.put_draft(value=_canvas(SETTLED_IDS))
        streamer.end()
        return _canvas(SETTLED_IDS)


def _checkpoint() -> FrameCheckpoint:
    return FrameCheckpoint(
        ids=torch.tensor(SETTLED_IDS, dtype=torch.long),
        canvas_index=0,
        rng=None,
        extra=DgemmaFrame(
            seen_revealed=frozenset(range(CANVAS_LENGTH)),
        ),
    )


def _drive(
    *,
    restore: bool,
    logits: bool = False,
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
    streamer._takes_logits = logits
    if restore:
        streamer.restore(_checkpoint())
    frames: List[Dict[str, Any]] = []

    async def drive() -> None:
        generator = _run_streamed(
            model=_StubModel(logits=logits),
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


def test_confidence_does_not_depend_on_the_resume() -> None:
    """What replaced two tests that pinned a carried counter.

    Confidence used to be derived from consecutive stable steps, so
    losing that counter across a resume made a settled canvas read
    as uncertain and the state had to be carried. It is now measured
    from each frame's logits, so a restored run and a fresh one
    report the same numbers for the same canvas, and there is
    nothing left for a resume to lose.
    """
    restored = _drive(restore=True, logits=True)[0]
    fresh = _drive(restore=False, logits=True)[0]

    restored_conf = [tok["c"] for tok in restored["tokens"]]
    fresh_conf = [tok["c"] for tok in fresh["tokens"]]
    assert restored_conf == fresh_conf
    assert restored["mean_conf"] == fresh["mean_conf"]


def test_the_resume_still_governs_what_reads_as_masked() -> None:
    """The half that does depend on carried state, kept adjacent to
    the half that no longer does so the two are not confused. Same
    frames as above: identical confidence, opposite mask flags."""
    restored = _drive(restore=True, logits=True)[0]
    fresh = _drive(restore=False, logits=True)[0]

    assert not any(tok["m"] for tok in restored["tokens"])
    assert all(tok["m"] for tok in fresh["tokens"])


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
    assert last.seen_revealed == frozenset(range(CANVAS_LENGTH))


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
    """The number is the model's own probability, which is what
    fades a mask by how sure the model is of the guess underneath
    it. No longer conditional on anything: this is the path every
    real run takes."""
    streamer, out_queue = _bare_streamer(takes_logits=True)
    streamer.put_draft(logits=_logits())

    tokens = out_queue.get()["tokens"]
    assert all(tok["m"] for tok in tokens)
    assert all("c" in tok for tok in tokens)


def test_a_frame_without_logits_records_no_confidence() -> None:
    """Nothing was measured, so nothing is claimed.

    This used to depend on which position you asked about: an
    unsettled one carried nothing, and a settled one carried a count
    of consecutive unchanged steps written into the same field as a
    probability. Both now record absent, which the client draws
    solid rather than fading by a number nobody took.
    """
    streamer, out_queue = _bare_streamer(takes_logits=False)
    # A canvas identical to the one before it, so these positions
    # read as settled and would have carried the old proxy.
    streamer.put_draft(value=_canvas(SETTLED_IDS))
    streamer.put_draft(value=_canvas(SETTLED_IDS))

    out_queue.get()
    tokens = out_queue.get()["tokens"]
    assert not any(tok["m"] for tok in tokens), (
        "the fixture must produce settled positions, or this is"
        " testing the unsettled case twice"
    )
    assert not any("c" in tok for tok in tokens)


def test_a_committed_canvas_still_carries_confidence() -> None:
    """The one place a number is written without being measured,
    and it is the model's own acceptance rather than a proxy for
    it. Pinned so the narrowed gate cannot take it too."""
    streamer, out_queue = _bare_streamer(takes_logits=False)
    streamer.put(_canvas(SETTLED_IDS))

    tokens = out_queue.get()["tokens"]
    assert all(tok["c"] == 1.0 for tok in tokens)


def test_mean_conf_is_a_mean_of_measured_probabilities() -> None:
    """A frame's mean_conf reaches the saved run and the Analytics
    confidence chart. It used to average whichever quantity the
    toggle selected, so the same field meant a probability on one
    run and a stability count on another and the two were plotted
    on the same axis. Now there is only the one quantity.
    """
    streamer, out_queue = _bare_streamer(takes_logits=True)
    streamer.put_draft(logits=_logits())

    frame = out_queue.get()
    per_token = [tok["c"] for tok in frame["tokens"]]
    expected = round(sum(per_token) / len(per_token), 4)
    assert frame["mean_conf"] == expected
    assert 0.0 < frame["mean_conf"] <= 1.0


def _bare_streamer(
    *, takes_logits: bool
) -> tuple[FrameQueueStreamer, Any]:
    """A streamer past its prompt, ready to emit.

    ``put`` swallows the first canvas it sees, which is the prompt
    echo transformers hands every streamer. A test that skipped
    this would block forever on an empty queue.
    """
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(_StubTokenizer(), out_queue)
    streamer._takes_logits = takes_logits
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    assert out_queue.qsize() == 0, "the prompt echo is not a frame"
    return streamer, out_queue


def _logits() -> torch.Tensor:
    """Per-position logits favouring one id, so conf is real."""
    values = torch.zeros(1, CANVAS_LENGTH, 32)
    for position, token_id in enumerate(SETTLED_IDS):
        values[0, position, token_id] = 4.0
    return values
