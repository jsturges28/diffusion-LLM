"""Tests for stopping a DiffusionGemma run partway through.

Strategy: no checkpoint and no GPU. A stub ``generate`` stands in
for the real one and drives the streamer the same way transformers
does, one ``put_draft`` per denoising step and one ``put`` per
committed canvas, so these tests exercise the exact hook a cancel
travels through.

That hook is the streamer rather than a stopping criterion on
purpose. ``generate`` consults an externally supplied criterion
once per canvas, in ``_finalize_canvas``, so a criterion cannot
interrupt a single-canvas run at all, while ``put_draft`` lands on
every denoising step.

Passing proves a cancelled run unwinds ``generate`` within one
denoising step rather than running to completion invisibly, that
the unwinding is reported as a cancellation rather than an error,
and that a stopped run reports the text the page received: every
canvas it was sent, and nothing the generate thread drafted after
the Stop. A run nobody stopped still reads its text off the
sequences ``generate`` returned.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest
import torch

from src.backends.text_adapter import DGEMMA_TEXT
from src.inference.dgemma_sampler import (
    FrameQueueStreamer,
    _run_streamed,
)
from src.inference.frame_queue import (
    FRAME_QUEUE_MAX_FRAMES,
    FrameQueueCancelled,
    frame_queue_create,
)

CANVAS_LENGTH = 4


class _StubTokenizer:
    """Decodes ids to a letter each, so text is easy to assert.

    Accepts a tensor as well as a list because the real code path
    hands it both: a single id from the streamer, and a sequence
    slice when a completed generate is decoded.
    """

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
    """A generate that denoises one canvas for a set step count."""

    def __init__(self, steps: int, canvases: int = 1) -> None:
        self.steps = steps
        self.canvases = canvases
        self.steps_run = 0
        self.device = "cpu"

    def generate(
        self,
        *,
        streamer: Any,
        **kwargs: Any,
    ) -> Any:
        streamer.put(_canvas([0] * CANVAS_LENGTH))
        for canvas in range(self.canvases):
            for step in range(self.steps):
                self.steps_run += 1
                streamer.put_draft(
                    value=_canvas(
                        [canvas + step] * CANVAS_LENGTH
                    )
                )
            streamer.put(
                _canvas([canvas + 100] * CANVAS_LENGTH)
            )
        streamer.end()
        return _canvas([1, 2, 3])


def _settled(step: int) -> List[int]:
    """A draft whose positions up to ``step`` hold still."""
    ids: List[int] = []
    for position in range(CANVAS_LENGTH):
        if position <= step:
            ids.append(10 + position)
        else:
            ids.append(50 + step)
    return ids


class _SettlingModel(_StubModel):
    """A generate whose drafts settle one position at a time.

    A position reads as decided once it holds still for a draft, so
    each of the first frames reads differently, and a text taken
    from the wrong frame cannot match the right one by luck.
    """

    def generate(
        self,
        *,
        streamer: Any,
        **kwargs: Any,
    ) -> Any:
        streamer.put(_canvas([0] * CANVAS_LENGTH))
        for step in range(self.steps):
            self.steps_run += 1
            streamer.put_draft(value=_canvas(_settled(step)))
        streamer.put(_canvas(_settled(CANVAS_LENGTH)))
        streamer.end()
        return _canvas([1, 2, 3])


def test_the_streamer_raises_once_the_run_is_cancelled() -> None:
    out_queue = frame_queue_create()
    stop = threading.Event()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT, out_queue, stop_event=stop
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))

    streamer.put_draft(value=_canvas([1] * CANVAS_LENGTH))
    stop.set()
    with pytest.raises(FrameQueueCancelled):
        streamer.put_draft(value=_canvas([2] * CANVAS_LENGTH))


def test_the_streamer_is_silent_while_the_run_is_wanted(
) -> None:
    out_queue = frame_queue_create()
    stop = threading.Event()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT, out_queue, stop_event=stop
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    for step in range(3):
        streamer.put_draft(
            value=_canvas([step] * CANVAS_LENGTH)
        )
    assert out_queue.qsize() == 3
    assert not stop.is_set()


def test_a_streamer_without_a_stop_event_never_cancels() -> None:
    # Cancellation is opt-in; a caller that wires no event must
    # not have its run read as already stopped.
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT, out_queue
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    streamer.put_draft(value=_canvas([1] * CANVAS_LENGTH))
    assert out_queue.qsize() == 1


def _drive(
    model: _StubModel,
    stop: Optional[threading.Event],
    *,
    cancel_after: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run the shared streaming path, collecting yielded frames."""
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT, out_queue, stop_event=stop
    )
    frames: List[Dict[str, Any]] = []

    async def drive() -> None:
        generator = _run_streamed(
            model=model,
            tokenizer=_StubTokenizer(),
            inputs={},
            prompt_len=0,
            streamer=streamer,
            out_queue=out_queue,
            generate_kwargs={},
            seed=-1,
            cancel_event=stop,
        )
        async for frame in generator:
            frames.append(frame)
            reached = cancel_after is not None and (
                len(frames) >= cancel_after
            )
            if reached and stop is not None:
                stop.set()

    asyncio.run(drive())
    return frames


def _sent(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The frames among what a run yielded, its terminal left out."""
    found: List[Dict[str, Any]] = []
    for frame in frames:
        if frame["type"] == "frame":
            found.append(frame)
    return found


def test_an_uncancelled_run_streams_every_step() -> None:
    model = _StubModel(steps=5)
    frames = _drive(model, threading.Event())
    assert model.steps_run == 5
    assert frames[-1]["type"] == "done"
    assert [f["type"] for f in frames[:-1]] == ["frame"] * 6


def test_a_cancel_stops_generate_rather_than_only_the_yield(
) -> None:
    """The finding's core: the model must stop computing."""
    model = _StubModel(steps=200)
    frames = _drive(model, threading.Event(), cancel_after=3)

    # The step budget was 200. It must not have run them all.
    assert model.steps_run < 200, "generate ran to completion"
    assert frames[-1]["type"] == "done"


def test_a_cancelled_run_reports_text_rather_than_an_error(
) -> None:
    model = _SettlingModel(steps=200)
    frames = _drive(model, threading.Event(), cancel_after=2)
    terminal = frames[-1]
    sent = _sent(frames)

    assert terminal["type"] == "done"
    assert terminal["cancelled"] is True
    # The text of the last frame the page received. The generate
    # thread had drafted past it before the Stop reached the queue,
    # and what it drafted there the page never saw.
    assert len(sent) == 2
    assert terminal["final_text"] == sent[-1]["text"]
    assert terminal["final_text"] != ""


def test_a_run_stopped_before_its_first_frame_reports_no_text(
) -> None:
    """Nothing reached the page, so there is no text to report. The
    frame the thread built as the Stop landed was refused by the
    queue, and reporting its text claimed a frame nobody saw."""
    stop = threading.Event()
    stop.set()
    frames = _drive(_SettlingModel(steps=5), stop)
    terminal = frames[-1]

    assert _sent(frames) == []
    assert terminal["cancelled"] is True
    assert terminal["final_text"] == ""


def test_a_stopped_run_keeps_every_canvas_it_sent() -> None:
    """Each canvas's last frame, in order: the first canvas's
    commit, then the second's draft as far as the page saw it. A
    stop in the second canvas used to report that canvas alone."""
    model = _StubModel(steps=2, canvases=3)
    frames = _drive(model, threading.Event(), cancel_after=4)
    sent = _sent(frames)

    canvases = [frame["canvas_index"] for frame in sent]
    assert canvases == [0, 0, 0, 1]
    assert frames[-1]["final_text"] == (
        sent[2]["text"] + sent[3]["text"]
    )


def test_a_finished_run_reports_its_decoded_output() -> None:
    """The other half: a run nobody stopped reads its text off the
    sequences generate returned, split into channels, as before."""
    frames = _drive(_StubModel(steps=2), threading.Event())
    terminal = frames[-1]
    _, answer = DGEMMA_TEXT.split_channels(
        _StubTokenizer().decode([1, 2, 3])
    )

    assert "cancelled" not in terminal
    assert terminal["final_text"] == answer


def test_a_genuine_failure_is_still_raised() -> None:
    """Cancellation is special-cased; breakage must not be."""

    class _BrokenModel(_StubModel):
        def generate(self, *, streamer: Any, **kwargs: Any) -> Any:
            raise RuntimeError("denoiser exploded")

    with pytest.raises(RuntimeError, match="denoiser exploded"):
        _drive(_BrokenModel(steps=1), threading.Event())


def test_the_queue_bound_holds_against_a_fast_denoiser() -> None:
    # A producer far faster than its consumer is the case the
    # bound exists for, and the streamer is that producer.
    out_queue = frame_queue_create()
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT, out_queue
    )
    streamer.put(_canvas([0] * CANVAS_LENGTH))
    for step in range(FRAME_QUEUE_MAX_FRAMES):
        streamer.put_draft(
            value=_canvas([step] * CANVAS_LENGTH)
        )
    assert out_queue.qsize() == FRAME_QUEUE_MAX_FRAMES
    assert out_queue.full()
