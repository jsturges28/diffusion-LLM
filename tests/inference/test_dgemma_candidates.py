"""A DiffusionGemma run's candidates cover exactly the drafts the
page received.

Strategy: a stub ``generate`` drives the real streamer and the real
``_run_streamed`` on CPU, drafting from seeded logits and committing
canvases the way the model does, over one canvas or several. The
assertions read the messages a worker would forward.

Passing proves the sampler's half of the contract. Every draft is
captured and no committed frame is, since a commit arrives without
logits; the candidates are the draft's own processed distribution,
with the draft's argmax as the held token; the raw step travels with
its frame and never leaves the process; and a stopped run's
candidates name only frames the consumer forwarded, although the
generate thread had run ahead of it. Each draft also carries, for
live cycling, the sets of exactly the positions that changed on it,
identical to the end-of-run message's, and nothing else does.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest
import torch

from src.backends.text_adapter import DGEMMA_TEXT
from src.inference import dgemma_sampler
from src.inference.dgemma_sampler import (
    CANDIDATES_KEY,
    LIVE_CANDIDATES_KEY,
    FrameQueueStreamer,
    _run_streamed,
)
from src.inference.frame_queue import frame_queue_create

CANVAS_LENGTH = 4
VOCAB = 12
DRAFTS = 3
# Drafts the thread queues before a test presses Stop, so frames the
# page will never receive are already waiting when it does.
AHEAD_DRAFTS = 3


class _StubTokenizer:
    def decode(
        self, ids: Any, skip_special_tokens: bool = False
    ) -> str:
        return "".join(f"<{int(i)}>" for i in ids)


def _draft_logits(step: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(500 + step)
    return torch.randn(
        1, CANVAS_LENGTH, VOCAB, generator=generator
    )


class _StubModel:
    """Drafts from seeded logits, then commits, per canvas, and says
    when it has queued ``AHEAD_DRAFTS`` drafts."""

    device = "cpu"

    def __init__(self, canvases: int, ahead: threading.Event) -> None:
        self.canvases = canvases
        self.ahead = ahead

    def generate(self, *, streamer: Any, **_: Any) -> Any:
        streamer.put(torch.zeros((1, 2), dtype=torch.long))
        step = 0
        for _canvas in range(self.canvases):
            for _draft in range(DRAFTS):
                streamer.put_draft(logits=_draft_logits(step))
                step += 1
                if step == AHEAD_DRAFTS:
                    self.ahead.set()
            streamer.put(torch.arange(CANVAS_LENGTH).unsqueeze(0))
        streamer.end()
        return torch.arange(CANVAS_LENGTH).unsqueeze(0)


class _RepeatingModel:
    """Drafts the same logits twice, so the second draft changes
    nothing, then commits."""

    device = "cpu"

    def generate(self, *, streamer: Any, **_: Any) -> Any:
        streamer.put(torch.zeros((1, 2), dtype=torch.long))
        streamer.put_draft(logits=_draft_logits(0))
        streamer.put_draft(logits=_draft_logits(0))
        streamer.put(torch.arange(CANVAS_LENGTH).unsqueeze(0))
        streamer.end()
        return torch.arange(CANVAS_LENGTH).unsqueeze(0)


def _run(
    *,
    alternatives: bool = True,
    canvases: int = 1,
    stop_after: int = -1,
    model: Any = None,
) -> List[Dict[str, Any]]:
    """Every message a worker would forward. ``stop_after`` names
    the frame on whose arrival the user presses Stop, which waits
    until the thread has queued drafts past it."""
    out_queue = frame_queue_create()
    stop = threading.Event()
    ahead = threading.Event()
    if model is None:
        model = _StubModel(canvases, ahead)
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT,
        out_queue,
        stop_event=stop,
        alternatives=alternatives,
    )
    streamer._takes_logits = True
    messages: List[Dict[str, Any]] = []

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
        async for message in generator:
            messages.append(message)
            if message.get("index") == stop_after:
                assert await asyncio.to_thread(ahead.wait, 5.0)
                stop.set()

    asyncio.run(drive())
    return messages


def _of_type(
    messages: List[Dict[str, Any]], kind: str
) -> List[Dict[str, Any]]:
    return [m for m in messages if m["type"] == kind]


def _candidates(
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    found = _of_type(messages, "candidates")
    assert len(found) <= 1, "candidates are sent once"
    return found[0] if found else None


# -- delivery --


def test_candidates_arrive_once_just_before_done() -> None:
    messages = _run()

    closing = [m["type"] for m in messages[-2:]]
    assert closing == ["candidates", "done"]


def test_a_run_without_alternatives_sends_none() -> None:
    assert _candidates(_run(alternatives=False)) is None


def test_the_draft_pass_reads_five_only_for_a_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every draft is read in one pass, and how many candidates it
    reads is the one thing the capture changes: a run with
    Alternatives off asks for one, the likeliest token it shows."""
    asked: List[int] = []
    real_pass = dgemma_sampler.draft_signals

    def spy(logits: torch.Tensor, k: int) -> Any:
        asked.append(k)
        return real_pass(logits, k)

    monkeypatch.setattr(dgemma_sampler, "draft_signals", spy)
    _run(alternatives=False)
    off = list(asked)
    asked.clear()
    _run(alternatives=True)

    assert off == [1] * DRAFTS
    assert asked == [5] * DRAFTS


def test_the_raw_capture_never_leaves_the_process() -> None:
    """The frame carries the raw step across the queue; the consumer
    takes it off before the frame leaves the process. What reaches
    the page is the decoded slice below, never the tensors."""
    for frame in _of_type(_run(), "frame"):
        assert CANDIDATES_KEY not in frame


# -- which frames --


def test_every_draft_is_captured_and_no_commit_is() -> None:
    """Two canvases of three drafts each: frames 0 to 2 are drafts,
    3 commits the first canvas, 4 to 6 draft the second, 7 commits
    it. A commit arrives without logits, so it has nothing of its
    own; the page shows it its canvas's last draft."""
    message = _candidates(_run(canvases=2))

    assert message is not None
    assert message["frames"] == [0, 1, 2, 4, 5, 6]
    for sets in message["sets"]:
        assert len(sets) == CANVAS_LENGTH


def test_a_stopped_run_names_only_frames_the_page_received() -> None:
    """The generate thread runs ahead of the consumer by the queue's
    depth, so frames it drafted after the Stop reached the queue but
    never the page. Offering at the consumer is what keeps them out.
    """
    messages = _run(canvases=3, stop_after=1)
    forwarded = [m["index"] for m in _of_type(messages, "frame")]
    message = _candidates(messages)

    assert forwarded == [0, 1]
    assert message is not None
    assert message["frames"] == forwarded
    assert messages[-1].get("cancelled") is True


# -- what each set says --


def test_a_drafts_candidates_are_its_processed_distribution() -> None:
    message = _candidates(_run())
    assert message is not None
    probs = torch.softmax(_draft_logits(0)[0], dim=-1)
    top_probs, top_ids = torch.topk(probs, 5, dim=-1)

    for position, entry in enumerate(message["sets"][0]):
        ids = [row["id"] for row in entry["c"]]
        assert ids == top_ids[position].tolist()
        for row, want in zip(
            entry["c"], top_probs[position].tolist(), strict=True
        ):
            assert row["p"] == pytest.approx(want, abs=1e-4)


def test_the_held_token_is_the_drafts_argmax() -> None:
    """A draft shows its argmax, so the marked row is the likeliest
    and no position ever needs the held token appended."""
    messages = _run()
    frames = {m["index"]: m for m in _of_type(messages, "frame")}
    message = _candidates(messages)
    assert message is not None

    for frame, sets in zip(
        message["frames"], message["sets"], strict=True
    ):
        for token, entry in zip(
            frames[frame]["tokens"], sets, strict=True
        ):
            assert entry["h"] == token["id"]
            assert entry["c"][0]["id"] == token["id"]
            assert len(entry["c"]) == 5
            # The frame's confidence is that first row's probability,
            # read in the same pass; both are rounded to four places.
            assert abs(token["c"] - entry["c"][0]["p"]) <= 1e-4


def test_candidate_text_is_the_raw_decode() -> None:
    message = _candidates(_run())
    assert message is not None

    row = message["sets"][0][0]["c"][0]
    assert row["t"] == f"<{row['id']}>"


# -- the sets each frame carries for live cycling --


def _unsettled(frame: Dict[str, Any]) -> List[int]:
    found: List[int] = []
    for position, token in enumerate(frame["tokens"]):
        if token["m"]:
            found.append(position)
    return found


def test_a_draft_carries_the_sets_of_exactly_its_changes() -> None:
    """The positions that changed on a draft are the ones the page
    cycles, so theirs are the sets that ride the frame."""
    frames = _of_type(_run(), "frame")
    drafts = [frame for frame in frames if _unsettled(frame)]

    assert drafts, "the fixture's drafts change something"
    for frame in drafts:
        live = frame[LIVE_CANDIDATES_KEY]
        assert live["positions"] == _unsettled(frame)
        assert len(live["sets"]) == len(live["positions"])


def test_each_live_set_is_the_end_of_run_set() -> None:
    """A position cycles live through exactly what its popover shows
    once the run has ended."""
    messages = _run()
    message = _candidates(messages)
    assert message is not None
    kept = dict(zip(message["frames"], message["sets"], strict=True))

    checked = 0
    for frame in _of_type(messages, "frame"):
        live = frame.get(LIVE_CANDIDATES_KEY)
        if live is None:
            continue
        for position, entry in zip(
            live["positions"], live["sets"], strict=True
        ):
            assert entry == kept[frame["index"]][position]
            checked += 1

    assert checked > 0


def test_a_draft_that_changed_nothing_carries_no_sets() -> None:
    frames = _of_type(_run(model=_RepeatingModel()), "frame")

    assert LIVE_CANDIDATES_KEY in frames[0]
    assert _unsettled(frames[1]) == []
    assert LIVE_CANDIDATES_KEY not in frames[1]


def test_a_commit_carries_no_sets() -> None:
    """Frames 3 and 7 commit their canvases, which arrives without
    logits and so without candidates."""
    frames = {
        m["index"]: m for m in _of_type(_run(canvases=2), "frame")
    }

    assert LIVE_CANDIDATES_KEY not in frames[3]
    assert LIVE_CANDIDATES_KEY not in frames[7]
    assert LIVE_CANDIDATES_KEY in frames[4]


def test_a_run_without_alternatives_carries_no_sets() -> None:
    for frame in _of_type(_run(alternatives=False), "frame"):
        assert LIVE_CANDIDATES_KEY not in frame


def test_the_live_texts_stay_within_their_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dgemma_sampler, "LIVE_TEXT_CACHE_LIMIT", 3)
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT,
        frame_queue_create(),
        alternatives=True,
    )

    texts = [streamer._live_decode(token) for token in range(10)]

    assert texts == [f"<{token}>" for token in range(10)]
    assert len(streamer._live_texts) <= 3


# -- the hand-off itself --


def test_a_frame_without_candidates_offers_nothing() -> None:
    streamer = FrameQueueStreamer(
        _StubTokenizer(),
        DGEMMA_TEXT,
        frame_queue_create(),
        alternatives=True,
    )
    streamer.offer_forwarded({"type": "frame", "index": 0})

    assert streamer.capture is not None
    assert streamer.capture.frames() == []


def test_candidates_without_a_capture_are_a_programmer_error(
) -> None:
    streamer = FrameQueueStreamer(
        _StubTokenizer(), DGEMMA_TEXT, frame_queue_create()
    )

    with pytest.raises(AssertionError):
        streamer.offer_forwarded({CANDIDATES_KEY: object()})
