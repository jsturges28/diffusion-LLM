"""Tests for the bounded model-thread to event-loop hand-off.

Strategy: the queue module is pure standard library, so these tests
drive it directly with real threads and a real ``queue.Queue``, no
model and no tokenizer. They cover the three properties that make
bounding the queue safe to do at all: the depth never exceeds the
bound no matter how far ahead the producer runs, a producer parked
against a full queue leaves promptly when the run is cancelled, and
a consumer that has stopped forwarding still lets the producer
thread reach its own cleanup.

Passing proves the bound is real (so a slow reader cannot turn a
quadratic payload into unbounded worker memory), that Stop reaches
a producer within one put timeout, and that adding the bound has
not introduced the deadlock it invites: a producer waiting for
space while the consumer waits for the producer.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any, List

import pytest

from src.inference.frame_queue import (
    FRAME_QUEUE_MAX_FRAMES,
    FRAME_QUEUE_PUT_SECONDS_MAX,
    FRAME_QUEUE_PUT_TIMEOUT_SECONDS,
    FrameQueueCancelled,
    frame_queue_close,
    frame_queue_create,
    frame_queue_drain_until_done,
    frame_queue_put,
)

# Enough puts to overrun the bound several times over if it were
# not enforced, while still finishing in well under a second.
OVERRUN_FRAMES = FRAME_QUEUE_MAX_FRAMES * 3


def test_the_queue_is_created_with_the_shared_bound() -> None:
    out_queue = frame_queue_create()
    assert out_queue.maxsize == FRAME_QUEUE_MAX_FRAMES
    assert out_queue.empty()


def test_a_bound_that_holds_nothing_would_be_a_bug() -> None:
    # The constant is load-bearing in two directions: zero would
    # deadlock every run, and unbounded is the thing being fixed.
    assert FRAME_QUEUE_MAX_FRAMES > 0
    assert FRAME_QUEUE_MAX_FRAMES < 1024


def test_a_producer_never_exceeds_the_bound() -> None:
    """The whole point: depth is capped even with no consumer."""
    out_queue = frame_queue_create()
    stop = threading.Event()
    depths: List[int] = []

    def produce() -> None:
        for i in range(OVERRUN_FRAMES):
            frame_queue_put(
                out_queue, {"i": i}, stop_event=stop
            )
            depths.append(out_queue.qsize())

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    # Let it run ahead as far as it can, then stop it.
    time.sleep(FRAME_QUEUE_PUT_TIMEOUT_SECONDS * 2)
    assert out_queue.qsize() <= FRAME_QUEUE_MAX_FRAMES
    stop.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive(), "producer did not leave"
    assert depths, "producer never delivered anything"
    assert max(depths) <= FRAME_QUEUE_MAX_FRAMES


def test_a_full_queue_stops_the_producer_rather_than_growing(
) -> None:
    out_queue = frame_queue_create()
    stop = threading.Event()
    for i in range(FRAME_QUEUE_MAX_FRAMES):
        assert frame_queue_put(
            out_queue, {"i": i}, stop_event=stop
        )
    assert out_queue.full()

    # The next put has nowhere to go. It must refuse rather than
    # make room, and it must not raise.
    stop.set()
    delivered = frame_queue_put(
        out_queue, {"i": "one too many"}, stop_event=stop
    )
    assert delivered is False
    assert out_queue.qsize() == FRAME_QUEUE_MAX_FRAMES


def test_a_cancelled_run_releases_a_parked_producer() -> None:
    """Stop must reach a producer waiting for queue space."""
    out_queue = frame_queue_create()
    stop = threading.Event()
    for i in range(FRAME_QUEUE_MAX_FRAMES):
        frame_queue_put(out_queue, {"i": i}, stop_event=stop)

    outcome: List[bool] = []
    started = threading.Event()

    def produce() -> None:
        started.set()
        outcome.append(
            frame_queue_put(
                out_queue, {"i": "blocked"}, stop_event=stop
            )
        )

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    assert started.wait(timeout=5.0)
    # It is now parked inside put, waiting for space.
    time.sleep(FRAME_QUEUE_PUT_TIMEOUT_SECONDS / 2)
    assert worker.is_alive()

    stop.set()
    worker.join(timeout=FRAME_QUEUE_PUT_TIMEOUT_SECONDS * 8)
    assert not worker.is_alive(), "cancel did not reach the put"
    assert outcome == [False]


def test_a_put_gives_up_long_before_it_hangs_forever() -> None:
    # The cap is what makes a lost consumer survivable even when
    # nothing ever sets the stop event. It is deliberately far
    # longer than any healthy send, so this asserts the contract
    # rather than waiting it out.
    assert FRAME_QUEUE_PUT_SECONDS_MAX >= 30.0
    assert FRAME_QUEUE_PUT_SECONDS_MAX <= 300.0


def test_a_put_with_no_stop_event_still_delivers() -> None:
    # Callers may pass None (no cancellation wired), and that must
    # not be read as "already cancelled".
    out_queue = frame_queue_create()
    assert frame_queue_put(out_queue, {"i": 0}, stop_event=None)
    assert out_queue.get_nowait() == {"i": 0}


def test_the_sentinel_is_not_a_frame() -> None:
    out_queue = frame_queue_create()
    with pytest.raises(AssertionError):
        frame_queue_put(out_queue, None, stop_event=None)


def test_close_delivers_the_end_of_stream_sentinel() -> None:
    out_queue = frame_queue_create()
    frame_queue_close(out_queue)
    assert out_queue.get_nowait() is None


def test_the_drain_lets_a_parked_producer_finish() -> None:
    """The deadlock a bounded queue invites, proven absent.

    A consumer that stops reading leaves the producer blocked for
    space; awaiting that producer without draining would hang.
    """
    out_queue = frame_queue_create()
    finished = threading.Event()

    def produce() -> None:
        # Deliberately more than the queue can hold, with no
        # stop event ever set: only the drain can free this.
        for i in range(OVERRUN_FRAMES):
            frame_queue_put(out_queue, {"i": i}, stop_event=None)
        frame_queue_close(out_queue)
        finished.set()

    async def drive() -> None:
        task = asyncio.create_task(asyncio.to_thread(produce))
        # Give it time to fill the queue and park.
        await asyncio.sleep(0.05)
        assert not task.done(), "producer should be parked"
        await asyncio.wait_for(
            frame_queue_drain_until_done(out_queue, task),
            timeout=10.0,
        )
        assert task.done()

    asyncio.run(drive())
    assert finished.is_set()


def test_the_drain_reraises_what_the_producer_failed_with(
) -> None:
    out_queue = frame_queue_create()

    def produce() -> None:
        raise RuntimeError("decode exploded")

    async def drive() -> None:
        task = asyncio.create_task(asyncio.to_thread(produce))
        await frame_queue_drain_until_done(out_queue, task)

    with pytest.raises(RuntimeError, match="decode exploded"):
        asyncio.run(drive())


def test_the_drain_returns_at_once_when_already_done() -> None:
    out_queue = frame_queue_create()

    def produce() -> None:
        return None

    async def drive() -> None:
        task = asyncio.create_task(asyncio.to_thread(produce))
        await task
        await asyncio.wait_for(
            frame_queue_drain_until_done(out_queue, task),
            timeout=5.0,
        )

    asyncio.run(drive())


def test_the_cancelled_signal_is_its_own_type() -> None:
    # Callers distinguish "the user stopped this" from "this
    # broke", so it must not be catchable only as a generic error.
    assert issubclass(FrameQueueCancelled, Exception)
    assert not issubclass(FrameQueueCancelled, ValueError)


def test_a_slow_consumer_is_waited_for_rather_than_dropped(
) -> None:
    """Backpressure, not loss: every frame arrives, in order."""
    out_queue: "queue.Queue[Any]" = frame_queue_create()
    stop = threading.Event()
    sent = OVERRUN_FRAMES
    received: List[Any] = []

    def produce() -> None:
        for i in range(sent):
            frame_queue_put(out_queue, {"i": i}, stop_event=stop)
        frame_queue_close(out_queue)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    while True:
        item = out_queue.get(timeout=10.0)
        if item is None:
            break
        received.append(item)
    worker.join(timeout=5.0)

    assert len(received) == sent
    assert [frame["i"] for frame in received] == list(
        range(sent)
    )
