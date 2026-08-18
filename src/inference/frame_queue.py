"""The bounded hand-off between a model thread and the loop.

Two of the three backends run their decode loop in a worker thread
and hand frames to the event loop through a queue. That queue used
to be unbounded, so a browser too slow to drain it could not slow
the producer down: the frames piled up in worker memory instead,
and for an autoregressive run the pile is quadratic, because frame
*n* carries all *n* tokens.

A bound turns that into backpressure, which is the right answer to
a slow reader and the wrong one to a reader that has gone away: a
producer parked forever on a full queue is a leaked thread holding
a model. So the bound comes with two guarantees, and neither is
optional.

The producer's put is bounded in time as well as in depth, and
re-reads the stop event on every wait, so a cancel reaches it
within one timeout. The consumer keeps draining while it waits for
the producer to finish, so the thread can always run to its own
cleanup even if nobody wants its output any more.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Optional

# How many frames a producer may run ahead of its consumer.
#
# What one frame costs is the thing being bounded, and for an
# autoregressive run frame *n* carries all *n* tokens. At the
# recommended 256-token ceiling, 32 frames is at most 8,192 token
# records; at the experimental 2,048 ceiling it is 65,536, against
# the 2,098,176 the whole run carries. So this is a burst absorber,
# not the fix for the quadratic payload: RUNTIME-01's append-only
# frame variant is that, and this bound only stops a slow reader
# from turning the payload into unbounded worker memory.
#
# The other side of the trade is jitter. At the ~51 tokens/second
# measured for SmolLM3 on this machine, 32 frames is about 0.6
# seconds of slack, which absorbs a paint or a collection pause
# without ever making a healthy client wait.
FRAME_QUEUE_MAX_FRAMES = 32

# How long one put waits before looking up to ask whether the run
# is still wanted. Short enough that Stop feels immediate, long
# enough that a busy consumer is not spun on.
FRAME_QUEUE_PUT_TIMEOUT_SECONDS = 0.25

# A ceiling on those waits, so a put terminates even if the stop
# event is never set. Reaching it means the consumer stopped
# draining without anyone saying so, which is a bug rather than a
# slow client: 60 seconds is thousands of times longer than a
# healthy send.
FRAME_QUEUE_PUT_WAITS_MAX = 240

# How long the drain sleeps between empty reads while it waits for
# a producer thread to finish.
FRAME_QUEUE_DRAIN_POLL_SECONDS = 0.01

assert FRAME_QUEUE_MAX_FRAMES > 0, "a bound of zero cannot hold"
assert FRAME_QUEUE_PUT_TIMEOUT_SECONDS > 0.0, "put must wait"
assert FRAME_QUEUE_PUT_WAITS_MAX > 0, "put must be allowed once"
assert FRAME_QUEUE_DRAIN_POLL_SECONDS > 0.0, "poll must yield"

# The total a put may spend before giving up, stated once so the
# comment above cannot drift from the numbers it describes.
FRAME_QUEUE_PUT_SECONDS_MAX = (
    FRAME_QUEUE_PUT_TIMEOUT_SECONDS * FRAME_QUEUE_PUT_WAITS_MAX
)

assert FRAME_QUEUE_PUT_SECONDS_MAX >= 30.0, (
    "a put must outlast any plausible client stall"
)


class FrameQueueCancelled(Exception):
    """Unwinds a model thread whose run is no longer wanted.

    Raised from inside a streamer callback, which is the only
    place some backends give us to stand. DiffusionGemma's
    ``generate`` consults an externally supplied stopping
    criterion once per canvas, in ``_finalize_canvas``, so a
    criterion cannot stop a single-canvas run at all; its
    streamer's ``put_draft`` is called on every denoising step.
    Raising from there is therefore the hook with the granularity
    a user expects from pressing Stop.

    Its own type because it is an expected outcome rather than a
    failure: the caller catches it and ends the run as cancelled,
    where any other exception ends the run as an error.
    """


def frame_queue_create() -> "queue.Queue[Any]":
    """A producer queue carrying the shared bound."""
    return queue.Queue(maxsize=FRAME_QUEUE_MAX_FRAMES)


def frame_queue_put(
    out_queue: "queue.Queue[Any]",
    item: Any,
    *,
    stop_event: Optional[threading.Event],
) -> bool:
    """Hand one frame over. False when nobody is reading.

    Called from the model thread, so the stop event is read across
    a thread boundary and must be a ``threading.Event``.

    A False return is not an error. It means the run was cancelled
    or the consumer stopped draining, and the caller should stop
    producing rather than keep decoding for a page that has gone.
    """
    assert out_queue is not None, "put needs a queue"
    assert item is not None, "None is the sentinel, not a frame"
    for _ in range(FRAME_QUEUE_PUT_WAITS_MAX):
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            out_queue.put(
                item, timeout=FRAME_QUEUE_PUT_TIMEOUT_SECONDS
            )
        except queue.Full:
            continue
        return True
    return False


def frame_queue_close(out_queue: "queue.Queue[Any]") -> None:
    """Deliver the end-of-stream sentinel, whatever it takes.

    Unlike a frame this one may not be dropped, because it is what
    releases the consumer's blocking read; losing it strands the
    consumer forever. A plain blocking put is safe here precisely
    because of the drain below: the consumer is either still
    reading, and takes it at once, or has stopped and is draining,
    and takes it just as readily.
    """
    assert out_queue is not None, "close needs a queue"
    out_queue.put(None)


async def frame_queue_drain_until_done(
    out_queue: "queue.Queue[Any]",
    task: "asyncio.Task[Any]",
) -> None:
    """Let the producer finish, discarding what it still emits.

    The consumer calls this once it has stopped forwarding frames,
    which happens on cancel, on a failed send, and at the ordinary
    end of a run. Draining is what makes the bound safe to add: a
    producer parked on a full queue would otherwise never reach
    its own cleanup, and awaiting that thread would deadlock.

    Re-raises whatever the producer failed with, because awaiting
    the task is still how that error reaches the caller.
    """
    assert out_queue is not None, "drain needs a queue"
    assert task is not None, "drain needs a producer"
    while not task.done():
        try:
            out_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(FRAME_QUEUE_DRAIN_POLL_SECONDS)
    await task
