"""The relay between a page and its worker, both ways and torn down.

Strategy: drive `server._pipe` directly with fake ends inside one
event loop, under a timeout, so a relay that never finishes fails
rather than hangs. No app, no route and no worker process: the route
around it is exercised in `tests/web/test_activation_identity.py`,
and this is the part every generation passes through.

What this pins is `A2-QUALITY-02`. Both halves of the process
boundary had their own tests, and nothing sent a message across it,
so a relay that dropped one direction or left a half running could
pass both suites. Passing proves a page's message reaches the worker
and the worker's reaches the page; that whichever end finishes first
ends the other; and that nothing is left running when the relay
returns, because the half that did not finish is waited for as well
as cancelled.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

from starlette.websockets import WebSocketDisconnect

from src.web import server

# Far beyond what a relay of a few in-memory messages takes, so only a
# relay that never finishes reaches it.
RELAY_TIMEOUT_SECONDS = 5.0


class _Page:
    """The browser end. Sends ``messages``, then leaves if ``leaves``,
    or else waits to be cancelled, as an open tab does."""

    def __init__(self, messages: List[str], *, leaves: bool) -> None:
        self._messages = list(messages)
        self._leaves = leaves
        self.received: List[str] = []

    async def receive_text(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        if self._leaves:
            raise WebSocketDisconnect(code=1000)
        await asyncio.Event().wait()
        raise AssertionError("an event nobody sets was set")

    async def send_text(self, message: str) -> None:
        self.received.append(message)


class _Worker:
    """The worker end. Answers each message with one of its own, and
    ends its stream after ``hang_up_after`` answers when that is set,
    as a worker that exits does."""

    def __init__(self, *, hang_up_after: Optional[int]) -> None:
        self.received: List[str] = []
        self._hang_up_after = hang_up_after
        self._replies: asyncio.Queue = asyncio.Queue()
        self.settled = False

    async def send(self, message: str) -> None:
        self.received.append(message)
        asked = json.loads(message).get("type")
        reply = json.dumps({"type": "answer", "to": asked})
        await self._replies.put(reply)
        if self._hang_up_after == len(self.received):
            await self._replies.put(None)

    async def __aiter__(self) -> Any:
        try:
            while True:
                reply = await self._replies.get()
                if reply is None:
                    return
                yield reply
        finally:
            self.settled = True


def _relay(page: _Page, worker: _Worker) -> List[asyncio.Task]:
    """Run the relay to its end and return what it left running."""

    async def scenario() -> List[asyncio.Task]:
        await asyncio.wait_for(
            server._pipe(page, worker), timeout=RELAY_TIMEOUT_SECONDS
        )
        current = asyncio.current_task()
        return [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]

    return asyncio.run(scenario())


def test_messages_cross_in_both_directions() -> None:
    page = _Page(['{"type": "generate"}'], leaves=False)
    worker = _Worker(hang_up_after=1)

    _relay(page, worker)

    assert worker.received == ['{"type": "generate"}']
    assert page.received == ['{"type": "answer", "to": "generate"}']


def test_a_page_leaving_ends_the_worker_half() -> None:
    """The page's half finishes first. The worker's half was reading
    a stream that would never end, and the relay must not return
    while it still is."""
    page = _Page([], leaves=True)
    worker = _Worker(hang_up_after=None)

    left_running = _relay(page, worker)

    assert left_running == []
    assert worker.settled


def test_a_worker_hanging_up_ends_the_page_half() -> None:
    """The worker's half finishes first, and the page's, waiting on a
    tab that stays open, is the one that must end with it."""
    page = _Page(['{"type": "generate"}'], leaves=False)
    worker = _Worker(hang_up_after=1)

    left_running = _relay(page, worker)

    assert left_running == []
    assert page.received, "the answer was lost in the teardown"
