"""How large one save may be, checked before its body is read.

A save is the one request here that carries a whole run, and nothing
bounded it: Starlette reads and parses the complete JSON before any
field is validated, so a single crafted POST could allocate far beyond
any run the app can make, and write as much under the results root
(`A2-TRUST-02`). The limits a parsed save answers to field by field
live with ``SaveRunRequest``; this is the layer in front of them,
which refuses on size alone, before the body is held in memory.

Its own module for the same reason as ``model_lease``: it is small, it
is policy, and nothing here needs the server.

## The ceiling

Today's experimental slider ranges are the limit for one run, so the
ceiling is sized to the largest save they allow: an edited LLaDA run
at 1,024 steps and 1,024 positions, which also carries the run as it
was before the edit. At the 62 bytes a token record measures on disk,
its two layers of 1,025 frames by 1,024 records are about 125 MiB, and
frame text and two candidate sidecars bring it to about 141 MiB.
256 MiB leaves room for token text longer than that measurement saw.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send

MIB = 2**20

SAVE_BODY_BYTES_MAX = 256 * MIB

# Read at request time rather than captured when the middleware is
# built, so a test can lower one without building a second app.
BODY_LIMITS: Dict[str, int] = {"/api/save": SAVE_BODY_BYTES_MAX}

# A bound on the pieces a body may arrive in, which a client sending
# empty chunks would otherwise never run out of. Far above what a
# real upload uses: 256 MiB in uvicorn's 64 KiB reads is 4,096.
BODY_CHUNKS_MAX = 1 << 20

assert SAVE_BODY_BYTES_MAX > 141 * MIB, (
    "the ceiling must hold the largest run the sliders allow"
)


class BodyLimit:
    """Refuse a request body past its route's limit, with a 413.

    A declared length past the limit is refused before a byte is read.
    A body that declares none, as a chunked one does, is counted as it
    arrives and refused at the first byte over. A body within the
    limit reaches the app unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        limit = BODY_LIMITS.get(scope["path"])
        if limit is None:
            await self._app(scope, receive, send)
            return
        assert limit > 0, "a body limit is a positive byte count"
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await _refuse(send, declared, limit)
            return
        body = await _read_capped(receive, limit)
        if body is None:
            await _refuse(send, None, limit)
            return
        await self._app(scope, _replay(body, receive), send)


def _declared_length(scope: Scope) -> Optional[int]:
    """The body length the client declared, or None."""
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


async def _read_capped(
    receive: Receive, limit: int
) -> Optional[bytes]:
    """The whole body, or None as soon as it passes ``limit``.

    A disconnect ends the read with what arrived, and the app then
    refuses that as the incomplete request it is.
    """
    chunks: List[bytes] = []
    total = 0
    for _ in range(BODY_CHUNKS_MAX):
        message = await receive()
        if message["type"] != "http.request":
            return b"".join(chunks)
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)
    return None


def _replay(body: bytes, receive: Receive) -> Receive:
    """A receive that hands the app the body already read.

    Later calls go to the real one, which is where a disconnect the
    app listens for after the body still comes from.
    """
    delivered = False

    async def replayed() -> Message:
        nonlocal delivered
        if delivered:
            return await receive()
        delivered = True
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    return replayed


async def _refuse(
    send: Send, size: Optional[int], limit: int
) -> None:
    """Answer 413 in the shape the page reads a failed save from."""
    ceiling = f"{limit / MIB:,.0f} MiB"
    if size is None:
        message = f"This save is over the {ceiling} one run can take."
    else:
        message = (
            f"This save is {size / MIB:,.1f} MiB, over the {ceiling}"
            " one run can take."
        )
    body = json.dumps({"success": False, "message": message}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
