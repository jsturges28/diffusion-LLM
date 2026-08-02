"""Tests that a What If branch never re-points the worker's run state.

Strategy: drive ``handle_substitute`` with a stubbed WebSocket and a
monkeypatched ``streaming_substitute`` (no model, no GPU), then assert
``last_run_state`` still describes the run the browser is showing.

This is the server half of the Retry bug: the client's Retry restores
its arrays to the pre-substitution run, so if the worker adopted the
branch instead, the two sides would validate against different
candidate sets and every position at or after the edit would be
rejected with "was not among the captured candidates". Passing proves
a Retry-then-pick round trip validates against the same run the user
sees, and that the pre-edit candidates survive a completed branch.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.backends import smollm3_worker
from src.backends.smollm3_worker import Smollm3Backend

# Two positions, each with two captured candidates. Position 1 is the
# "after the edit" position that the bug used to reject.
ORIGINAL_ALTS: List[List[Dict[str, Any]]] = [
    [
        {"id": 5, "t": "he", "p": 0.7},
        {"id": 7, "t": "she", "p": 0.2},
    ],
    [
        {"id": 11, "t": " ran", "p": 0.6},
        {"id": 13, "t": " sat", "p": 0.3},
    ],
]

BRANCH_ALTS: List[List[Dict[str, Any]]] = [
    [
        {"id": 7, "t": "she", "p": 0.2},
        {"id": 5, "t": "he", "p": 0.7},
    ],
    [
        {"id": 99, "t": " flew", "p": 0.8},
        {"id": 98, "t": " dove", "p": 0.1},
    ],
]


class _StubWebSocket:
    """Collects the JSON the worker would have sent."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


class _StubStreamer:
    """Drains the generator the way FrameStreamer would."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []

    async def run(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
    ) -> bool:
        async for frame in generator:
            self.frames.append(frame)
        return True


def _run_state() -> Dict[str, Any]:
    """The recorded run a substitution re-enters."""
    return {
        "ids": [5, 11],
        "confidences": [0.7, 0.6],
        "entropies": [0.31, 0.44],
        "alternatives": [
            [dict(c) for c in ORIGINAL_ALTS[0]],
            [dict(c) for c in ORIGINAL_ALTS[1]],
        ],
        "prompt": "p",
        "max_new_tokens": 8,
        "thinking": False,
        "seed": -1,
        "alternatives_enabled": True,
    }


def _install_branch_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the sampler with one that fills any sink it is given.

    Mimics the real generator's contract: it fills ``state_sink``
    when one is passed, which is what the worker must not ask for.
    """

    async def fake_substitute(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "frame", "index": 1}
        sink = kwargs.get("state_sink")
        if sink is not None:
            sink["ids"] = [7, 99]
            sink["confidences"] = [0.2, 0.8]
            sink["entropies"] = [0.9, 0.1]
            sink["alternatives"] = BRANCH_ALTS
        yield {"type": "done", "final_text": "she flew"}

    monkeypatch.setattr(
        smollm3_worker, "streaming_substitute", fake_substitute
    )


def _substitute(
    backend: Smollm3Backend, position: int, token_id: int
) -> _StubWebSocket:
    ws = _StubWebSocket()
    asyncio.run(
        backend.handle_substitute(
            ws,  # type: ignore[arg-type]
            {"position": position, "token_id": token_id},
            asyncio.Event(),
            _StubStreamer(),  # type: ignore[arg-type]
        )
    )
    return ws


def test_branch_does_not_replace_the_run_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_branch_stub(monkeypatch)
    backend = Smollm3Backend()
    backend.last_run_state = _run_state()

    ws = _substitute(backend, position=0, token_id=7)

    assert not any(
        m.get("type") == "error" for m in ws.sent
    )
    state = backend.last_run_state
    assert state is not None
    assert state["ids"] == [5, 11]
    assert state["alternatives"] == ORIGINAL_ALTS


def test_retry_can_pick_a_position_after_the_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression itself: substitute, then Retry and pick later.

    Position 1 sits after the edit at position 0, so its candidates
    only still validate because the branch was discarded.
    """
    _install_branch_stub(monkeypatch)
    backend = Smollm3Backend()
    backend.last_run_state = _run_state()

    _substitute(backend, position=0, token_id=7)
    ws = _substitute(backend, position=1, token_id=13)

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors == []


def test_a_branch_only_candidate_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative space: pinning the run must not accept anything.

    Token 99 was only ever a branch candidate, so it is not a real
    counterfactual of the recorded run and must be refused.
    """
    _install_branch_stub(monkeypatch)
    backend = Smollm3Backend()
    backend.last_run_state = _run_state()

    _substitute(backend, position=0, token_id=7)
    ws = _substitute(backend, position=1, token_id=99)

    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "error"
    assert "not among the captured candidates" in (
        ws.sent[0]["message"]
    )


def test_substitute_without_a_run_reports_an_error() -> None:
    backend = Smollm3Backend()

    ws = _substitute(backend, position=0, token_id=5)

    assert ws.sent[0]["type"] == "error"
    assert "No previous generation" in ws.sent[0]["message"]


def test_out_of_range_position_reports_an_error() -> None:
    backend = Smollm3Backend()
    backend.last_run_state = _run_state()

    ws = _substitute(backend, position=2, token_id=5)

    assert ws.sent[0]["type"] == "error"
    assert "out of range" in ws.sent[0]["message"]
