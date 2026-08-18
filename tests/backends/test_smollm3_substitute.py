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
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
)

import pytest

from src.backends import smollm3_worker
from src.backends.protocol import (
    ERROR_SCOPE_REQUEST,
    ERROR_STALE_RUN,
)
from src.backends.smollm3_worker import Smollm3Backend

# A tiny vocabulary for the typed-token path. Words in it resolve to
# one entry; anything else falls apart into one entry per character,
# which is the multi-piece case the feature has to refuse.
_VOCAB: Dict[str, int] = {
    "he": 5,
    "she": 7,
    " ran": 11,
    " sat": 13,
}
_REVERSE: Dict[int, str] = {
    value: key for key, value in _VOCAB.items()
}


class _StubTokenizer:
    """Deterministic stand-in with a known one-token vocabulary."""

    is_fast = True
    name_or_path = "stub/tokenizer"
    vocab_size = 128

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> List[int]:
        if text in _VOCAB:
            return [_VOCAB[text]]
        return [ord(char) % 128 for char in text]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        return "".join(
            _REVERSE.get(int(i), chr(int(i))) for i in ids
        )

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
        # Opaque here on purpose: the worker's only job with the cache
        # is to hand the run's own along, and the sampler owns every
        # decision about whether it can be used.
        "cache": _CACHE_SENTINEL,
    }


_CACHE_SENTINEL = object()


def _install_branch_stub(
    monkeypatch: pytest.MonkeyPatch,
    calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Replace the sampler with one that fills any sink it is given.

    Mimics the real generator's contract: it fills ``state_sink``
    when one is passed, which is what the worker must not ask for.
    ``calls`` collects the keyword arguments, so a test can assert
    what the worker decided to hand the sampler.
    """

    async def fake_substitute(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if calls is not None:
            calls.append(kwargs)
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


def _seed_run(
    backend: Smollm3Backend, state: Dict[str, Any]
) -> None:
    """Retain a run the way a finished generation would.

    State alone is no longer enough: `check_run_token` refuses a
    request that does not name the retained run, and it runs before
    anything reads the state, so a test seeding one has to mint the
    identity too.
    """
    backend.last_run_state = state
    backend.run_counter += 1


def _substitute(
    backend: Smollm3Backend,
    position: int,
    token_id: int,
    **extra: Any,
) -> _StubWebSocket:
    ws = _StubWebSocket()
    payload: Dict[str, Any] = {
        "position": position,
        "token_id": token_id,
        # Since LIFE-01 a stateful request names the run it belongs
        # to. `extra` can override it, which is how the stale cases
        # below send a token this backend does not hold.
        "run_token": backend.run_token,
    }
    payload.update(extra)
    asyncio.run(
        backend.handle_substitute(
            ws,  # type: ignore[arg-type]
            payload,
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
    _seed_run(backend, _run_state())

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
    _seed_run(backend, _run_state())

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
    _seed_run(backend, _run_state())

    _substitute(backend, position=0, token_id=7)
    ws = _substitute(backend, position=1, token_id=99)

    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "error"
    assert "not among the captured candidates" in (
        ws.sent[0]["message"]
    )


def test_substitution_is_handed_the_runs_own_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded run's attention state, not the branch's.

    Without it every substitution prefills the whole kept prefix again
    before producing a single token, which is the operation's dominant
    cost. It has to be the pinned run's for the same reason
    ``state_sink`` is None: each substitution re-enters the run the
    user is still looking at.
    """
    calls: List[Dict[str, Any]] = []
    _install_branch_stub(monkeypatch, calls)
    backend = Smollm3Backend()
    _seed_run(backend, _run_state())

    _substitute(backend, position=0, token_id=7)

    assert len(calls) == 1
    assert calls[0]["cache"] is _CACHE_SENTINEL


def test_a_run_without_a_cache_still_substitutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs recorded before caches were kept, and runs whose cache
    passed the ceiling, both reach here with the key absent. The
    sampler prefills in that case, so the absence must arrive as None
    rather than as a KeyError."""
    calls: List[Dict[str, Any]] = []
    _install_branch_stub(monkeypatch, calls)
    backend = Smollm3Backend()
    state = _run_state()
    del state["cache"]
    _seed_run(backend, state)

    ws = _substitute(backend, position=0, token_id=7)

    assert not any(
        m.get("type") == "error" for m in ws.sent
    )
    assert calls[0]["cache"] is None


def test_substitute_without_a_run_reports_an_error() -> None:
    """Asserts the code, not the sentence. Holding no run is the
    degenerate case of holding a different one, so it now answers
    with the stale code, and pinning prose would make this test fail
    on a reworded message rather than on a behaviour change."""
    backend = Smollm3Backend()

    ws = _substitute(backend, position=0, token_id=5)

    assert ws.sent[0]["type"] == "error"
    assert ws.sent[0]["code"] == ERROR_STALE_RUN


def test_out_of_range_position_reports_an_error() -> None:
    backend = Smollm3Backend()
    _seed_run(backend, _run_state())

    ws = _substitute(backend, position=2, token_id=5)

    assert ws.sent[0]["type"] == "error"
    assert "out of range" in ws.sent[0]["message"]


# -- the typed-token path --
#
# A second, explicitly flagged branch through the same validation.
# These tests exist to prove it is a branch and not a loosening: the
# captured path above must keep rejecting everything it rejected
# before, and the typed path must do its own checking rather than
# trusting the client that set the flag.


def _typed_backend() -> Smollm3Backend:
    backend = Smollm3Backend()
    _seed_run(backend, _run_state())
    backend.tokenizer = _StubTokenizer()
    return backend


def test_typed_token_is_accepted_when_it_is_one_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"she" is one vocabulary entry, so it may be forced."""
    _install_branch_stub(monkeypatch)
    backend = _typed_backend()

    ws = _substitute(
        backend,
        position=0,
        token_id=7,
        typed=True,
        typed_text="she",
    )

    assert not any(m.get("type") == "error" for m in ws.sent)


def test_typed_token_leaves_confidence_for_the_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed token has no recorded probability to report.

    Passing None is what tells the sampler to measure the real one
    off the distribution it computes anyway. Inventing a number here
    would quietly make the readout a fiction.
    """
    calls: List[Dict[str, Any]] = []
    _install_branch_stub(monkeypatch, calls)
    backend = _typed_backend()

    _substitute(
        backend,
        position=0,
        token_id=7,
        typed=True,
        typed_text="she",
    )

    assert len(calls) == 1
    assert calls[0]["forced_conf"] is None


def test_captured_token_keeps_its_recorded_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair to the test above: the old path is unchanged."""
    calls: List[Dict[str, Any]] = []
    _install_branch_stub(monkeypatch, calls)
    backend = _typed_backend()

    _substitute(backend, position=0, token_id=7)

    assert len(calls) == 1
    assert calls[0]["forced_conf"] == pytest.approx(0.2)


def test_multi_token_text_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one token, because everything downstream is indexed.

    A replacement of any other length would shift every position
    after it, which the diff, the entropy chart and the edit marker
    all read positionally.
    """
    _install_branch_stub(monkeypatch)
    backend = _typed_backend()

    ws = _substitute(
        backend,
        position=0,
        token_id=7,
        typed=True,
        typed_text="hex",
    )

    assert ws.sent[0]["type"] == "error"
    assert "3 tokens" in ws.sent[0]["message"]


def test_typed_id_must_match_the_resolved_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative space: a stale preview cannot force a stray id.

    The client sends both the text and the id it believes the text
    resolves to. Re-resolving server side and comparing is what stops
    a reply that arrived for an earlier draft from substituting a
    token the user never saw.
    """
    _install_branch_stub(monkeypatch)
    backend = _typed_backend()

    ws = _substitute(
        backend,
        position=0,
        token_id=5,
        typed=True,
        typed_text="she",
    )

    assert ws.sent[0]["type"] == "error"
    assert "resolves to token 7" in ws.sent[0]["message"]


def test_typed_with_no_text_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_branch_stub(monkeypatch)
    backend = _typed_backend()

    ws = _substitute(
        backend,
        position=0,
        token_id=7,
        typed=True,
        typed_text="",
    )

    assert ws.sent[0]["type"] == "error"
    assert "No text was typed" in ws.sent[0]["message"]


def test_unflagged_request_still_needs_a_captured_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed path must not become a back door.

    Token 42 is a real vocabulary id but was never captured at this
    position, so without the typed flag it is still refused.
    """
    _install_branch_stub(monkeypatch)
    backend = _typed_backend()

    ws = _substitute(backend, position=0, token_id=42)

    assert ws.sent[0]["type"] == "error"
    assert "not among the captured candidates" in (
        ws.sent[0]["message"]
    )


# -- handle_probe --
#
# The measurement itself belongs to probe_token and is covered in
# tests/inference/test_ar_sampler.py. What is checked here is the
# worker's half: validation before any forward pass is scheduled, and
# a reply the client can match to the request that asked for it.


def _install_probe_stub(
    monkeypatch: pytest.MonkeyPatch,
    calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Replace the measurement with a fixed, recognizable answer."""

    def fake_probe(**kwargs: Any) -> Dict[str, Any]:
        if calls is not None:
            calls.append(kwargs)
        return {
            "probability": 0.0123,
            "rank": 407,
            "vocab_size": 128,
        }

    monkeypatch.setattr(
        smollm3_worker, "probe_token", fake_probe
    )


def _probe(
    backend: Smollm3Backend, **payload: Any
) -> _StubWebSocket:
    ws = _StubWebSocket()
    request: Dict[str, Any] = {"run_token": backend.run_token}
    request.update(payload)
    asyncio.run(
        backend.handle_probe(
            ws,  # type: ignore[arg-type]
            request,
        )
    )
    return ws


def test_probe_reports_the_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_stub(monkeypatch)
    backend = _typed_backend()
    backend.model = object()

    ws = _probe(
        backend, position=1, token_id=42, request_id=3
    )

    assert len(ws.sent) == 1
    reply = ws.sent[0]
    assert reply["type"] == "probe_result"
    assert reply["probability"] == pytest.approx(0.0123)
    assert reply["rank"] == 407
    assert reply["vocab_size"] == 128


def test_probe_echoes_what_it_was_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request id and token come back so a stale reply is dropped.

    A Retry between request and reply changes which token the row
    describes, and without the echo the client would label the new
    token with the old one's odds.
    """
    _install_probe_stub(monkeypatch)
    backend = _typed_backend()
    backend.model = object()

    ws = _probe(
        backend, position=1, token_id=42, request_id=9
    )

    assert ws.sent[0]["request_id"] == 9
    assert ws.sent[0]["token_id"] == 42
    assert ws.sent[0]["position"] == 1


def test_probe_prefills_only_up_to_the_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefix must stop before the position being asked about.

    Handing over one token too many would measure the position after
    it, which is the one mistake here that still returns a plausible
    number rather than an error.
    """
    calls: List[Dict[str, Any]] = []
    _install_probe_stub(monkeypatch, calls)
    backend = _typed_backend()
    backend.model = object()

    _probe(backend, position=1, token_id=42, request_id=1)

    assert len(calls) == 1
    assert calls[0]["prefix_ids"] == [5]


def test_probe_is_handed_the_runs_own_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With it, a token the run captured measures to its recorded
    probability exactly. Without it the two land a bf16 rounding step
    apart, which is visible in the UI as a percentage that disagrees
    with the row above it."""
    calls: List[Dict[str, Any]] = []
    _install_probe_stub(monkeypatch, calls)
    backend = _typed_backend()
    backend.model = object()

    _probe(backend, position=1, token_id=42, request_id=1)

    assert calls[0]["cache"] is _CACHE_SENTINEL


def test_probe_rejects_a_position_outside_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated exactly as a substitution is, and before the pass.

    A probe that answered for a position the substitution would
    refuse is worse than no answer: the strip would quote odds for an
    edit that cannot be made.
    """
    _install_probe_stub(monkeypatch)
    backend = _typed_backend()
    backend.model = object()

    ws = _probe(
        backend, position=9, token_id=42, request_id=1
    )

    assert ws.sent[0]["type"] == "error"
    assert "out of range" in ws.sent[0]["message"]


def test_probe_rejects_a_negative_token_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lower bound, which is the only one checkable here."""
    _install_probe_stub(monkeypatch)
    backend = _typed_backend()
    backend.model = object()

    ws = _probe(
        backend, position=0, token_id=-1, request_id=1
    )

    assert ws.sent[0]["type"] == "error"
    assert "not valid" in ws.sent[0]["message"]


def test_probe_without_a_run_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operating error, not a crash: nothing has run yet. Scoped
    to the request, so a page in the middle of What If is told the
    measurement failed and nothing else is disturbed."""
    _install_probe_stub(monkeypatch)
    backend = Smollm3Backend()
    backend.model = object()

    ws = _probe(
        backend, position=0, token_id=7, request_id=1
    )

    assert ws.sent[0]["type"] == "error"
    assert ws.sent[0]["code"] == ERROR_STALE_RUN
    assert ws.sent[0]["scope"] == ERROR_SCOPE_REQUEST
