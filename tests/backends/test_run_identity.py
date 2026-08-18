"""Tests for the run token that fences retained worker state.

Strategy: drive `Backend` directly with a stub that records what it
was asked, because the property under test is about which requests
are answered at all, not about what a model computes. One backend
instance is one worker process, which is exactly the sharing that
makes this a problem: every browser window is proxied to the same
one, and it keeps exactly one run.

What passing proves is `LIFE-01`'s clause. A second window completing
a generation replaces the state behind the first window's still
visible run, and before this the first window's resume, substitution
or probe was answered from whatever the last generation left. The
report is specific about why that is worse than an error: if the two
runs' shapes happen to agree the operation succeeds against the wrong
prompt, and the result looks valid. So the interleaving tests below
give both runs identical shapes on purpose, which is the case a
bounds check alone cannot catch.

The nonce is the other half. A bare counter restarts at one in a
fresh worker, and a page is not always reloaded when its worker is
replaced: `handleResident` forces that only on a model or device
change, so reloading the same model leaves a browser holding a token
a bare counter would hand out again.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from src.backends.protocol import (
    ERROR_SCOPE_REQUEST,
    ERROR_SCOPE_RUN,
    ERROR_STALE_RUN,
)
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    StaleRunError,
)


class _StubWebSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


class _Backend(Backend):
    """The smallest thing that can hold a run.

    Deliberately not one of the three real backends: this is about
    the base class's contract, which all three inherit, and a stub
    keeps a torch import out of a test about bookkeeping.
    """

    def __init__(self) -> None:
        self.model_info = None  # type: ignore[assignment]

    def load(self, *, device: str = "cuda") -> None:
        raise NotImplementedError

    async def handle_generate(
        self,
        ws: Any,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """What every real handle_generate does, minus the model."""
        self.begin_run()
        self.last_run_state = {"prompt": data.get("prompt", "")}


def _finish_run(backend: _Backend, prompt: str) -> str:
    """Complete a generation and return the token it minted."""
    asyncio.run(
        backend.handle_generate(
            _StubWebSocket(),
            {"prompt": prompt},
            asyncio.Event(),
            None,
        )
    )
    return backend.run_token


# -- minting --


def test_a_fresh_worker_holds_no_run() -> None:
    assert _Backend().run_token == ""


def test_finishing_a_run_names_it() -> None:
    backend = _Backend()

    token = _finish_run(backend, "one")

    assert token != ""
    assert backend.run_token == token


def test_each_run_gets_its_own_name() -> None:
    backend = _Backend()

    first = _finish_run(backend, "one")
    second = _finish_run(backend, "two")

    assert first != second


def test_two_workers_never_agree() -> None:
    """The nonce, and the reason for it. Both counters are at one
    here, which is precisely the collision a bare counter allows
    after a worker is replaced by one loading the same model."""
    first = _Backend()
    second = _Backend()

    first_token = _finish_run(first, "one")
    second_token = _finish_run(second, "one")

    assert first.run_counter == second.run_counter == 1
    assert first_token != second_token


def test_beginning_a_run_discards_the_previous_state() -> None:
    """State and token retire together. A token outliving its state
    would name a run the worker cannot answer for, and state
    outliving its token would be reachable by a stale request."""
    backend = _Backend()
    _finish_run(backend, "one")

    backend.begin_run()

    assert backend.last_run_state is None


# -- checking --


def test_the_holder_of_the_token_is_admitted() -> None:
    backend = _Backend()
    token = _finish_run(backend, "one")

    backend.check_run_token({"run_token": token})


def test_a_superseded_token_is_refused() -> None:
    backend = _Backend()
    stale = _finish_run(backend, "one")
    _finish_run(backend, "two")

    with pytest.raises(StaleRunError):
        backend.check_run_token({"run_token": stale})


def test_a_request_naming_no_run_is_refused() -> None:
    """Silence is not consent. A client that sends no token cannot be
    assumed to mean the current run, because that assumption is
    exactly the old behaviour."""
    backend = _Backend()
    _finish_run(backend, "one")

    with pytest.raises(StaleRunError):
        backend.check_run_token({})


def _refusal_message(backend: _Backend, data: Dict[str, Any]) -> str:
    with pytest.raises(StaleRunError) as caught:
        backend.check_run_token(data)
    return str(caught.value)


def test_a_missing_token_reads_differently_from_a_stale_one() -> None:
    """Both are refused and both carry the same code, since the
    client's recourse is the same. The sentences differ because the
    situations do: a superseded token is an ordinary consequence of
    using two windows, while a missing one means a client that never
    learned which run it was looking at, which is a bug to report
    rather than an outcome to explain.

    Asserted as a difference rather than as two exact strings, so
    rewording either stays free.
    """
    backend = _Backend()
    stale = _finish_run(backend, "one")
    _finish_run(backend, "two")

    missing = _refusal_message(backend, {})
    superseded = _refusal_message(backend, {"run_token": stale})

    assert missing != superseded


def test_a_worker_with_no_run_refuses_everything() -> None:
    backend = _Backend()

    with pytest.raises(StaleRunError):
        backend.check_run_token({"run_token": "anything"})


def test_a_token_from_another_worker_is_refused() -> None:
    backend = _Backend()
    other = _Backend()
    foreign = _finish_run(other, "one")
    _finish_run(backend, "one")

    with pytest.raises(StaleRunError):
        backend.check_run_token({"run_token": foreign})


def test_the_empty_token_does_not_pass_for_no_run() -> None:
    """A page that never saw a terminal frame sends "", which must
    not match a worker whose own token is "" for having no run."""
    backend = _Backend()

    with pytest.raises(StaleRunError):
        backend.check_run_token({"run_token": ""})


# -- two windows, one worker --


def test_a_second_windows_run_locks_out_the_first() -> None:
    """The finding, at the level the base class owns it."""
    worker = _Backend()
    window_one = _finish_run(worker, "the first prompt")
    window_two = _finish_run(worker, "the second prompt")

    with pytest.raises(StaleRunError):
        worker.check_run_token({"run_token": window_one})
    worker.check_run_token({"run_token": window_two})


def test_matching_shapes_do_not_let_a_stale_request_through() -> None:
    """The clause's "repeat with equal frame counts and equal output
    lengths". Identical runs are the dangerous case, because every
    check that works on shape passes and the answer looks right."""
    worker = _Backend()
    window_one = _finish_run(worker, "identical")
    window_two = _finish_run(worker, "identical")

    assert worker.last_run_state == {"prompt": "identical"}
    with pytest.raises(StaleRunError):
        worker.check_run_token({"run_token": window_one})
    worker.check_run_token({"run_token": window_two})


def test_the_state_reached_is_the_state_the_token_names() -> None:
    """Not just that the stale one is refused, but that the admitted
    one reaches the run it asked for."""
    worker = _Backend()
    _finish_run(worker, "replaced")
    current = _finish_run(worker, "current")

    worker.check_run_token({"run_token": current})

    assert worker.last_run_state is not None
    assert worker.last_run_state["prompt"] == "current"


# -- what a refusal looks like on the wire --


def _refusal(request_type: str) -> Dict[str, Any]:
    from src.backends.protocol import request_error

    return dict(
        request_error(
            message="stale",
            code=ERROR_STALE_RUN,
            request_type=request_type,
        )
    )


def test_a_stale_resume_ends_the_run_it_was_editing() -> None:
    """Run-scoped, because the client truncated the run on screen
    before asking, so a refusal has to roll that back."""
    assert _refusal("resume")["scope"] == ERROR_SCOPE_RUN


def test_a_stale_probe_disturbs_nothing() -> None:
    """Request-scoped. A measurement that cannot be taken is not a
    reason to close What If."""
    assert _refusal("probe")["scope"] == ERROR_SCOPE_REQUEST


def test_a_refusal_is_told_apart_from_a_bad_request() -> None:
    """Distinct codes, because the client's recourse differs: a stale
    run wants a fresh generation, a malformed one wants correcting."""
    from src.backends.protocol import ERROR_INVALID_REQUEST

    assert ERROR_STALE_RUN != ERROR_INVALID_REQUEST


# -- the terminal frame is where the token reaches the client --


def test_the_streamer_names_the_run_on_every_done() -> None:
    """One place stamps it, because a path that forgot would hand the
    browser a run it could never act on."""
    ws = _StubWebSocket()
    backend = _Backend()
    _finish_run(backend, "one")
    stream = FrameStreamer(
        ws, run_token=lambda: backend.run_token  # type: ignore[arg-type]
    )

    asyncio.run(
        stream.send_done({"type": "done", "final_text": "x"}, 0.0)
    )

    assert ws.sent[0]["run_token"] == backend.run_token


def test_a_streamed_done_is_named_too() -> None:
    """The ordinary path, alongside the synthesized one above."""
    ws = _StubWebSocket()
    backend = _Backend()
    _finish_run(backend, "one")
    stream = FrameStreamer(
        ws, run_token=lambda: backend.run_token  # type: ignore[arg-type]
    )

    async def frames() -> Any:
        yield {"type": "frame", "index": 0}
        yield {"type": "done", "final_text": "x"}

    asyncio.run(stream.run(frames(), 0.0))

    assert "run_token" not in ws.sent[0]
    assert ws.sent[1]["run_token"] == backend.run_token


def test_a_streamer_with_no_source_stamps_nothing() -> None:
    """The stub streamers in the other backend tests pass no token
    provider, and must not gain a null field for it."""
    ws = _StubWebSocket()
    stream = FrameStreamer(ws)  # type: ignore[arg-type]

    asyncio.run(
        stream.send_done({"type": "done", "final_text": "x"}, 0.0)
    )

    assert "run_token" not in ws.sent[0]
