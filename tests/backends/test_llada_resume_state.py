"""Tests that a failed LLaDA resume leaves the retained run intact.

Strategy: drive ``handle_resume`` with a stubbed WebSocket, a stub
streamer reproducing ``FrameStreamer.run``, and a monkeypatched
``streaming_resume`` (no model, no GPU), injecting failure at each
point the resume can break: before the first frame, midway, while
forwarding the sampler's terminal frame, and while sending the
worker's own terminal frame after a guided run stopped at its
budget.

A resume used to shorten ``tensor_history`` before doing any work,
so any of those failures left the worker holding a truncated run
while the browser rolled back to its full pre-edit snapshot. The
two then disagreed about which frames exist, and the next retry
either fell out of range or branched from a different frame than
the one on screen. Passing proves the history and its step count
survive every failure byte for byte, that a second resume from any
original frame still works, and that the outcomes which *are*
terminal (a completed run, a guided stop, a cancel) still commit.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
)

import pytest
import torch

from src.backends import llada_worker
from src.backends.llada_worker import (
    LladaBackend,
    _commit_resume,
)

# The recorded run every test resumes from: five frames, so the
# last resumable one is frame 3 (frame 4 is the final frame and is
# refused by design).
ORIGINAL_FRAMES = 5
ORIGINAL_TOTAL_STEPS = ORIGINAL_FRAMES - 1
GEN_LENGTH = 8

assert ORIGINAL_TOTAL_STEPS > 1, (
    "the run needs more than one resumable frame"
)


def _tensor(value: int) -> torch.Tensor:
    """One recognizable generation-region frame."""
    return torch.full((1, GEN_LENGTH), value, dtype=torch.long)


class _StubTokenizer:
    """Decodes a frame to a string naming the tensor it came from."""

    def batch_decode(
        self,
        tokens: torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> List[str]:
        return ["frame-" + str(int(tokens[0][0]))]


class _StubWebSocket:
    """Collects payloads, optionally failing on a chosen send.

    ``fail_on`` is consulted before the payload is recorded, so an
    injected failure is indistinguishable from a socket that died
    mid-send: the client never received that message.
    """

    def __init__(
        self,
        fail_on: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        self.sent: List[Dict[str, Any]] = []
        self._fail_on = fail_on

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if self._fail_on is not None and self._fail_on(payload):
            raise RuntimeError("socket send failed")
        self.sent.append(dict(payload))


class _StubStreamer:
    """``FrameStreamer.run`` without a real WebSocket.

    Reproduces the two behaviors the resume path depends on: every
    frame is forwarded to the socket, so an injected send failure
    surfaces exactly where it would in production, and the loop
    stops once ``max_frames`` frames have gone out, returning
    whether a terminal frame was sent.
    """

    def __init__(self, ws: _StubWebSocket) -> None:
        self._ws = ws

    async def run(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
        *,
        max_frames: Optional[int] = None,
    ) -> bool:
        del start_time  # Elapsed timings are not under test.
        frame_count = 0
        done_sent = False
        async for frame in generator:
            await self._ws.send_json(frame)
            if frame.get("type") == "done":
                done_sent = True
            elif frame.get("type") == "frame":
                frame_count += 1
                if (
                    max_frames is not None
                    and frame_count >= max_frames
                ):
                    break
        return done_sent


def _install_resume_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: int,
    fail_after: Optional[int] = None,
    yield_done: bool = True,
) -> None:
    """Replace the sampler with a scripted frame sequence.

    Mirrors the real generator's contract, which is what makes the
    assertions below mean in a test what they would mean in
    production: each frame's tensor is appended to the caller's
    ``tensor_history`` *before* that frame is yielded, so the sink
    is always one step ahead of what the client has seen.

    ``fail_after`` raises once that many frames have been yielded,
    standing in for an inference or streaming failure at a chosen
    point; zero raises before the first frame. ``yield_done`` off
    models a cancelled run, which returns without a terminal frame
    exactly as ``streaming_resume`` does when its cancel event is
    set.
    """

    async def fake_resume(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        sink: Optional[List[torch.Tensor]] = kwargs.get(
            "tensor_history"
        )
        if fail_after == 0:
            raise RuntimeError("inference failed")
        for index in range(frames):
            if sink is not None:
                sink.append(_tensor(100 + index))
            yield {"type": "frame", "index": index}
            if (
                fail_after is not None
                and index + 1 >= fail_after
            ):
                raise RuntimeError("inference failed")
        if yield_done:
            yield {"type": "done", "final_text": "resumed"}

    monkeypatch.setattr(
        llada_worker, "streaming_resume", fake_resume
    )


def _backend() -> LladaBackend:
    """A backend holding the recorded run, ready to resume."""
    backend = LladaBackend()
    backend.tokenizer = _StubTokenizer()
    backend.last_run_state = {
        "tensor_history": [
            _tensor(index) for index in range(ORIGINAL_FRAMES)
        ],
        "prompt_ids": torch.zeros((1, 2), dtype=torch.long),
        "attention_mask": torch.ones(
            (1, 2 + GEN_LENGTH), dtype=torch.long
        ),
        "gen_length": GEN_LENGTH,
        "total_steps": ORIGINAL_TOTAL_STEPS,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
    }
    return backend


def _resume(
    backend: LladaBackend,
    ws: _StubWebSocket,
    *,
    frame_index: int = 2,
    max_frames: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "frame_index": frame_index,
        "remask_positions": [0, 1],
    }
    if max_frames is not None:
        payload["max_frames"] = max_frames
    asyncio.run(
        backend.handle_resume(
            ws,  # type: ignore[arg-type]
            payload,
            asyncio.Event(),
            _StubStreamer(ws),  # type: ignore[arg-type]
        )
    )


def _snapshot(backend: LladaBackend) -> List[torch.Tensor]:
    state = backend.last_run_state
    assert state is not None
    return list(state["tensor_history"])


def _assert_run_untouched(
    backend: LladaBackend, original: List[torch.Tensor]
) -> None:
    """The history and its step count are byte for byte the same.

    Identity rather than equality on purpose: a rebuilt list of
    equal tensors would still mean the worker had thrown the
    originals away, and the frames the browser can resume from are
    the original objects.
    """
    state = backend.last_run_state
    assert state is not None
    history = state["tensor_history"]
    assert len(history) == len(original), (
        "a failed resume changed the frame count"
    )
    for actual, expected in zip(
        history, original, strict=True
    ):
        assert actual is expected, (
            "a failed resume replaced a retained frame"
        )
    assert state["total_steps"] == ORIGINAL_TOTAL_STEPS, (
        "a failed resume moved the step count"
    )


def _assert_reported_error(ws: _StubWebSocket) -> None:
    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert len(errors) == 1, "the failure was not reported once"


# -- the four injected failures --


def test_failure_before_the_first_frame_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resume_stub(monkeypatch, frames=3, fail_after=0)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _resume(backend, ws)

    _assert_run_untouched(backend, original)
    _assert_reported_error(ws)


def test_failure_midway_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that used to lose the most: frames streamed, so the
    browser is mid-edit, and then inference died."""
    _install_resume_stub(monkeypatch, frames=4, fail_after=2)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _resume(backend, ws)

    _assert_run_untouched(backend, original)
    _assert_reported_error(ws)


def test_failure_forwarding_the_terminal_frame_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every frame arrived and the sampler finished; the socket
    then died on the one message that makes the run terminal."""
    _install_resume_stub(monkeypatch, frames=3)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket(
        fail_on=lambda p: p.get("type") == "done"
    )

    _resume(backend, ws)

    _assert_run_untouched(backend, original)
    _assert_reported_error(ws)


def test_failure_sending_the_guided_terminal_frame_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-resume target: the streamer stopped at the
    budget, so the terminal message is the worker's own."""
    _install_resume_stub(monkeypatch, frames=4)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket(
        fail_on=lambda p: p.get("type") == "done"
    )

    _resume(backend, ws, max_frames=2)

    _assert_run_untouched(backend, original)
    _assert_reported_error(ws)


@pytest.mark.parametrize("frame_index", [0, 3])
def test_a_second_resume_succeeds_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, frame_index: int
) -> None:
    """The point of preserving the history: the user retries.

    Both ends of the resumable range, because a truncating failure
    used to make the later frames unreachable while leaving the
    earlier ones pointing at a run that no longer existed.
    """
    _install_resume_stub(monkeypatch, frames=3, fail_after=1)
    backend = _backend()
    original = _snapshot(backend)
    _resume(backend, _StubWebSocket(), frame_index=frame_index)
    _assert_run_untouched(backend, original)

    _install_resume_stub(monkeypatch, frames=3)
    retry = _StubWebSocket()
    _resume(backend, retry, frame_index=frame_index)

    assert not any(
        m.get("type") == "error" for m in retry.sent
    ), "the retry was rejected"
    state = backend.last_run_state
    assert state is not None
    assert len(state["tensor_history"]) == frame_index + 3


# -- the outcomes that do commit --


def test_a_completed_resume_replaces_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resume_stub(monkeypatch, frames=3)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2)

    state = backend.last_run_state
    assert state is not None
    history = state["tensor_history"]
    assert len(history) == 5
    assert state["total_steps"] == 4
    assert not any(
        m.get("type") == "error" for m in ws.sent
    )
    # The surviving prefix is the original frames themselves, which
    # is what makes staging free: nothing was copied to roll back.
    for index in range(2):
        assert history[index] is original[index]


def test_a_guided_resume_commits_without_moving_total_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "run to here" stops short on purpose, so the step count
    still describes the run it branched from."""
    _install_resume_stub(monkeypatch, frames=4)
    backend = _backend()
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2, max_frames=2)

    state = backend.last_run_state
    assert state is not None
    # Two frames reached the client, and the sink holds exactly
    # those, so worker and browser agree on the new length.
    assert len(state["tensor_history"]) == 4
    assert state["total_steps"] == ORIGINAL_TOTAL_STEPS


def test_a_cancelled_resume_keeps_the_frames_the_client_saw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation is a terminal outcome, not a failure.

    The sampler returns without a terminal frame and the browser
    keeps what it received, so discarding here would recreate the
    same disagreement in the other direction.
    """
    _install_resume_stub(
        monkeypatch, frames=2, yield_done=False
    )
    backend = _backend()
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2)

    state = backend.last_run_state
    assert state is not None
    assert len(state["tensor_history"]) == 4
    assert state["total_steps"] == ORIGINAL_TOTAL_STEPS
    assert ws.sent[-1]["type"] == "done"


# -- a rejected request never reaches the staging at all --


def test_an_out_of_range_frame_leaves_the_run_alone() -> None:
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=ORIGINAL_FRAMES + 1)

    _assert_run_untouched(backend, original)
    assert "out of range" in ws.sent[0]["message"]


def test_the_final_frame_is_refused() -> None:
    """Negative space: there are no steps left to run from it."""
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=ORIGINAL_FRAMES - 1)

    _assert_run_untouched(backend, original)
    assert "final frame" in ws.sent[0]["message"]


# -- the commit itself --


def test_commit_refuses_an_empty_candidate() -> None:
    """A terminal outcome always carries frames, so an empty one
    is a broken caller. Committing it would cut the run back to
    the surviving prefix, which is the loss being prevented."""
    state: Dict[str, Any] = {
        "tensor_history": [_tensor(0), _tensor(1)],
        "total_steps": 1,
    }
    kept = list(state["tensor_history"])

    _commit_resume(state, [_tensor(0)], [], done=True)

    assert state["tensor_history"] == kept
    assert state["total_steps"] == 1
