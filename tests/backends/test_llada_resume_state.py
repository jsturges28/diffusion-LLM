"""Tests that a failed LLaDA resume leaves the retained run intact.

Strategy: drive ``handle_resume`` with a stubbed WebSocket, a stub
streamer reproducing ``FrameStreamer.run``, and a monkeypatched
``streaming_resume`` (no model, no GPU), injecting failure at each
point the resume can break: before the first frame, midway, while
forwarding the sampler's terminal frame, and while sending the
worker's own terminal frame after a guided run stopped at its
budget.

A resume used to shorten ``frame_checkpoints`` before any work,
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
import threading
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
from src.inference.checkpoint import (
    FrameCheckpoint,
    LladaFrame,
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


def _checkpoint(value: int) -> FrameCheckpoint:
    """One recognizable frame, packaged as a checkpoint.

    The confidence is derived from the same value so a test can tell
    which frame a resumed run branched from by either field.
    """
    return FrameCheckpoint(
        ids=_tensor(value),
        canvas_index=0,
        rng=None,
        extra=LladaFrame(
            reveal_conf=torch.full(
                (GEN_LENGTH,), float(value), dtype=torch.float
            )
        ),
    )


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

    async def send_done(
        self,
        frame: Dict[str, Any],
        start_time: float,
    ) -> None:
        """The worker's own terminal frame, forwarded like the rest.

        Present because the resume path sends one whenever the
        sampler stopped at a frame budget, and because an injected
        send failure there has to surface the same way it does for a
        sampler-produced frame.
        """
        del start_time  # Elapsed timings are not under test.
        await self._ws.send_json(frame)

    async def send_cancelled(
        self, final_text: str, start_time: float
    ) -> None:
        """The terminal frame for a run the user stopped.

        A separate method on the real streamer, and separate here,
        because the resume path has to choose between the two: a
        guided edit stopping at its frame budget looks identical
        from the sampler's side and is not a cancellation.
        """
        del start_time  # Elapsed timings are not under test.
        await self._ws.send_json(
            {
                "type": "done",
                "final_text": final_text,
                "cancelled": True,
            }
        )


def _install_resume_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: int,
    fail_after: Optional[int] = None,
    yield_done: bool = True,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Replace the sampler with a scripted frame sequence.

    Mirrors the real generator's contract, which is what makes the
    assertions below mean in a test what they would mean in
    production: each frame's checkpoint is appended to the caller's
    ``frame_checkpoints`` *before* that frame is yielded, so the
    sink is always one step ahead of what the client has seen.

    ``fail_after`` raises once that many frames have been yielded,
    standing in for an inference or streaming failure at a chosen
    point; zero raises before the first frame. ``yield_done`` off
    models a cancelled run, which returns without a terminal frame
    exactly as ``streaming_resume`` does when its cancel event is
    set.

    ``entries`` collects the keyword arguments of each call, so a
    test can ask which checkpoint a resume actually branched from
    rather than inferring it from the frames that came back.
    """

    async def fake_resume(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if entries is not None:
            entries.append(kwargs)
        sink: Optional[List[FrameCheckpoint]] = kwargs.get(
            "frame_checkpoints"
        )
        if fail_after == 0:
            raise RuntimeError("inference failed")
        for index in range(frames):
            if sink is not None:
                sink.append(_checkpoint(100 + index))
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
    generated = [
        _checkpoint(index) for index in range(ORIGINAL_FRAMES)
    ]
    backend.last_run_state = {
        "frame_checkpoints": list(generated),
        # What a rewind restores to, written by _store_state after a
        # real generation. Two lists over one set of checkpoints, so
        # a resume replacing the first leaves this one alone.
        "generated_checkpoints": generated,
        "generated_total_steps": ORIGINAL_TOTAL_STEPS,
        "prompt_ids": torch.zeros((1, 2), dtype=torch.long),
        "attention_mask": torch.ones(
            (1, 2 + GEN_LENGTH), dtype=torch.long
        ),
        "gen_length": GEN_LENGTH,
        "total_steps": ORIGINAL_TOTAL_STEPS,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "seed": 0,
    }
    # Since LIFE-01 a retained run is state plus an identity, and a
    # resume that does not name it is refused before the state is
    # read. A hand-built run has to mint one the way a completed
    # generation would.
    backend.run_counter += 1
    return backend


def _resume(
    backend: LladaBackend,
    ws: _StubWebSocket,
    *,
    frame_index: int = 2,
    max_frames: Optional[int] = None,
    cancelled: bool = False,
) -> None:
    payload: Dict[str, Any] = {
        "frame_index": frame_index,
        "remask_positions": [0, 1],
        "run_token": backend.run_token,
    }
    if max_frames is not None:
        payload["max_frames"] = max_frames
    stop = threading.Event()
    if cancelled:
        stop.set()
    asyncio.run(
        backend.handle_resume(
            ws,  # type: ignore[arg-type]
            payload,
            stop,
            _StubStreamer(ws),  # type: ignore[arg-type]
        )
    )


def _snapshot(backend: LladaBackend) -> List[FrameCheckpoint]:
    state = backend.last_run_state
    assert state is not None
    return list(state["frame_checkpoints"])


def _assert_run_untouched(
    backend: LladaBackend, original: List[FrameCheckpoint]
) -> None:
    """The history and its step count are byte for byte the same.

    Identity rather than equality on purpose: a rebuilt list of
    equal tensors would still mean the worker had thrown the
    originals away, and the frames the browser can resume from are
    the original objects.
    """
    state = backend.last_run_state
    assert state is not None
    history = state["frame_checkpoints"]
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
    assert len(state["frame_checkpoints"]) == frame_index + 3


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
    history = state["frame_checkpoints"]
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
    assert len(state["frame_checkpoints"]) == 4
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

    _resume(backend, ws, frame_index=2, cancelled=True)

    state = backend.last_run_state
    assert state is not None
    assert len(state["frame_checkpoints"]) == 4
    assert state["total_steps"] == ORIGINAL_TOTAL_STEPS
    assert ws.sent[-1]["type"] == "done"
    assert ws.sent[-1]["cancelled"] is True


def test_a_guided_edit_at_its_budget_is_not_a_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction that shares this code path.

    A guided "run to here" stops at the frame the user asked for
    and returns without a terminal frame, exactly as a cancelled
    resume does. Only the stop signal separates them, and marking
    this one stopped would tell the page a completed request had
    been cut short.
    """
    _install_resume_stub(monkeypatch, frames=6)
    backend = _backend()
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2, max_frames=2)

    terminal = ws.sent[-1]
    assert terminal["type"] == "done"
    assert "cancelled" not in terminal


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
        "frame_checkpoints": [
            _checkpoint(0),
            _checkpoint(1),
        ],
        "total_steps": 1,
    }
    kept = list(state["frame_checkpoints"])

    _commit_resume(state, [_checkpoint(0)], [], done=True)

    assert state["frame_checkpoints"] == kept
    assert state["total_steps"] == 1


# -- the rewind --
#
# A resume that succeeds is supposed to replace the retained run.
# What was missing is the way back: every route out of an edit
# session restored the browser and told the worker nothing, so the
# two disagreed about which frames exist and the next edit at the
# same or a later frame branched from the discarded canvas while
# the user clicked tokens on the original one.


def _same_objects(
    left: List[FrameCheckpoint],
    right: List[FrameCheckpoint],
) -> bool:
    """Identity, element for element.

    A checkpoint holds tensors, so ``==`` on two of them raises
    rather than answering. Identity is the stronger claim anyway:
    an equal frame rebuilt from the branch would still mean the
    worker had thrown the original away.
    """
    if len(left) != len(right):
        return False
    return all(
        a is b for a, b in zip(left, right, strict=True)
    )


def _rewind(
    backend: LladaBackend,
    ws: _StubWebSocket,
    *,
    run_token: Optional[str] = None,
) -> None:
    token = (
        backend.run_token if run_token is None else run_token
    )
    asyncio.run(
        backend.handle_rewind(
            ws,  # type: ignore[arg-type]
            {"run_token": token},
        )
    )


def test_a_rewind_restores_the_generated_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resume_stub(monkeypatch, frames=3)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()
    _resume(backend, ws, frame_index=2)
    assert not _same_objects(_snapshot(backend), original), (
        "the resume must commit, or the rewind proves nothing"
    )

    _rewind(backend, ws)

    _assert_run_untouched(backend, original)


def test_a_rewind_undoes_a_whole_chain_of_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guided session commits one resume per Run to Here. The
    browser rolls all of them back from a single snapshot taken
    when the session opened, so the worker has to as well."""
    _install_resume_stub(monkeypatch, frames=3)
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()
    _resume(backend, ws, frame_index=1, max_frames=2)
    _resume(backend, ws, frame_index=2)

    _rewind(backend, ws)

    _assert_run_untouched(backend, original)


def test_one_edit_branches_from_one_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the whole change exists for, and the one a
    hardware pass could not confirm before it.

    Repeating an edit across a rewind must re-enter the identical
    checkpoint, because that object carries the random state
    `XAI-01` retains. Identity rather than equality: an equal
    tensor rebuilt from the branch would still be the wrong canvas.
    """
    entries: List[Dict[str, Any]] = []
    _install_resume_stub(monkeypatch, frames=3, entries=entries)
    backend = _backend()
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2)
    _rewind(backend, ws)
    _resume(backend, ws, frame_index=2)

    assert len(entries) == 2
    assert entries[0]["base_tokens"] is entries[1]["base_tokens"]
    assert entries[0]["base_conf"] is entries[1]["base_conf"]
    assert entries[0]["base_rng"] is entries[1]["base_rng"]


def test_without_a_rewind_the_second_edit_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative space, and a description of the bug. Two edits at
    one frame with nothing between them branch from different
    canvases, which is what made a repeated edit irreproducible."""
    entries: List[Dict[str, Any]] = []
    _install_resume_stub(monkeypatch, frames=3, entries=entries)
    backend = _backend()
    ws = _StubWebSocket()

    _resume(backend, ws, frame_index=2)
    _resume(backend, ws, frame_index=2)

    assert (
        entries[0]["base_tokens"]
        is not entries[1]["base_tokens"]
    )


def test_a_rewind_restores_the_step_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled resume commits a short history without moving
    total_steps, so the figure outlives the run it described."""
    _install_resume_stub(
        monkeypatch, frames=2, yield_done=False
    )
    backend = _backend()
    ws = _StubWebSocket()
    _resume(backend, ws, frame_index=2, cancelled=True)

    _rewind(backend, ws)

    state = backend.last_run_state
    assert state is not None
    assert state["total_steps"] == ORIGINAL_TOTAL_STEPS
    assert len(state["frame_checkpoints"]) == ORIGINAL_FRAMES


def test_a_rewind_before_any_edit_changes_nothing() -> None:
    """Sent on every session open, including the first, so the
    no-op case is the common one."""
    backend = _backend()
    original = _snapshot(backend)
    ws = _StubWebSocket()

    _rewind(backend, ws)

    _assert_run_untouched(backend, original)
    assert ws.sent == []


def test_a_stale_window_cannot_rewind_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal that matters: another window may be mid-edit on
    this run, and rewinding for it would discard live work."""
    _install_resume_stub(monkeypatch, frames=3)
    backend = _backend()
    ws = _StubWebSocket()
    _resume(backend, ws, frame_index=2)
    after_edit = _snapshot(backend)
    ws.sent.clear()

    _rewind(backend, ws, run_token="someone-elses-run")

    _assert_run_untouched(backend, after_edit)
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "error"


def test_a_rewind_without_a_run_is_not_an_error() -> None:
    """A page can open an edit session against a worker that has
    been restarted under it. Refusing here would surface as an
    error the user cannot act on; the resume that follows is what
    tells them the run is gone."""
    backend = LladaBackend()
    backend.tokenizer = _StubTokenizer()
    backend.run_counter += 1
    ws = _StubWebSocket()

    _rewind(backend, ws)

    assert backend.last_run_state is None
    assert ws.sent == []
