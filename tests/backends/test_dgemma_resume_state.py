"""Tests that a DiffusionGemma resume keeps worker and page agreed.

Strategy: drive ``DgemmaBackend.handle_resume`` and ``handle_rewind``
with no model and no GPU. The run is six hand-built checkpoints,
stored the way a finished generation stores them, and every message
goes out through the real ``FrameStreamer`` to a socket that records
it, so each terminal frame carries the provenance and run token a
page would receive. Most tests replace the sampler with a scripted
one that keeps the real sampler's contract. The tests that stop a
resume partway run the real ``streaming_resume`` over a stub model
instead, because the hand-off between its model thread and the
worker is where a stop takes effect.

The worker cannot be imported here as it ships. Its NF4 loader
imports ``bitsandbytes`` at module level, which only ``.venv-dgemma``
installs, so the ``worker`` fixture stands in for that one module.
Nothing a resume or a rewind does reaches the loader.

LLaDA has had a suite like this since a failed resume truncated its
history (``test_llada_resume_state.py``). DiffusionGemma had only
sampler tests, which cannot see the history the worker keeps or the
terminal frame it builds for a guided edit. Passing proves that a
resume keeps exactly the frames the page received, and changes
nothing when none did, whether it completes, stops at a guided
budget, is stopped by the user or fails; that a guided edit ends
with the text of the last frame the page received, marked stopped
only when a stop cut it short; that a rewind returns the generated
run, object for object; and that a request the worker cannot
honour, a multi-canvas run or a malformed budget among them, is
refused before the sampler runs.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import pytest
import torch

from src.backends.context_pack import MessageRecord
from src.backends.protocol import (
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_STALE_RUN,
    RESUME_CONTINUE,
    TERMINAL_CANCELLED,
)
from src.backends.text_adapter import DGEMMA_TEXT
from src.backends.worker_base import FrameStreamer
from src.inference.checkpoint import DgemmaFrame, FrameCheckpoint

if TYPE_CHECKING:
    # For annotations only. Importing the worker for real needs the
    # stand-in the ``worker`` fixture installs.
    from src.backends.dgemma_worker import DgemmaBackend

NF4_MODULE = "src.inference.dgemma_nf4"
WORKER_MODULE = "src.backends.dgemma_worker"

# The recorded run every test resumes from, on a four-position canvas.
CANVAS_LENGTH = 4
ORIGINAL_FRAMES = 6
LAST_FRAME = ORIGINAL_FRAMES - 1
# Mid-run, so a resume has recorded frames on both sides of it.
RESUME_FRAME = 3

# A scripted branch's checkpoints hold this value and up, which no
# recorded frame does, so a test can tell the two apart by content.
BRANCH_VALUE = 100

# The stub model's drafts hold this value and up, one more per step.
DRAFT_VALUE = 20
# Comfortably more drafts than any test lets reach the page.
DRAFT_STEPS = 6
PROMPT_LENGTH = 2
VOCAB_SIZE = 64

assert 0 < RESUME_FRAME < LAST_FRAME, "the edit sits mid-run"
assert BRANCH_VALUE > LAST_FRAME, "branch frames are recognizable"
assert DRAFT_VALUE > LAST_FRAME, "drafted frames are recognizable"
assert DRAFT_VALUE + DRAFT_STEPS < VOCAB_SIZE, "drafts are token ids"

# What a finished generation would have stored. A negative seed
# leaves the process's random state alone.
RUN_PARAMS: Dict[str, Any] = {
    "prompt": "continue the story",
    "t_max": 0.8,
    "t_min": 0.4,
    "confidence_threshold": 0.005,
    "stability_threshold": 1,
    "thinking": False,
    "seed": -1,
    "alternatives": False,
    "max_denoising_steps": 48,
}

PROVENANCE: Dict[str, str] = {"model_id": "stub"}

# The scripted sampler's own terminal text, unlike any frame's.
SAMPLER_TEXT = "sampler-final"


# -- the worker, without its loader --


def _refuse_load(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("a resume test must never load weights")


@pytest.fixture(scope="module")
def worker() -> Iterator[ModuleType]:
    """The DiffusionGemma worker, imported over a stand-in loader.

    Only the NF4 module is replaced; everything else the worker
    imports is the code that ships. Both entries leave
    ``sys.modules`` afterwards, so no later test can import the
    stand-in by accident.
    """
    stand_in = ModuleType(NF4_MODULE)
    stand_in.load_quantized = (  # type: ignore[attr-defined]
        _refuse_load
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, NF4_MODULE, stand_in)
        # Fresh, so the module under test is bound to the stand-in
        # even if something imported the worker earlier.
        patch.delitem(sys.modules, WORKER_MODULE, raising=False)
        module = importlib.import_module(WORKER_MODULE)
        assert module.load_quantized is _refuse_load, (
            "the worker must be bound to the stand-in loader"
        )
        try:
            yield module
        finally:
            sys.modules.pop(WORKER_MODULE, None)


# -- what the page receives --


class _RecordingSocket:
    """Collects what the worker sends, as the page would receive it.

    ``fail_on`` is consulted before a payload is recorded, so an
    injected failure is a message the page never received.
    ``stop_after`` sets ``stop`` once that many frames have arrived,
    which is the user pressing Stop with that many on screen.
    """

    def __init__(
        self,
        *,
        fail_on: Optional[Callable[[Dict[str, Any]], bool]] = None,
        stop: Optional[threading.Event] = None,
        stop_after: Optional[int] = None,
    ) -> None:
        if stop_after is not None:
            assert stop is not None, "a stop needs an event to set"
            assert stop_after > 0, (
                "to stop before any frame, set the event up front"
            )
        self.sent: List[Dict[str, Any]] = []
        self._fail_on = fail_on
        self._stop = stop
        self._stop_after = stop_after

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if self._fail_on is not None and self._fail_on(payload):
            raise RuntimeError("socket send failed")
        self.sent.append(dict(payload))
        self._press_stop_when_due()

    def _press_stop_when_due(self) -> None:
        if self._stop is None or self._stop_after is None:
            return
        if len(self.frames()) >= self._stop_after:
            self._stop.set()

    def kinds(self) -> List[str]:
        return [str(message["type"]) for message in self.sent]

    def frames(self) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m["type"] == "frame"]

    def terminals(self) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m["type"] == "done"]

    def errors(self) -> List[Dict[str, Any]]:
        return [m for m in self.sent if m["type"] == "error"]


def _streamer(
    backend: DgemmaBackend, socket: _RecordingSocket
) -> FrameStreamer:
    """The streamer the worker's socket loop builds, over
    ``socket``."""
    return FrameStreamer(
        socket,  # type: ignore[arg-type]
        provenance=lambda: dict(PROVENANCE),
        run_token=lambda: backend.run_token,
    )


# -- the recorded run --


def _canvas_of(index: int, *, canvases: int) -> int:
    """The canvas recorded frame ``index`` sits on.

    A two-canvas run commits its first canvas halfway through.
    """
    assert canvases in (1, 2), "one canvas or two"
    if canvases == 1:
        return 0
    if index < ORIGINAL_FRAMES // 2:
        return 0
    return 1


def _checkpoint(
    value: int, *, canvas_index: int = 0
) -> FrameCheckpoint:
    """One recognizable frame: every position holds ``value``."""
    return FrameCheckpoint(
        ids=torch.full((CANVAS_LENGTH,), value, dtype=torch.long),
        canvas_index=canvas_index,
        rng=None,
        extra=DgemmaFrame(
            seen_revealed=frozenset(range(CANVAS_LENGTH)),
        ),
    )


def _backend(
    worker: ModuleType,
    *,
    canvases: int = 1,
    alternatives: bool = False,
) -> DgemmaBackend:
    """A backend holding the recorded run, ready to resume.

    Stored through ``_store_state``, the call a finished generation
    makes, so the retained run has the shape a real one has.
    """
    backend: DgemmaBackend = worker.DgemmaBackend()
    history: List[FrameCheckpoint] = []
    for index in range(ORIGINAL_FRAMES):
        canvas = _canvas_of(index, canvases=canvases)
        history.append(_checkpoint(index, canvas_index=canvas))
    params = dict(RUN_PARAMS)
    params["alternatives"] = alternatives
    backend._store_state(params, history)
    # A retained run is state plus an identity, and a resume that
    # does not name it is refused before the state is read. A
    # finished generation mints one; a hand-built run has to.
    backend.run_counter += 1
    return backend


# -- the real sampler, over a stub model --


class _LetterTokenizer:
    """Decodes each id to one letter, which is all the streamer
    needs to build a frame's text."""

    def decode(
        self, ids: Any, skip_special_tokens: bool = False
    ) -> str:
        del skip_special_tokens  # Every id decodes the same way.
        if isinstance(ids, torch.Tensor):
            values = ids.tolist()
        else:
            values = list(ids)
        letters = [chr(ord("a") + int(i) % 26) for i in values]
        return "".join(letters)


class _PromptAdapter:
    """DiffusionGemma's text conventions over a fixed prompt.

    Two prompt tokens rather than a chat template, so the real
    sampler can build its inputs without a tokenizer that has one.
    """

    def build_inputs(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        *,
        thinking: bool,
    ) -> Dict[str, torch.Tensor]:
        del tokenizer, model, prompt, thinking  # The prompt is fixed.
        ids = torch.zeros((1, PROMPT_LENGTH), dtype=torch.long)
        return {"input_ids": ids}

    def sanitize(self, text: str) -> str:
        return DGEMMA_TEXT.sanitize(text)

    def split_channels(self, raw: str) -> Tuple[str, str]:
        return DGEMMA_TEXT.split_channels(raw)


class _DenoisingModel:
    """A ``generate`` that drafts one canvas ``DRAFT_STEPS`` times.

    Drives the streamer the way transformers does: the prompt echo,
    one ``put_draft`` per denoising step, then ``end``. A stop takes
    effect inside ``put_draft``, exactly as it does on the real one.
    """

    config = SimpleNamespace(
        canvas_length=CANVAS_LENGTH,
        text_config=SimpleNamespace(vocab_size=VOCAB_SIZE),
    )
    device = "cpu"

    def generate(
        self, *, streamer: Any, **_kwargs: Any
    ) -> torch.Tensor:
        echo = torch.zeros((1, CANVAS_LENGTH), dtype=torch.long)
        streamer.put(echo)
        for step in range(DRAFT_STEPS):
            draft = torch.full(
                (1, CANVAS_LENGTH),
                DRAFT_VALUE + step,
                dtype=torch.long,
            )
            streamer.put_draft(value=draft)
        streamer.end()
        return torch.zeros(
            (1, PROMPT_LENGTH + CANVAS_LENGTH), dtype=torch.long
        )


class _SeedRecordingModel(_DenoisingModel):
    """The stub model, keeping the canvas each ``generate`` was
    seeded with, which is the frame a resume re-enters."""

    def __init__(self) -> None:
        self.seeds: List[List[int]] = []

    def generate(
        self, *, streamer: Any, **kwargs: Any
    ) -> torch.Tensor:
        self.seeds.append(kwargs["decoder_input_ids"][0].tolist())
        return super().generate(streamer=streamer, **kwargs)


def _with_real_sampler(backend: DgemmaBackend) -> DgemmaBackend:
    """Let ``handle_resume`` run the real ``streaming_resume``.

    Its model thread, bounded queue and consumer are where a stop
    takes effect, so the tests about stopping drive them for real.
    """
    backend.model = _DenoisingModel()
    backend.tokenizer = _LetterTokenizer()
    backend.text_adapter = (  # type: ignore[assignment]
        _PromptAdapter()
    )
    return backend


def _drafted_values(count: int) -> List[int]:
    """The values of the first ``count`` frames the stub model
    drafts, which is what their checkpoints hold."""
    return [DRAFT_VALUE + step for step in range(count)]


# -- a scripted sampler --


def _branch_text(index: int) -> str:
    """What scripted resume frame ``index`` reads."""
    return "branch-" + str(index)


def _branch_values(count: int) -> List[int]:
    """The values of the first ``count`` scripted branch frames."""
    return [BRANCH_VALUE + index for index in range(count)]


def _scripted_frame(index: int) -> Dict[str, Any]:
    """A resumed frame, shaped the way the sampler shapes one."""
    return {
        "type": "frame",
        "index": index,
        "total_steps": None,
        "canvas_index": 0,
        "mean_conf": 0.5,
        "text": _branch_text(index),
        "tokens": [],
        "revealed": [],
    }


def _scripted_terminal(stop: threading.Event) -> Dict[str, Any]:
    """The sampler's own ``done``, as ``_terminal_frame`` builds
    it."""
    terminal: Dict[str, Any] = {
        "type": "done",
        "final_text": SAMPLER_TEXT,
        "thinking": "",
        "prompt_len": PROMPT_LENGTH,
    }
    if stop.is_set():
        terminal[TERMINAL_CANCELLED] = True
    return terminal


def _install_scripted_sampler(
    monkeypatch: pytest.MonkeyPatch,
    worker: ModuleType,
    *,
    frames: int,
    fail_after: Optional[int] = None,
    calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Replace the sampler with ``frames`` scripted frames.

    It keeps the contract ``_run_streamed`` keeps, which is what lets
    an assertion here mean what it would in production. The stop is
    read before each frame, and once it is set nothing more is
    yielded or recorded. Each frame's checkpoint is appended before
    the frame is yielded. The run always ends with exactly one
    ``done``, marked cancelled when the stop was set, and a run that
    captures alternatives sends its ``candidates`` just before it.

    ``fail_after`` raises once that many frames have been yielded,
    standing in for inference failing; zero raises before the first.
    ``calls`` collects each call's keyword arguments, so a test can
    ask which checkpoint a resume branched from.
    """

    async def scripted(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if calls is not None:
            calls.append(kwargs)
        sink: List[FrameCheckpoint] = kwargs["frame_history"]
        stop: threading.Event = kwargs["cancel_event"]
        if fail_after == 0:
            raise RuntimeError("inference failed")
        for index in range(frames):
            if stop.is_set():
                break
            sink.append(_checkpoint(BRANCH_VALUE + index))
            yield _scripted_frame(index)
            if fail_after is not None and index + 1 >= fail_after:
                raise RuntimeError("inference failed")
        if kwargs["alternatives"]:
            yield {"type": "candidates", "frames": [], "sets": {}}
        yield _scripted_terminal(stop)

    monkeypatch.setattr(worker, "streaming_resume", scripted)


# -- requests, as the page sends them --


def _resume(
    backend: DgemmaBackend,
    socket: _RecordingSocket,
    stop: threading.Event,
    *,
    frame_index: int = RESUME_FRAME,
    max_frames: object = None,
    run_token: Optional[str] = None,
    positions: Tuple[int, ...] = (0, 1),
    continuing: bool = False,
) -> None:
    """Send one resume request and wait for it to finish.

    ``max_frames`` is typed loosely on purpose: what arrives off the
    wire is whatever the client sent.
    """
    payload: Dict[str, Any] = {
        "type": "resume",
        "frame_index": frame_index,
        "remask_positions": list(positions),
        "run_token": (
            backend.run_token if run_token is None else run_token
        ),
    }
    if max_frames is not None:
        payload["max_frames"] = max_frames
    if continuing:
        payload[RESUME_CONTINUE] = True
    asyncio.run(
        backend.handle_resume(
            socket,  # type: ignore[arg-type]
            payload,
            stop,
            _streamer(backend, socket),
        )
    )


def _rewind(
    backend: DgemmaBackend,
    socket: _RecordingSocket,
    *,
    run_token: Optional[str] = None,
) -> None:
    token = backend.run_token if run_token is None else run_token
    asyncio.run(
        backend.handle_rewind(
            socket,  # type: ignore[arg-type]
            {"type": "rewind", "run_token": token},
        )
    )


# -- reading the result --


def _history(backend: DgemmaBackend) -> List[FrameCheckpoint]:
    """The retained history, copied so a later resume cannot move
    it under the test."""
    state = backend.last_run_state
    assert state is not None, "the backend holds a run"
    return list(state["frame_history"])


def _values(history: List[FrameCheckpoint]) -> List[int]:
    """Each checkpoint's value, which says which frame it is."""
    return [int(checkpoint.ids[0]) for checkpoint in history]


def _assert_same_objects(
    actual: List[FrameCheckpoint],
    expected: List[FrameCheckpoint],
) -> None:
    """Identity, element for element.

    The stronger claim, and the one that matters: a checkpoint
    carries the random state a repeated edit re-enters, so an equal
    frame rebuilt from somewhere else would still be the wrong one.
    Equality is not on offer anyway, since ``==`` on tensors raises.
    """
    assert len(actual) == len(expected), (
        f"expected {len(expected)} frames, found {len(actual)}"
    )
    for got, want in zip(actual, expected, strict=True):
        assert got is want, "a retained frame was replaced"


def _assert_branch_kept(
    backend: DgemmaBackend,
    original: List[FrameCheckpoint],
    *,
    frame_index: int,
    branch: List[int],
) -> None:
    """The recorded frames before the edit, as they were, then the
    branch frames whose values are ``branch``, and nothing else."""
    history = _history(backend)
    _assert_same_objects(
        history[:frame_index], original[:frame_index]
    )
    assert _values(history[frame_index:]) == branch


def _assert_one_terminal(
    socket: _RecordingSocket,
) -> Dict[str, Any]:
    terminals = socket.terminals()
    assert len(terminals) == 1, f"expected one done, got {terminals}"
    return terminals[0]


def _assert_one_error(
    socket: _RecordingSocket, *, code: str
) -> Dict[str, Any]:
    errors = socket.errors()
    assert len(errors) == 1, f"expected one error, got {errors}"
    assert errors[0]["code"] == code, errors[0]
    return errors[0]


# -- a stopped resume keeps what reached the page --


def test_a_stop_after_two_frames_keeps_exactly_those(
    worker: ModuleType,
) -> None:
    """The real sampler, stopped with two frames on screen.

    The page keeps the frames it received, so the worker holds the
    same ones: the run up to the edit, untouched, then the two that
    arrived, and nothing its model thread drafted after the stop.
    """
    backend = _with_real_sampler(_backend(worker))
    original = _history(backend)
    stop = threading.Event()
    socket = _RecordingSocket(stop=stop, stop_after=2)

    _resume(backend, socket, stop)

    assert socket.errors() == []
    assert len(socket.frames()) == 2
    _assert_branch_kept(
        backend,
        original,
        frame_index=RESUME_FRAME,
        branch=_drafted_values(2),
    )
    terminal = _assert_one_terminal(socket)
    assert terminal[TERMINAL_CANCELLED] is True


def test_a_stop_before_the_first_frame_keeps_the_run(
    worker: ModuleType,
) -> None:
    """The real sampler, stopped before its first draft landed.

    DiffusionGemma's first resumed frame needs a denoising step, so
    unlike LLaDA's resume, which sends the remasked canvas before it
    can see a stop, this one can end having sent nothing. No frame of
    a branch reached the page, so there is no branch to adopt, and
    the worker holds the run it held before.
    """
    backend = _with_real_sampler(_backend(worker))
    original = _history(backend)
    stop = threading.Event()
    stop.set()
    socket = _RecordingSocket()

    _resume(backend, socket, stop)

    assert socket.frames() == []
    assert socket.errors() == []
    terminal = _assert_one_terminal(socket)
    assert terminal[TERMINAL_CANCELLED] is True
    _assert_same_objects(_history(backend), original)


def test_a_guided_stop_before_the_first_frame_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The same stop on a Run to Here, whose terminal frame is the
    worker's own rather than the sampler's."""
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    original = _history(backend)
    stop = threading.Event()
    stop.set()
    socket = _RecordingSocket()

    _resume(backend, socket, stop, max_frames=2)

    assert socket.frames() == []
    assert socket.errors() == []
    _assert_one_terminal(socket)
    _assert_same_objects(_history(backend), original)


def test_a_guided_stop_keeps_what_arrived(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Stopped one frame into a Run to Here with a budget of four."""
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    original = _history(backend)
    stop = threading.Event()
    socket = _RecordingSocket(stop=stop, stop_after=1)

    _resume(backend, socket, stop, max_frames=4)

    assert len(socket.frames()) == 1
    _assert_branch_kept(
        backend,
        original,
        frame_index=RESUME_FRAME,
        branch=_branch_values(1),
    )
    _assert_one_terminal(socket)


# -- a guided edit keeps its budget --


def test_run_to_here_keeps_only_the_frames_it_sent(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """A budget of two frames out of a branch of six.

    The sampler keeps drafting past the budget, because its model
    thread has to finish before the request does, but the page
    receives two frames, so the worker keeps two. A guided edit's
    candidates would name frames the page never received, so none go
    out, even on a run that captures them.
    """
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker, alternatives=True)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event(), max_frames=2)

    texts = [frame["text"] for frame in socket.frames()]
    assert texts == [_branch_text(0), _branch_text(1)]
    _assert_branch_kept(
        backend,
        original,
        frame_index=RESUME_FRAME,
        branch=_branch_values(2),
    )
    assert "candidates" not in socket.kinds()
    terminal = _assert_one_terminal(socket)
    assert terminal["run_token"] == backend.run_token
    assert terminal["provenance"] == PROVENANCE


# -- a guided edit's terminal frame names its branch --
#
# On Run to Here the sampler's own terminal frame describes drafts
# past the budget, which the page never receives, so the worker ends
# the run itself. The page adopts any text that frame carries, and a
# save, the rescue when another window takes the model included,
# writes it beside the branch's frames, so it has to be the text of
# the branch the page is showing.


def test_run_to_here_ends_with_its_last_frames_text(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """A budget of two out of six: the terminal frame carries the
    second frame's text, rather than the sampler's or none."""
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event(), max_frames=2)

    terminal = _assert_one_terminal(socket)
    assert terminal["final_text"] == socket.frames()[-1]["text"]
    assert terminal["final_text"] == _branch_text(1)
    assert TERMINAL_CANCELLED not in terminal


def test_run_to_here_stopped_short_says_it_stopped(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Stopped one frame into a budget of four, so the request was
    cut short, and the terminal frame says so as every stopped run
    does (``LIFE-04``). The page then reads Stopped rather than Done,
    and a save records the run as partial."""
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    stop = threading.Event()
    socket = _RecordingSocket(stop=stop, stop_after=1)

    _resume(backend, socket, stop, max_frames=4)

    terminal = _assert_one_terminal(socket)
    assert terminal.get(TERMINAL_CANCELLED) is True
    assert terminal["final_text"] == _branch_text(0)


def test_a_stop_after_the_target_is_not_a_cancellation(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The distinction this path has to keep (manual item 170).

    Once the budget's frames are out, the worker still drains the
    sampler until its model thread finishes, and the page waits for
    the terminal frame meanwhile. A stop pressed then arrives after
    the request was met: it ends the drain early, and the edit still
    reads as completed.
    """
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    stop = threading.Event()
    socket = _RecordingSocket(stop=stop, stop_after=2)

    _resume(backend, socket, stop, max_frames=2)

    terminal = _assert_one_terminal(socket)
    assert TERMINAL_CANCELLED not in terminal
    assert terminal["final_text"] == _branch_text(1)


def test_a_branch_that_ends_inside_its_budget_is_done(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The branch settled after two frames of a budget of four, so
    the request ran out of work rather than being cut short."""
    _install_scripted_sampler(monkeypatch, worker, frames=2)
    backend = _backend(worker)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event(), max_frames=4)

    terminal = _assert_one_terminal(socket)
    assert TERMINAL_CANCELLED not in terminal
    assert terminal["final_text"] == _branch_text(1)


def test_run_to_here_stopped_before_any_frame_names_no_text(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Nothing reached the page, so the terminal frame is a stop
    that names no text, and the page keeps the text it holds."""
    _install_scripted_sampler(monkeypatch, worker, frames=6)
    backend = _backend(worker)
    stop = threading.Event()
    stop.set()
    socket = _RecordingSocket()

    _resume(backend, socket, stop, max_frames=2)

    terminal = _assert_one_terminal(socket)
    assert terminal.get(TERMINAL_CANCELLED) is True
    assert terminal["final_text"] == ""


# -- a completed resume --


def test_a_completed_resume_replaces_the_tail(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Every frame reached the page, then the branch's candidates
    and the sampler's own terminal frame, unchanged."""
    _install_scripted_sampler(monkeypatch, worker, frames=3)
    backend = _backend(worker, alternatives=True)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event())

    _assert_branch_kept(
        backend,
        original,
        frame_index=RESUME_FRAME,
        branch=_branch_values(3),
    )
    assert socket.kinds() == [
        "frame", "frame", "frame", "candidates", "done",
    ]
    terminal = _assert_one_terminal(socket)
    assert terminal["final_text"] == SAMPLER_TEXT
    assert TERMINAL_CANCELLED not in terminal


# -- a failure keeps the run as it was --
#
# Nothing commits until the terminal frame has reached the page. A
# failure before that is answered with an error, the page rolls back
# to the run it showed before the edit, and the worker has to be
# holding that same run.


def test_a_failure_before_the_first_frame_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, fail_after=0
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event())

    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_GENERATION_FAILED)


def test_a_failure_midway_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Frames streamed, so the page is mid-edit, and then inference
    died."""
    _install_scripted_sampler(
        monkeypatch, worker, frames=4, fail_after=2
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event())

    assert len(socket.frames()) == 2
    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_GENERATION_FAILED)


def test_a_failed_frame_send_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The socket died on the second frame."""
    _install_scripted_sampler(monkeypatch, worker, frames=4)
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket(
        fail_on=lambda payload: (
            payload["type"] == "frame" and payload["index"] == 1
        )
    )

    _resume(backend, socket, threading.Event())

    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_GENERATION_FAILED)


def test_a_failed_terminal_send_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Every frame arrived, and the socket then died on the one
    message that makes the run terminal."""
    _install_scripted_sampler(monkeypatch, worker, frames=3)
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket(
        fail_on=lambda payload: payload["type"] == "done"
    )

    _resume(backend, socket, threading.Event())

    assert len(socket.frames()) == 3
    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_GENERATION_FAILED)


def test_a_failed_guided_terminal_send_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The same on Run to Here, where the terminal frame is the
    worker's own rather than the sampler's."""
    _install_scripted_sampler(monkeypatch, worker, frames=4)
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket(
        fail_on=lambda payload: payload["type"] == "done"
    )

    _resume(backend, socket, threading.Event(), max_frames=2)

    assert len(socket.frames()) == 2
    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_GENERATION_FAILED)


@pytest.mark.parametrize("frame_index", [0, LAST_FRAME])
def test_a_retry_after_a_failure_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    worker: ModuleType,
    frame_index: int,
) -> None:
    """The point of keeping the run: the user tries again. Both ends
    of the resumable range, because a truncating failure strands the
    later frames and repoints the earlier ones."""
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, fail_after=1
    )
    backend = _backend(worker)
    original = _history(backend)
    _resume(
        backend,
        _RecordingSocket(),
        threading.Event(),
        frame_index=frame_index,
    )
    _assert_same_objects(_history(backend), original)

    _install_scripted_sampler(monkeypatch, worker, frames=3)
    retry = _RecordingSocket()
    _resume(
        backend, retry, threading.Event(), frame_index=frame_index
    )

    assert retry.errors() == []
    _assert_branch_kept(
        backend,
        original,
        frame_index=frame_index,
        branch=_branch_values(3),
    )


# -- the rewind --
#
# Every edit session opens by sending one, so the worker is back on
# the run the page shows before the session's first resume.


def test_a_rewind_restores_the_generated_run(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    _install_scripted_sampler(monkeypatch, worker, frames=3)
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()
    _resume(backend, socket, threading.Event())
    assert _values(_history(backend)) != _values(original), (
        "the resume must commit, or the rewind proves nothing"
    )

    _rewind(backend, socket)

    _assert_same_objects(_history(backend), original)


def test_a_rewind_undoes_a_chain_of_run_to_here_commits(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """A guided session commits one resume per Run to Here, and the
    page rolls all of them back from the one snapshot it took when
    the session opened."""
    _install_scripted_sampler(monkeypatch, worker, frames=3)
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()
    _resume(
        backend,
        socket,
        threading.Event(),
        frame_index=1,
        max_frames=2,
    )
    _resume(backend, socket, threading.Event(), frame_index=2)

    _rewind(backend, socket)

    _assert_same_objects(_history(backend), original)


def test_a_rewind_before_any_edit_changes_nothing(
    worker: ModuleType,
) -> None:
    """Sent on every session open, including the first, so the
    no-op is the common case rather than the odd one."""
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _rewind(backend, socket)

    _assert_same_objects(_history(backend), original)
    assert socket.sent == []


def test_a_stale_window_cannot_rewind(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Another window may be mid-edit on this run, and rewinding on
    its behalf would throw its work away."""
    _install_scripted_sampler(monkeypatch, worker, frames=3)
    backend = _backend(worker)
    socket = _RecordingSocket()
    _resume(backend, socket, threading.Event())
    after_edit = _history(backend)
    socket.sent.clear()

    _rewind(backend, socket, run_token="someone-elses-run")

    _assert_same_objects(_history(backend), after_edit)
    _assert_one_error(socket, code=ERROR_STALE_RUN)


def test_one_edit_branches_from_one_checkpoint(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Repeating an edit across a rewind re-enters the identical
    checkpoint, whose random state is what makes the repeat match
    the first attempt (``XAI-01``)."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event())
    _rewind(backend, socket)
    _resume(backend, socket, threading.Event())

    assert len(calls) == 2
    assert calls[0]["base"] is original[RESUME_FRAME]
    assert calls[1]["base"] is original[RESUME_FRAME]


def test_without_a_rewind_the_second_edit_moves(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Negative space for the test above. With nothing between them,
    two edits at one frame branch from different canvases, because
    the first one's branch now occupies that frame."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event())
    _resume(backend, socket, threading.Event())

    assert len(calls) == 2
    assert calls[1]["base"] is not calls[0]["base"]


def test_a_resume_leaves_the_step_budget_alone(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Why a rewind restores only the history here.

    LLaDA's rewind restores a step count as well, because its resume
    moves one. DiffusionGemma derives what an edit may run from
    ``max_denoising_steps``, which no resume writes, so an edit at a
    given frame gets the same budget however many resumes came
    before it.
    """
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    socket = _RecordingSocket()
    budget = int(RUN_PARAMS["max_denoising_steps"])

    _resume(backend, socket, threading.Event(), max_frames=2)
    _resume(
        backend,
        socket,
        threading.Event(),
        frame_index=RESUME_FRAME + 1,
    )

    state = backend.last_run_state
    assert state is not None
    assert state["max_denoising_steps"] == budget
    assert calls[0]["remaining_steps"] == budget - RESUME_FRAME
    assert calls[1]["remaining_steps"] == budget - RESUME_FRAME - 1
    assert worker.DgemmaBackend.REWIND_KEYS == (
        ("frame_history", "generated_frame_history"),
    )


def test_resume_reuses_the_retained_packed_input(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """A resume templates the same included conversation suffix."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=2, calls=calls
    )
    backend = _backend(worker)
    state = backend.last_run_state
    assert state is not None
    packed = (
        MessageRecord("user", "first", "1"),
        MessageRecord("assistant", "answer", "2"),
        MessageRecord("user", "next", "3"),
    )
    state["prompt"] = packed

    _resume(
        backend,
        _RecordingSocket(),
        threading.Event(),
    )

    assert calls[0]["prompt"] == packed


# -- carrying a stopped branch on --


def test_a_continue_resumes_with_nothing_remasked(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Continue carries a stopped branch on from a frame as it was:
    the sampler is asked to renoise nothing, and the frames that
    come back commit like any resume's."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=2, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(
        backend,
        socket,
        threading.Event(),
        positions=(),
        continuing=True,
    )

    assert socket.errors() == []
    assert calls[0]["remask_positions"] == []
    _assert_branch_kept(
        backend,
        original,
        frame_index=RESUME_FRAME,
        branch=_branch_values(2),
    )


def test_a_continue_reenters_the_frame_as_it_was(
    worker: ModuleType,
) -> None:
    """The real sampler: with nothing renoised, ``generate`` is
    seeded with the frame's own canvas."""
    backend = _with_real_sampler(_backend(worker))
    model = _SeedRecordingModel()
    backend.model = model
    socket = _RecordingSocket()

    _resume(
        backend,
        socket,
        threading.Event(),
        positions=(),
        continuing=True,
    )

    assert socket.errors() == []
    assert model.seeds == [[RESUME_FRAME] * CANVAS_LENGTH]


def test_a_continue_that_names_positions_is_refused(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Either an edit or a continue, never both, so the request is
    malformed rather than read one way or the other."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(
        backend,
        socket,
        threading.Event(),
        positions=(0,),
        continuing=True,
    )

    assert calls == []
    _assert_same_objects(_history(backend), original)
    error = _assert_one_error(socket, code=ERROR_INVALID_REQUEST)
    assert "remasks nothing" in error["message"]


def test_an_edit_with_no_positions_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """An empty list is not a continue: an edit that lost its
    positions on the way must not run as one."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(backend, socket, threading.Event(), positions=())

    assert calls == []
    _assert_same_objects(_history(backend), original)
    error = _assert_one_error(socket, code=ERROR_INVALID_REQUEST)
    assert "non-empty" in error["message"]


# -- refused before the sampler runs --


def test_a_multi_canvas_run_is_refused(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """Resume re-enters one canvas, so a run that spans two cannot
    be resumed, even from a frame on its first canvas.

    The page withholds Edit Frames from such a run
    (``generator_edit_gate.test.js``); this is the worker's half of
    the same contract, for a request that arrives anyway.
    """
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker, canvases=2)
    original = _history(backend)
    socket = _RecordingSocket()
    assert _canvas_of(1, canvases=2) == 0, "the frame is on canvas 0"

    _resume(backend, socket, threading.Event(), frame_index=1)

    assert calls == []
    _assert_same_objects(_history(backend), original)
    error = _assert_one_error(socket, code=ERROR_INVALID_REQUEST)
    assert "single-canvas" in error["message"]
    assert socket.frames() == []
    assert socket.terminals() == []


def test_an_out_of_range_frame_is_refused(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(
        backend,
        socket,
        threading.Event(),
        frame_index=ORIGINAL_FRAMES,
    )

    assert calls == []
    _assert_same_objects(_history(backend), original)
    error = _assert_one_error(socket, code=ERROR_INVALID_REQUEST)
    assert "out of range" in error["message"]


@pytest.mark.parametrize(
    "max_frames",
    [0, -1, True, "2", 1.5],
    ids=["zero", "negative", "a flag", "a string", "a fraction"],
)
def test_a_budget_that_is_not_a_frame_count_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    worker: ModuleType,
    max_frames: object,
) -> None:
    """A Run to Here budget counts the frames the page will receive,
    so anything but a positive whole number is a malformed request.
    It is refused before the model runs rather than discovered after,
    when the frames it would have counted are already drafted."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(
        backend, socket, threading.Event(), max_frames=max_frames
    )

    assert calls == []
    _assert_same_objects(_history(backend), original)
    error = _assert_one_error(socket, code=ERROR_INVALID_REQUEST)
    assert "max_frames" in error["message"]


def test_a_stale_window_cannot_resume(
    monkeypatch: pytest.MonkeyPatch, worker: ModuleType
) -> None:
    """The request names a run the worker no longer holds, because
    another window's generation replaced it. Answering it would
    branch from a run this page is not showing."""
    calls: List[Dict[str, Any]] = []
    _install_scripted_sampler(
        monkeypatch, worker, frames=3, calls=calls
    )
    backend = _backend(worker)
    original = _history(backend)
    socket = _RecordingSocket()

    _resume(
        backend,
        socket,
        threading.Event(),
        run_token="someone-elses-run",
    )

    assert calls == []
    _assert_same_objects(_history(backend), original)
    _assert_one_error(socket, code=ERROR_STALE_RUN)
