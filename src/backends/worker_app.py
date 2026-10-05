"""The socket shell around one worker's backend.

A worker is one process holding one model, and this module is the
part of it the supervisor talks to: the FastAPI app, its ``/health``
and ``/params`` routes, the WebSocket that carries every request, the
load that runs when the app starts, and the one generation slot that
two windows share. What a request does once it is routed is the
backend's business, and that side of the worker is ``worker_base``.

The dependency runs one way. This module imports ``worker_base`` and
nothing there imports this, so a backend can be built and tested
without an app.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Optional,
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.backends.protocol import (
    ERROR_BUSY,
    ERROR_MODEL_LOAD_FAILED,
    ERROR_SCOPE_FATAL,
    ERROR_SCOPE_REQUEST,
    ERROR_UNKNOWN_MESSAGE,
    MSG_CANCEL,
    MSG_COUNT_PROMPT,
    MSG_DETECT_WATERMARK,
    MSG_GENERATE,
    MSG_MODEL_STATUS,
    MSG_PROBE,
    MSG_RESUME,
    MSG_REWIND,
    MSG_SUBSTITUTE,
    MSG_TOKENIZE,
    request_error,
    request_id_of,
    wire_error,
)
from src.backends.resource_sampler import CpuSampler
from src.backends.worker_base import (
    Backend,
    FrameStreamer,
    describe_context_length,
    describe_tokenizer,
    library_versions,
    provenance_envelope,
    pump_resource_samples,
    worker_envelope,
)

logger = logging.getLogger("diffusion_worker")


# How long a closing socket waits for its own generation to
# notice the stop before saying so. Every sampler checks between
# steps, and the slowest step this project runs is far under a
# second, so reaching this means a backend is ignoring the signal
# rather than that it is merely busy.
SETTLE_WARN_SECONDS = 30.0
AUXILIARY_TASKS_PER_SESSION_MAX = 1

assert SETTLE_WARN_SECONDS > 0.0, "a warning must have a delay"
assert AUXILIARY_TASKS_PER_SESSION_MAX > 0


def _log_generation_outcome(
    task: "asyncio.Task[None]",
) -> None:
    """Report a spawned generation's failure.

    Needed because the socket loop no longer awaits its handler.
    An exception in a task whose result nobody retrieves is
    discarded silently, so a generation that died would be
    indistinguishable from one still running.
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("generation failed", exc_info=error)


def _log_detector_outcome(
    task: "asyncio.Task[None]",
) -> None:
    """Report a detached detector failure.

    The socket stays open because this task owns one request only.
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("watermark detector failed", exc_info=error)


class _Generation:
    """The one generation a worker may have in flight.

    This replaces the lock the socket loop used to hold, and the
    difference is the whole point of the change. A lock made the
    loop *wait* for the generation, which is exactly what stopped
    it reading, so a Cancel sat unread until the run it was meant
    to stop had already finished. A task reference lets the loop
    ask whether the worker is busy without ever blocking on it.

    One at a time is still enforced, just by asking rather than by
    queueing: a second request is refused as busy instead of
    silently waiting its turn behind a run the user cannot see.

    Held for the whole worker rather than per connection, because
    the thing being serialized is a device with one model on it.
    Two browser windows share it, and the second is told the
    worker is busy exactly as it was before.
    """

    def __init__(self) -> None:
        self._task: Optional["asyncio.Task[None]"] = None

    def busy(self) -> bool:
        """Is a generation running right now?"""
        if self._task is None:
            return False
        return not self._task.done()

    def start(
        self, work: "Coroutine[Any, Any, None]"
    ) -> "asyncio.Task[None]":
        """Run *work* alongside the loop that spawned it.

        Returns the task so the caller can wait for its own
        generation later without waiting for somebody else's.
        """
        assert not self.busy(), "one generation at a time"
        task = asyncio.create_task(work)
        task.add_done_callback(_log_generation_outcome)
        self._task = task
        return task


class _DetectorGate:
    """The one CPU-heavy detector job this worker may run."""

    def __init__(self) -> None:
        self._task: Optional["asyncio.Task[None]"] = None

    def busy(self) -> bool:
        if self._task is None:
            return False
        return not self._task.done()

    def start(
        self, work: "Coroutine[Any, Any, None]"
    ) -> "asyncio.Task[None]":
        assert not self.busy(), "one detector job at a time"
        task = asyncio.create_task(work)
        task.add_done_callback(_log_detector_outcome)
        self._task = task
        return task


async def _settle_generation(
    task: Optional["asyncio.Task[None]"],
) -> None:
    """Wait for one socket's own generation to actually end.

    Called as a socket goes away. Returning before the work stops
    is what LIFE-04 calls hidden work: the page is gone, the
    supervisor believes the worker is idle, and a model still
    holds the device.

    The wait is unbounded on purpose. Abandoning the task would
    restore exactly the invisibility being fixed, so a backend
    that ignores cancellation must show up as a slow close rather
    than as a worker that lies about being idle. The bounded log
    below is how it shows up.
    """
    if task is None:
        return
    if task.done():
        return
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=SETTLE_WARN_SECONDS,
        )
        return
    except TimeoutError:
        logger.error(
            "generation still running %.0fs after cancel;"
            " the backend is not honouring the stop signal",
            SETTLE_WARN_SECONDS,
        )
    except Exception:  # noqa: BLE001
        # Already reported by the done callback; swallowed here
        # because the peer is, by construction, gone.
        logger.debug("generation ended with an error")
        return
    try:
        await task
    except Exception:  # noqa: BLE001
        logger.debug("generation ended with an error")


def resolve_load_status(
    *,
    failed: bool,
    ready: bool,
    progress: Optional[Dict[str, Any]],
) -> str:
    """Pick the ``/health`` status from the worker's load signals.

    Lifted out of the endpoint because it is the one real decision
    there and it is worth testing on its own.

    The Hub download and the read into memory both report through the
    backend's one ``load_progress`` attribute, so the dict carries a
    ``phase`` saying which it is. A missing phase means download:
    ``hf_download`` predates the distinction and its payload has no
    such key, and treating its absence as a load would relabel every
    download.
    """
    if failed:
        return "error"
    if ready:
        return "ready"
    if not isinstance(progress, dict):
        return "loading"
    if str(progress.get("phase", "download")) == "load":
        return "loading"
    return "downloading"


async def _send_busy(
    ws: WebSocket, request_type: str, data: Dict[str, Any]
) -> None:
    """Refuse a request because the generation lock is held.

    Takes the request it is refusing because that decides how far the
    refusal reaches. Turning away a second generation ends a run;
    turning away a probe should leave What If exactly as it was, and
    for a long time it did not, because both arrived as the same
    unscoped error.
    """
    await ws.send_json(
        request_error(
            message=("A generation is already running. Please wait."),
            code=ERROR_BUSY,
            request_type=request_type,
            request_id=request_id_of(data),
        )
    )


async def _await_model_ready(
    ws: WebSocket,
    model_ready: asyncio.Event,
    load_failed: asyncio.Event,
    load_error: Dict[str, str],
    model_id: str,
) -> bool:
    """Hold the socket until the model is usable.

    False means it never will be, and the reason has already been
    sent, so the caller only has to stop. Lifted out of the socket
    handler because it is a distinct phase with its own three-way
    outcome, and inlining it put four branches in front of the
    message loop that has nothing to do with them.

    ``model_id`` rides on the status frames so the page can tell
    which model is answering it. The supervisor sends its own
    statement of that on connect; this is the worker's, and the two
    disagreeing is the only way to catch a proxy pointed at the
    wrong worker.
    """
    if load_failed.is_set():
        await _send_load_error(ws, load_error)
        return False
    if model_ready.is_set():
        return True

    await ws.send_json(
        {
            "type": MSG_MODEL_STATUS,
            "status": "loading",
            "model": model_id,
        }
    )
    ready_task = asyncio.ensure_future(model_ready.wait())
    failed_task = asyncio.ensure_future(load_failed.wait())
    _done, pending = await asyncio.wait(
        {ready_task, failed_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if load_failed.is_set():
        await _send_load_error(ws, load_error)
        return False
    return True


async def _send_load_error(
    ws: WebSocket, load_error: Dict[str, str]
) -> None:
    """Fatal: there is no model, so no request can be attempted."""
    await ws.send_json(
        wire_error(
            message=load_error.get(
                "message", "Model failed to load."
            ),
            code=ERROR_MODEL_LOAD_FAILED,
            scope=ERROR_SCOPE_FATAL,
        )
    )


# The backend method a request type reaches.
_Handler = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class _LoadState:
    """A worker's load, which its routes share: whether the model is
    ready, whether it failed, and why."""

    ready: asyncio.Event = field(default_factory=asyncio.Event)
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    error: Dict[str, str] = field(default_factory=dict)
    # asyncio keeps only a weak reference to a running task, so the
    # load is held here or it could be collected before it finishes.
    task: Optional["asyncio.Task[None]"] = None


@dataclass
class _Session:
    """One socket's side of the worker.

    Its own cancel flag, frame streamer and in-flight generation,
    which is not necessarily the worker's: another window may hold
    that one, and closing this page must not wait for theirs.
    """

    ws: WebSocket
    cancel_event: threading.Event
    stream: FrameStreamer
    streaming: Dict[str, _Handler]
    concurrent: Dict[str, _Handler]
    background: Dict[str, _Handler]
    exclusive: Dict[str, _Handler]
    mine: Optional["asyncio.Task[None]"] = None
    auxiliary: Dict["asyncio.Task[None]", threading.Event] = field(
        default_factory=dict
    )


def create_worker_app(
    backend: Backend, *, device: str = "cuda"
) -> FastAPI:
    """Build the FastAPI app hosting a single model worker.

    ``device`` is forwarded to ``backend.load`` so the supervisor can
    place a model on CPU or GPU per activation. The routes only
    register here; what each one does is in the functions below.
    """
    load = _LoadState()
    # Worker-scoped, like the lock it replaces: one model on one
    # device, so two connected windows contend for the same slot.
    generation = _Generation()
    # Detector scoring is CPU-heavy and independent of generation,
    # but admitting one per socket would create an unbounded queue of
    # green-list work across windows.
    detector = _DetectorGate()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        load.task = asyncio.create_task(
            _load_model(backend, device, load)
        )
        yield

    app = FastAPI(
        title=f"worker:{backend.model_info.id}", lifespan=_lifespan
    )

    @app.get("/health")
    async def _health() -> JSONResponse:
        return JSONResponse(_health_payload(backend, load))

    @app.get("/params")
    async def _params() -> JSONResponse:
        return JSONResponse(backend.model_info.model_dump())

    @app.websocket("/ws")
    async def _ws(ws: WebSocket) -> None:
        await _serve_socket(ws, backend, load, generation, detector)

    return app


async def _load_model(
    backend: Backend, device: str, load: _LoadState
) -> None:
    """Load the model off the event loop and record how it went."""
    try:
        await asyncio.to_thread(backend.load, device=device)
    except Exception as exc:  # noqa: BLE001
        load.error["message"] = str(exc)
        load.failed.set()
        logger.exception(
            "model %s failed to load", backend.model_info.id
        )
        return
    load.ready.set()
    logger.info("model %s ready", backend.model_info.id)


def _health_payload(
    backend: Backend, load: _LoadState
) -> Dict[str, Any]:
    """What ``/health`` reports: the load's status, and once it is
    ready, what only a loaded model can say about itself."""
    progress = getattr(backend, "load_progress", None)
    status = resolve_load_status(
        failed=load.failed.is_set(),
        ready=load.ready.is_set(),
        progress=progress,
    )
    payload: Dict[str, Any] = {
        "status": status,
        "id": backend.model_info.id,
        "versions": library_versions(),
    }
    # Only once ready: there is no tokenizer to describe before the
    # load finishes, and the supervisor caches this on the same
    # transition it caches versions on.
    if status == "ready":
        payload.update(_ready_details(backend))
    # "loading" is reported with or without progress: the sampler
    # only attaches once the load starts and can measure the
    # checkpoint, and everything before that is still a load.
    if status in ("downloading", "loading") and progress:
        payload["progress"] = progress
    if status == "error":
        payload["message"] = load.error.get(
            "message", "Model failed to load."
        )
    return payload


def _ready_details(backend: Backend) -> Dict[str, Any]:
    """The loaded model's tokenizer, device and context window."""
    model = getattr(backend, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    details: Dict[str, Any] = {
        "tokenizer": describe_tokenizer(tokenizer, model),
        # Where the model landed, which is not always where it was
        # sent: the supervisor knows only what it asked for, and a
        # CUDA request on a GPU-less host becomes CPU here.
        "device": backend.effective_device or "unknown",
    }
    # Omitted rather than sent as null when unreadable, so the
    # client's "is there a ceiling to check against" test is a plain
    # key check and cannot mistake null for zero.
    context = describe_context_length(model, tokenizer)
    if context is not None:
        details["context_length"] = context
    return details


def _open_session(ws: WebSocket, backend: Backend) -> _Session:
    """A socket's cancel flag, streamer and request tables."""
    return _Session(
        ws=ws,
        # A threading.Event rather than an asyncio one because the
        # readers are model threads: the autoregressive decode loop
        # and DiffusionGemma's streamer both check it from inside the
        # thread running the forward pass, and only the event loop
        # ever sets it.
        cancel_event=threading.Event(),
        stream=FrameStreamer(
            ws,
            provenance=lambda: provenance_envelope(backend),
            run_token=lambda: backend.run_token,
            opening=lambda: worker_envelope(backend),
        ),
        # The three that stream frames. Identical but for the method
        # they reach, so they share one path rather than three copies
        # of the same busy check and spawn.
        streaming={
            MSG_GENERATE: backend.handle_generate,
            MSG_RESUME: backend.handle_resume,
            MSG_SUBSTITUTE: backend.handle_substitute,
        },
        # Answered even while a generation runs: these are tokenizer
        # reads and neither performs a model forward.
        #
        # They are the only requests that can write to this socket
        # alongside a streaming generation. That is safe because each
        # reply is a single complete WebSocket text frame and the
        # transport writes frames in order, so a reply lands between
        # two frames rather than inside one.
        concurrent={
            MSG_TOKENIZE: backend.handle_tokenize,
            MSG_COUNT_PROMPT: backend.handle_count_prompt,
        },
        # Started rather than awaited so the receive loop can read a
        # Cancel while scoring. The worker-wide gate bounds this work
        # across sockets, and the session owns the task for cleanup.
        background={
            MSG_DETECT_WATERMARK: backend.handle_detect_watermark,
        },
        # Refused while a generation runs, unlike the two above, and
        # for a different reason each. The probe runs a forward pass,
        # so admitting it alongside a generation would put two passes
        # on one device and its memory. The rewind rewrites the
        # retained history, which is the very thing a running resume
        # is in the middle of deciding.
        #
        # Awaited inline once accepted, which holds the loop for one
        # pass and cannot deadlock, because nothing else can be
        # running by then.
        exclusive={
            MSG_PROBE: backend.handle_probe,
            MSG_REWIND: backend.handle_rewind,
        },
    )


async def _serve_socket(
    ws: WebSocket,
    backend: Backend,
    load: _LoadState,
    generation: _Generation,
    detector: _DetectorGate,
) -> None:
    """One socket, from accepting it to settling its generation."""
    await ws.accept()
    session = _open_session(ws, backend)
    # This socket's resource meter, started before the readiness wait
    # so it also runs for a client that arrives while this worker is
    # still loading.
    #
    # That case is narrow, and the comment used to claim more. The
    # supervisor's proxy refuses a socket until ``load_state`` is
    # "ready", which it only becomes once this worker reports its
    # model loaded, so a browser cannot watch a load through it and
    # the meter does not cover one. Reaching this handler mid-load
    # means connecting to the worker directly. The placement is kept
    # because it costs nothing and is honest about the case it
    # serves; it is not a view of a load.
    #
    # Created just before the ``try`` rather than inside it, so the
    # ``finally`` can stop it without first asking whether it exists.
    # Nothing can happen in between, and a task needing a guard would
    # be one the cleanup could miss.
    meter = asyncio.create_task(
        pump_resource_samples(ws, backend, CpuSampler())
    )
    try:
        if not await _greet(ws, backend, load):
            return
        while True:
            data = await ws.receive_json()
            await _dispatch(session, generation, detector, data)
    except WebSocketDisconnect:
        logger.info("worker client disconnected")
    finally:
        # Cancelled without being awaited, unlike the generation
        # below. Its only await is a sleep, so it stops at once and
        # holds nothing; waiting on it would add a step to every
        # disconnect to settle a task that owns no device.
        meter.cancel()
        session.cancel_event.set()
        await _settle_auxiliary(session)
        # The socket is going away for some reason, and every reason
        # means nothing will read this run's frames again. Stopping
        # and then waiting is what makes the disconnect bounded
        # rather than hidden: without the wait, the supervisor
        # believes this worker is idle while a model still holds the
        # device.
        await _settle_generation(session.mine)


async def _settle_auxiliary(session: _Session) -> None:
    """Cancel cooperatively and await this socket's detector work."""
    owned = list(session.auxiliary.items())
    for _task, cancel_event in owned:
        cancel_event.set()
    if not owned:
        return
    await asyncio.gather(
        *(task for task, _event in owned),
        return_exceptions=True,
    )


async def _greet(
    ws: WebSocket, backend: Backend, load: _LoadState
) -> bool:
    """Hold the socket until the model is usable, then say so.

    False when it never will be, the reason already sent.
    """
    ready = await _await_model_ready(
        ws,
        load.ready,
        load.failed,
        load.error,
        backend.model_info.id,
    )
    if not ready:
        return False
    await ws.send_json(
        {
            "type": MSG_MODEL_STATUS,
            "status": "ready",
            "model": backend.model_info.id,
        }
    )
    return True


async def _dispatch(
    session: _Session,
    generation: _Generation,
    detector: _DetectorGate,
    data: Dict[str, Any],
) -> None:
    """Route one message by its type."""
    mtype = data.get("type")
    if mtype == MSG_CANCEL:
        # Reachable during a run, which is the whole point: the loop
        # is parked on receive_json rather than inside the handler it
        # would stop.
        session.cancel_event.set()
        return
    if mtype in session.streaming:
        await _start_streaming(session, generation, mtype, data)
        return
    if mtype in session.concurrent:
        await session.concurrent[mtype](session.ws, data)
        return
    if mtype in session.background:
        await _start_background(session, detector, mtype, data)
        return
    if mtype in session.exclusive:
        await _run_exclusive(session, generation, mtype, data)
        return
    # A client bug rather than a worker one, and scoped to the
    # request so it disturbs nothing: there is no owner to route it
    # to, and a page that has just sent something unrecognisable is
    # not helped by also losing whatever it was doing.
    await session.ws.send_json(
        wire_error(
            message=f"Unknown message type: {mtype}",
            code=ERROR_UNKNOWN_MESSAGE,
            scope=ERROR_SCOPE_REQUEST,
            request_id=request_id_of(data),
        )
    )


async def _start_background(
    session: _Session,
    detector: _DetectorGate,
    mtype: str,
    data: Dict[str, Any],
) -> None:
    """Start one bounded auxiliary request without blocking reads."""
    if detector.busy():
        await session.ws.send_json(
            request_error(
                message=(
                    "Watermark detection is already running."
                    " Wait for it to finish and try again."
                ),
                code=ERROR_BUSY,
                request_type=mtype,
                request_id=request_id_of(data),
            )
        )
        return
    if len(session.auxiliary) >= AUXILIARY_TASKS_PER_SESSION_MAX:
        await session.ws.send_json(
            request_error(
                message="This session already has detector work.",
                code=ERROR_BUSY,
                request_type=mtype,
                request_id=request_id_of(data),
            )
        )
        return
    cancel_event = threading.Event()
    task = detector.start(
        session.background[mtype](
            session.ws,
            data,
            cancel_event=cancel_event,
        )
    )
    session.auxiliary[task] = cancel_event
    task.add_done_callback(
        lambda finished: session.auxiliary.pop(finished, None)
    )


async def _start_streaming(
    session: _Session,
    generation: _Generation,
    mtype: str,
    data: Dict[str, Any],
) -> None:
    """Start a frame-streaming run, unless one holds the device."""
    if generation.busy():
        await _send_busy(session.ws, mtype, data)
        return
    session.cancel_event.clear()
    session.mine = generation.start(
        session.streaming[mtype](
            session.ws, data, session.cancel_event, session.stream
        )
    )


async def _run_exclusive(
    session: _Session,
    generation: _Generation,
    mtype: str,
    data: Dict[str, Any],
) -> None:
    """Run a request that may not share the device, if it is free."""
    if generation.busy():
        await _send_busy(session.ws, mtype, data)
        return
    await session.exclusive[mtype](session.ws, data)
