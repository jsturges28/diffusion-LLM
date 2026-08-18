"""Shared FastAPI scaffolding for per-model diffusion workers.

A worker loads exactly one model and exposes a WebSocket
streaming interface identical to the original single-process
server. The supervisor proxies browser traffic to it. All
model-specific logic (loading, validation, sampling, resume)
lives in a ``Backend`` supplied by each worker module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.backends.protocol import (
    ERROR_BUSY,
    ERROR_MODEL_LOAD_FAILED,
    ERROR_NO_TOKENIZER,
    ERROR_SCOPE_FATAL,
    ERROR_SCOPE_REQUEST,
    ERROR_UNKNOWN_MESSAGE,
    MSG_CANCEL,
    MSG_COUNT_PROMPT,
    MSG_COUNT_PROMPT_RESULT,
    MSG_GENERATE,
    MSG_MODEL_STATUS,
    MSG_PROBE,
    MSG_RESUME,
    MSG_SUBSTITUTE,
    MSG_TOKENIZE,
    MSG_TOKENIZE_RESULT,
    ModelInfo,
    request_error,
    request_id_of,
    wire_error,
)

logger = logging.getLogger("diffusion_worker")

# Upper bound on a typed-token preview request. The feature resolves
# one token, and no vocabulary entry approaches this, so anything
# longer is either a paste accident or an attempt to make the worker
# do work. Truncating beats rejecting: the preview still shows the
# user what their first few tokens would be.
TOKENIZE_TEXT_MAX_CHARS = 200

# Upper bound on a prompt being counted. Far larger than the preview
# cap above, because the whole point is to answer for a file somebody
# imported, and a bound that rejected those would leave the readout
# silent exactly when it matters. Still bounded: this is a per-request
# encode on the worker, and the client caps imports well below it, so
# reaching this means something is wrong rather than large. Truncating
# beats rejecting, and the reply reports that it truncated so the
# readout can say the count is a floor.
COUNT_PROMPT_MAX_CHARS = 200_000

assert COUNT_PROMPT_MAX_CHARS > TOKENIZE_TEXT_MAX_CHARS, (
    "counting a prompt must allow more than previewing a token"
)


class FrameStreamer:
    """Forwards frames from an async generator to a WebSocket.

    Also the single place a terminal frame is stamped with what the
    worker attests about the run. Every ``done`` leaves through
    ``run`` or ``send_done``, so a backend cannot finish a run
    without saying which model, device and tokenizer produced it.
    """

    def __init__(
        self,
        ws: WebSocket,
        provenance: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._ws = ws
        # A callable rather than a dict because a worker's tokenizer
        # is not describable until its load finishes, and the socket
        # is accepted before that. Called once per terminal frame,
        # which is once per run.
        self._provenance = provenance

    async def run(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
        *,
        max_frames: Optional[int] = None,
    ) -> bool:
        """Stream frames; return True if a ``done`` was sent.

        Stops early (returning False) when *max_frames* frame
        messages have been forwarded.
        """
        frame_count = 0
        done_sent = False
        async for frame in generator:
            elapsed = time.monotonic() - start_time
            frame["elapsed"] = round(elapsed, 2)
            if frame.get("type") == "done":
                self._stamp_provenance(frame)
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
        """Send a terminal frame the generator did not produce.

        A guided edit stops its sampler at a frame budget, so the
        worker ends the run itself. Routed through here so those
        runs carry the same ``elapsed`` and provenance as every
        other one; they used to be assembled by hand at each site
        and disagree about which fields a done frame has.
        """
        assert frame.get("type") == "done", frame.get("type")
        frame["elapsed"] = round(
            time.monotonic() - start_time, 2
        )
        self._stamp_provenance(frame)
        await self._ws.send_json(frame)

    def _stamp_provenance(self, frame: Dict[str, Any]) -> None:
        if self._provenance is None:
            return
        frame["provenance"] = self._provenance()


def provenance_envelope(backend: Backend) -> Dict[str, Any]:
    """What the worker attests about the run it just finished.

    Taken from the loaded model rather than from what the supervisor
    was asked for. The supervisor answers these from whichever worker
    is active when a save arrives, so a run finished before a model
    switch was being described by the model that replaced it. Here
    the facts travel with the run.

    ``device`` in particular is the placement the model actually got,
    which is not always the one requested: two backends fall back to
    CPU when CUDA is unavailable.
    """
    envelope: Dict[str, Any] = {
        "model_id": backend.model_info.id,
        "checkpoint": backend.model_info.checkpoint,
        "device": backend.effective_device or "unknown",
        "versions": library_versions(),
        "tokenizer": describe_tokenizer(
            getattr(backend, "tokenizer", None),
            getattr(backend, "model", None),
        ),
    }
    # Omitted rather than null when unreadable, matching /health, so
    # a consumer's "is there a ceiling" test stays a plain key check.
    context = describe_context_length(
        getattr(backend, "model", None),
        getattr(backend, "tokenizer", None),
    )
    if context is not None:
        envelope["context_length"] = context
    return envelope


def library_versions() -> Dict[str, str]:
    """Torch and transformers as this worker's venv has them.

    Read here rather than in the supervisor because the three venvs
    hold deliberately incompatible versions, so the supervisor's own
    imports would describe the wrong environment.
    """
    try:
        import torch
        import transformers

        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    except Exception:  # noqa: BLE001
        # A worker that cannot import its own libraries has bigger
        # problems, and none of them are fixed by refusing to report
        # health. An empty block reads as "not recorded".
        return {}


class Backend(ABC):
    """Model-specific worker logic (loading + streaming)."""

    model_info: ModelInfo
    # Set to a progress dict ({fraction, downloaded_bytes, total_bytes})
    # while weights download during ``load``, then back to None. Read by
    # ``/health`` to report a "downloading" state to the supervisor.
    load_progress: Optional[Dict[str, Any]] = None
    # Where the model actually ended up, set by ``load``. Not the same
    # as the device it was asked for: two of the three backends fall
    # back to CPU when CUDA was requested and is unavailable, and a
    # run saved as "GPU" when it ran on CPU misreports the one fact
    # its timings should be read against. None means ``load`` has not
    # finished, or a backend has not been taught to say.
    effective_device: Optional[str] = None

    @abstractmethod
    def load(self, *, device: str = "cuda") -> None:
        """Blocking model + tokenizer load (runs in a thread).

        ``device`` is the requested placement ("cuda" or "cpu"),
        chosen per activation so CPU-only hosts can still run small
        models. GPU-only backends may ignore it.
        """

    @abstractmethod
    async def handle_generate(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """Validate, run, and stream a generation request."""

    async def handle_resume(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """Resume from a saved frame (override if supported)."""
        raise NotImplementedError("resume not supported")

    async def handle_substitute(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: asyncio.Event,
        stream: FrameStreamer,
    ) -> None:
        """Force one position to an alternative, then regenerate.

        The autoregressive counterfactual: truncate at a position,
        substitute a captured candidate there, and continue forward
        (override if supported).
        """
        raise NotImplementedError(
            "substitution not supported"
        )

    async def handle_tokenize(
        self, ws: WebSocket, data: Dict[str, Any]
    ) -> None:
        """Resolve a typed string against the loaded vocabulary.

        Backs the What If typed-token preview. Implemented here
        rather than per backend because every backend holds a
        tokenizer, so the diffusion models inherit a working preview
        for free if What If ever reaches them.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            await ws.send_json(
                request_error(
                    message="No tokenizer is loaded.",
                    code=ERROR_NO_TOKENIZER,
                    request_type=MSG_TOKENIZE,
                    request_id=request_id_of(data),
                )
            )
            return
        raw = data.get("text", "")
        text = raw[:TOKENIZE_TEXT_MAX_CHARS] if (
            isinstance(raw, str)
        ) else ""
        pieces = tokenize_pieces(tokenizer, text)
        await ws.send_json(
            {
                "type": MSG_TOKENIZE_RESULT,
                # Echoed so the client can drop a reply that a later
                # keystroke has already made irrelevant.
                "request_id": int(data.get("request_id", 0)),
                "text": text,
                "pieces": pieces,
                "count": len(pieces),
            }
        )

    async def handle_count_prompt(
        self, ws: WebSocket, data: Dict[str, Any]
    ) -> None:
        """Report how many tokens a prompt becomes.

        Implemented on the base class like ``handle_tokenize``,
        because every backend templates a prompt and the answer is
        just an encode. The per-model part is
        ``prompt_token_count``.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            await ws.send_json(
                request_error(
                    message="No tokenizer is loaded.",
                    code=ERROR_NO_TOKENIZER,
                    request_type=MSG_COUNT_PROMPT,
                    request_id=request_id_of(data),
                )
            )
            return
        raw = data.get("text", "")
        text = raw[:COUNT_PROMPT_MAX_CHARS] if (
            isinstance(raw, str)
        ) else ""
        count = self.prompt_token_count(
            text, thinking=bool(data.get("thinking", False))
        )
        await ws.send_json(
            {
                "type": MSG_COUNT_PROMPT_RESULT,
                # Echoed for the same reason the tokenize reply echoes
                # them: a later keystroke makes an in-flight answer
                # wrong rather than merely late.
                "request_id": int(data.get("request_id", 0)),
                "chars": len(text),
                "truncated": len(text) < len(raw) if (
                    isinstance(raw, str)
                ) else False,
                "count": count,
            }
        )

    def prompt_token_count(
        self, prompt: str, *, thinking: bool = False
    ) -> int:
        """Tokens the templated prompt occupies before generation.

        Counts the *templated* sequence, not the raw text, which is
        the whole point of doing this on the worker instead of in the
        browser. A chat template wraps the prompt in system and role
        markers, and ``enable_thinking`` changes that wrapping, so a
        count of the user's characters would understate what actually
        reaches the model, by a margin that grows with the template
        rather than with the prompt.

        The default mirrors ``_build_inputs`` in the autoregressive
        and DiffusionGemma samplers exactly, minus the device move
        this does not need, so SmolLM3 and DiffusionGemma inherit a
        count of the tokens their runs really build. LLaDA templates
        in two steps and takes no thinking flag, so it overrides.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        if prompt == "":
            return 0
        tokenizer = getattr(self, "tokenizer", None)
        assert tokenizer is not None, "no tokenizer loaded"
        chat = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        count = int(encoded["input_ids"].shape[-1])
        assert count > 0, "a templated prompt has tokens"
        return count

    async def handle_probe(
        self, ws: WebSocket, data: Dict[str, Any]
    ) -> None:
        """Report what the model gave a token at one position.

        Not implemented here, unlike ``handle_tokenize``: answering
        needs the last run's committed prefix and a forward pass
        against it, and only the autoregressive backend keeps a run
        state shaped for that. Diffusion positions are revealed out
        of order, so there is no prefix to prefill up to.
        """
        raise NotImplementedError("probing not supported")


def tokenize_pieces(
    tokenizer: Any, text: str
) -> List[Dict[str, Any]]:
    """Split a string into the vocabulary entries it resolves to.

    This is a standalone encode, not a re-tokenization of the
    sequence, and that distinction is what makes the preview
    trustworthy. Substitution keeps the prefix ids verbatim and
    continues decoding from them, so the sequence is never
    re-encoded and the boundary effects that make BPE
    context-sensitive cannot arise. The only question left is
    whether the typed text is exactly one vocabulary entry.

    ``add_special_tokens=False`` because this is a fragment being
    spliced into an existing sequence; the BOS a tokenizer would
    otherwise prepend is not something the user typed.

    Each piece decodes exactly the way a captured candidate does
    (see ``_top_alternatives``), so one frontend helper renders both.
    """
    assert isinstance(text, str), "text must be a string"
    if text == "":
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces: List[Dict[str, Any]] = []
    for token_id in ids:
        pieces.append(
            {
                "id": int(token_id),
                "t": tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                ),
            }
        )
    assert len(pieces) == len(ids), "dropped a piece"
    return pieces


def describe_tokenizer(
    tokenizer: Any, model: Any = None
) -> Dict[str, Any]:
    """Identify a loaded tokenizer, for the UI and saved runs.

    Read off the live object rather than declared in the registry.
    A hand-maintained name is free to drift from whatever the
    checkpoint actually loads, and this is shown to the user as a
    statement of fact about their run, so drift is the one failure
    that matters here.

    ``vocab_size`` is the base vocabulary, deliberately not
    ``len(tokenizer)``, which adds the special tokens grafted on
    afterwards. The base figure is the one the entropy scale refers
    to, since its natural log is the most a position can carry.

    ``model_vocab_size`` is the model's output width, which is a
    property of the checkpoint rather than of the tokenizer, and is
    reported here anyway because the two figures only mean anything
    next to each other: they differ wherever a checkpoint pads its
    embedding for alignment (128,256 against 128,000 for SmolLM3),
    and that difference is what a token's rank is measured against.
    Sharing one dict also means the pair reaches the UI and a saved
    run through the plumbing the tokenizer already has.

    Every field is optional, because this runs against whatever
    objects a backend happens to hold. An unrecognizable tokenizer
    reports only its class rather than failing a health check.
    """
    if tokenizer is None:
        return {}
    described: Dict[str, Any] = {
        "class": type(tokenizer).__name__
    }
    assert described["class"], "tokenizer class name is empty"

    name = getattr(tokenizer, "name_or_path", None)
    if name:
        described["name_or_path"] = str(name)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int):
        assert vocab_size > 0, "vocab_size must be positive"
        described["vocab_size"] = vocab_size
    is_fast = getattr(tokenizer, "is_fast", None)
    if isinstance(is_fast, bool):
        described["is_fast"] = is_fast
    width = describe_output_width(model)
    if width is not None:
        described["model_vocab_size"] = width
    return described


def describe_output_width(model: Any) -> Optional[int]:
    """The model's output width, or None when it cannot be read.

    Taken from the config rather than from ``lm_head.out_features``,
    because a tied-embedding checkpoint may not expose a linear head
    to measure, and the config figure is what the head is built from
    either way.
    """
    if model is None:
        return None
    config = getattr(model, "config", None)
    width = getattr(config, "vocab_size", None)
    if not isinstance(width, int):
        return None
    assert width > 0, "model vocab_size must be positive"
    return width


def describe_context_length(
    model: Any, tokenizer: Any = None
) -> Optional[int]:
    """How many tokens the checkpoint can attend to at once.

    Read off the loaded objects for the same reason
    ``describe_tokenizer`` is: a figure declared in the registry is a
    hand-maintained string free to drift from whatever the checkpoint
    actually loads, and this one is used to tell the user their prompt
    will not fit, which is a claim that has to be true.

    The config comes first because it is the architectural fact.
    ``model_max_length`` is a tokenizer-side convention and is
    frequently a sentinel rather than a measurement: a checkpoint
    shipping no ``model_max_length`` gets ``int(1e30)`` from
    transformers, and older ones sometimes carry a stale small number
    from a predecessor. So it is consulted only as a fallback, and
    only when it lands inside ``CONTEXT_LENGTH_SANE_MAX``.

    Returns None rather than a guess when neither source is usable.
    A missing readout is honest; an invented ceiling would silently
    approve a prompt that overflows or refuse one that fits.
    """
    for candidate in (
        getattr(getattr(model, "config", None),
                "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if _is_sane_context_length(candidate):
            return int(candidate)
    return None


# Above this, a reported context length is a sentinel rather than a
# measurement. transformers uses int(1e30) for "unspecified", and no
# real checkpoint is within orders of magnitude of this bound, so it
# separates the two cases without having to name the sentinel.
CONTEXT_LENGTH_SANE_MAX = 1 << 25

assert CONTEXT_LENGTH_SANE_MAX > 1_000_000, (
    "the ceiling must clear every real context length"
)


def _is_sane_context_length(value: Any) -> bool:
    """Whether a reported length is a measurement, not a sentinel."""
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if value < 1:
        return False
    return value <= CONTEXT_LENGTH_SANE_MAX


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
            message=(
                "A generation is already running. Please wait."
            ),
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


def create_worker_app(
    backend: Backend, *, device: str = "cuda"
) -> FastAPI:
    """Build the FastAPI app hosting a single model worker.

    ``device`` is forwarded to ``backend.load`` so the supervisor can
    place a model on CPU or GPU per activation.
    """
    app = FastAPI(title=f"worker:{backend.model_info.id}")
    model_ready = asyncio.Event()
    load_failed = asyncio.Event()
    load_error: Dict[str, str] = {}
    gen_lock = asyncio.Lock()

    @app.on_event("startup")
    async def _startup() -> None:
        async def _load() -> None:
            try:
                await asyncio.to_thread(
                    backend.load, device=device
                )
            except Exception as exc:  # noqa: BLE001
                load_error["message"] = str(exc)
                load_failed.set()
                logger.exception(
                    "model %s failed to load",
                    backend.model_info.id,
                )
                return
            model_ready.set()
            logger.info(
                "model %s ready", backend.model_info.id
            )

        asyncio.create_task(_load())

    @app.get("/health")
    async def _health() -> JSONResponse:
        progress = getattr(backend, "load_progress", None)
        status = resolve_load_status(
            failed=load_failed.is_set(),
            ready=model_ready.is_set(),
            progress=progress,
        )
        payload: Dict[str, Any] = {
            "status": status,
            "id": backend.model_info.id,
            "versions": library_versions(),
        }
        # Only once ready: there is no tokenizer to describe before
        # the load finishes, and the supervisor caches this on the
        # same transition it caches versions on.
        if status == "ready":
            payload["tokenizer"] = describe_tokenizer(
                getattr(backend, "tokenizer", None),
                getattr(backend, "model", None),
            )
            # Where the model landed, which is not always where it
            # was sent: the supervisor knows only what it asked for,
            # and a CUDA request on a GPU-less host becomes CPU here.
            payload["device"] = (
                backend.effective_device or "unknown"
            )
            # Omitted rather than sent as null when unreadable, so the
            # client's "is there a ceiling to check against" test is a
            # plain key check and cannot mistake null for zero.
            context = describe_context_length(
                getattr(backend, "model", None),
                getattr(backend, "tokenizer", None),
            )
            if context is not None:
                payload["context_length"] = context
        # "loading" is reported with or without progress: the sampler
        # only attaches once the load starts and can measure the
        # checkpoint, and everything before that is still a load.
        if status in ("downloading", "loading") and progress:
            payload["progress"] = progress
        if status == "error":
            payload["message"] = load_error.get(
                "message", "Model failed to load."
            )
        return JSONResponse(payload)

    @app.get("/params")
    async def _params() -> JSONResponse:
        return JSONResponse(backend.model_info.model_dump())

    @app.websocket("/ws")
    async def _ws(ws: WebSocket) -> None:
        await ws.accept()
        cancel_event = asyncio.Event()
        stream = FrameStreamer(
            ws, provenance=lambda: provenance_envelope(backend)
        )
        # The three that stream frames. Identical but for the method
        # they reach, so they share one branch below rather than three
        # copies of the same busy check and lock acquisition.
        streaming = {
            MSG_GENERATE: backend.handle_generate,
            MSG_RESUME: backend.handle_resume,
            MSG_SUBSTITUTE: backend.handle_substitute,
        }
        # Answered without gen_lock: both are tokenizer reads costing
        # microseconds, and the lock exists to serialize generation,
        # not to guard the tokenizer. Taking it would stall a preview
        # or a prompt count behind a running model, which is exactly
        # when the user is still typing.
        lockless = {
            MSG_TOKENIZE: backend.handle_tokenize,
            MSG_COUNT_PROMPT: backend.handle_count_prompt,
        }
        try:
            ready = await _await_model_ready(
                ws,
                model_ready,
                load_failed,
                load_error,
                backend.model_info.id,
            )
            if not ready:
                return
            await ws.send_json(
                {
                    "type": MSG_MODEL_STATUS,
                    "status": "ready",
                    "model": backend.model_info.id,
                }
            )
            while True:
                data = await ws.receive_json()
                mtype = data.get("type")

                if mtype == MSG_CANCEL:
                    cancel_event.set()
                    continue

                if mtype in streaming:
                    if gen_lock.locked():
                        await _send_busy(ws, mtype, data)
                        continue
                    async with gen_lock:
                        cancel_event.clear()
                        await streaming[mtype](
                            ws, data, cancel_event, stream
                        )
                    continue

                if mtype in lockless:
                    await lockless[mtype](ws, data)
                    continue

                # Inside gen_lock, unlike the two above it: this one
                # runs a forward pass, so letting it in alongside a
                # generation would have two passes contending for the
                # same device and its memory.
                if mtype == MSG_PROBE:
                    if gen_lock.locked():
                        await _send_busy(ws, MSG_PROBE, data)
                        continue
                    async with gen_lock:
                        await backend.handle_probe(ws, data)
                    continue

                # A client bug rather than a worker one, and scoped
                # to the request so it disturbs nothing: there is no
                # owner to route it to, and a page that has just sent
                # something unrecognisable is not helped by also
                # losing whatever it was doing.
                await ws.send_json(
                    wire_error(
                        message=f"Unknown message type: {mtype}",
                        code=ERROR_UNKNOWN_MESSAGE,
                        scope=ERROR_SCOPE_REQUEST,
                        request_id=request_id_of(data),
                    )
                )
        except WebSocketDisconnect:
            cancel_event.set()
            logger.info("worker client disconnected")

    return app
