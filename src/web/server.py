"""Supervisor server for the multi-model diffusion visualizer.

Responsibilities:
  - Serve the shared frontend and the analytics API.
  - Hold the one ModelManager, from ``model_manager``, which runs the
    model worker subprocesses (one active at a time), each in its
    own venv so incompatible dependency stacks (e.g. Transformers
    4.38.2 vs v5) never collide.
  - Proxy the browser WebSocket to the active worker's /ws.

The supervisor itself never imports torch or transformers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Sized,
    Tuple,
)

import websockets
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analytics.metrics import (
    CONVERGENCE_BASIS_CHARACTERS,
    CONVERGENCE_BASIS_SETTLEMENT,
    CONVERGENCE_BASIS_TOKENS,
    UnsupportedRunVersionError,
    canvas_boundaries,
    compute_convergence,
    convergence_from_positions,
    convergence_from_records,
    convergence_from_settlement,
    list_runs,
    load_run_frames,
    load_run_metadata,
    masks_are_real,
    read_frame_texts,
    records_match_frames,
    run_schema_version,
    tokens_produced_series,
    total_elapsed_seconds,
)
from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    CANDIDATES_PER_POSITION,
    ERROR_NO_MODEL_ACTIVE,
    ERROR_SCOPE_FATAL,
    ERROR_WORKER_UNREACHABLE,
    PROMPT_CHARS_MAX,
    SAVED_MODEL_TYPE_AUTOREGRESSIVE,
    SAVED_MODEL_TYPE_DIFFUSION,
    ModelInfo,
    ParamSpec,
    is_hub_checkpoint,
    saved_model_type,
    wire_error,
)
from src.backends.params import ParamValue, coerce, default_of
from src.backends.registry import (
    DEFAULT_MODEL,
    REGISTRY,
    run_bounds,
)
from src.inference.render_gif import history_to_gif
from src.inference.vision_encoders import (
    EncoderUnavailable,
    VisionEncoder,
    declared as vision_declared,
    find as vision_find,
    is_cached as vision_is_cached,
    load_geometry as vision_load_geometry,
)
from src.inference.vision_geometry import (
    MAX_DIMENSION as VISION_MAX_DIMENSION,
    MIN_DIMENSION as VISION_MIN_DIMENSION,
    geometry as vision_image_geometry,
)
from src.web import collections as collection_ops
from src.web import model_manager
from src.web import run_store
from src.web.save_limits import (
    FREEFORM_JSON_CHARS_MAX,
    IDENTIFIER_CHARS_MAX,
    TOKEN_TEXT_CHARS_MAX,
    BodyLimit,
)
from src.web.data_root import (
    RESULTS_DIR_ENV,
    resolve_results_dir,
)
from src.web.model_manager import ActivationRefused, ModelManager
from src.web.ui_state import (
    load_ui_state,
    mutate_ui_state_key,
    set_ui_state_key,
)


logger = logging.getLogger("diffusion_supervisor")

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Resolved once, here, rather than inherited from wherever the
# process happened to be started (see src/web/data_root.py).
RESULTS_DIR = resolve_results_dir(
    os.environ.get(RESULTS_DIR_ENV), repo_root=REPO_ROOT
)

# What this server calls itself when asked. Read by the desktop
# launcher to tell its own supervisor from an unrelated process
# holding the same port. A constant rather than a version string:
# the question is "is this us", and pinning it to a version would
# make two builds of the same app fail to recognise each other.
APP_IDENTITY = "diffusion-llm-supervisor"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


manager = ModelManager()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Say where the data is before anything reads or writes it. The
    # incident this guards against was silent: two result trees, no
    # error, and a repository that looked like no work had happened.
    source = (
        "from " + RESULTS_DIR_ENV
        if os.environ.get(RESULTS_DIR_ENV, "").strip()
        else "default"
    )
    logger.info(
        "results directory: %s (%s)", RESULTS_DIR, source
    )
    # Reap any worker orphaned by a prior crashed supervisor before we
    # start serving, so stale workers cannot keep holding VRAM.
    await asyncio.to_thread(model_manager.sweep_orphan_workers)
    # In a finally, so the manager stops even when the lifespan ends
    # in an error: a worker left running keeps its VRAM after the
    # supervisor is gone.
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(title="Diffusion LLM Visualizer", lifespan=_lifespan)
# Ahead of every route, so a save past its ceiling is refused before
# Starlette reads the body it would otherwise parse whole.
app.add_middleware(BodyLimit)


# -- Model API --


def _model_headroom_gib(
    info: ModelInfo,
    *,
    free_vram_gib: Optional[float],
    resident_reclaimable_gib: float,
) -> Optional[float]:
    """Signed VRAM headroom in GiB: (free + reclaimable) - required.

    A resident GPU model's VRAM counts as reclaimable, since the
    supervisor stops the current worker before spawning the next
    (see ``_preflight_vram``). Positive means it fits with that much
    to spare; negative means it is short by that much. None when the
    model needs no VRAM or free VRAM is unreadable.
    """
    if info.min_vram_gib <= 0:
        return None
    if free_vram_gib is None:
        return None
    return round(
        (free_vram_gib + resident_reclaimable_gib)
        - info.min_vram_gib,
        1,
    )


def _model_fits(
    info: ModelInfo,
    *,
    status: str,
    headroom_gib: Optional[float],
) -> bool:
    """Whether ``info`` can be activated, derived from headroom.

    Unreadable free VRAM / no requirement (headroom None) is treated
    as "fits", mirroring the pre-flight's skip-on-unreadable behavior.
    """
    if status == "active":
        return True
    if headroom_gib is None:
        return True
    return headroom_gib >= 0


def _model_entry(model_id: str, info: Any) -> Dict[str, Any]:
    """One model as the frontend sees it, before any probing.

    Shared with the generator's inlined boot state, which needs the
    registry half of a snapshot and none of the VRAM half, so that the
    two cannot describe the same model differently.
    """
    data = info.model_dump()
    data.pop("worker_module", None)
    data.pop("environment", None)
    data["status"] = manager.status(model_id)
    return data


def _models_snapshot() -> Dict[str, Any]:
    """Registry plus live GPU/VRAM info for the Main Menu.

    Runs the blocking ``nvidia-smi`` probes here so the endpoint can
    offload it to a thread and keep the event loop responsive.
    """
    free_vram_gib = model_manager.free_vram_gib()
    active_id = manager.active_id
    # Only a resident GPU worker reclaims VRAM when stopped; a
    # CPU-resident model frees no VRAM, so it must not inflate the
    # free pool (which previously made GPU models look "Available").
    resident_reclaimable_gib = 0.0
    if (
        active_id is not None
        and manager.status(active_id) == "active"
        and active_id in REGISTRY
        and manager.active_device == "cuda"
    ):
        resident_reclaimable_gib = REGISTRY[
            active_id
        ].min_vram_gib

    models: List[Dict[str, Any]] = []
    for model_id, info in REGISTRY.items():
        data = _model_entry(model_id, info)
        status = data["status"]
        headroom = _model_headroom_gib(
            info,
            free_vram_gib=free_vram_gib,
            resident_reclaimable_gib=resident_reclaimable_gib,
        )
        data["vram_headroom_gib"] = headroom
        data["fits"] = _model_fits(
            info, status=status, headroom_gib=headroom
        )
        data["downloadable"] = is_hub_checkpoint(
            info.checkpoint
        )
        data["downloaded"] = model_manager.is_downloaded(
            info.checkpoint, info.revision, info.companion
        )
        data["partial"] = model_manager.is_partial(info.checkpoint)
        models.append(data)
    gpu = model_manager.gpu_name()
    # Only classify the failure reason when the name is unreadable, so
    # a healthy system pays no extra nvidia-smi call.
    gpu_status = (
        "ok" if gpu is not None else model_manager.gpu_status()
    )
    return {
        "models": models,
        "active": active_id,
        "active_device": manager.active_device,
        # Empty until a worker reports ready. Lives here rather than
        # on each model's capabilities because it describes the one
        # resident load, which is the only tokenizer that exists.
        "active_tokenizer": dict(manager.active_tokenizer),
        # None when the checkpoint did not report a readable one. The
        # prompt readout treats that as "no ceiling to check against"
        # and shows a bare count rather than inventing a denominator.
        "active_context_length": manager.active_context_length,
        "default": DEFAULT_MODEL,
        "gpu_name": gpu,
        "free_vram_gib": free_vram_gib,
        "gpu_status": gpu_status,
        "cpu_name": model_manager.cpu_name(),
        "free_ram_gib": model_manager.free_ram_gib(),
    }


@app.get("/api/app")
async def app_identity() -> JSONResponse:
    """Say what is listening here, for a launcher deciding to start.

    Exists because "is port 8760 free" and "is *this app* already on
    port 8760" need opposite answers. A bind that fails could be our
    own supervisor, in which case a second one must not be started,
    or something unrelated, in which case the desktop app should get
    out of its way and take another port. Only this can tell them
    apart.

    Deliberately the cheapest route on the server: no GPU probe, no
    disk, no manager state. It is called on a launch path where the
    user is waiting for a window to appear.
    """
    return JSONResponse({"app": APP_IDENTITY, "pid": os.getpid()})


@app.get("/api/models")
async def list_models() -> JSONResponse:
    snapshot = await asyncio.to_thread(_models_snapshot)
    return JSONResponse(snapshot)


class ActivateRequest(BaseModel):
    """Optional activation body: pick CPU/GPU placement.

    Body-less activation (the generator's model switch) leaves
    ``device`` None, letting the manager auto-select.
    """

    device: Optional[str] = None


@app.post("/api/models/{model_id}/activate")
async def activate_model(
    model_id: str, body: Optional[ActivateRequest] = None
) -> JSONResponse:
    device = body.device if body is not None else None
    try:
        operation = await manager.activate(model_id, device=device)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": f"unknown model: {model_id}",
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": str(exc)},
        )
    except ActivationRefused as exc:
        # Expected, and already explained. Logged as a line rather
        # than a stack trace so real faults stay findable, and 409
        # rather than 500 because nothing here is the server's
        # fault: the request asked for something this machine
        # cannot currently do.
        logger.info("activation refused: %s", exc)
        return JSONResponse(
            status_code=409,
            content={"ok": False, "message": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("activation failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": str(exc)},
        )
    # Non-blocking: the worker is spawned and loading in the
    # background. The client polls /api/models/activation for
    # progress, and carries the operation id so it can tell its own
    # load's outcome from one another window started.
    return JSONResponse(
        {
            "ok": True,
            "active": manager.active_id,
            "state": manager.load_state,
            "operation": operation,
        }
    )


@app.get("/api/models/activation")
async def activation_status() -> JSONResponse:
    """Current activation progress for the client's loading poll."""
    return JSONResponse(
        {
            "active": manager.active_id,
            "device": manager.active_device,
            "state": manager.load_state,
            "progress": manager.load_progress,
            "message": manager.load_error,
            # Which activation this state describes. A client that
            # started one compares it, so a second window's load
            # cannot be mistaken for the first window's finishing.
            "operation": manager.activation_id,
        }
    )


class CancelActivationRequest(BaseModel):
    """Which activation the caller believes it is cancelling.

    Optional so the endpoint stays parseable for a caller that sends
    nothing, but an absent operation is refused just as a stale one
    is: not naming an activation is not the same as owning it.
    """

    operation: Optional[int] = None


@app.post("/api/models/activate/cancel")
async def cancel_activation(
    body: Optional[CancelActivationRequest] = None,
) -> JSONResponse:
    """Cancel an in-flight load: stop the worker and free its VRAM."""
    operation = body.operation if body is not None else None
    try:
        await manager.cancel_activation(operation)
    except ActivationRefused as exc:
        logger.info("cancel refused: %s", exc)
        return JSONResponse(
            status_code=409,
            content={"ok": False, "message": str(exc)},
        )
    return JSONResponse({"ok": True})


@app.post("/api/models/{model_id}/download")
async def download_model(model_id: str) -> JSONResponse:
    """Pre-fetch a model's weights (no VRAM). Client polls status."""
    try:
        operation = manager.start_download(model_id)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": f"unknown model: {model_id}",
            },
        )
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": str(exc)},
        )
    return JSONResponse(
        {
            "ok": True,
            "state": manager.download_state,
            "operation": operation,
        }
    )


@app.get("/api/models/download-status")
async def download_status() -> JSONResponse:
    """Current pre-fetch progress for the download veneer's poll."""
    target = manager.download_target
    target_name: Optional[str] = None
    if target is not None and target in REGISTRY:
        target_name = REGISTRY[target].display_name
    return JSONResponse(
        {
            "target": target,
            "target_name": target_name,
            "state": manager.download_state,
            "progress": manager.download_progress,
            "message": manager.download_error,
            # Which download this state describes, on the same terms
            # as an activation's: a second window's fetch finishing
            # must not read as this window's.
            "operation": manager.download_id,
        }
    )


class CancelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Optional[int] = None


@app.post("/api/models/download/cancel")
async def cancel_download(
    body: Optional[CancelDownloadRequest] = None,
) -> JSONResponse:
    """Stop a fetch, leaving its parts on disk so it can resume."""
    operation = body.operation if body is not None else None
    try:
        await manager.cancel_download(operation)
    except ActivationRefused as exc:
        return JSONResponse(
            status_code=409,
            content={"ok": False, "message": str(exc)},
        )
    return JSONResponse({"ok": True})


@app.post("/api/models/download/ack")
async def ack_download() -> JSONResponse:
    """Clear a finished pre-fetch (done/error -> idle), so the
    completion toast and menu re-attach fire exactly once."""
    manager.ack_download()
    return JSONResponse({"ok": True})


# -- WebSocket proxy to the active worker --


async def _pipe(browser: WebSocket, worker: Any) -> None:
    """Bidirectionally forward text frames browser <-> worker."""

    async def browser_to_worker() -> None:
        try:
            while True:
                message = await browser.receive_text()
                await worker.send(message)
        except Exception:
            return

    async def worker_to_browser() -> None:
        try:
            async for message in worker:
                await browser.send_text(message)
        except Exception:
            return

    task_b2w = asyncio.create_task(browser_to_worker())
    task_w2b = asyncio.create_task(worker_to_browser())
    _done, pending = await asyncio.wait(
        {task_b2w, task_w2b},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if not pending:
        return
    # Cancelling only asks. The caller closes the worker's connection
    # as soon as this returns, so the half still reading from it has
    # to have stopped first (`A2-QUALITY-02`). ``wait`` rather than
    # ``gather``: were this task cancelled meanwhile, gather would
    # re-raise the half's cancellation instead of this task's own,
    # which a cancel scope around the handler would not recognise.
    await asyncio.wait(pending)
    assert all(task.done() for task in pending), "a half outlived it"


@app.websocket("/ws")
async def websocket_proxy(browser: WebSocket) -> None:
    await browser.accept()
    active_id = manager.active_id
    if active_id is None or not manager.is_serving(active_id):
        # Model selection happens on the Main Menu; the generator
        # never auto-boots a worker. Tell the client to go back.
        await browser.send_json(
            wire_error(
                message=(
                    "No model is active. Return to the menu"
                    " to select one."
                ),
                code=ERROR_NO_MODEL_ACTIVE,
                scope=ERROR_SCOPE_FATAL,
            )
        )
        await browser.close()
        return

    url = manager.ws_url()
    try:
        async with websockets.connect(
            url, max_size=None
        ) as worker:
            # Who the page is actually talking to, sent before any
            # worker traffic so the answer is the first thing it
            # reads. A generator caches its model, device, capability
            # flags and whole parameter form at boot and only
            # refreshes them by reloading, so a page whose worker was
            # replaced from another window would otherwise go on
            # labelling and parameterising requests for a model that
            # is no longer there.
            await browser.send_json(
                {
                    "type": "resident",
                    "model": active_id,
                    "device": manager.active_device,
                    "operation": manager.activation_id,
                    # What a page compares to notice that the same
                    # model and device are now a different worker,
                    # which holds none of the runs it is showing.
                    "worker": manager.worker_identity(),
                }
            )
            await _pipe(browser, worker)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("proxy error")
        # Best-effort notification: the browser socket may already be
        # gone, which is why we are here. The error itself is logged
        # above, so nothing is lost if this cannot be delivered.
        with contextlib.suppress(Exception):
            await browser.send_json(
                wire_error(
                    message=str(exc),
                    code=ERROR_WORKER_UNREACHABLE,
                    scope=ERROR_SCOPE_FATAL,
                )
            )


# -- Save endpoint (model-agnostic) --


# Every model on the save boundary refuses fields it does not declare.
# Pydantic's default is to drop them silently, which turns "somebody
# added a signal to the client and forgot the server" into a run saved
# without it and an HTTP 200 saying otherwise. A 422 naming the field
# is the whole point: the failure should be the rollout, not the data.
STRICT = ConfigDict(extra="forbid")


class RemaskEdit(BaseModel):
    model_config = STRICT

    frame_index: int
    token_positions: List[int]


class TokenRecord(BaseModel):
    """One persisted per-token record for durable overlays.

    Mirrors the live protocol shape ``{t, m, id, c?, e?}``: ``t`` is
    the display text, ``m`` marks an unresolved position, ``id`` is
    the vocab id, ``c`` is the reveal confidence (absent for masked
    positions), and ``e`` is the entropy in nats.

    Both floats, and what they vary over is not visible from here:
    ``e`` is one value per position on an autoregressive run, whose
    positions are decided once, and a value per denoising step on a
    diffusion run, whose positions are re-decided. That is what the
    signal manifest says and this shape cannot, which is the whole
    reason the manifest exists. See ``SignalChannel``.

    A new signal must be declared here to reach ``tokens.json``. It
    used to be dropped silently; now the request fails and says which
    key it did not recognize. ``f`` is the one Mamba-3 adds: what
    reading the token erased from its recurrent state, a fraction.
    """

    model_config = STRICT

    t: str = Field(max_length=TOKEN_TEXT_CHARS_MAX)
    m: bool
    id: int
    c: Optional[float] = None
    e: Optional[float] = None
    f: Optional[float] = None


class TokenAlternative(BaseModel):
    """One competing candidate token at a single position.

    ``p`` is the candidate's probability under the untempered
    softmax at the step that position was sampled.

    ``rank`` is absent for the captured set, whose rank is its order
    in the list, and present only on the entry appended for a token
    the position committed from outside that set. There the two part
    company: it is last in the list and may be thousandth in the
    distribution.
    """

    model_config = STRICT

    id: int
    t: str = Field(max_length=TOKEN_TEXT_CHARS_MAX)
    p: float
    rank: Optional[int] = None


class CandidateSet(BaseModel):
    """What one diffusion position was weighing at one captured step.

    ``h`` is the token the position held at that step, which is the
    row the popover marks: the step's guess where the position was
    masked, its token where it had settled. ``c`` is the likeliest
    candidates, then the held token with its ``rank`` when they omit
    it, exactly as an autoregressive position's list reads.
    """

    model_config = STRICT

    h: int
    c: List[TokenAlternative] = Field(min_length=1)


class FrameCandidates(BaseModel):
    """A diffusion run's candidates: a set per position for each
    captured frame.

    Not ``alternatives``, which is one set per position, because a
    diffusion position is re-decided at every step, so what it was
    weighing is a trajectory. The capture thins to a stride past
    ``CANDIDATE_BUDGET_RECORDS``, and ``stride`` is where it got to.

    ``frames`` are the run's own frame indices, ascending.
    ``segments`` are the frames where a stream of candidates begins:
    0, then the first frame of each resumed edit, so a reader never
    lends a resumed frame the candidates of the run the edit
    replaced.
    """

    model_config = STRICT

    k: int = Field(ge=1)
    stride: int = Field(ge=1)
    frames: List[int] = Field(min_length=1)
    segments: List[int] = Field(min_length=1)
    sets: List[List[CandidateSet]]

    @model_validator(mode="after")
    def _placeable(self) -> "FrameCandidates":
        _check_frame_candidates(self)
        return self


def _ascending(values: List[int]) -> bool:
    return all(
        earlier < later
        for earlier, later in zip(values, values[1:], strict=False)
    )


def _check_frame_candidates(value: FrameCandidates) -> None:
    """Refuse candidates no reader could place, or that outgrew their
    budget. A ValueError, which the save reports as a 422 naming it.
    """
    if len(value.sets) != len(value.frames):
        raise ValueError("candidates need one list of sets per frame")
    if value.frames[0] < 0 or not _ascending(value.frames):
        raise ValueError("candidate frames must ascend from 0")
    if value.segments[0] != 0 or not _ascending(value.segments):
        raise ValueError("candidate segments must ascend from 0")
    rows = value.k + 1
    for sets in value.sets:
        if any(len(entry.c) > rows for entry in sets):
            raise ValueError(f"a candidate set holds over {rows}")
    records = sum(len(sets) for sets in value.sets) * value.k
    if records > CANDIDATE_BUDGET_RECORDS:
        raise ValueError(
            f"candidates hold {records} records, over the budget of"
            f" {CANDIDATE_BUDGET_RECORDS}"
        )


# Per-frame, per-token stream. A frame may be ``None`` when a model
# emitted no token detail for it.
FrameTokens = List[Optional[List[TokenRecord]]]

# The same run, flat: one record per position rather than one array
# per frame. Sent by a model whose output only grows, where frame N
# is the first N+1 positions and the per-frame arrays above are that
# same information written out N times.
RunPositions = List[TokenRecord]


def expand_positions(
    positions: List[TokenRecord],
) -> FrameTokens:
    """Rebuild the per-frame token arrays from a flat run.

    The browser sends positions because holding N(N+1)/2 records is
    what made a long run degrade there. Disk has no such pressure and
    every reader downstream, Analytics included, already understands
    the per-frame form, so the expansion happens here and the saved
    run is byte-for-byte what a snapshot client would have written.

    Prefixes share nothing on purpose. A frame's list is its own, so
    a later mutation of one cannot reach into another, and the cost
    is a shallow list of references rather than copies of the
    records.
    """
    frames: FrameTokens = []
    for count in range(1, len(positions) + 1):
        frames.append(list(positions[:count]))
    return frames


def expand_position_text(positions: List[TokenRecord]) -> List[str]:
    """The per-frame rendered text, for the same reason.

    Accumulated rather than re-joined per frame: joining each prefix
    separately is the quadratic again, in the one place that has the
    whole run in hand at once.
    """
    texts: List[str] = []
    running: List[str] = []
    for token in positions:
        running.append(token.t)
        texts.append("".join(running))
    return texts


class RunProvenance(BaseModel):
    """What the worker attested when it finished the run.

    Travels with the run from the terminal frame, through the
    browser's snapshot, back to the save. Everything here used to be
    read from whichever worker happened to be active when the save
    arrived, so a run finished before a model switch was described by
    the model that replaced it.

    Not strict, unlike its siblings. This one is echoed back by the
    client from a worker payload, and the workers are the part of the
    system most likely to gain a field ahead of the supervisor; a
    save must not start failing because a worker learned to attest
    something new. Only the declared fields are read.
    """

    model_id: str
    checkpoint: str = ""
    # The commit the weights were read from. Defaults to empty rather
    # than being required because runs saved before TRUST-03 have no
    # commit to report, and a schema that rejected them would lose the
    # corpus to gain a field. Empty means "not recorded", which is
    # what those runs honestly are.
    revision: str = ""
    # The placement the model actually got, which is not always the
    # one requested: LLaDA and SmolLM3 fall back to CPU when CUDA is
    # unavailable, so a run that ran on CPU could be saved as GPU.
    device: str = "unknown"
    versions: Dict[str, str] = Field(default_factory=dict)
    tokenizer: Dict[str, Any] = Field(default_factory=dict)
    context_length: Optional[int] = None
    # What the run's signals measure and vary over, as the worker
    # declared them. A list of dicts rather than parsed channels: this
    # model is deliberately not strict because a worker may gain a
    # field ahead of the supervisor, and validating the channels here
    # would reintroduce exactly the coupling that permits.
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    # What the run cost the device, as the worker measured it. A dict
    # rather than named fields for the same reason as ``signals``: the
    # worker owns what it can measure, and a CPU run or an older one
    # sends nothing at all rather than zeros.
    resources: Dict[str, Any] = Field(default_factory=dict)


class SaveRunRequest(BaseModel):
    """One run, sent whole to be saved.

    Held to what one run of its model can be (`A2-TRUST-02`): text
    fields to their caps here, and every list to the bounds the
    registry reads off the model's sliders, once the fields have
    parsed. A save past either is a 422 naming the field.
    """

    model_config = STRICT

    model: str = Field(
        default=DEFAULT_MODEL, max_length=IDENTIFIER_CHARS_MAX
    )
    prompt: str = Field(max_length=PROMPT_CHARS_MAX)
    params: Dict[str, Any] = Field(default_factory=dict)
    # One of two ways to describe the same frames. ``frames`` plus
    # ``frame_tokens`` is the per-frame form a snapshot model sends;
    # ``frame_positions`` is the flat form an append model sends,
    # which ``normalized`` below expands into the first. Exactly one
    # arrives, and everything past this model sees only the first.
    frames: Optional[List[str]] = None
    frame_positions: Optional[RunPositions] = None
    original_frame_positions: Optional[RunPositions] = None
    final_text: str
    elapsed_seconds: Optional[float] = None
    per_frame_elapsed: Optional[List[float]] = None
    # Durable per-token records for the commit-order / diff /
    # confidence overlays. ``frame_tokens`` is the primary (possibly
    # edited) run; ``original_frame_tokens`` is the pre-edit snapshot,
    # sent only for edited runs so the counterfactual diff survives.
    frame_tokens: Optional[FrameTokens] = None
    original_frame_tokens: Optional[FrameTokens] = None
    # Per-position candidate sets (index = token position, not frame),
    # sent only when the opt-in capture ran. A position with no
    # capture is None, so the list stays aligned with positions.
    alternatives: Optional[
        List[Optional[List[TokenAlternative]]]
    ] = None
    canvas_index: Optional[List[int]] = None
    mean_conf: Optional[List[Optional[float]]] = None
    remask_edits: Optional[List[RemaskEdit]] = None
    # The pre-edit run's own signals, sent alongside
    # ``original_frame_tokens`` for edited runs. They let Analytics
    # compare original against edited on timing, confidence, and
    # candidates, not just on token text. Absent on unedited runs and
    # on edited runs saved before these fields existed.
    original_per_frame_elapsed: Optional[List[float]] = None
    original_elapsed_seconds: Optional[float] = None
    original_mean_conf: Optional[List[Optional[float]]] = None
    original_alternatives: Optional[
        List[Optional[List[TokenAlternative]]]
    ] = None
    # A diffusion run's candidates, per captured frame. Absent when
    # the capture was off, for an autoregressive run, and for runs
    # saved before it existed.
    candidates: Optional[FrameCandidates] = None
    # The pre-edit run's, sent only for an edited run, so the popover
    # can page between the two runs from the edit on. Unlike its
    # tokens these cannot be rebuilt later: an edit truncates the live
    # capture at the frame it branched from.
    original_candidates: Optional[FrameCandidates] = None
    # What the worker said about itself when this run finished,
    # echoed back from the terminal frame. A run whose connection
    # dropped has none, and sends what its opening frame carried
    # instead: the same envelope less what the run cost. Absent for a
    # run whose snapshot predates this field, which then falls back to
    # the supervisor's current view, as every save used to do.
    provenance: Optional[RunProvenance] = None
    # Which generation produced this run, from the same terminal
    # frame (`LIFE-01`). The store publishes under it, so a save that
    # was already made once lands on the run it made rather than on a
    # second copy. Absent for a run whose snapshot predates it.
    run_token: Optional[str] = Field(
        default=None, max_length=IDENTIFIER_CHARS_MAX
    )
    # When set, replace this existing run instead of creating a new
    # one. Used when a saved run is edited-and-resumed: the edited
    # (bundled) run replaces its pre-edit original so it is a single
    # Analytics row rather than two.
    run_id: Optional[str] = Field(
        default=None, max_length=IDENTIFIER_CHARS_MAX
    )
    # The revision the client believes it is replacing, echoed from
    # the save that produced it. The replacement is refused if the run
    # has moved on since, so two windows editing one run cannot have
    # the later writer silently erase the earlier. Absent means "I did
    # not look", which is accepted for a client that predates this
    # field and for the runs saved before revisions existed.
    expected_revision: Optional[int] = Field(default=None, ge=0)
    # Tokens the templated prompt occupied, as reported by the sampler
    # on its ``done`` frame rather than counted by the client, so the
    # saved figure is the one the run really built. None for a run
    # whose sampler predates the field.
    prompt_len: Optional[int] = Field(default=None, ge=0)
    # True when the run was stopped rather than finished, taken from
    # the ``cancelled`` flag on its terminal frame (`LIFE-04`).
    # Defaulted rather than optional because absent and false mean
    # the same thing here: only a run that says it was stopped was.
    partial: bool = False

    @model_validator(mode="after")
    def _within_bounds(self) -> "SaveRunRequest":
        _check_run_bounds(self)
        return self

    def normalized(self) -> "SaveRunRequest":
        """This request with its per-frame text filled in.

        Only the text. The positions stay flat all the way to disk
        now, which is the whole of stage two: the per-frame token
        arrays this used to build were 93% of a long run's bytes and
        every one of them was a prefix of the next.

        The text is still expanded because `history.txt` and
        `frames.jsonl` are read by things that have nothing to do
        with tokens, and 21 MiB of a 282 MiB run is not where the
        problem was.

        Returns self untouched for a request that sent per-frame
        arrays, so a diffusion save costs nothing.
        """
        # Emptiness is checked before shape, and deliberately: an
        # empty list is not a description of a run in either form.
        # `frames` used to carry `min_length=1`, which stopped being
        # expressible as a field constraint when there were two ways
        # to send the same thing, so the guarantee moved here rather
        # than lapsing. Without it a save with no frames reached the
        # store and failed later, in the GIF renderer, as an
        # assertion about something else.
        if not self.frames and not self.frame_positions:
            raise ValueError(
                "a run must carry frames or frame_positions"
            )
        if self.frames and self.frame_positions:
            # Both would be two descriptions of one run with nothing
            # deciding which is true, and quietly preferring either
            # is worse than refusing.
            raise ValueError(
                "a run carries frames or frame_positions, not both"
            )
        if not self.frame_positions:
            return self
        expanded = self.model_copy(
            update={
                "frames": expand_position_text(self.frame_positions),
            }
        )
        assert expanded.frames, "expansion produced no frames"
        assert len(expanded.frames) == len(self.frame_positions), (
            "one frame per position, or the run is not the run"
        )
        return expanded


# A field as a refusal names it, and what it holds, if anything.
CountedField = Tuple[str, Optional[Sized]]


def _check_run_bounds(body: SaveRunRequest) -> None:
    """Refuse a save holding more than one run of its model can.

    Counted, not cross-checked: whether the frames agree with each
    other is checked later, once the run is known to be one the app
    could have made. A ValueError, which the save reports as a 422
    naming the field.
    """
    bounds = run_bounds(body.model)
    run_text = bounds.positions_max * TOKEN_TEXT_CHARS_MAX
    frame_text = bounds.frame_positions_max * TOKEN_TEXT_CHARS_MAX
    checks: Tuple[Tuple[List[CountedField], int], ...] = (
        (_per_frame(body), bounds.frames_max),
        (_per_position(body), bounds.positions_max),
        (_per_canvas(body), bounds.frame_positions_max),
        (_per_alternative_set(body), CANDIDATES_PER_POSITION + 1),
        ([("final_text", body.final_text)], run_text),
        (_frame_texts(body), frame_text),
        (_carried_through(body), FREEFORM_JSON_CHARS_MAX),
    )
    for fields, limit in checks:
        _check_counts(fields=fields, limit=limit, model=body.model)


def _check_counts(
    *, fields: List[CountedField], limit: int, model: str
) -> None:
    """Refuse the first of ``fields`` holding more than ``limit``."""
    assert limit >= 1, "every bound admits something"
    for name, values in fields:
        if values is None:
            continue
        count = len(values)
        if count > limit:
            raise ValueError(
                f"{name} holds {count:,}, past the {limit:,} one"
                f" {model} run can hold"
            )


def _per_frame(body: SaveRunRequest) -> List[CountedField]:
    """Everything kept one per frame, the pre-edit layer's too."""
    fields: List[CountedField] = [
        ("frames", body.frames),
        ("frame_tokens", body.frame_tokens),
        ("original_frame_tokens", body.original_frame_tokens),
        ("per_frame_elapsed", body.per_frame_elapsed),
        (
            "original_per_frame_elapsed",
            body.original_per_frame_elapsed,
        ),
        ("mean_conf", body.mean_conf),
        ("original_mean_conf", body.original_mean_conf),
        ("canvas_index", body.canvas_index),
        ("remask_edits", body.remask_edits),
    ]
    # A capture's record budget bounds its candidates but not its
    # frames, and a capture of empty frames costs nothing against it.
    for name, capture in _captures(body):
        fields.append((f"{name}.frames", capture.frames))
        fields.append((f"{name}.segments", capture.segments))
    return fields


def _per_position(body: SaveRunRequest) -> List[CountedField]:
    """Everything kept one per position, across the whole run."""
    return [
        ("frame_positions", body.frame_positions),
        ("original_frame_positions", body.original_frame_positions),
        ("alternatives", body.alternatives),
        ("original_alternatives", body.original_alternatives),
    ]


def _per_canvas(body: SaveRunRequest) -> List[CountedField]:
    """What each frame holds: one canvas, on a diffusion run."""
    fields: List[CountedField] = []
    layers = (
        ("frame_tokens", body.frame_tokens),
        ("original_frame_tokens", body.original_frame_tokens),
    )
    for name, layer in layers:
        for index, tokens in enumerate(layer or []):
            fields.append((f"{name}[{index}]", tokens))
    for index, edit in enumerate(body.remask_edits or []):
        field = f"remask_edits[{index}]"
        fields.append((field, edit.token_positions))
    for name, capture in _captures(body):
        for index, sets in enumerate(capture.sets):
            fields.append((f"{name}.sets[{index}]", sets))
    return fields


def _per_alternative_set(body: SaveRunRequest) -> List[CountedField]:
    """Each position's alternatives: the captured candidates, then the
    committed token where they missed it, as the sampler adds it."""
    fields: List[CountedField] = []
    layers = (
        ("alternatives", body.alternatives),
        ("original_alternatives", body.original_alternatives),
    )
    for name, layer in layers:
        for index, entries in enumerate(layer or []):
            fields.append((f"{name}[{index}]", entries))
    return fields


def _frame_texts(body: SaveRunRequest) -> List[CountedField]:
    return [
        (f"frames[{index}]", text)
        for index, text in enumerate(body.frames or [])
    ]


def _carried_through(body: SaveRunRequest) -> List[CountedField]:
    """The blocks a save keeps as they came, measured as JSON."""
    provenance: Optional[str] = None
    if body.provenance is not None:
        provenance = body.provenance.model_dump_json()
    return [
        ("params", json.dumps(body.params)),
        ("provenance", provenance),
    ]


def _captures(
    body: SaveRunRequest,
) -> List[Tuple[str, FrameCandidates]]:
    captures: List[Tuple[str, FrameCandidates]] = []
    if body.candidates is not None:
        captures.append(("candidates", body.candidates))
    if body.original_candidates is not None:
        captures.append(
            ("original_candidates", body.original_candidates)
        )
    return captures


def _display_run_path(run_dir: Path) -> str:
    """Run folder as written in the repo, for the UI's status line."""
    return run_store.display_path(run_dir, REPO_ROOT)


def _dump_positions(
    positions: Optional[RunPositions],
) -> List[Dict[str, Any]]:
    """Serialize a flat run, dropping absent confidence.

    The same projection ``_dump_frame_tokens`` applies per frame,
    over the one list an append run has.
    """
    assert positions, "a flat run has at least one position"
    return [
        record.model_dump(exclude_none=True) for record in positions
    ]


def _dump_frame_tokens(
    frames: FrameTokens,
) -> List[Optional[List[Dict[str, Any]]]]:
    """Serialize frame token records, dropping absent confidence.

    ``exclude_none`` keeps masked tokens compact (no ``c`` key),
    matching the live protocol payload.
    """
    dumped: List[Optional[List[Dict[str, Any]]]] = []
    for frame in frames:
        if frame is None:
            dumped.append(None)
            continue
        dumped.append(
            [
                record.model_dump(exclude_none=True)
                for record in frame
            ]
        )
    return dumped


def _dump_alternatives(
    positions: List[Optional[List[TokenAlternative]]],
) -> List[Optional[List[Dict[str, Any]]]]:
    """Serialize per-position candidate sets for persistence.

    Keeps the index alignment with token positions: a position that
    captured nothing stays None rather than collapsing the list.

    ``exclude_none`` for the same reason ``_dump_frame_tokens`` uses
    it: ``rank`` is set on at most one entry per position, and a null
    on the other five would be pure weight in a file that already
    runs to tens of kilobytes.
    """
    dumped: List[Optional[List[Dict[str, Any]]]] = []
    for entry in positions:
        if entry is None:
            dumped.append(None)
            continue
        dumped.append(
            [
                candidate.model_dump(exclude_none=True)
                for candidate in entry
            ]
        )
    return dumped


def _context_metadata(
    prompt_len: Optional[int],
    provenance: Optional[RunProvenance],
) -> Dict[str, Any]:
    """The context block for a saved run, or empty when unknowable.

    Two figures, together because either alone answers nothing useful:
    a prompt length means little without the window it competed for,
    and the window means little without a prompt to place inside it.

    The window comes from the run's own provenance when it has one.
    Older snapshots fall back to the resident model's, which was the
    only source before and is right whenever nothing has changed
    since the run finished.
    """
    if prompt_len is None:
        return {}
    assert prompt_len >= 0, "prompt_len must be non-negative"
    block: Dict[str, Any] = {"prompt_tokens": prompt_len}
    if provenance is not None:
        window = provenance.context_length
    else:
        window = manager.active_context_length
    if window is not None:
        block["context_length"] = window
    return block


def _resources_metadata(
    provenance: Optional[RunProvenance],
) -> Dict[str, Any]:
    """What the run cost the device, or empty when unmeasured.

    Its own block, not part of ``_reproducibility_block``, which
    answers what it would take to run this again. What a run cost is
    an observation about the run, not an input to reproducing it, and
    folding the two together is how a block stops having one job.

    Read only from the run's own envelope, with no fall back to the
    supervisor's view. There is nothing to fall back to: the peak is a
    measurement of one run over one interval, and the resident model
    cannot be asked afterwards what a finished run held.
    """
    if provenance is None:
        return {}
    return dict(provenance.resources)


# Request fields copied into metadata verbatim when the client sent
# them. Absent stays absent: the readers distinguish "this run never
# recorded it" from "it recorded zero".
_OPTIONAL_METADATA_FIELDS = (
    "elapsed_seconds",
    "per_frame_elapsed",
    "canvas_index",
    "mean_conf",
    "original_per_frame_elapsed",
    "original_elapsed_seconds",
    "original_mean_conf",
)


def _describe_processor(
    provenance: Optional[RunProvenance],
) -> Tuple[str, Optional[str]]:
    """What ran the model, for the Processor column and the header.

    The run's own attested device when it has one. That is stronger
    than what the supervisor knows in two ways: it survives a model
    switch between finishing and saving, and it is where the model
    actually landed rather than where it was sent, which differ
    whenever CUDA was asked for on a host without it.

    Falls back to the supervisor's current device for snapshots taken
    before runs carried provenance.
    """
    if provenance is not None:
        device = provenance.device
    else:
        device = manager.active_device
    if device == "cuda":
        return "GPU", model_manager.gpu_name()
    if device == "cpu":
        return "CPU", model_manager.cpu_name()
    return "Unknown", None


def _attested_model_id(body: SaveRunRequest) -> str:
    """Which model produced this run, worker's word over client's.

    They agree for every ordinary save. When they do not, the client
    has told us about a different model than the one that generated
    the frames, and the worker is the one that was there. Logged
    rather than refused: the run itself is real and complete, and
    losing it to a disagreement about its label would be the worse
    outcome.
    """
    claimed = body.model or DEFAULT_MODEL
    if body.provenance is None:
        return claimed
    attested = body.provenance.model_id
    if not attested:
        return claimed
    if attested != claimed:
        logger.warning(
            "save claims model %s but the run was produced by %s;"
            " recording the latter",
            claimed,
            attested,
        )
    return attested


def _build_metadata(body: SaveRunRequest) -> Dict[str, Any]:
    """Assemble the metadata a saved run records.

    Split out of the save so the write path is about writing.

    Everything describing *how* the run was produced comes from the
    run's own provenance envelope, attested by the worker at the
    moment it finished. The supervisor's current state is consulted
    only for runs whose snapshot predates that envelope. This is
    `DATA-04`: two windows share one supervisor, so the model that
    is active when a save arrives is not necessarily the model that
    produced the run being saved.
    """
    provenance = body.provenance
    model_id = _attested_model_id(body)
    entry = REGISTRY.get(model_id)
    checkpoint = entry.checkpoint if entry else ""
    if provenance is not None and provenance.checkpoint:
        checkpoint = provenance.checkpoint
    # The one place the axes become the on-disk field. Shape rather
    # than family, because every reader of it asks whether the run has
    # a masked canvas to converge, not what the architecture was.
    model_type = (
        saved_model_type(entry.capabilities.generation_shape)
        if entry
        else SAVED_MODEL_TYPE_DIFFUSION
    )
    processor, processor_name = _describe_processor(provenance)
    metadata: Dict[str, Any] = {
        "backend": model_id,
        "model": checkpoint or model_id,
        # Lets the analytics suite gate diffusion-only charts (e.g.
        # convergence) off for autoregressive runs. Absent on runs
        # saved before this field existed, all of which are diffusion.
        "model_type": model_type,
        # GPU / CPU / Unknown, plus the device name for the timing
        # header.
        "processor": processor,
        "processor_name": processor_name,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prompt": body.prompt,
        "final_text": body.final_text,
        "params": body.params,
    }
    # Copied only when present, so an older run and a run that
    # measured nothing stay distinguishable in the saved file. A table
    # rather than a chain of ifs: they all do the same thing, and the
    # chain was most of this function's complexity.
    for name in _OPTIONAL_METADATA_FIELDS:
        value = getattr(body, name)
        if value is not None:
            metadata[name] = value
    if body.remask_edits:
        metadata["remask_edits"] = [
            edit.model_dump() for edit in body.remask_edits
        ]
    # Recorded only when true, so an older run stays distinguishable
    # from one measured as complete. This is what stops a run the
    # user stopped partway from reading, months later in Analytics,
    # exactly like a run that finished: the text ends where it ends
    # either way, and nothing else in the record says which.
    if body.partial:
        metadata["partial"] = True
    # How `tokens.json` is arranged, written down rather than left to
    # be inferred from whether the file's entries happen to be lists.
    # A reader that guessed would be one legitimately empty frame away
    # from reading a per-frame run as a flat one.
    if body.frame_positions:
        metadata[run_store.FRAME_SHAPE_KEY] = (
            run_store.FRAME_SHAPE_APPEND
        )
    # The run token is deliberately not set here. The store stamps it
    # beside the revision, so the identity a run is published under
    # and the identity recorded in its metadata cannot disagree.
    # Absent, not zeroed, when the length is unknown: an older run and
    # a run with an empty prompt must stay distinguishable, and the
    # Analytics rows are built to skip a missing block.
    context = _context_metadata(body.prompt_len, provenance)
    if context:
        metadata["context"] = context
    # Absent for a CPU run, for a run saved before this existed, and
    # for one whose worker could not read the device. All three are
    # honestly "not measured", which is what no block says.
    resources = _resources_metadata(provenance)
    if resources:
        metadata["resources"] = resources
    # The signal manifest, as its own key rather than folded into
    # ``capture``. That one answers "which files were written" and is
    # read as booleans per sidecar, both by the staging validator and
    # by the analytics reader; this answers "what do the channels mean
    # and what do they vary over". Written only when the run declared
    # any, so absent keeps meaning "infer as before" for every run
    # already on disk.
    if provenance is not None and provenance.signals:
        metadata[run_store.SIGNALS_KEY] = provenance.signals
    metadata["reproducibility"] = _reproducibility_block(
        body, provenance
    )
    return metadata


def _reproducibility_block(
    body: SaveRunRequest,
    provenance: Optional[RunProvenance],
) -> Dict[str, Any]:
    """What it would take to run this again and get this back.

    The environment half comes from the run's own envelope when it
    has one. The host half (GPU name, git commit) is still read at
    save time, because neither is something a worker attests and
    both are properties of the machine rather than the run.
    """
    if provenance is not None:
        versions = dict(provenance.versions)
        tokenizer = dict(provenance.tokenizer)
    else:
        versions = dict(manager.active_versions)
        tokenizer = dict(manager.active_tokenizer)
    return {
        "seed": body.params.get("seed"),
        "gpu": model_manager.gpu_name(),
        "git_commit": _git_commit(),
        # The model's commit, beside the app's. The pair is the whole
        # answer to "what produced this": ``git_commit`` pins the code
        # that ran, ``model_revision`` pins the weights it ran. Only
        # one of them used to be here, which made the block look
        # complete while leaving half the inputs unnamed. Empty for a
        # local checkpoint and for runs saved before this existed.
        "model_revision": (
            provenance.revision if provenance is not None else ""
        ),
        "versions": versions,
        # Which tokenizer produced these ids. Persisted per run so an
        # old run still answers the question after the model it used
        # has been swapped out or its checkpoint has moved on.
        "tokenizer": tokenizer,
        # Whether the two fields above describe the run or the
        # supervisor's state at save time. Recorded because the two
        # are not equally trustworthy and a reader cannot otherwise
        # tell which one it is looking at.
        "attested": provenance is not None,
    }


def _build_bundle(body: SaveRunRequest) -> run_store.RunBundle:
    """Turn a save request into the content of a run directory.

    ``tokens.json`` holds whichever shape the run has: a flat list of
    positions for a run that only grows, one array per frame for one
    whose positions change. Which it is, is recorded in metadata by
    ``_build_metadata`` rather than left for a reader to infer from
    the nesting.
    """
    return run_store.RunBundle(
        metadata=_build_metadata(body),
        final_text=body.final_text,
        frames=list(body.frames),
        frame_tokens=(
            _dump_positions(body.frame_positions)
            if body.frame_positions
            else (
                None
                if body.frame_tokens is None
                else _dump_frame_tokens(body.frame_tokens)
            )
        ),
        original_frame_tokens=(
            _dump_positions(body.original_frame_positions)
            if body.original_frame_positions
            else (
                None
                if body.original_frame_tokens is None
                else _dump_frame_tokens(body.original_frame_tokens)
            )
        ),
        alternatives=(
            None
            if body.alternatives is None
            else _dump_alternatives(body.alternatives)
        ),
        original_alternatives=(
            None
            if body.original_alternatives is None
            else _dump_alternatives(body.original_alternatives)
        ),
        candidates=_dump_candidates(body.candidates),
        original_candidates=_dump_candidates(
            body.original_candidates
        ),
    )


def _dump_candidates(
    candidates: Optional[FrameCandidates],
) -> Optional[Dict[str, Any]]:
    """A diffusion run's candidates for their sidecar, or None when
    the run has none. ``exclude_none`` so the rank appears only on
    the row that carries one."""
    if candidates is None:
        return None
    return candidates.model_dump(exclude_none=True)


def _save_run_blocking(body: SaveRunRequest) -> Dict[str, Any]:
    """Publish a run and describe it back to the client.

    Returns the id and revision as well as the display path, because
    an edited run has to be able to replace what it just saved, and
    doing that safely means quoting the revision it is replacing.

    Which run is written is the store's decision, not this one: it
    resolves the run token first, then the client's run id while that
    still names a run, and makes a new run when neither does.

    Expanded first, on this thread, because rebuilding a long run's
    frames is real work and the event loop is not where it belongs.
    """
    body = body.normalized()
    bundle = _build_bundle(body)
    run_id, revision = run_store.save(
        RESULTS_DIR,
        bundle,
        model_id=body.model or DEFAULT_MODEL,
        run_id=body.run_id or None,
        expected_revision=body.expected_revision,
        run_token=body.run_token,
    )
    run_dir = RESULTS_DIR / run_id

    # After publication, deliberately. The GIF is a derivative, and a
    # failure rendering one must not cost the user the run's text and
    # token data.
    try:
        _render_run_gif(body, bundle.metadata, run_dir)
    except Exception:  # noqa: BLE001
        logger.exception(
            "GIF rendering failed for %s; run is saved", run_id
        )

    return {
        "path": _display_run_path(run_dir),
        "run_id": run_id,
        "revision": revision,
    }


def _render_run_gif(
    body: SaveRunRequest,
    metadata: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Draw the run's preview, labelled with the model that ran it.

    Reads the label out of the metadata that was just written rather
    than off the request, so the picture and the record cannot
    disagree about which model this was; `DATA-04` may have preferred
    the worker's word over the client's claim.
    """
    model_id = str(metadata.get("backend", ""))
    entry = REGISTRY.get(model_id)
    history_to_gif(
        body.frames,
        run_dir / "diffusion.gif",
        header_text=body.prompt,
        model_label=(
            entry.display_name if entry else model_id or None
        ),
        model_type=str(
            metadata.get("model_type", "diffusion")
        ),
    )


@app.exception_handler(RequestValidationError)
async def _validation_refusal(
    request: Request, exc: RequestValidationError
) -> Response:
    """A refused save, in the shape the page reads one from.

    FastAPI's own 422 carries a ``detail`` list and no ``message``,
    which is the field the page shows, so every save refused for its
    shape read as "Save failed: unknown". Other routes keep the
    default.
    """
    if request.url.path != "/api/save":
        return await request_validation_exception_handler(
            request, exc
        )
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": _first_problem(exc)},
    )


def _first_problem(exc: RequestValidationError) -> str:
    """The first thing wrong with a request, in words.

    Built from where and what, never from the offending input, which
    for a run can be megabytes.
    """
    errors = exc.errors()
    if not errors:
        return "The save request was not valid."
    first = errors[0]
    where = ".".join(
        str(part) for part in first.get("loc", ()) if part != "body"
    )
    reason = str(first.get("msg", "not valid"))
    reason = reason.removeprefix("Value error, ")
    if where:
        return f"{where}: {reason}"
    return reason


@app.post("/api/save")
async def save_run(body: SaveRunRequest) -> JSONResponse:
    try:
        saved = await asyncio.to_thread(_save_run_blocking, body)
    except run_store.RevisionConflictError as exc:
        # Someone else wrote this run since the client last read it.
        # A conflict, not a failure: the client can reload and decide.
        logger.info("save conflict: %s", exc)
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": str(exc),
                "run_id": exc.run_id,
                "revision": exc.actual,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to save run")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    logger.info("saved run to %s", saved["path"])
    return JSONResponse(content={"success": True, **saved})


# -- Analytics endpoints --


@app.get("/api/vision/encoders")
async def vision_encoders_list() -> JSONResponse:
    """The encoders the tokeniser view can compare.

    Whether each one's configuration is on disk is part of the answer,
    so the page can offer a download before a reader picks one rather
    than after a request fails. Reading a cache is a filesystem walk,
    hence the thread.
    """
    async def describe(encoder: VisionEncoder) -> Dict[str, Any]:
        cached = await asyncio.to_thread(vision_is_cached, encoder)
        return {
            "id": encoder.id,
            "display_name": encoder.display_name,
            "repo_id": encoder.repo_id,
            "revision": encoder.revision,
            "summary": encoder.summary,
            "cached": cached,
        }

    return JSONResponse(content={
        "encoders": [
            await describe(encoder) for encoder in vision_declared()
        ]
    })


@app.get("/api/vision/geometry")
async def vision_geometry(
    encoder: str, width: int, height: int
) -> JSONResponse:
    """How this encoder would turn an image of this size into tokens.

    Dimensions rather than an image, which is the whole shape of this
    feature: the geometry depends on nothing else, so the picture
    never leaves the browser and there is no upload to own.

    No worker, no residency claim and no eviction either. Only two
    small JSON files are read, so this answers while a model is
    resident and mid-run, and a reader can compare both encoders
    without disturbing it.
    """
    found = vision_find(encoder)
    if found is None:
        known = [item.id for item in vision_declared()]
        return JSONResponse(
            status_code=404,
            content={
                "error": f"unknown encoder {encoder!r}",
                "known": known,
            },
        )

    for name, value in (("width", width), ("height", height)):
        if not VISION_MIN_DIMENSION <= value <= VISION_MAX_DIMENSION:
            return JSONResponse(
                status_code=400,
                content={"error": (
                    f"{name} {value} is outside"
                    f" {VISION_MIN_DIMENSION}"
                    f"..{VISION_MAX_DIMENSION}"
                )},
            )

    try:
        encoder_geometry = await asyncio.to_thread(
            vision_load_geometry, found
        )
    except EncoderUnavailable as exc:
        # Not cached and not fetchable, which is an operating error on
        # a first run with no network. 503 rather than 500: the
        # request was fine and may work later.
        return JSONResponse(
            status_code=503, content={"error": str(exc)}
        )

    image = vision_image_geometry(encoder_geometry, width, height)
    return JSONResponse(content={
        "encoder": {
            "id": found.id,
            "display_name": found.display_name,
            "longest_edge": encoder_geometry.longest_edge,
            "tile": encoder_geometry.tile,
            "patch": encoder_geometry.patch,
            "scale": encoder_geometry.scale,
            "patch_side": encoder_geometry.patch_side,
            "unseen_edge": encoder_geometry.unseen_edge,
            "token_side": encoder_geometry.token_side,
            "tokens_per_tile": encoder_geometry.tokens_per_tile,
            "patches_per_token": encoder_geometry.patches_per_token,
        },
        "image": {
            "source_width": image.source_width,
            "source_height": image.source_height,
            "resized_width": image.resized_width,
            "resized_height": image.resized_height,
            "fitted_width": image.fitted_width,
            "fitted_height": image.fitted_height,
            "tile_rows": image.tile_rows,
            "tile_cols": image.tile_cols,
            "tile_count": image.tile_count,
            "aspect_changed": image.aspect_changed,
            "total_tokens": image.total_tokens,
        },
    })


@app.get("/api/analytics/runs")
async def analytics_list_runs() -> JSONResponse:
    runs = await asyncio.to_thread(list_runs, RESULTS_DIR)
    return JSONResponse(content=runs)


# How many runs one comparison may carry. The chart is a legend and
# a handful of lines; past this it is unreadable before it is slow,
# and the list used to be unbounded, so a crafted request could ask
# the server to read the whole archive in one breath.
COMPARE_RUNS_MAX = 12

# Parameters a legend label may name before it stops being a label.
COMPARE_LABEL_PARAMS_MAX = 3

# What became of one selection. Every id gets exactly one of these,
# which is the difference from silently returning fewer lines than
# the user asked for.
COMPARE_STATUS_DATA = "data"
COMPARE_STATUS_UNAVAILABLE = "unavailable"
COMPARE_STATUS_ERROR = "error"

# Why a selection carries no data. Separate from the message so the
# browser can group or style them without matching on prose.
COMPARE_NOT_FOUND = "not_found"
COMPARE_INVALID_ID = "invalid_id"
COMPARE_UNSUPPORTED = "unsupported_version"
COMPARE_UNREADABLE = "unreadable"
COMPARE_NO_CURVE = "no_curve"

COMPARE_REASONS = (
    COMPARE_NOT_FOUND,
    COMPARE_INVALID_ID,
    COMPARE_UNSUPPORTED,
    COMPARE_UNREADABLE,
    COMPARE_NO_CURVE,
)

assert COMPARE_RUNS_MAX > 1, "a comparison needs two runs"
assert len(set(COMPARE_REASONS)) == len(COMPARE_REASONS)


def _compute_run_metrics(run_id: str) -> Dict[str, Any]:
    # Through the store's resolver like every other run-id endpoint.
    # This one used to join the path unguarded, so a crafted id could
    # walk out of the data root while its three siblings refused.
    run_dir = run_store.resolve_run_dir(RESULTS_DIR, run_id)
    meta = load_run_metadata(run_dir)
    # Which file holds the frames is the schema version's business,
    # so the check for its absence belongs to the reader too.
    frames = read_frame_texts(run_dir, meta)
    convergence, basis, produced_from = _run_convergence(
        run_dir, frames, meta.get("canvas_index")
    )

    result: Dict[str, Any] = {
        "run_id": run_id,
        "convergence": convergence,
        # Named so the chart can caption a weaker measure rather than
        # present it as the stronger one.
        "convergence_basis": basis,
        "total_frames": len(frames),
        # Carried so compare can decide what a run can contribute
        # without consulting the catalog. The browser used to look
        # this up in its in-memory run list, which coupled the two
        # endpoints for one string.
        "model_type": str(
            meta.get("model_type", "diffusion")
        ),
        # What to call this run's model in prose. The convergence
        # caption names it, and only the server can turn a registry
        # id into something worth reading. Falls back to the id, so
        # a run from a model this build no longer knows still reads
        # as itself rather than as nothing.
        "model_label": _model_label(meta),
    }
    for key in (
        "per_frame_elapsed",
        "elapsed_seconds",
        "remask_edits",
        "mean_conf",
        "original_per_frame_elapsed",
        "original_elapsed_seconds",
        "original_mean_conf",
    ):
        if key in meta:
            result[key] = meta[key]
    # Same repair list_runs applies, so the two endpoints cannot
    # disagree about how long an edited run took.
    repaired = total_elapsed_seconds(
        meta.get("per_frame_elapsed")
    )
    if repaired is not None:
        result["elapsed_seconds"] = repaired
    canvas_index = meta.get("canvas_index")
    if canvas_index:
        result["canvas_boundaries"] = canvas_boundaries(
            canvas_index
        )
    # Computed here rather than in the browser because it needs the
    # canvas each frame belongs to, and getting it wrong is invisible:
    # the old client-side version read plausibly and undercounted a
    # whole committed canvas.
    #
    # Fed the sampler's own resolution counts, which is not always the
    # series above. The two charts answer different questions: how
    # settled the canvas is, and how fast the model produced. Only the
    # second has a live counterpart, and the generator's footer counts
    # what the sampler emitted, so feeding this the settlement series
    # would make the same run read as two speeds again.
    result["tokens_produced"] = tokens_produced_series(
        produced_from, canvas_index
    )
    return result


def _model_label(meta: Dict[str, Any]) -> str:
    """The display name for the model that produced a run."""
    backend = str(meta.get("backend", ""))
    entry = REGISTRY.get(backend)
    if entry is None:
        return backend
    return entry.display_name


def _run_convergence(
    run_dir: Path,
    frames: List[str],
    canvas_index: Any = None,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """A run's convergence series, how it was measured, and the
    series the throughput chart should count from.

    Three measures, and which one a run gets is a property of the run
    rather than a preference. Where the mask is a real token the flag
    is ground truth and is used. Where the sampler inferred it from a
    position holding still, the flag overstates badly, so agreement
    with what the canvas committed is used instead. A run that saved
    no usable records falls back to counting mask glyphs against
    characters, which is roughly a tenth of the archive here.

    The third return value exists because the throughput chart must
    keep counting what the sampler resolved even when the convergence
    chart stops. Only throughput has a live counterpart, and the
    generator's footer counts the sampler's own reveals, so the two
    would disagree again if this handed back the settlement series.

    A malformed token stream falls back rather than raising. The
    weaker curve is worth more than no page, and the basis says which
    one the reader is looking at.
    """
    try:
        loaded = load_run_frames(run_dir)
    except (ValueError, OSError):
        logger.warning(
            "token records unreadable for %s; counting characters",
            run_dir.name,
        )
        loaded = None

    if loaded is not None and loaded.get("records_available"):
        positions = loaded.get("positions")
        if positions is not None and len(positions) == len(frames):
            # A run that only grows has no masked position and
            # nothing behind the newest one moves, so its curve
            # follows from the count alone. Taken before the branches
            # below because those exist to tell apart two ways a
            # position can change, and here none of them do.
            by_count = convergence_from_positions(len(positions))
            return (
                by_count, CONVERGENCE_BASIS_TOKENS, by_count
            )
        token_frames = loaded.get("frames")
        if records_match_frames(token_frames, len(frames)):
            by_mask = convergence_from_records(token_frames)
            if masks_are_real(token_frames):
                return (
                    by_mask, CONVERGENCE_BASIS_TOKENS, by_mask
                )
            return (
                convergence_from_settlement(
                    token_frames, canvas_index
                ),
                CONVERGENCE_BASIS_SETTLEMENT,
                by_mask,
            )
    by_chars = compute_convergence(frames)
    return (by_chars, CONVERGENCE_BASIS_CHARACTERS, by_chars)


def _unsupported_version_response(
    exc: UnsupportedRunVersionError,
) -> JSONResponse:
    """Answer a run this build cannot read with a plain explanation.

    Separate from the generic malformed-run 400 so the browser can
    say "update the app" rather than "this run is broken". The run is
    almost certainly fine; this build is the old one.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": (
                "This run was saved by a newer version of the app"
                f" (format {exc.version}), which this build cannot"
                " read. Update to open it."
            ),
            "unsupported_version": True,
        },
    )


@app.get("/api/analytics/runs/{run_id}/metrics")
async def analytics_run_metrics(run_id: str) -> JSONResponse:
    try:
        result = await asyncio.to_thread(
            _compute_run_metrics, run_id
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404, content={"error": str(exc)}
        )
    except UnsupportedRunVersionError as exc:
        return _unsupported_version_response(exc)
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"error": str(exc)}
        )
    return JSONResponse(content=result)


@app.get("/api/analytics/runs/{run_id}/metadata")
async def analytics_run_metadata(run_id: str) -> JSONResponse:
    """Everything about one run that the catalog no longer carries.

    The list used to hand back whole metadata files, so the detail
    panel could build its rows from a row it already had. It cannot
    any more, and that is the point: the list pays for every run and
    this pays for the one the user opened.
    """
    try:
        meta = await asyncio.to_thread(_run_metadata, run_id)
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404, content={"error": str(exc)}
        )
    except UnsupportedRunVersionError as exc:
        return _unsupported_version_response(exc)
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"error": str(exc)}
        )
    return JSONResponse(content=meta)


def _run_metadata(run_id: str) -> Dict[str, Any]:
    """One run's full metadata, guarded like every other read."""
    run_dir = run_store.resolve_run_dir(RESULTS_DIR, run_id)
    meta = load_run_metadata(run_dir)
    # The version is checked here rather than trusted, so a run this
    # build cannot read is refused instead of rendered from fields it
    # does not understand.
    run_schema_version(meta)
    # Both are computed rather than stored, and the detail panel
    # shows them, so they travel with the metadata rather than
    # leaving the panel to work out which run list to consult.
    meta["has_diff"] = (
        run_dir / "original_tokens.json"
    ).is_file()
    repaired = total_elapsed_seconds(
        meta.get("per_frame_elapsed")
    )
    if repaired is not None:
        meta["elapsed_seconds"] = repaired
    return meta


def _compute_run_frames(run_id: str) -> Dict[str, Any]:
    """Load durable token streams for the overlay viewer.

    Kept separate from ``_compute_run_metrics`` because token streams
    are large; the analytics UI fetches this only when a run's overlay
    viewer opens.
    """
    run_dir = run_store.resolve_run_dir(RESULTS_DIR, run_id)
    meta = load_run_metadata(run_dir)
    data = load_run_frames(run_dir)
    # A run that only grows goes out flat and the page rebuilds each
    # frame as a prefix, which is the same slice the generator does
    # live. At 2,048 tokens that is the difference between a 123 MiB
    # download and under a megabyte, and it is the download rather
    # than the file that the reader waits on.
    #
    # Old runs get it too when their frames turn out to be prefixes,
    # though only for the wire: the file still has to be parsed to
    # discover that, so an old long run is quicker to draw and no
    # quicker to open.
    positions = data["positions"]
    return {
        "run_id": run_id,
        "frames": None if positions is not None else data["frames"],
        "positions": positions,
        "original_frames": (
            None
            if data["original_positions"] is not None
            else data["original_frames"]
        ),
        "original_positions": data["original_positions"],
        "records_available": data["records_available"],
        "alternatives": data["alternatives"],
        "alternatives_available": data[
            "alternatives_available"
        ],
        "original_alternatives": data[
            "original_alternatives"
        ],
        "candidates": data["candidates"],
        "original_candidates": data["original_candidates"],
        "remask_edits": meta.get("remask_edits", []),
        "canvas_index": meta.get("canvas_index"),
        "stop_rule": _stop_rule(meta),
        # What each signal varies over, as the run declared it, so
        # the page reads a channel by its axes. None for a run saved
        # before manifests, which the page reads as it always has.
        "signals": meta.get(run_store.SIGNALS_KEY),
    }


# The parameters a model that stops adaptively declares, and that a
# saved run's rule is read back from. The step budget rides along
# because the readout's verdict on a committed canvas compares its
# length with it.
STOP_RULE_PARAMS: Tuple[str, ...] = (
    "confidence_threshold",
    "stability_threshold",
    "max_denoising_steps",
)


def _stop_rule(
    meta: Dict[str, Any],
) -> Optional[Dict[str, ParamValue]]:
    """The stopping rule a saved run ran under, or None.

    None for a model that does not stop adaptively, which is how the
    Analytics page knows to offer neither the readout nor the
    Stopping chart. Each value comes from the run's own parameters,
    held to the experimental bounds, the widest a run could have
    used. One that is missing or malformed takes the registry
    default, so a run saved before the rule was a parameter reads as
    0.005 and 1: what the checkpoint applied to it.
    """
    entry = REGISTRY.get(str(meta.get("backend", "")))
    if entry is None:
        return None
    if not entry.capabilities.adaptive_stopping:
        return None
    saved = meta.get("params")
    if not isinstance(saved, dict):
        saved = {}
    specs = {spec.name: spec for spec in entry.param_specs}
    rule: Dict[str, ParamValue] = {}
    for name in STOP_RULE_PARAMS:
        assert name in specs, (
            f"{entry.id} stops adaptively without {name}"
        )
        rule[name] = _stop_rule_value(specs[name], saved.get(name))
    return rule


def _stop_rule_value(spec: ParamSpec, given: Any) -> ParamValue:
    """One saved value of the rule, or its default."""
    default = default_of(spec, device=None)
    if given is None:
        return default
    try:
        return coerce(spec, given, device=None, experimental=True)
    except ValueError:
        return default


@app.get("/api/analytics/runs/{run_id}/frames")
async def analytics_run_frames(run_id: str) -> JSONResponse:
    try:
        result = await asyncio.to_thread(
            _compute_run_frames, run_id
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404, content={"error": str(exc)}
        )
    except UnsupportedRunVersionError as exc:
        return _unsupported_version_response(exc)
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"error": str(exc)}
        )
    return JSONResponse(content=result)


@app.get("/api/analytics/compare")
async def analytics_compare(ids: str = "") -> JSONResponse:
    """Compare a bounded set of runs, accounting for every one.

    The contract is that a selection is never silently dropped. Each
    id comes back as exactly one record saying what happened to it,
    because a chart with fewer lines than the user ticked, and
    nothing explaining which are missing, is worse than an error.
    """
    run_ids = _compare_selection(ids)
    if len(run_ids) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "ids parameter is required"},
        )
    if len(run_ids) > COMPARE_RUNS_MAX:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Compare accepts up to {COMPARE_RUNS_MAX}"
                    f" runs; {len(run_ids)} were selected."
                )
            },
        )
    results = [
        await _compare_one(run_id) for run_id in run_ids
    ]
    assert len(results) == len(run_ids), "one record per id"
    return JSONResponse(content=results)


def _compare_selection(ids: str) -> List[str]:
    """The ids to compare: trimmed, non-empty, first occurrence.

    Deduplicated because the same run twice is one line drawn twice,
    and it would otherwise count against the cap below while adding
    nothing.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for raw in ids.split(","):
        run_id = raw.strip()
        if not run_id:
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        ordered.append(run_id)
    return ordered


async def _compare_one(run_id: str) -> Dict[str, Any]:
    """One selection's outcome: data, unavailable, or an error.

    Every failure is caught here rather than escaping. The batch used
    to survive only the two exception types it named, so a run whose
    frames were corrupt in an unanticipated way took down the whole
    comparison, including the runs that were fine.
    """
    try:
        record = await asyncio.to_thread(
            _compute_run_metrics, run_id
        )
    except run_store.RunNotFoundError:
        return _compare_error(
            run_id, COMPARE_NOT_FOUND, "This run no longer exists."
        )
    except run_store.InvalidRunIdError:
        return _compare_error(
            run_id, COMPARE_INVALID_ID, "Not a valid run id."
        )
    except UnsupportedRunVersionError:
        return _compare_error(
            run_id,
            COMPARE_UNSUPPORTED,
            "Saved by a newer version of this app.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("compare failed for %s", run_id)
        return _compare_error(
            run_id, COMPARE_UNREADABLE, f"Could not be read: {exc}"
        )

    record["status"] = COMPARE_STATUS_DATA
    record["label"] = _compare_label(run_id)
    if (
        record.get("model_type")
        == SAVED_MODEL_TYPE_AUTOREGRESSIVE
    ):
        # Real run, no comparable curve: an autoregressive run has no
        # masked canvas to converge. Said out loud rather than
        # dropped, which is what the chart used to do.
        record["status"] = COMPARE_STATUS_UNAVAILABLE
        record["reason"] = COMPARE_NO_CURVE
        record["message"] = (
            "Autoregressive runs have no convergence curve."
        )
    return record


def _compare_error(
    run_id: str, reason: str, message: str
) -> Dict[str, Any]:
    """One refused selection, in the shape the chart legend reads."""
    assert reason in COMPARE_REASONS, reason
    return {
        "run_id": run_id,
        "status": COMPARE_STATUS_ERROR,
        "reason": reason,
        "message": message,
        # Kept so the legend can name the run it could not draw.
        "label": run_id,
    }


def _compare_label(run_id: str) -> str:
    """A legend label built from the model's own parameters.

    The browser used to assemble this from ``steps``, ``gen_length``
    and ``block_length``, which only LLaDA has, so a DiffusionGemma
    or SmolLM3 run was labelled with the word ``undefined`` three
    times. The registry knows each model's parameters and what to
    call them, and only the server can read the registry, so the
    label is built here.
    """
    try:
        run_dir = run_store.resolve_run_dir(RESULTS_DIR, run_id)
        meta = load_run_metadata(run_dir)
    except (ValueError, OSError):
        return run_id

    entry = REGISTRY.get(str(meta.get("backend", "")))
    if entry is None:
        return run_id

    params = meta.get("params")
    if not isinstance(params, dict):
        return entry.display_name

    parts: List[str] = []
    for spec in entry.param_specs:
        if len(parts) >= COMPARE_LABEL_PARAMS_MAX:
            break
        if spec.name not in params:
            continue
        parts.append(f"{spec.label}={params[spec.name]}")
    if not parts:
        return entry.display_name
    return entry.display_name + " " + " ".join(parts)


@app.get("/api/analytics/system")
async def analytics_system_info() -> JSONResponse:
    """GPU name and data root for the analytics UI.

    The GPU name because the supervisor has no torch. The data root
    because the delete confirmation used to spell it ``results/``
    from a hardcoded string, which stopped being true the moment
    the root became configurable, and a dialog about permanent
    deletion is the worst place to name the wrong directory.
    """
    return JSONResponse(
        content={
            "gpu_name": model_manager.gpu_name(),
            "results_dir": _display_run_path(RESULTS_DIR),
        }
    )


def _delete_run_blocking(run_id: str) -> None:
    """Delete one saved run directory under the data root."""
    run_store.delete(RESULTS_DIR, run_id)


@app.delete("/api/analytics/runs/{run_id}")
async def analytics_delete_run(run_id: str) -> JSONResponse:
    try:
        await asyncio.to_thread(_delete_run_blocking, run_id)
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": str(exc)},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except OSError as exc:
        logger.exception("failed to delete run %s", run_id)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    logger.info("deleted run %s", run_id)
    return JSONResponse(content={"success": True})


# -- Durable UI state (origin-independent frontend preferences) --

# Collections still live in the ui-state file, because that file
# already has the interprocess lock and the atomic replace this needs.
# What changed is who may write the key: the generic PUT refuses it,
# and the operations below are the only way in.
COLLECTIONS_KEY = "diffusion_collections"


class UiStateValue(BaseModel):
    """A UI-state value, verbatim as its localStorage string."""

    value: str


def _reconcile_new_runs(state: Dict[str, str]) -> Dict[str, str]:
    """Prune the "new run" cue to run IDs whose folders still exist.

    The cue accumulates IDs of saved-but-unviewed runs. A run deleted
    outside the app (or before per-delete clearing existed) would
    linger as an orphan and inflate the generator/menu count forever,
    since it no longer appears in Analytics to open or delete.
    Reconciling here, on the endpoint every page hydrates from, makes
    the count self-heal everywhere. A freshly saved run is never
    pruned: its folder exists before its ID is added to the cue.

    Read and write happen under one lock, because this derives a new
    value from the stored one: pruning a snapshot taken before a
    concurrent PUT and then writing the result would undo that PUT.
    The run scan is done first so the lock is not held across it.
    """
    if not state.get("diffusion_new_runs"):
        return state
    existing = _existing_run_ids()

    def prune(raw: Optional[str]) -> Optional[str]:
        ids = _decode_id_list(raw)
        if ids is None:
            # Corrupt: leave it for load_ui_state to drop.
            return None
        kept = [run_id for run_id in ids if run_id in existing]
        if len(kept) == len(ids):
            return None
        return json.dumps(kept)

    try:
        return mutate_ui_state_key(
            RESULTS_DIR, "diffusion_new_runs", prune
        )
    except (KeyError, ValueError, OSError):
        logger.exception("failed to reconcile new-run cue")
        return state


def _decode_id_list(raw: Optional[str]) -> Optional[List[Any]]:
    """Parse a stored JSON list, or ``None`` if it is not one."""
    if not raw:
        return None
    try:
        ids = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(ids, list):
        return None
    return ids


def _existing_run_ids() -> Set[str]:
    """Run IDs with a saved run on disk (the folder name is the ID).

    Through the store, so "is this a run" is decided in one place.
    Slightly stricter than the directory scan this replaced: a folder
    with no metadata is a half-written save, and counting one as a
    live run is how the reconciliation would keep a cue alive for
    something Analytics cannot open.
    """
    return set(run_store.list_run_ids(RESULTS_DIR))


def _reconcile_collections(state: Dict[str, str]) -> Dict[str, str]:
    """Drop deleted runs from every collection they were filed into.

    Same reasoning as ``_reconcile_new_runs``, with a sharper failure
    if skipped: a collection is a list the user reads, so an id whose
    folder is gone would show as a row that cannot be opened, and the
    tab's count would overstate what is in it. Runs are deleted from
    the table, from another window, or from the filesystem, and only
    the first of those can prune client-side.

    Empty collections survive: the user made them, and one whose runs
    have been deleted is still a place they intend to file more.

    Under one lock for the same reason as the cue above, and it
    matters more here: this key holds filing the user did by hand,
    which nothing on disk can reconstruct.
    """
    if not state.get(COLLECTIONS_KEY):
        return state
    existing = _existing_run_ids()

    def prune(raw: Optional[str]) -> Optional[str]:
        current = collection_ops.decode(raw)
        kept, dropped = collection_ops.prune_missing(
            current, existing
        )
        if dropped == 0 and kept == current:
            return None
        return collection_ops.encode(kept)

    try:
        return mutate_ui_state_key(
            RESULTS_DIR, COLLECTIONS_KEY, prune
        )
    except (KeyError, ValueError, OSError):
        logger.exception("failed to reconcile collections")
        return state


@app.get("/api/ui-state")
async def get_ui_state() -> JSONResponse:
    """Return durable UI state (Settings, analytics "new run" cue,
    prompt history, collections, generate teaser). The frontend
    hydrates localStorage from this on boot so the values survive
    restarts whatever the window origin (see src/web/ui_state.py).
    The "new run" cue and the collections are both reconciled against
    existing runs, so a deleted run neither lingers in the count nor
    shows as an unopenable row in a collection.
    """
    state = await asyncio.to_thread(load_ui_state, RESULTS_DIR)
    state = await asyncio.to_thread(_reconcile_new_runs, state)
    state = await asyncio.to_thread(_reconcile_collections, state)
    return JSONResponse(content=state)


@app.put("/api/ui-state/{key}")
async def put_ui_state(
    key: str, body: UiStateValue
) -> JSONResponse:
    if key == COLLECTIONS_KEY:
        # The one key with no whole-value write. Collections are the
        # only durable value that is intent rather than cache, and
        # replacing the array wholesale is how one window used to
        # drop another's filing: both read the same list, both wrote
        # a different successor, and the later write won. The
        # operations below say what changed instead, so the lost
        # update is unrepresentable rather than merely unlikely.
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "reason": "use_collection_operations",
                "message": (
                    "collections are changed through"
                    " /api/collections, not by replacement"
                ),
            },
        )
    try:
        state = await asyncio.to_thread(
            set_ui_state_key, RESULTS_DIR, key, body.value
        )
    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": str(exc)},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except OSError as exc:
        logger.exception("failed to write ui-state key %s", key)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    return JSONResponse(content={"success": True, "state": state})


# -- Collections: one endpoint per gesture --
#
# Each takes what the user did, not what the list should become, and
# applies it to whatever is stored at the moment it runs. That is the
# whole of DATA-02's chosen fork: a client that cannot name a
# successor state cannot overwrite one it never saw.
#
# Every response carries the full list afterwards, so the caller
# adopts rather than merges, and a window that was behind is level
# again the moment it acts.


class CollectionName(BaseModel):
    """A collection's display name, for create and rename.

    The run fields are create-only, and they are here so that naming
    a collection from the filing dialog is one gesture: both halves
    land under a single lock, or neither does. ``run_ids`` is the
    same idea for a selection, so naming a collection for six runs
    cannot leave it made and empty.
    """

    name: str
    run_id: Optional[str] = None
    run_ids: Optional[List[str]] = None

    def ids(self) -> List[str]:
        if self.run_ids is not None:
            return self.run_ids
        if self.run_id is not None:
            return [self.run_id]
        return []


class CollectionRun(BaseModel):
    """A run id, for filing and for the star."""

    run_id: str


class CollectionRuns(BaseModel):
    """One run or several, for filing.

    Either field, so the single-run path keeps the shape it had and
    the table's multi-row selection does not have to send one request
    per row. Both go to the same operation, which files all of them
    or none.
    """

    run_id: Optional[str] = None
    run_ids: Optional[List[str]] = None

    def ids(self) -> List[str]:
        if self.run_ids is not None:
            return self.run_ids
        if self.run_id is not None:
            return [self.run_id]
        return []


def _collections_apply(
    operation: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Run one operation with the state file held against everyone.

    The transform runs inside ``mutate_ui_state_key``, so the read it
    works from and the write it produces cannot be separated by
    another process. Returning the list rather than the ui-state
    mapping keeps the endpoints from re-parsing what they just wrote.
    """
    settled: Dict[str, List[Dict[str, Any]]] = {}

    def mutate(raw: Optional[str]) -> Optional[str]:
        current = collection_ops.decode(raw)
        updated = operation(current)
        settled["value"] = updated
        if updated == current:
            return None  # A no-op gesture does not rewrite the file.
        return collection_ops.encode(updated)

    mutate_ui_state_key(RESULTS_DIR, COLLECTIONS_KEY, mutate)
    assert "value" in settled, "the operation did not run"
    return settled["value"]


async def _collections_respond(
    operation: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> JSONResponse:
    """Apply an operation off the event loop and answer with the list.

    ``CollectionError`` is the client asking for something the
    contract refuses, so it carries a reason the browser can act on
    rather than a bare status. ``ValueError`` here is the ui-state
    size bound, which is the aggregate limit no single operation can
    see coming.
    """
    try:
        value = await asyncio.to_thread(_collections_apply, operation)
    except collection_ops.CollectionError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "reason": exc.reason,
                "message": exc.message,
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "reason": "collections_full",
                "message": str(exc),
            },
        )
    except OSError as exc:
        logger.exception("failed to write collections")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    return JSONResponse(
        content={"success": True, "collections": value}
    )


@app.get("/api/collections")
async def get_collections() -> JSONResponse:
    """The stored collections, reconciled against runs on disk.

    The same prune the hydrate does, exposed on its own so a window
    can resync without reloading the page.
    """
    state = await asyncio.to_thread(load_ui_state, RESULTS_DIR)
    state = await asyncio.to_thread(_reconcile_collections, state)
    value = collection_ops.decode(state.get(COLLECTIONS_KEY))
    return JSONResponse(
        content={"success": True, "collections": value}
    )


@app.post("/api/collections")
async def create_collection(body: CollectionName) -> JSONResponse:
    existing = await asyncio.to_thread(_existing_run_ids)
    run_ids = body.ids()

    def operation(
        current: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        made = collection_ops.create(current, body.name)
        if not run_ids:
            return made
        # The id is the server's, and create appends, so the new
        # collection is the last one. Composing the two pure
        # operations here is what makes the pair atomic.
        return collection_ops.add_runs(
            made, made[-1]["id"], run_ids, existing
        )

    return await _collections_respond(operation)


@app.post("/api/collections/favorite")
async def favorite_collection_run(
    body: CollectionRun,
) -> JSONResponse:
    """The star, which is one gesture with two meanings.

    Declared above the ``{collection_id}`` routes because FastAPI
    matches in definition order and "favorite" would otherwise be
    read as a collection id.
    """
    existing = await asyncio.to_thread(_existing_run_ids)
    return await _collections_respond(
        lambda current: collection_ops.toggle_favorite(
            current, body.run_id, existing
        )
    )


@app.post("/api/collections/{collection_id}/rename")
async def rename_collection(
    collection_id: str, body: CollectionName
) -> JSONResponse:
    return await _collections_respond(
        lambda current: collection_ops.rename(
            current, collection_id, body.name
        )
    )


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str) -> JSONResponse:
    return await _collections_respond(
        lambda current: collection_ops.delete(
            current, collection_id
        )
    )


@app.post("/api/collections/{collection_id}/runs")
async def add_collection_runs(
    collection_id: str, body: CollectionRuns
) -> JSONResponse:
    """File one run or a selection of them.

    ``favorites`` is accepted here even before it exists, so the
    table's bulk star is a single request: the two operations compose
    inside one lock rather than needing a create first.
    """
    existing = await asyncio.to_thread(_existing_run_ids)
    run_ids = body.ids()

    def operation(
        current: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if collection_id == collection_ops.FAVORITES_ID:
            current = collection_ops.ensure_favorites(current)
        return collection_ops.add_runs(
            current, collection_id, run_ids, existing
        )

    return await _collections_respond(operation)


@app.delete("/api/collections/{collection_id}/runs/{run_id}")
async def remove_collection_run(
    collection_id: str, run_id: str
) -> JSONResponse:
    return await _collections_respond(
        lambda current: collection_ops.remove_run(
            current, collection_id, run_id
        )
    )


# -- HTML pages with automatic asset cache-busting --

# Local CSS/JS references (external CDN/font URLs, which are not
# root-relative, are left untouched).
_ASSET_REF_RE = re.compile(
    r'(?P<attr>href|src)="(?P<path>/[^"?#]+\.(?:css|js))"'
)

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _stamp_asset_versions(html: str) -> str:
    """Append ``?v=<mtime>`` to local CSS/JS refs.

    The version is each asset file's modification time, so the browser
    re-fetches a file exactly when it changes: automatic, per-file
    cache-busting with no manual version bumping and no reliance on
    the browser honoring ``no-store``.
    """

    def _replace(match: "re.Match[str]") -> str:
        path = match.group("path")
        asset = STATIC_DIR / path.lstrip("/")
        try:
            version = asset.stat().st_mtime_ns
        except OSError:
            # Unknown file: leave the reference as it is.
            return match.group(0)
        return f'{match.group("attr")}="{path}?v={version}"'

    return _ASSET_REF_RE.sub(_replace, html)


# The Generation nav link ships hidden and is revealed only when a
# worker is resident, because the generator is gated on one. Analytics
# and Settings used to ask /api/models for that single boolean, which
# cost two nvidia-smi subprocesses per page load and moved every link
# to its right when the answer arrived. The supervisor already knows.
_GENERATION_LINK_RE = re.compile(
    r'<a\b[^>]*\bid="link-generation"[^>]*>'
)
_HIDDEN_ATTR_RE = re.compile(r"\s+hidden(?=[\s>])")


def _active_model_is_serving() -> bool:
    """Whether a worker is resident and answering requests."""
    active_id = manager.active_id
    if active_id is None:
        return False
    return manager.is_serving(active_id)


def _reveal_generation_link(html: str, *, resident: bool) -> str:
    """Unhide the Generation nav link when a model is serving.

    Pages without the link are returned untouched, so this can run for
    every page rather than being wired per route.
    """
    if not resident:
        return html

    def _replace(match: "re.Match[str]") -> str:
        return _HIDDEN_ATTR_RE.sub("", match.group(0), count=1)

    return _GENERATION_LINK_RE.sub(_replace, html)


# State a page needs before its first paint, inlined ahead of the
# scripts so they can read it synchronously instead of fetching it and
# rebuilding the page around the answer. Every consumer falls back to
# fetching when it is absent, which is what keeps the vm test harness
# (and opening a file directly) working.
_BOOT_GLOBAL = "window.__BOOT__"


def _boot_script(state: Dict[str, Any]) -> str:
    """Serialise boot state as an inline script tag.

    ``<`` is escaped so a value containing ``</script>`` cannot close
    the tag early. The escape is ordinary JSON, so what the browser
    parses is unchanged.
    """
    payload = json.dumps(state, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    return f"<script>{_BOOT_GLOBAL}={payload};</script>"


def _inject_boot_state(
    html: str, state: Optional[Dict[str, Any]]
) -> str:
    """Inline boot state ahead of the page's scripts.

    Placed before ``</head>`` so it is defined however the scripts are
    ordered. A page with no state, or no head, is left alone.
    """
    if not state:
        return html
    head_end = html.find("</head>")
    if head_end < 0:
        return html
    return (
        html[:head_end] + _boot_script(state) + html[head_end:]
    )


def _serve_stamped_page(
    filename: str, boot: Optional[Dict[str, Any]] = None
) -> HTMLResponse:
    html = (STATIC_DIR / filename).read_text(encoding="utf-8")
    html = _stamp_asset_versions(html)
    html = _reveal_generation_link(
        html, resident=_active_model_is_serving()
    )
    html = _inject_boot_state(html, boot)
    return HTMLResponse(html, headers=dict(_NO_STORE_HEADERS))


@app.get("/")
async def serve_menu() -> HTMLResponse:
    """Landing page: the model-selection Main Menu."""
    return _serve_stamped_page("menu.html")


def _models_boot_state() -> Dict[str, Any]:
    """The `/api/models` answer minus everything a probe would cost.

    Shaped exactly like the endpoint's payload so the page can feed it
    to the same code, and deliberately missing the VRAM fields, which
    would put an `nvidia-smi` call on every navigation. Those describe
    a hover popover inside a closed dropdown, so they are fetched
    after first paint instead; `buildOptionInfo` already draws the row
    without them. The GPU name stays because it is cached, and because
    whether a CPU/GPU pill is offered is first-paint state.
    """
    return {
        "models": [
            _model_entry(model_id, info)
            for model_id, info in REGISTRY.items()
        ],
        "active": manager.active_id,
        "active_device": manager.active_device,
        "active_tokenizer": dict(manager.active_tokenizer),
        "active_context_length": manager.active_context_length,
        "default": DEFAULT_MODEL,
        "gpu_name": model_manager.gpu_name(),
    }


def _generator_boot_state() -> Dict[str, Any]:
    """Everything the generator used to chain two fetches to learn.

    `/api/ui-state` then `/api/models`, in that order because the
    second callback read what the first wrote, so the page could not
    be correct until both had answered and it rebuilt itself around
    them. Both are cheap to produce here.
    """
    ui_state = load_ui_state(RESULTS_DIR)
    ui_state = _reconcile_new_runs(ui_state)
    ui_state = _reconcile_collections(ui_state)
    return {
        "ui_state": ui_state,
        "models": _models_boot_state(),
    }


@app.get("/generate")
async def serve_generate() -> Response:
    """Generator page, gated behind model selection.

    The Main Menu is the single entry point: reaching the generator
    without an active model (e.g. a direct URL hit) redirects back to
    the menu to choose one, rather than silently booting a default.
    """
    if not _active_model_is_serving():
        return RedirectResponse(url="/", status_code=307)
    boot = await asyncio.to_thread(_generator_boot_state)
    return _serve_stamped_page("index.html", boot=boot)


@app.get("/index.html")
async def serve_index_html() -> RedirectResponse:
    """Back-compat: the generator now lives at ``/generate``."""
    return RedirectResponse(url="/generate", status_code=307)


def _analytics_boot_state() -> Dict[str, Any]:
    """The catalog, the collections and the data root, up front.

    All three are cheap to produce: the catalog reads one metadata
    file per run, around 25ms for 240 of them, and the reconciliation
    the collections need has already walked the same directory. The
    GPU name is deliberately not here even though it is now cached,
    because it is only a fallback for a run that did not record its
    own processor, in a detail view nobody has opened yet.
    """
    runs = list_runs(RESULTS_DIR)
    ui_state = load_ui_state(RESULTS_DIR)
    ui_state = _reconcile_new_runs(ui_state)
    ui_state = _reconcile_collections(ui_state)
    return {
        # Read by persistHydrate, the same as on the generator, and
        # for the same reason: it gates the first render, so fetching
        # it puts a round trip in front of the table.
        "ui_state": ui_state,
        "runs": runs,
        "collections": collection_ops.decode(
            ui_state.get(COLLECTIONS_KEY)
        ),
        "results_dir": _display_run_path(RESULTS_DIR),
    }


@app.get("/analytics.html")
async def serve_analytics_page() -> HTMLResponse:
    boot = await asyncio.to_thread(_analytics_boot_state)
    return _serve_stamped_page("analytics.html", boot=boot)


def _active_model_family() -> Optional[str]:
    """Architecture family of the resident model, or None.

    Family rather than generation shape: the glow settings are keyed
    by model class, so a state-space model wants its own pair even
    though it appends like an autoregressive one.

    Registry data, so this costs no GPU probe. Settings needs it to
    open its glow preview on the right class instead of playing the
    diffusion default and switching a moment later.
    """
    active_id = manager.active_id
    if active_id is None or active_id not in REGISTRY:
        return None
    return REGISTRY[active_id].capabilities.family


@app.get("/settings.html")
async def serve_settings_page() -> HTMLResponse:
    """Shared, model-agnostic Settings page (always available)."""
    return _serve_stamped_page(
        "settings.html",
        boot={"active_model_family": _active_model_family()},
    )


@app.get("/vision.html")
async def serve_vision_page() -> HTMLResponse:
    """The image tokeniser view, always available.

    Not gated on a resident model, unlike `/generate`, because it
    loads no weights: it reads two small config files per
    encoder and does integer arithmetic. So it works with nothing
    loaded, and works without disturbing anything that is.

    The encoder list is inlined as boot state for the same reason the
    other pages inline theirs: it is cheap to produce here and saves
    the page a round trip before its first paint.
    """
    encoders = await asyncio.to_thread(_vision_boot_state)
    return _serve_stamped_page("vision.html", boot=encoders)


def _vision_boot_state() -> Dict[str, Any]:
    """The encoder list, shaped exactly like `/api/vision/encoders`.

    Same shape so the page feeds it to the same code path, which is
    the convention `_models_boot_state` established.
    """
    return {
        "encoders": [
            {
                "id": encoder.id,
                "display_name": encoder.display_name,
                "repo_id": encoder.repo_id,
                "revision": encoder.revision,
                "summary": encoder.summary,
                "cached": vision_is_cached(encoder),
            }
            for encoder in vision_declared()
        ]
    }


class _NoCacheStaticFiles(StaticFiles):
    """Serve static assets with no-store so the browser never holds a
    stale CSS/JS copy between edits (this is a local dev tool).

    Beyond the ``Cache-Control`` header, the validator headers
    (``ETag`` / ``Last-Modified``) are stripped so the browser cannot
    issue a conditional request and be handed a ``304 Not Modified``
    for a stale asset (observed with cached CSS in Firefox).
    """

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        for validator in ("etag", "last-modified"):
            if validator in response.headers:
                del response.headers[validator]
        return response


app.mount(
    "/",
    _NoCacheStaticFiles(directory=str(STATIC_DIR), html=True),
    name="static",
)
