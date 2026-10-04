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
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

import websockets
from fastapi import (
    BackgroundTasks,
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
from pydantic import BaseModel, ConfigDict

from src.analytics.metrics import list_runs
from src.backends.protocol import (
    ERROR_NO_MODEL_ACTIVE,
    ERROR_SCOPE_FATAL,
    ERROR_WORKER_UNREACHABLE,
    ModelInfo,
    is_hub_checkpoint,
    wire_error,
)
from src.backends.registry import (
    DEFAULT_MODEL,
    REGISTRY,
)
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
from src.web import analytics_api
from src.web import collections as collection_ops
from src.web import model_manager
from src.web import run_store
from src.web import save_pipeline
from src.web.save_limits import BodyLimit
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


def _current_results_dir() -> Path:
    """Read the live root so tests and embeddings can replace it."""
    return RESULTS_DIR


def _current_gpu_name() -> Optional[str]:
    """Probe through the module so replacements remain visible."""
    return model_manager.gpu_name()


app.include_router(
    analytics_api.create_analytics_router(
        analytics_api.AnalyticsApiDependencies(
            results_dir=_current_results_dir,
            repo_root=REPO_ROOT,
            gpu_name=_current_gpu_name,
        )
    )
)


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


def _current_save_model_facts() -> save_pipeline.CurrentModelFacts:
    """Snapshot resident facts used by legacy save requests."""
    return save_pipeline.CurrentModelFacts(
        device=manager.active_device,
        versions=dict(manager.active_versions),
        tokenizer=dict(manager.active_tokenizer),
        context_length=manager.active_context_length,
    )


def _save_pipeline_context() -> save_pipeline.SavePipelineContext:
    """Bind supervisor-owned dependencies to the save pipeline."""
    return save_pipeline.SavePipelineContext(
        results_dir=RESULTS_DIR,
        repo_root=REPO_ROOT,
        current_model_facts=_current_save_model_facts,
        gpu_name=model_manager.gpu_name,
        cpu_name=model_manager.cpu_name,
        git_commit=_git_commit,
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
async def save_run(
    body: save_pipeline.SaveRunRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Publish a run, answer, and only then draw its preview.

    The preview is a derivative of a run already published, and
    drawing a long one takes seconds that the page should not wait
    through to be told its run is safe. A background task draws it
    after the response is sent.
    """
    try:
        saved, preview = await save_pipeline.publish_run(
            body, _save_pipeline_context()
        )
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
    save_pipeline.schedule_preview(preview, background_tasks.add_task)
    logger.info("saved run to %s", saved["path"])
    return JSONResponse(content={"success": True, **saved})


# -- Vision endpoints --


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
        "results_dir": run_store.display_path(
            RESULTS_DIR, REPO_ROOT
        ),
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
