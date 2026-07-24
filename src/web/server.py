"""Supervisor server for the multi-model diffusion visualizer.

Responsibilities:
  - Serve the shared frontend and the analytics API.
  - Manage model worker subprocesses (one active at a time),
    each running in its own venv so incompatible dependency
    stacks (e.g. Transformers 4.38.2 vs v5) never collide.
  - Proxy the browser WebSocket to the active worker's /ws.

The supervisor itself never imports torch or transformers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import websockets
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analytics.metrics import (
    canvas_boundaries,
    compute_convergence,
    list_runs,
    load_run_frames,
    load_run_metadata,
    parse_history,
)
from src.backends.protocol import ModelInfo
from src.backends.registry import DEFAULT_MODEL, REGISTRY
from src.inference.render_gif import history_to_gif

logger = logging.getLogger("diffusion_supervisor")

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path("Results")

WORKER_START_TIMEOUT_S = 180.0
WORKER_STOP_TIMEOUT_S = 30.0
# Grace period for a stopped worker's VRAM to be reclaimed
# before the pre-flight check refuses the next activation.
VRAM_SETTLE_TIMEOUT_S = 8.0


# -- Model worker manager --


def _gpu_name() -> Optional[str]:
    """Best-effort GPU name via nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if lines:
                return lines[0].strip()
    except Exception:
        return None
    return None


def _free_vram_gib() -> Optional[float]:
    """Free GPU memory in GiB via nvidia-smi (None if unknown)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if lines:
                return float(lines[0].strip()) / 1024.0
    except Exception:
        return None
    return None


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


def _venv_cuda_lib_dirs(python_path: Path) -> List[str]:
    """Bundled CUDA lib dirs for a venv (for bitsandbytes etc.).

    ``<venv>/lib/pythonX.Y/site-packages/nvidia/*/lib`` — native
    extensions like bitsandbytes need these on LD_LIBRARY_PATH,
    since the dynamic linker resolves them at process start.
    """
    venv_root = python_path.parent.parent
    lib_root = venv_root / "lib"
    if not lib_root.is_dir():
        return []
    dirs: List[str] = []
    for site in lib_root.glob("python*/site-packages/nvidia"):
        for lib in sorted(site.glob("*/lib")):
            if lib.is_dir():
                dirs.append(str(lib))
    return dirs


class ModelManager:
    """Spawns/stops one model worker subprocess at a time.

    Only one worker is ever alive, since a single ~15-16 GB model
    already saturates the 24 GB GPU.
    """

    def __init__(self) -> None:
        self.active_id: Optional[str] = None
        self.active_versions: Dict[str, str] = {}
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _free_port() -> int:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
        )

    def status(self, model_id: str) -> str:
        if self.active_id == model_id and self._alive():
            return "active"
        return "inactive"

    def ws_url(self) -> str:
        assert self._port is not None
        return f"ws://127.0.0.1:{self._port}/ws"

    async def activate(self, model_id: str) -> None:
        if model_id not in REGISTRY:
            raise KeyError(model_id)
        async with self._lock:
            if self.active_id == model_id and self._alive():
                return
            await self._stop_locked()
            info = REGISTRY[model_id]
            python = REPO_ROOT / info.venv_python
            if not python.exists():
                raise RuntimeError(
                    f"venv python not found: {python}"
                )
            await self._preflight_vram(info)
            port = self._free_port()
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            lib_dirs = _venv_cuda_lib_dirs(python)
            if lib_dirs:
                existing = env.get("LD_LIBRARY_PATH", "")
                parts = lib_dirs + (
                    [existing] if existing else []
                )
                env["LD_LIBRARY_PATH"] = ":".join(parts)
            logger.info(
                "spawning worker %s on port %d",
                model_id,
                port,
            )
            proc = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "src.backends.run_worker",
                    "--model",
                    model_id,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=str(REPO_ROOT),
                env=env,
            )
            self._proc = proc
            self._port = port
            self.active_id = model_id
            try:
                await self._await_health(port, proc)
            except Exception:
                await self._stop_locked()
                raise

    async def _preflight_vram(self, info: ModelInfo) -> None:
        """Refuse activation if the model cannot fit in VRAM.

        Runs after the previous worker is stopped, so it briefly
        waits for that VRAM to be reclaimed before deciding.
        """
        required = info.min_vram_gib
        if required <= 0:
            return
        deadline = time.monotonic() + VRAM_SETTLE_TIMEOUT_S
        free = _free_vram_gib()
        while (
            free is not None
            and free < required
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.5)
            free = _free_vram_gib()
        if free is None:
            logger.warning(
                "free VRAM unreadable; skipping pre-flight"
                " check for %s",
                info.id,
            )
            return
        if free < required:
            raise RuntimeError(
                f"Not enough free GPU memory to load"
                f" {info.display_name}: needs about"
                f" {required:.0f} GiB but only {free:.1f} GiB"
                f" is free. Close other GPU processes and"
                f" try again."
            )

    async def _await_health(
        self, port: int, proc: subprocess.Popen
    ) -> None:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + WORKER_START_TIMEOUT_S
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        "worker exited during startup"
                        f" (code {proc.returncode})"
                    )
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        body = resp.json()
                        status = body.get("status")
                        if status == "error":
                            raise RuntimeError(
                                body.get(
                                    "message",
                                    "model failed to load",
                                )
                            )
                        if status == "ready":
                            self.active_versions = body.get(
                                "versions", {}
                            )
                            return
                except RuntimeError:
                    raise
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeError("worker health check timed out")

    async def ensure_default(self) -> None:
        if not self._alive():
            await self.activate(DEFAULT_MODEL)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self._proc is not None and self._alive():
            logger.info("stopping worker %s", self.active_id)
            self._proc.terminate()
            try:
                await asyncio.to_thread(
                    self._proc.wait, WORKER_STOP_TIMEOUT_S
                )
            except Exception:
                self._proc.kill()
        self._proc = None
        self._port = None
        self.active_id = None
        self.active_versions = {}


manager = ModelManager()
app = FastAPI(title="Diffusion LLM Visualizer")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await manager.stop()


# -- Model API --


@app.get("/api/models")
async def list_models() -> JSONResponse:
    models: List[Dict[str, Any]] = []
    for model_id, info in REGISTRY.items():
        data = info.model_dump()
        data.pop("worker_module", None)
        data.pop("venv_python", None)
        data["status"] = manager.status(model_id)
        models.append(data)
    return JSONResponse(
        {
            "models": models,
            "active": manager.active_id,
            "default": DEFAULT_MODEL,
        }
    )


@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str) -> JSONResponse:
    try:
        await manager.activate(model_id)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": f"unknown model: {model_id}",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("activation failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": str(exc)},
        )
    return JSONResponse(
        {"ok": True, "active": manager.active_id}
    )


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


@app.websocket("/ws")
async def websocket_proxy(browser: WebSocket) -> None:
    await browser.accept()
    try:
        await manager.ensure_default()
    except Exception as exc:  # noqa: BLE001
        logger.exception("could not start default worker")
        await browser.send_json(
            {
                "type": "error",
                "message": f"model start failed: {exc}",
            }
        )
        await browser.close()
        return

    url = manager.ws_url()
    try:
        async with websockets.connect(
            url, max_size=None
        ) as worker:
            await _pipe(browser, worker)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("proxy error")
        try:
            await browser.send_json(
                {"type": "error", "message": str(exc)}
            )
        except Exception:
            pass


# -- Save endpoint (model-agnostic) --


class RemaskEdit(BaseModel):
    frame_index: int
    token_positions: List[int]


class TokenRecord(BaseModel):
    """One persisted per-token record for durable overlays.

    Mirrors the live protocol shape ``{t, m, id, c?}``: ``t`` is the
    display text, ``m`` marks an unresolved position, ``id`` is the
    vocab id, and ``c`` is the reveal confidence (absent for masked
    positions).
    """

    t: str
    m: bool
    id: int
    c: Optional[float] = None


# Per-frame, per-token stream. A frame may be ``None`` when a model
# emitted no token detail for it.
FrameTokens = List[Optional[List[TokenRecord]]]


class SaveRunRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    params: Dict[str, Any] = Field(default_factory=dict)
    frames: List[str] = Field(min_length=1)
    final_text: str
    elapsed_seconds: Optional[float] = None
    per_frame_elapsed: Optional[List[float]] = None
    # Durable per-token records for the commit-order / diff /
    # confidence overlays. ``frame_tokens`` is the primary (possibly
    # edited) run; ``original_frame_tokens`` is the pre-edit snapshot,
    # sent only for edited runs so the counterfactual diff survives.
    frame_tokens: Optional[FrameTokens] = None
    original_frame_tokens: Optional[FrameTokens] = None
    canvas_index: Optional[List[int]] = None
    mean_conf: Optional[List[Optional[float]]] = None
    remask_edits: Optional[List[RemaskEdit]] = None


def _make_run_dir(base: Path, model_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = model_id.replace("/", "_")
    run_dir = base / f"{timestamp}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def _save_run_blocking(body: SaveRunRequest) -> str:
    model_id = body.model or DEFAULT_MODEL
    checkpoint = ""
    if model_id in REGISTRY:
        checkpoint = REGISTRY[model_id].checkpoint
    run_dir = _make_run_dir(RESULTS_DIR, model_id)

    metadata: Dict[str, Any] = {
        "backend": model_id,
        "model": checkpoint or model_id,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prompt": body.prompt,
        "final_text": body.final_text,
        "params": body.params,
    }
    if body.elapsed_seconds is not None:
        metadata["elapsed_seconds"] = body.elapsed_seconds
    if body.per_frame_elapsed is not None:
        metadata["per_frame_elapsed"] = body.per_frame_elapsed
    if body.canvas_index is not None:
        metadata["canvas_index"] = body.canvas_index
    if body.mean_conf is not None:
        metadata["mean_conf"] = body.mean_conf
    if body.remask_edits:
        metadata["remask_edits"] = [
            edit.model_dump() for edit in body.remask_edits
        ]
    metadata["reproducibility"] = {
        "seed": body.params.get("seed"),
        "gpu": _gpu_name(),
        "git_commit": _git_commit(),
        "versions": dict(manager.active_versions),
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "final.txt").write_text(
        body.final_text, encoding="utf-8"
    )
    with (run_dir / "history.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        for index, frame_text in enumerate(body.frames):
            handle.write(f"\n===== FRAME {index} =====\n")
            handle.write(frame_text)
            handle.write("\n")

    if body.frame_tokens is not None:
        (run_dir / "tokens.json").write_text(
            json.dumps(
                _dump_frame_tokens(body.frame_tokens),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if body.original_frame_tokens is not None:
        (run_dir / "original_tokens.json").write_text(
            json.dumps(
                _dump_frame_tokens(
                    body.original_frame_tokens
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    history_to_gif(
        body.frames,
        run_dir / "diffusion.gif",
        header_text=body.prompt,
    )
    return str(run_dir)


@app.post("/api/save")
async def save_run(body: SaveRunRequest) -> JSONResponse:
    try:
        run_path = await asyncio.to_thread(
            _save_run_blocking, body
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to save run")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    logger.info("saved run to %s", run_path)
    return JSONResponse(
        content={"success": True, "path": run_path}
    )


# -- Analytics endpoints --


@app.get("/api/analytics/runs")
async def analytics_list_runs() -> JSONResponse:
    runs = await asyncio.to_thread(list_runs, RESULTS_DIR)
    return JSONResponse(content=runs)


def _compute_run_metrics(run_id: str) -> Dict[str, Any]:
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    history_path = run_dir / "history.txt"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"history.txt missing for run {run_id}"
        )

    meta = load_run_metadata(run_dir)
    frames = parse_history(history_path)
    convergence = compute_convergence(frames)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "convergence": convergence,
        "total_frames": len(frames),
    }
    for key in (
        "per_frame_elapsed",
        "elapsed_seconds",
        "remask_edits",
        "mean_conf",
    ):
        if key in meta:
            result[key] = meta[key]
    canvas_index = meta.get("canvas_index")
    if canvas_index:
        result["canvas_boundaries"] = canvas_boundaries(
            canvas_index
        )
    return result


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
    return JSONResponse(content=result)


def _compute_run_frames(run_id: str) -> Dict[str, Any]:
    """Load durable token streams for the overlay viewer.

    Kept separate from ``_compute_run_metrics`` because token streams
    are large; the analytics UI fetches this only when a run's overlay
    viewer opens. Guards against path traversal like the delete path.
    """
    results_root = RESULTS_DIR.resolve()
    run_dir = (results_root / run_id).resolve()
    if run_dir.parent != results_root:
        raise ValueError(f"invalid run id: {run_id}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")

    meta = load_run_metadata(run_dir)
    data = load_run_frames(run_dir)
    return {
        "run_id": run_id,
        "frames": data["frames"],
        "original_frames": data["original_frames"],
        "records_available": data["records_available"],
        "remask_edits": meta.get("remask_edits", []),
        "canvas_index": meta.get("canvas_index"),
    }


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
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"error": str(exc)}
        )
    return JSONResponse(content=result)


@app.get("/api/analytics/compare")
async def analytics_compare(ids: str = "") -> JSONResponse:
    run_ids = [
        rid.strip() for rid in ids.split(",") if rid.strip()
    ]
    if len(run_ids) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "ids parameter is required"},
        )
    results: List[Dict[str, Any]] = []
    for run_id in run_ids:
        try:
            metrics = await asyncio.to_thread(
                _compute_run_metrics, run_id
            )
            results.append(metrics)
        except FileNotFoundError:
            results.append(
                {
                    "run_id": run_id,
                    "error": f"Run not found: {run_id}",
                }
            )
    return JSONResponse(content=results)


@app.get("/api/analytics/system")
async def analytics_system_info() -> JSONResponse:
    """GPU name for the analytics UI (no torch in supervisor)."""
    return JSONResponse(content={"gpu_name": _gpu_name()})


def _delete_run_blocking(run_id: str) -> None:
    """Delete one saved run directory under Results/.

    Guards against path traversal: the resolved directory must be a
    direct child of RESULTS_DIR.
    """
    results_root = RESULTS_DIR.resolve()
    run_dir = (results_root / run_id).resolve()
    if run_dir.parent != results_root:
        raise ValueError(f"invalid run id: {run_id}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    shutil.rmtree(run_dir)


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
    re-fetches a file exactly when it changes -- automatic, per-file
    cache-busting with no manual version bumping and no reliance on the
    browser honoring ``no-store``.
    """

    def _replace(match: "re.Match[str]") -> str:
        path = match.group("path")
        asset = STATIC_DIR / path.lstrip("/")
        try:
            version = asset.stat().st_mtime_ns
        except OSError:
            return match.group(0)  # Unknown file: leave the ref as-is.
        return f'{match.group("attr")}="{path}?v={version}"'

    return _ASSET_REF_RE.sub(_replace, html)


def _serve_stamped_page(filename: str) -> HTMLResponse:
    html = (STATIC_DIR / filename).read_text(encoding="utf-8")
    return HTMLResponse(
        _stamp_asset_versions(html),
        headers=dict(_NO_STORE_HEADERS),
    )


@app.get("/")
async def serve_index() -> HTMLResponse:
    return _serve_stamped_page("index.html")


@app.get("/index.html")
async def serve_index_html() -> HTMLResponse:
    return _serve_stamped_page("index.html")


@app.get("/analytics.html")
async def serve_analytics_page() -> HTMLResponse:
    return _serve_stamped_page("analytics.html")


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
