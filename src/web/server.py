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
import ctypes
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import websockets
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analytics.metrics import (
    canvas_boundaries,
    compute_convergence,
    list_runs,
    load_run_frames,
    load_run_metadata,
    parse_history,
    total_elapsed_seconds,
)
from src.backends.protocol import ModelInfo
from src.backends.registry import DEFAULT_MODEL, REGISTRY
from src.inference.render_gif import history_to_gif
from src.web.data_root import (
    RESULTS_DIR_ENV,
    resolve_results_dir,
)
from src.web.ui_state import load_ui_state, set_ui_state_key

# Disable the Xet download client before the first huggingface_hub
# import. Here huggingface_hub is imported lazily (in _is_downloaded /
# the download task), so setting the flag now, at module load, still
# precedes it. The flag is cached in hf constants at import time, so
# setting it any later is a no-op; Xet bypasses our tqdm progress hook,
# whereas the classic downloader routes through it, so the menu's
# download bar fills smoothly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger("diffusion_supervisor")

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Resolved once, here, rather than inherited from wherever the
# process happened to be started (see src/web/data_root.py).
RESULTS_DIR = resolve_results_dir(
    os.environ.get(RESULTS_DIR_ENV), repo_root=REPO_ROOT
)

WORKER_START_TIMEOUT_S = 180.0
WORKER_STOP_TIMEOUT_S = 30.0
# How often the startup monitor reads the worker's /health. Two
# cadences, because the two halves of startup want different things.
# Before the worker answers at all, every poll is a refused connection
# during its torch import, so there is nothing to gain by hurrying.
# Once it is reporting progress, this is the client's only source of
# it, and the browser polls on top of this: a slow read here plus a
# slow read there is what left a short load looking like it stopped
# part way.
WORKER_HEALTH_POLL_S = 0.5
WORKER_PROGRESS_POLL_S = 0.25
# Grace period for a stopped worker's VRAM to be reclaimed
# before the pre-flight check refuses the next activation.
VRAM_SETTLE_TIMEOUT_S = 8.0


# -- Model worker manager --


# nvidia-smi is often absent from PATH when the app is launched from a
# desktop entry (a minimal session PATH), which silently made GPU info
# unavailable. Resolve it explicitly with common fallbacks, cached, and
# log the outcome once so a missing binary is diagnosable.
_NVIDIA_SMI_FALLBACKS = (
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
)
_nvidia_smi_resolved = False
_nvidia_smi_path_cached: Optional[str] = None


def _nvidia_smi_path() -> Optional[str]:
    """Resolve the nvidia-smi binary (PATH, then common paths). Cached."""
    global _nvidia_smi_resolved, _nvidia_smi_path_cached
    if _nvidia_smi_resolved:
        return _nvidia_smi_path_cached
    found = shutil.which("nvidia-smi")
    if not found:
        for candidate in _NVIDIA_SMI_FALLBACKS:
            if Path(candidate).is_file():
                found = candidate
                break
    _nvidia_smi_path_cached = found
    _nvidia_smi_resolved = True
    if found is None:
        logger.warning(
            "nvidia-smi not found on PATH or common paths"
            " (%s); GPU info will be unavailable",
            ", ".join(_NVIDIA_SMI_FALLBACKS),
        )
    else:
        logger.info("using nvidia-smi at %s", found)
    return found


def _nvidia_smi_query(field: str) -> Optional[str]:
    """Return one --query-gpu field's first-GPU value, or None.

    Failures are logged (not swallowed) so a broken GPU probe does not
    silently masquerade as "no GPU".
    """
    binary = _nvidia_smi_path()
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [
                binary,
                "--query-gpu=" + field,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort GPU probe
        logger.warning("nvidia-smi query failed: %s", exc)
        return None
    if out.returncode != 0:
        logger.warning(
            "nvidia-smi exited %d: %s",
            out.returncode,
            out.stderr.strip(),
        )
        return None
    lines = out.stdout.strip().splitlines()
    if not lines:
        return None
    return lines[0].strip()


def _gpu_name() -> Optional[str]:
    """Best-effort GPU name via nvidia-smi."""
    return _nvidia_smi_query("name")


def _free_vram_gib() -> Optional[float]:
    """Free GPU memory in GiB via nvidia-smi (None if unknown)."""
    raw = _nvidia_smi_query("memory.free")
    if raw is None:
        return None
    try:
        return float(raw) / 1024.0
    except ValueError:
        return None


def _gpu_status() -> str:
    """Classify GPU availability for a clearer Main Menu message.

    Returns one of: "ok", "no_nvidia_smi", "mismatch" (driver/library
    version mismatch, e.g. after an NVIDIA update pending a reboot), or
    "error". Only called when the GPU name is unreadable, to explain why.
    """
    binary = _nvidia_smi_path()
    if binary is None:
        return "no_nvidia_smi"
    try:
        out = subprocess.run(
            [binary, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort GPU probe
        logger.warning("nvidia-smi status probe failed: %s", exc)
        return "error"
    if out.returncode == 0:
        return "ok"
    combined = (out.stderr + " " + out.stdout).lower()
    if "mismatch" in combined or "nvml" in combined:
        return "mismatch"
    return "error"


def _cpu_name() -> Optional[str]:
    """Best-effort CPU model name (Linux /proc/cpuinfo, then platform).

    Returned to the Main Menu so a GPU-less user can see what will run
    the CPU-capable models. Optional, mirroring the GPU probes.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            name = value.strip()
            if name:
                return name
    fallback = platform.processor() or platform.machine()
    return fallback or None


def _free_ram_gib() -> Optional[float]:
    """Available system RAM in GiB (Linux /proc/meminfo), or None."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        # Format: "MemAvailable:   12345678 kB".
        if len(parts) < 2:
            return None
        try:
            kib = float(parts[1])
        except ValueError:
            return None
        return kib / (1024.0 * 1024.0)
    return None


_WORKER_CMD_MARKER = "src.backends.run_worker"


def _proc_ppid(pid_dir: Path) -> Optional[int]:
    """Parent PID for a /proc entry, or None if unreadable."""
    try:
        status = (pid_dir / "status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _sweep_orphan_workers() -> None:
    """Terminate leftover worker processes orphaned by a prior crash.

    A worker whose supervisor died is reparented to init (ppid 1) yet
    may still hold VRAM (the PDEATHSIG guard covers most cases, but not
    e.g. a supervisor that predates it). We match our worker command
    line and terminate only orphans (ppid == 1), never a worker still
    owned by a live supervisor, so a browser and desktop instance can
    coexist. Best-effort and Linux-only (/proc); a no-op elsewhere.
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode(
            "utf-8", "replace"
        )
        if _WORKER_CMD_MARKER not in cmdline:
            continue
        if _proc_ppid(entry) != 1:
            continue  # still owned by a live supervisor
        try:
            os.kill(int(entry.name), signal.SIGTERM)
            logger.warning(
                "swept orphaned worker pid %s", entry.name
            )
        except OSError:  # noqa: PERF203 - best-effort
            pass


def _is_repo_checkpoint(checkpoint: str) -> bool:
    """True when the checkpoint is an HF repo id, not a local path.

    Repo-id checkpoints (e.g. ``org/name``) download from the Hub;
    local paths (``~/models/...``) are produced offline (e.g. the
    DiffusionGemma quantize script) and are not UI-downloadable.
    """
    value = checkpoint.strip()
    if not value or value.startswith(("~", "/", ".")):
        return False
    return value.count("/") == 1


def _is_downloaded(checkpoint: str) -> bool:
    """Whether the checkpoint's files are fully present locally.

    A partial cache (an interrupted download leaving ``*.incomplete``
    parts) counts as not-downloaded so the menu keeps the "Click to
    Download" veneer and a re-click resumes, rather than the model being
    marked ready and hanging on load.
    """
    if _is_repo_checkpoint(checkpoint):
        try:
            from src.inference.hf_download import is_repo_cached

            return is_repo_cached(checkpoint)
        except Exception:  # noqa: BLE001 - probe failure: treat as not cached
            return False
    return Path(checkpoint).expanduser().is_dir()


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

    ``<venv>/lib/pythonX.Y/site-packages/nvidia/*/lib``: native
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


def _set_pdeathsig() -> None:
    """Child-side: ask the kernel to SIGTERM this worker if the
    supervisor dies (Linux ``PR_SET_PDEATHSIG``).

    Belt-and-suspenders against orphaned workers holding VRAM: even if
    the supervisor is hard-killed (e.g. the desktop window closes mid
    load before the graceful stop can run), the worker is signalled.
    Best-effort and Linux-only; a failure here must not block spawn.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except Exception:  # noqa: BLE001 - best-effort orphan guard
        pass


def _worker_popen_kwargs() -> Dict[str, Any]:
    """Extra ``Popen`` kwargs to keep workers from being orphaned.

    On Linux, put the worker in its own session and arm PDEATHSIG.
    Elsewhere, return nothing (``preexec_fn`` is POSIX-only and the
    app targets Linux).
    """
    if sys.platform.startswith("linux"):
        return {
            "start_new_session": True,
            "preexec_fn": _set_pdeathsig,
        }
    return {}


class ModelManager:
    """Spawns/stops one model worker subprocess at a time.

    Only one worker is ever alive, since a single ~15-16 GB model
    already saturates the 24 GB GPU.
    """

    def __init__(self) -> None:
        self.active_id: Optional[str] = None
        self.active_device: Optional[str] = None
        self.active_versions: Dict[str, str] = {}
        # Identity of the resident model's tokenizer, reported by the
        # worker off the loaded object (see describe_tokenizer). Kept
        # beside the versions because it is the same kind of fact: a
        # property of what is loaded right now, not of the registry.
        self.active_tokenizer: Dict[str, Any] = {}
        # How many tokens the resident checkpoint can attend to, or
        # None when it could not be read (see
        # describe_context_length). None rather than a default, so the
        # prompt readout can say nothing instead of quoting a ceiling
        # nobody measured.
        self.active_context_length: Optional[int] = None
        # Activation is non-blocking: activate() spawns the worker and
        # returns; a background monitor task tracks these until ready,
        # and the client polls them. States: idle | starting |
        # downloading | loading | ready | error.
        self.load_state: str = "idle"
        self.load_progress: Optional[Dict[str, Any]] = None
        self.load_error: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # Download-only state (pre-fetch weights without loading into
        # VRAM). Independent of the worker, so it can run alongside a
        # resident model. States: idle | downloading | done | error.
        self.download_state: str = "idle"
        self.download_target: Optional[str] = None
        self.download_progress: Optional[Dict[str, Any]] = None
        self.download_error: Optional[str] = None
        # Held so the fire-and-forget download task is not GC'd mid-run.
        self._download_task: Optional[asyncio.Task] = None

    @staticmethod
    def _free_port() -> int:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Normalize the requested device to "cuda" or "cpu".

        A None request (body-less activate, e.g. the generator's
        in-header model switch) auto-selects the GPU when one is
        detected and CPU otherwise, so a GPU-less host still works.
        """
        if device is None:
            return "cuda" if _gpu_name() is not None else "cpu"
        if device not in ("cuda", "cpu"):
            raise ValueError(
                f"invalid device: {device!r}"
                " (expected 'cuda' or 'cpu')"
            )
        return device

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

    async def activate(
        self, model_id: str, *, device: Optional[str] = None
    ) -> None:
        """Spawn the worker and return immediately (non-blocking).

        A background monitor task then tracks startup (download /
        load / ready / error), which the client polls via
        ``/api/models/activation``. Keeping the load off the lock lets
        ``stop`` / ``cancel_activation`` terminate a still-loading
        worker instead of deadlocking behind a held lock.
        """
        if model_id not in REGISTRY:
            raise KeyError(model_id)
        device = self._resolve_device(device)
        async with self._lock:
            if (
                self.active_id == model_id
                and self.active_device == device
                and self._alive()
            ):
                return
            await self._stop_locked()
            info = REGISTRY[model_id]
            python = REPO_ROOT / info.venv_python
            if not python.exists():
                raise RuntimeError(
                    f"venv python not found: {python}"
                )
            # CPU placement has no VRAM cost, so skip the GPU
            # pre-flight (which would otherwise block on nvidia-smi).
            if device != "cpu":
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
                "spawning worker %s on port %d (device=%s)",
                model_id,
                port,
                device,
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
                    "--device",
                    device,
                ],
                cwd=str(REPO_ROOT),
                env=env,
                **_worker_popen_kwargs(),
            )
            self._proc = proc
            self._port = port
            self.active_id = model_id
            self.active_device = device
            self.active_versions = {}
            self.active_tokenizer = {}
            self.active_context_length = None
            self.load_state = "starting"
            self.load_progress = None
            self.load_error = None
            self._monitor_task = asyncio.create_task(
                self._monitor_startup(proc, port)
            )

    async def _monitor_startup(
        self, proc: subprocess.Popen, port: int
    ) -> None:
        """Poll the worker's /health until ready/error/exit.

        Updates ``load_state`` / ``load_progress`` so the client poll
        reflects downloading vs loading, and caches versions on ready.
        The startup deadline only guards reaching the first response;
        once the worker is answering (loading/downloading), there is no
        wall-clock cap so long first-time downloads are not cut off
        (the user can cancel instead).
        """
        url = f"http://127.0.0.1:{port}/health"
        startup_deadline = (
            time.monotonic() + WORKER_START_TIMEOUT_S
        )
        responded = False
        async with httpx.AsyncClient() as client:
            while True:
                if proc.poll() is not None:
                    self.load_state = "error"
                    self.load_error = (
                        "worker exited during startup"
                        f" (code {proc.returncode})"
                    )
                    return
                if (
                    not responded
                    and time.monotonic() > startup_deadline
                ):
                    self.load_state = "error"
                    self.load_error = (
                        "worker did not start in time"
                    )
                    return
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        responded = True
                        if self._apply_health(resp.json()):
                            return
                except Exception:  # noqa: BLE001 - worker still coming up
                    pass
                await asyncio.sleep(
                    WORKER_PROGRESS_POLL_S
                    if responded
                    else WORKER_HEALTH_POLL_S
                )

    def _apply_health(self, body: Dict[str, Any]) -> bool:
        """Fold one /health body into load state. True when terminal."""
        status = body.get("status")
        if status == "error":
            self.load_state = "error"
            self.load_error = body.get(
                "message", "model failed to load"
            )
            return True
        if status == "ready":
            self.active_versions = body.get("versions", {})
            self.active_tokenizer = body.get("tokenizer", {})
            self.active_context_length = _read_context_length(body)
            self.load_progress = None
            self.load_state = "ready"
            return True
        if status == "downloading":
            self.load_state = "downloading"
            self.load_progress = body.get("progress")
        else:
            # A load carries progress too, once the weights start
            # arriving. Before that the worker sends none and this
            # falls back to None, which the client shows as an
            # indeterminate spinner.
            self.load_state = "loading"
            self.load_progress = body.get("progress")
        return False

    async def cancel_activation(self) -> None:
        """Cancel an in-flight activation and free the worker/VRAM.

        Safe to call anytime: it stops the current worker (and its
        monitor). The lock is free during load, so this never
        deadlocks against ``activate``.
        """
        async with self._lock:
            await self._stop_locked()

    # -- download-only (pre-fetch weights, no VRAM) --

    def start_download(self, model_id: str) -> None:
        """Begin downloading a model's weights without loading them.

        Runs in a background task so a resident model keeps serving.
        Raises for an unknown / non-downloadable model, or if a
        download is already running.
        """
        if model_id not in REGISTRY:
            raise KeyError(model_id)
        checkpoint = REGISTRY[model_id].checkpoint
        if not _is_repo_checkpoint(checkpoint):
            raise ValueError(
                f"{model_id} is not downloadable from the Hub"
            )
        if self.download_state == "downloading":
            raise RuntimeError("a download is already running")
        self.download_target = model_id
        self.download_state = "downloading"
        self.download_progress = None
        self.download_error = None
        self._download_task = asyncio.create_task(
            self._run_download(model_id, checkpoint)
        )

    async def _run_download(
        self, model_id: str, checkpoint: str
    ) -> None:
        from src.inference.hf_download import (
            download_with_progress,
        )

        def _sink(progress: Dict[str, Any]) -> None:
            self.download_progress = progress

        try:
            await asyncio.to_thread(
                download_with_progress,
                checkpoint,
                sink=_sink,
            )
        except Exception as exc:  # noqa: BLE001
            self.download_state = "error"
            self.download_error = str(exc)
            logger.exception("download failed for %s", model_id)
            return
        self.download_progress = None
        self.download_state = "done"

    def ack_download(self) -> None:
        """Clear a finished pre-fetch so its completion notice fires once.

        Resets only a terminal state (done/error) back to idle; a no-op
        while a download is still running. Called when the user
        acknowledges the veneer's "Ok" (or dismisses the toast), so the
        cross-page download toast and re-attach do not keep re-firing.
        """
        if self.download_state in ("done", "error"):
            self.download_state = "idle"
            self.download_target = None
            self.download_progress = None
            self.download_error = None

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

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._monitor_task = None
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
        self.active_device = None
        self.active_versions = {}
        self.active_tokenizer = {}
        self.active_context_length = None
        self.load_state = "idle"
        self.load_progress = None
        self.load_error = None


def _read_context_length(
    body: Dict[str, Any],
) -> Optional[int]:
    """The context length from a ready /health body, if it sent one.

    Validated here rather than trusted, because the worker is a
    separate process on its own transformers version: this is a
    boundary, and a malformed value should degrade to "unknown" the
    way a missing key does rather than reach the UI as a ceiling.
    """
    value = body.get("context_length")
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 1:
        return None
    return value


manager = ModelManager()
app = FastAPI(title="Diffusion LLM Visualizer")


@app.on_event("startup")
async def _startup() -> None:
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
    await asyncio.to_thread(_sweep_orphan_workers)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await manager.stop()


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


def _models_snapshot() -> Dict[str, Any]:
    """Registry plus live GPU/VRAM info for the Main Menu.

    Runs the blocking ``nvidia-smi`` probes here so the endpoint can
    offload it to a thread and keep the event loop responsive.
    """
    free_vram_gib = _free_vram_gib()
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
        data = info.model_dump()
        data.pop("worker_module", None)
        data.pop("venv_python", None)
        status = manager.status(model_id)
        headroom = _model_headroom_gib(
            info,
            free_vram_gib=free_vram_gib,
            resident_reclaimable_gib=resident_reclaimable_gib,
        )
        data["status"] = status
        data["vram_headroom_gib"] = headroom
        data["fits"] = _model_fits(
            info, status=status, headroom_gib=headroom
        )
        data["downloadable"] = _is_repo_checkpoint(
            info.checkpoint
        )
        data["downloaded"] = _is_downloaded(info.checkpoint)
        models.append(data)
    gpu = _gpu_name()
    # Only classify the failure reason when the name is unreadable, so a
    # healthy system pays no extra nvidia-smi call.
    gpu_status = "ok" if gpu is not None else _gpu_status()
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
        "cpu_name": _cpu_name(),
        "free_ram_gib": _free_ram_gib(),
    }


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
        await manager.activate(model_id, device=device)
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("activation failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": str(exc)},
        )
    # Non-blocking: the worker is spawned and loading in the
    # background. The client polls /api/models/activation for progress.
    return JSONResponse(
        {
            "ok": True,
            "active": manager.active_id,
            "state": manager.load_state,
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
        }
    )


@app.post("/api/models/activate/cancel")
async def cancel_activation() -> JSONResponse:
    """Cancel an in-flight load, stopping the worker and freeing VRAM."""
    await manager.cancel_activation()
    return JSONResponse({"ok": True})


@app.post("/api/models/{model_id}/download")
async def download_model(model_id: str) -> JSONResponse:
    """Pre-fetch a model's weights (no VRAM). Client polls status."""
    try:
        manager.start_download(model_id)
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
        {"ok": True, "state": manager.download_state}
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
        }
    )


@app.post("/api/models/download/ack")
async def ack_download() -> JSONResponse:
    """Clear a finished pre-fetch (done/error -> idle) so the completion
    toast and menu re-attach fire exactly once."""
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


@app.websocket("/ws")
async def websocket_proxy(browser: WebSocket) -> None:
    await browser.accept()
    active_id = manager.active_id
    if active_id is None or manager.status(active_id) != "active":
        # Model selection happens on the Main Menu; the generator
        # never auto-boots a worker. Tell the client to go back.
        await browser.send_json(
            {
                "type": "error",
                "message": (
                    "No model is active. Return to the menu"
                    " to select one."
                ),
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

    Mirrors the live protocol shape ``{t, m, id, c?, e?}``: ``t`` is
    the display text, ``m`` marks an unresolved position, ``id`` is
    the vocab id, ``c`` is the reveal confidence (absent for masked
    positions), and ``e`` is the sampling-time entropy in nats
    (autoregressive runs only, so absent elsewhere).

    Unlisted keys are dropped by pydantic, so a new signal must be
    declared here or it silently never reaches ``tokens.json``.
    """

    t: str
    m: bool
    id: int
    c: Optional[float] = None
    e: Optional[float] = None


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

    id: int
    t: str
    p: float
    rank: Optional[int] = None


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
    # When set, update this existing run folder in place instead of
    # creating a new one. Used when a saved run is edited-and-resumed:
    # the edited (bundled) run replaces its pre-edit original so it is a
    # single Analytics row rather than two.
    run_id: Optional[str] = None
    # Tokens the templated prompt occupied, as reported by the sampler
    # on its ``done`` frame rather than counted by the client, so the
    # saved figure is the one the run really built. None for a run
    # whose sampler predates the field.
    prompt_len: Optional[int] = Field(default=None, ge=0)


def _make_run_dir(base: Path, model_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = model_id.replace("/", "_")
    run_dir = base / f"{timestamp}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _display_run_path(run_dir: Path) -> str:
    """Run folder as written in the repo, for the UI's status line.

    Every save branch now produces an absolute path, since
    ``RESULTS_DIR`` is resolved once at startup, but the status line
    reads better as "results/2026-...". Shortening at the one point
    the branches meet keeps the traversal guards in
    ``_existing_run_dir`` intact and gives every message the same
    short form.

    Falls back to the path as given when it is not under the repo,
    which is exactly what should happen for a ``--results-dir``
    pointing elsewhere: naming the full path is how the UI tells the
    user their runs are not in the usual place. An operating
    condition, not a broken invariant, so it degrades to a longer
    message rather than raising.
    """
    try:
        return str(
            run_dir.resolve().relative_to(REPO_ROOT)
        )
    except ValueError:
        return str(run_dir)


def _existing_run_dir(run_id: str) -> Path:
    """Resolve an existing run folder for in-place update.

    Path-guarded like the delete/frames endpoints: the resolved dir
    must be a direct child of RESULTS_DIR and already exist.
    """
    results_root = RESULTS_DIR.resolve()
    run_dir = (results_root / run_id).resolve()
    if run_dir.parent != results_root:
        raise ValueError(f"invalid run id: {run_id}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
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
) -> Dict[str, Any]:
    """The context block for a saved run, or empty when unknowable.

    Two figures, together because either alone answers nothing useful:
    a prompt length means little without the window it competed for,
    and the window means little without a prompt to place inside it.

    The window is the resident model's, which is sound because a save
    always follows a run on the model that is still loaded; a switch
    reloads the page and discards the unsaved run.
    """
    if prompt_len is None:
        return {}
    assert prompt_len >= 0, "prompt_len must be non-negative"
    block: Dict[str, Any] = {"prompt_tokens": prompt_len}
    window = manager.active_context_length
    if window is not None:
        block["context_length"] = window
    return block


def _save_run_blocking(body: SaveRunRequest) -> str:
    model_id = body.model or DEFAULT_MODEL
    checkpoint = ""
    if model_id in REGISTRY:
        checkpoint = REGISTRY[model_id].checkpoint
    run_dir: Optional[Path] = None
    if body.run_id:
        # Update the pre-edit run's folder in place; if it is gone
        # (e.g. deleted), fall back to a fresh folder rather than fail.
        try:
            run_dir = _existing_run_dir(body.run_id)
        except (ValueError, FileNotFoundError):
            run_dir = None
    if run_dir is None:
        run_dir = _make_run_dir(RESULTS_DIR, model_id)

    model_type = "diffusion"
    if model_id in REGISTRY:
        model_type = REGISTRY[model_id].capabilities.model_type
    # Which device the resident worker ran on, for the analytics
    # Processor column and per-run timing header. "Unknown" only when
    # no device is recorded (older runs / an unexpected save path).
    device = manager.active_device
    if device == "cuda":
        processor = "GPU"
        processor_name = _gpu_name()
    elif device == "cpu":
        processor = "CPU"
        processor_name = _cpu_name()
    else:
        processor = "Unknown"
        processor_name = None
    metadata: Dict[str, Any] = {
        "backend": model_id,
        "model": checkpoint or model_id,
        # Lets the analytics suite gate diffusion-only charts (e.g.
        # convergence) off for autoregressive runs. Absent on runs
        # saved before this field existed (all of which are diffusion).
        "model_type": model_type,
        # GPU / CPU / Unknown, plus the device name for the timing header.
        "processor": processor,
        "processor_name": processor_name,
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
    if body.original_per_frame_elapsed is not None:
        metadata["original_per_frame_elapsed"] = (
            body.original_per_frame_elapsed
        )
    if body.original_elapsed_seconds is not None:
        metadata["original_elapsed_seconds"] = (
            body.original_elapsed_seconds
        )
    if body.original_mean_conf is not None:
        metadata["original_mean_conf"] = body.original_mean_conf
    if body.remask_edits:
        metadata["remask_edits"] = [
            edit.model_dump() for edit in body.remask_edits
        ]
    # Absent, not zeroed, when the length is unknown: an older run and
    # a run with an empty prompt must stay distinguishable, and the
    # Analytics rows are built to skip a missing block.
    context = _context_metadata(body.prompt_len)
    if context:
        metadata["context"] = context
    metadata["reproducibility"] = {
        "seed": body.params.get("seed"),
        "gpu": _gpu_name(),
        "git_commit": _git_commit(),
        "versions": dict(manager.active_versions),
        # Which tokenizer produced these ids. Persisted per run so an
        # old run still answers the question after the model it used
        # has been swapped out or its checkpoint has moved on.
        "tokenizer": dict(manager.active_tokenizer),
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
    if body.alternatives is not None:
        (run_dir / "alternatives.json").write_text(
            json.dumps(
                _dump_alternatives(body.alternatives),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if body.original_alternatives is not None:
        (run_dir / "original_alternatives.json").write_text(
            json.dumps(
                _dump_alternatives(body.original_alternatives),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    history_to_gif(
        body.frames,
        run_dir / "diffusion.gif",
        header_text=body.prompt,
    )
    return _display_run_path(run_dir)


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
        "alternatives": data["alternatives"],
        "alternatives_available": data[
            "alternatives_available"
        ],
        "original_alternatives": data[
            "original_alternatives"
        ],
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
    """GPU name and data root for the analytics UI.

    The GPU name because the supervisor has no torch. The data root
    because the delete confirmation used to spell it ``results/``
    from a hardcoded string, which stopped being true the moment
    the root became configurable, and a dialog about permanent
    deletion is the worst place to name the wrong directory.
    """
    return JSONResponse(
        content={
            "gpu_name": _gpu_name(),
            "results_dir": _display_run_path(RESULTS_DIR),
        }
    )


def _delete_run_blocking(run_id: str) -> None:
    """Delete one saved run directory under results/.

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


# -- Durable UI state (origin-independent frontend preferences) --


class UiStateValue(BaseModel):
    """One UI-state value, stored verbatim as its localStorage string."""

    value: str


def _reconcile_new_runs(state: Dict[str, str]) -> Dict[str, str]:
    """Prune the "new run" cue to run IDs whose folders still exist.

    The cue accumulates IDs of saved-but-unviewed runs. A run deleted
    outside the app (or before per-delete clearing existed) would linger
    as an orphan and inflate the generator/menu count forever, since it
    no longer appears in Analytics to open or delete. Reconciling here,
    on the endpoint every page hydrates from, makes the count self-heal
    everywhere. A freshly saved run is never pruned: its folder exists
    before its ID is added to the cue.
    """
    raw = state.get("diffusion_new_runs")
    if not raw:
        return state
    try:
        ids = json.loads(raw)
    except ValueError:
        return state  # Corrupt value: leave it for load_ui_state to drop.
    if not isinstance(ids, list):
        return state

    existing = _existing_run_ids()
    pruned = [run_id for run_id in ids if run_id in existing]
    if len(pruned) == len(ids):
        return state

    new_raw = json.dumps(pruned)
    try:
        set_ui_state_key(RESULTS_DIR, "diffusion_new_runs", new_raw)
    except (KeyError, ValueError, OSError):
        logger.exception("failed to reconcile new-run cue")
        return state
    state["diffusion_new_runs"] = new_raw
    return state


def _existing_run_ids() -> Set[str]:
    """Run IDs with a folder on disk (the folder name is the ID)."""
    if not RESULTS_DIR.is_dir():
        return set()
    return {
        child.name
        for child in RESULTS_DIR.iterdir()
        if child.is_dir()
    }


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
    """
    raw = state.get("diffusion_collections")
    if not raw:
        return state
    try:
        collections = json.loads(raw)
    except ValueError:
        return state  # Corrupt value: load_ui_state will drop it.
    if not isinstance(collections, list):
        return state

    existing = _existing_run_ids()
    pruned, dropped = _prune_collection_runs(collections, existing)
    if dropped == 0:
        return state

    new_raw = json.dumps(pruned)
    try:
        set_ui_state_key(
            RESULTS_DIR, "diffusion_collections", new_raw
        )
    except (KeyError, ValueError, OSError):
        logger.exception("failed to reconcile collections")
        return state
    state["diffusion_collections"] = new_raw
    return state


def _prune_collection_runs(
    collections: List[Any], existing: Set[str]
) -> Tuple[List[Any], int]:
    """Filter each collection's runs; return the list and drop count.

    Malformed entries are passed through untouched rather than
    repaired. This endpoint's job is to remove ids for runs that are
    gone, and a client that wrote a shape the client itself does not
    understand is not a problem the server can fix by guessing.
    """
    pruned: List[Any] = []
    dropped = 0
    for entry in collections:
        if not isinstance(entry, dict):
            pruned.append(entry)
            continue
        runs = entry.get("runs")
        if not isinstance(runs, list):
            pruned.append(entry)
            continue
        kept = [
            run_id for run_id in runs if run_id in existing
        ]
        dropped += len(runs) - len(kept)
        updated = dict(entry)
        updated["runs"] = kept
        pruned.append(updated)
    assert len(pruned) == len(collections), "lost a collection"
    return pruned, dropped


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
async def serve_menu() -> HTMLResponse:
    """Landing page: the model-selection Main Menu."""
    return _serve_stamped_page("menu.html")


@app.get("/generate")
async def serve_generate() -> Response:
    """Generator page, gated behind model selection.

    The Main Menu is the single entry point: reaching the generator
    without an active model (e.g. a direct URL hit) redirects back to
    the menu to choose one, rather than silently booting a default.
    """
    active_id = manager.active_id
    if active_id is None or manager.status(active_id) != "active":
        return RedirectResponse(url="/", status_code=307)
    return _serve_stamped_page("index.html")


@app.get("/index.html")
async def serve_index_html() -> RedirectResponse:
    """Back-compat: the generator now lives at ``/generate``."""
    return RedirectResponse(url="/generate", status_code=307)


@app.get("/analytics.html")
async def serve_analytics_page() -> HTMLResponse:
    return _serve_stamped_page("analytics.html")


@app.get("/settings.html")
async def serve_settings_page() -> HTMLResponse:
    """Shared, model-agnostic Settings page (always available)."""
    return _serve_stamped_page("settings.html")


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
