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
import contextlib
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
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

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
from pydantic import BaseModel, ConfigDict, Field

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
    ERROR_NO_MODEL_ACTIVE,
    ERROR_SCOPE_FATAL,
    ERROR_WORKER_UNREACHABLE,
    ModelInfo,
    wire_error,
)
from src.backends.registry import DEFAULT_MODEL, REGISTRY
from src.inference.render_gif import history_to_gif
from src.web import collections as collection_ops
from src.web import run_store
from src.web.data_root import (
    RESULTS_DIR_ENV,
    resolve_results_dir,
)
from src.web.ui_state import (
    load_ui_state,
    mutate_ui_state_key,
    set_ui_state_key,
)
from src.inference.download_main import (
    DOWNLOAD_EXIT_OK,
    DOWNLOAD_EXIT_UNREACHABLE,
)
from src.web.worker_process import (
    WorkerHandle,
    download_command,
    spawn_worker,
    worker_command,
)

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

# What this server calls itself when asked. Read by the desktop
# launcher to tell its own supervisor from an unrelated process
# holding the same port. A constant rather than a version string:
# the question is "is this us", and pinning it to a version would
# make two builds of the same app fail to recognise each other.
APP_IDENTITY = "diffusion-llm-supervisor"

WORKER_START_TIMEOUT_S = 180.0
WORKER_STOP_TIMEOUT_S = 30.0
# How long to wait for a killed worker to actually be gone. Short,
# because SIGKILL is not refusable: anything still here after this is
# stuck in the kernel and waiting longer will not change that. It
# exists so the supervisor can say it waited rather than assumed.
WORKER_KILL_TIMEOUT_S = 5.0
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
# How often the supervisor measures a download's cache directory
# while a child process fetches it. Matches the cadence the
# in-process sampler used, which the progress bar was tuned against.
DOWNLOAD_PROGRESS_POLL_S = 0.5
# A ceiling on that sampling, so the loop is finite. Six hours is far
# past any real fetch on any plausible connection; reaching it means
# the child is wedged, which is reported rather than waited out.
DOWNLOAD_POLL_SECONDS_MAX = 6 * 60 * 60
DOWNLOAD_POLL_ITERATIONS_MAX = int(
    DOWNLOAD_POLL_SECONDS_MAX / DOWNLOAD_PROGRESS_POLL_S
)

assert DOWNLOAD_PROGRESS_POLL_S > 0.0, "a poll must advance"
assert DOWNLOAD_POLL_ITERATIONS_MAX > 0, "sample at least once"


# -- Model worker manager --

# The two things `ModelManager` does to the outside world that a test
# cannot afford to do for real: start a process, and read a socket.
# Named so the injection points below read as contracts rather than
# as "some callable".
class ActivationRefused(RuntimeError):
    """This model cannot be activated, and we knew before trying.

    Distinct from a fault so the route can answer with the reason
    instead of a 500 and a stack trace. A missing interpreter or a
    model that cannot fit is an ordinary answer to an ordinary
    request; logging it as a server error buries the real ones.
    """


SpawnWorker = Callable[..., WorkerHandle]
ProbeHealth = Callable[
    [str], Awaitable[Optional[Dict[str, Any]]]
]


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


def _is_partial(checkpoint: str) -> bool:
    """Whether an interrupted fetch left parts of this one behind.

    Reported beside ``downloaded`` because that flag alone cannot
    tell a model never fetched from one stopped part way, and the
    two want different words on the row: offering to start a
    download over is wrong when clicking it will resume.

    Always false for a local path. Those are produced offline rather
    than fetched, so there is no partial state for them to be in.
    """
    if not _is_repo_checkpoint(checkpoint):
        return False
    try:
        from src.inference.hf_download import (
            has_partial_download,
        )

        return has_partial_download(checkpoint)
    except Exception:  # noqa: BLE001 - probe failure: assume not
        return False


def _is_downloaded(checkpoint: str) -> bool:
    """Whether the checkpoint's files are fully present locally.

    A partial cache (an interrupted download leaving ``*.incomplete``
    parts) counts as not-downloaded so the menu keeps its download
    veneer and a re-click resumes, rather than the model being
    marked ready and hanging on load. ``_is_partial`` above is what
    lets that veneer say "resume" rather than "download".
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


async def _probe_health(
    url: str,
) -> Optional[Dict[str, Any]]:
    """One read of a worker's /health, or None if it did not answer.

    A worker that is still importing torch refuses the connection,
    which is expected rather than exceptional for the first several
    seconds of every activation. None says "no answer yet"; the
    caller decides whether that has gone on too long.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=2.0)
        except Exception:  # noqa: BLE001 - worker still coming up
            return None
    if response.status_code != 200:
        return None
    body: Dict[str, Any] = response.json()
    return body


class ModelManager:
    """Spawns/stops one model worker subprocess at a time.

    Only one worker is ever alive, since a single ~15-16 GB model
    already saturates the 24 GB GPU.
    """

    def __init__(
        self,
        *,
        spawn: SpawnWorker = spawn_worker,
        probe: ProbeHealth = _probe_health,
        start_timeout_s: float = WORKER_START_TIMEOUT_S,
        stop_timeout_s: float = WORKER_STOP_TIMEOUT_S,
        kill_timeout_s: float = WORKER_KILL_TIMEOUT_S,
        vram_settle_timeout_s: float = VRAM_SETTLE_TIMEOUT_S,
        health_poll_s: float = WORKER_HEALTH_POLL_S,
        progress_poll_s: float = WORKER_PROGRESS_POLL_S,
        download_poll_s: float = DOWNLOAD_PROGRESS_POLL_S,
    ) -> None:
        # Injected so a test can drive the lifecycle without a real
        # subprocess, a real socket, or a three-minute deadline. The
        # defaults are the production values, so nothing that builds
        # a manager the old way behaves differently.
        self._spawn = spawn
        self._probe = probe
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._kill_timeout_s = kill_timeout_s
        self._vram_settle_timeout_s = vram_settle_timeout_s
        self._health_poll_s = health_poll_s
        self._progress_poll_s = progress_poll_s
        self._download_poll_s = download_poll_s
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
        # Which activation the current state describes. Monotonic and
        # never reset, including by finalization: a client polling
        # for the outcome of a load that failed needs to recognise
        # the failure as its own, so the number has to outlive the
        # worker exactly the way the error message does. Zero means
        # nothing has ever been activated.
        self.activation_id: int = 0
        self._proc: Optional[WorkerHandle] = None
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
        # The child doing the fetching. A download used to be threads
        # inside this process with nothing able to reach them; this is
        # what makes cancel and shutdown mean something.
        self._download_proc: Optional[WorkerHandle] = None
        # Which download the state describes, on the same terms as
        # activation_id: monotonic, never reset, so a window can tell
        # its own download's outcome from another window's.
        self.download_id: int = 0

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
        """Whether this model's worker process exists right now.

        Deliberately still about the process rather than about
        readiness. Its callers are the menu's residency label and the
        VRAM accounting in ``_models_snapshot``, and a worker that is
        halfway through loading really is holding that memory. Asking
        "can this serve a request" is a different question with a
        different answer; see ``is_serving``.
        """
        if self.active_id == model_id and self._alive():
            return "active"
        return "inactive"

    def is_serving(self, model_id: str) -> bool:
        """Whether this model can answer a request right now.

        The gates in front of the generator page and the WebSocket
        proxy used to ask ``status``, which is only "a process
        exists". A worker that timed out or reported a load failure
        stayed alive, so both gates let traffic through to a model
        that was never going to answer. Readiness is the question
        they were always trying to ask.
        """
        return (
            self.active_id == model_id
            and self._alive()
            and self.load_state == "ready"
        )

    def ws_url(self) -> str:
        assert self._port is not None
        return f"ws://127.0.0.1:{self._port}/ws"

    async def activate(
        self, model_id: str, *, device: Optional[str] = None
    ) -> int:
        """Spawn the worker and return immediately (non-blocking).

        A background monitor task then tracks startup (download /
        load / ready / error), which the client polls via
        ``/api/models/activation``. Keeping the load off the lock lets
        ``stop`` / ``cancel_activation`` terminate a still-loading
        worker instead of deadlocking behind a held lock.

        Four phases, in this order for a reason. Everything knowable
        without freeing anything is checked first, so a switch to a
        model that cannot run leaves the working one running. The
        resident worker is evicted only once the target has passed;
        anything that can only be known after eviction (the real VRAM
        reading) follows it.

        Returns the activation's operation id, which is how the
        caller later recognises the outcome as its own. Two browser
        windows share one supervisor, so "is this load finished" is
        not a question with a single answer.
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
                # Nothing to do, so nothing new to number: the caller
                # is handed the activation that produced the worker
                # already running.
                return self.activation_id
            info = REGISTRY[model_id]
            python = self._validate_target(info, device)
            await self._stop_locked()
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
            proc = self._spawn(
                worker_command(
                    python=python,
                    model_id=model_id,
                    port=port,
                    device=device,
                ),
                cwd=REPO_ROOT,
                env=env,
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
            # Clears the previous failure, which `_finalize` keeps
            # around so the menu can explain a redirect. Trying again
            # is the moment it stops being news.
            self.load_error = None
            self.activation_id += 1
            self._monitor_task = asyncio.create_task(
                self._monitor_startup(proc, port)
            )
            return self.activation_id

    def _validate_target(
        self, info: ModelInfo, device: str
    ) -> Path:
        """Everything knowable before anything is freed.

        Activation used to stop the resident worker first and only
        then look at the target, so picking a model that could never
        have run cost the user a loaded model and the run on screen
        in front of it, for an error that needed no VRAM to discover.
        Every check here raises, and raising here means nothing has
        been evicted.

        Returns the interpreter to launch, since finding it is one of
        the checks.
        """
        python = REPO_ROOT / info.venv_python
        if not python.exists():
            raise ActivationRefused(
                f"{info.display_name} is not installed:"
                f" no interpreter at {info.venv_python}."
            )
        supported = info.capabilities.supported_devices
        if device not in supported:
            raise ActivationRefused(
                f"{info.display_name} cannot run on"
                f" {device.upper()}; it supports"
                f" {', '.join(d.upper() for d in supported)}."
            )
        # Only for checkpoints that are a directory on this machine.
        # A Hub id is not checked here: an uncached one downloads on
        # first activation, which is a supported path rather than a
        # failure, and the menu already marks it.
        if not _is_repo_checkpoint(info.checkpoint):
            path = Path(info.checkpoint).expanduser()
            if not path.is_dir():
                raise ActivationRefused(
                    f"{info.display_name} checkpoint not found"
                    f" at {path}."
                )
        self._validate_headroom(info, device)
        return python

    def _validate_headroom(
        self, info: ModelInfo, device: str
    ) -> None:
        """Refuse a model that cannot fit even after the switch.

        Non-destructive, which is the whole point: it counts the
        resident worker's VRAM as reclaimable rather than reclaiming
        it to find out. ``_preflight_vram`` still runs after eviction
        and remains the authority; this only catches the case that
        was already hopeless.
        """
        if device == "cpu" or info.min_vram_gib <= 0:
            return
        free = _free_vram_gib()
        if free is None:
            return  # unreadable; the post-eviction check will say so
        reclaimable = 0.0
        if (
            self.active_id is not None
            and self._alive()
            and self.active_device == "cuda"
            and self.active_id in REGISTRY
        ):
            reclaimable = REGISTRY[self.active_id].min_vram_gib
        if free + reclaimable < info.min_vram_gib:
            raise ActivationRefused(
                f"Not enough GPU memory for {info.display_name}:"
                f" needs about {info.min_vram_gib:.0f} GiB, and"
                f" only {free + reclaimable:.1f} GiB would be free"
                " after unloading the current model. The current"
                " model is still loaded."
            )

    async def _monitor_startup(
        self, proc: WorkerHandle, port: int
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
            time.monotonic() + self._start_timeout_s
        )
        responded = False
        while True:
            code = proc.poll()
            if code is not None:
                await self._fail_startup(
                    proc,
                    f"worker exited during startup (code {code})",
                )
                return
            if (
                not responded
                and time.monotonic() > startup_deadline
            ):
                await self._fail_startup(
                    proc, "worker did not start in time"
                )
                return
            body = await self._probe(url)
            if body is not None:
                responded = True
                failure = self._apply_health(body)
                if failure is not None:
                    await self._fail_startup(proc, failure)
                    return
                if self.load_state == "ready":
                    return
            await asyncio.sleep(
                self._progress_poll_s
                if responded
                else self._health_poll_s
            )

    async def _fail_startup(
        self, proc: WorkerHandle, message: str
    ) -> None:
        """End a worker that will never become ready.

        Called from inside the monitor, so it must not cancel the
        monitor task (that is this task) and must not take the lock
        (``_stop_locked`` awaits this task while holding it, which
        would deadlock). ``_finalize``'s identity check is what makes
        both omissions safe.

        Before this existed, all three of these exits set the state
        to "error" and returned with the worker still running: it
        kept its VRAM, and the page gates, which asked only whether a
        process was alive, went on letting traffic through to it.
        """
        logger.error(
            "worker %s failed to start: %s",
            self.active_id,
            message,
        )
        await self._finalize(proc, error=message)

    def _apply_health(
        self, body: Dict[str, Any]
    ) -> Optional[str]:
        """Fold one /health body into load state.

        Returns the failure message when the worker reports one, and
        None otherwise. The caller ends the run on a message or on
        reaching "ready"; a message additionally means the worker has
        to be terminated, which is why it is returned rather than
        just recorded.
        """
        status = body.get("status")
        if status == "error":
            return str(
                body.get("message", "model failed to load")
            )
        if status == "ready":
            self.active_versions = body.get("versions", {})
            self.active_tokenizer = body.get("tokenizer", {})
            self.active_context_length = _read_context_length(body)
            self.load_progress = None
            self.load_state = "ready"
            return None
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
        return None

    async def cancel_activation(
        self, operation: Optional[int] = None
    ) -> None:
        """Cancel an in-flight activation and free the worker/VRAM.

        ``operation`` is the id the caller was given when it started
        the activation. It has to match, because this used to stop
        whatever worker was loading regardless of who asked for it:
        two windows share one supervisor, so one window's Cancel
        could kill the other's load, which is half of `LIFE-03`.

        Cancelling when nothing is loading stays a no-op rather than
        a refusal. There is nothing to protect, and a stale window
        tidying up after itself should not be told off for it.

        The lock is free during load, so this never deadlocks against
        ``activate``.
        """
        async with self._lock:
            if not self._alive():
                return
            if operation != self.activation_id:
                raise ActivationRefused(self._not_yours_message())
            await self._stop_locked()

    def _not_yours_message(self) -> str:
        """Why a cancel was refused, in terms of what is loading."""
        entry = (
            REGISTRY.get(self.active_id)
            if self.active_id is not None
            else None
        )
        name = (
            entry.display_name
            if entry is not None
            else str(self.active_id)
        )
        return (
            f"{name} is loading, and it was started somewhere"
            " else. Cancel it from the window that started it."
        )

    # -- download-only (pre-fetch weights, no VRAM) --

    def start_download(self, model_id: str) -> int:
        """Begin downloading a model's weights without loading them.

        Runs as a child process so a resident model keeps serving and
        so the fetch has an owner: see ``cancel_download``. Returns
        the operation number naming it. Raises for an unknown or
        non-downloadable model, or if a download is already running.
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
        handle = self._spawn(
            download_command(
                python=Path(sys.executable), repo_id=checkpoint
            ),
            cwd=REPO_ROOT,
            env=dict(os.environ),
        )
        self._download_proc = handle
        self.download_target = model_id
        self.download_state = "downloading"
        self.download_progress = None
        self.download_error = None
        self.download_id += 1
        self._download_task = asyncio.create_task(
            self._watch_download(checkpoint, handle)
        )
        return self.download_id

    async def _watch_download(
        self, checkpoint: str, handle: WorkerHandle
    ) -> None:
        """Sample progress from disk until the child exits.

        The child reports nothing, and needs no channel to: progress
        is the size of the cache directory, which this process can
        measure while another does the fetching. That is the whole
        reason a download could move out of process cheaply.
        """
        from src.inference.hf_download import (
            repo_progress,
            repo_total_bytes,
        )

        total = await asyncio.to_thread(
            repo_total_bytes, checkpoint
        )
        code: Optional[int] = None
        for _ in range(DOWNLOAD_POLL_ITERATIONS_MAX):
            code = handle.poll()
            if code is not None:
                break
            self.download_progress = await asyncio.to_thread(
                repo_progress, checkpoint, total
            )
            await asyncio.sleep(self._download_poll_s)
        self._settle_download(checkpoint, code)

    def _settle_download(
        self, checkpoint: str, code: Optional[int]
    ) -> None:
        """Turn the child's exit status into a reportable outcome.

        The status is the entire protocol between the two processes,
        so this is where it is read. ``None`` means the sampler hit
        its ceiling with the child still running, which is a bug
        rather than a slow download: the ceiling is hours.
        """
        self._download_proc = None
        self.download_progress = None
        if code == DOWNLOAD_EXIT_OK:
            self.download_state = "done"
            return
        self.download_state = "error"
        if code == DOWNLOAD_EXIT_UNREACHABLE:
            from src.inference.hf_download import (
                describe_unreachable,
            )

            self.download_error = describe_unreachable(checkpoint)
            return
        if code is None:
            self.download_error = (
                "the download is still running but is no longer"
                " being watched; restart the app"
            )
            logger.error(
                "download sampler gave up on %s while it ran",
                checkpoint,
            )
            return
        self.download_error = (
            f"the download failed (exit {code}). The log has the"
            " underlying error."
        )

    async def cancel_download(
        self, operation: Optional[int] = None
    ) -> None:
        """Stop an in-flight download and leave its parts on disk.

        Refuses an operation that is not the current one, the way
        ``cancel_activation`` does, so a stale window cannot end a
        download somebody else started.

        The partial blobs stay exactly where they are. That is what
        makes a re-click resume rather than restart, and deleting the
        cache was rejected in the finding's own Direction because a
        valid snapshot in it may be shared with another process.
        """
        if self.download_state != "downloading":
            return
        if operation is not None and operation != self.download_id:
            raise ActivationRefused(
                "That download has already finished or belongs to"
                " another window."
            )
        await self._end_download()
        self.download_state = "idle"
        self.download_target = None
        self.download_progress = None
        self.download_error = None

    async def _end_download(self) -> None:
        """Stop watching, then stop the child, in that order.

        The watcher first: it would otherwise see the exit it was
        never told to expect and report a cancellation as a failed
        download.
        """
        task = self._download_task
        self._download_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - reported, not raised
                logger.exception("download watcher failed")
        handle = self._download_proc
        self._download_proc = None
        if handle is not None:
            await self._end_process(handle, "download")

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
        deadline = (
            time.monotonic() + self._vram_settle_timeout_s
        )
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
            raise ActivationRefused(
                f"Not enough free GPU memory to load"
                f" {info.display_name}: needs about"
                f" {required:.0f} GiB but only {free:.1f} GiB"
                f" is free. Close other GPU processes and"
                f" try again."
            )

    async def stop(self) -> None:
        # Outside the lock, and before it: a download is independent
        # of the worker (it can run alongside a resident model), and
        # taking the lock to end one would make a shutdown wait on
        # whatever activation happened to hold it.
        await self._end_download()
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop the resident worker and prove it is gone.

        The monitor is cancelled first because this is not the
        monitor calling; a failure detected inside the monitor takes
        the same finalization without that step (see
        ``_monitor_startup``), since a task cannot await its own
        cancellation.
        """
        await self._cancel_monitor()
        await self._finalize(self._proc, error=None)

    async def _cancel_monitor(self) -> None:
        """Stop watching a worker's startup, if we still are."""
        task = self._monitor_task
        if task is None:
            return
        self._monitor_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - reported, not raised
            # A monitor that died of a real fault used to be
            # swallowed here with nothing logged, so a bug in startup
            # tracking looked like a worker that never became ready.
            logger.exception("startup monitor failed")

    async def _finalize(
        self, handle: Optional[WorkerHandle], *, error: Optional[str]
    ) -> None:
        """End a worker and clear the state that described it.

        The one terminal path. Every way a worker stops, a switch, a
        cancel, a shutdown, a startup timeout, a load failure, comes
        through here, so "the process is gone" and "the supervisor
        says it is gone" cannot disagree.

        ``error`` carries the reason when there is one. It outlives
        the process on purpose: the page that would have shown it is
        often a redirect away, and clearing it here is what used to
        leave the menu with nothing to say. The next activation
        clears it.

        Safe to call from the startup monitor, which is why the state
        clearing is guarded by an identity check rather than by the
        lock: by the time a slow termination finishes, a newer
        activation may already own the manager's fields, and this
        must not wipe them.
        """
        if handle is not None:
            await self._end_process(handle)
        if handle is not None and self._proc is not handle:
            # Superseded while we were terminating. The process we
            # were asked to end is gone, which was the job; the state
            # now describes somebody else's worker.
            return
        self._proc = None
        self._port = None
        self.active_id = None
        self.active_device = None
        self.active_versions = {}
        self.active_tokenizer = {}
        self.active_context_length = None
        self.load_progress = None
        self.load_state = "error" if error else "idle"
        self.load_error = error

    async def _end_process(
        self, handle: WorkerHandle, what: str = "worker"
    ) -> None:
        """Terminate, escalate to kill, and wait for the exit.

        The wait after the kill is the point. Without it the manager
        cleared every field the instant it signalled, so a
        replacement could be spawned against VRAM whose release
        nothing had confirmed, and the eight-second settle window in
        ``_preflight_vram`` was left standing in for a wait that
        never happened.

        Shared with downloads since `TRUST-04`, which is why ``what``
        exists: one escalation policy, two kinds of child. A second
        ladder written for downloads would be a second place for the
        timeouts to drift.
        """
        if handle.poll() is not None:
            return
        logger.info(
            "stopping %s (pid %s)",
            what,
            handle.pid,
        )
        handle.terminate()
        if await self._await_exit(handle, self._stop_timeout_s):
            return
        logger.warning(
            "%s (pid %s) ignored SIGTERM; killing",
            what,
            handle.pid,
        )
        handle.kill()
        if await self._await_exit(handle, self._kill_timeout_s):
            return
        # Nothing further to try: SIGKILL is not refusable, so a
        # process still here is stuck in the kernel (uninterruptible
        # I/O, or a wedged GPU driver call). Say so loudly rather
        # than reporting a clean stop that did not happen.
        logger.error(
            "%s (pid %s) survived SIGKILL; its resources are not"
            " confirmed released",
            what,
            handle.pid,
        )

    async def _await_exit(
        self, handle: WorkerHandle, timeout_s: float
    ) -> bool:
        """Wait for one process to exit. True if it did."""
        try:
            await asyncio.to_thread(handle.wait, timeout_s)
        except Exception:  # noqa: BLE001 - timeout or reap race
            return handle.poll() is not None
        return True


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
        data["partial"] = _is_partial(info.checkpoint)
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
    """Cancel an in-flight load, stopping the worker and freeing VRAM."""
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
    positions), and ``e`` is the sampling-time entropy in nats
    (autoregressive runs only, so absent elsewhere).

    A new signal must be declared here to reach ``tokens.json``. It
    used to be dropped silently; now the request fails and says which
    key it did not recognize.
    """

    model_config = STRICT

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

    model_config = STRICT

    id: int
    t: str
    p: float
    rank: Optional[int] = None


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
    # The placement the model actually got, which is not always the
    # one requested: LLaDA and SmolLM3 fall back to CPU when CUDA is
    # unavailable, so a run that ran on CPU could be saved as GPU.
    device: str = "unknown"
    versions: Dict[str, str] = Field(default_factory=dict)
    tokenizer: Dict[str, Any] = Field(default_factory=dict)
    context_length: Optional[int] = None


class SaveRunRequest(BaseModel):
    model_config = STRICT

    model: str = DEFAULT_MODEL
    prompt: str
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
    # What the worker said about itself when this run finished,
    # echoed back from the terminal frame. Absent for a run whose
    # snapshot predates this field, which then falls back to the
    # supervisor's current view, as every save used to do.
    provenance: Optional[RunProvenance] = None
    # Which generation produced this run, from the same terminal
    # frame (`LIFE-01`). The store publishes under it, so a save that
    # was already made once lands on the run it made rather than on a
    # second copy. Absent for a run whose snapshot predates it.
    run_token: Optional[str] = None
    # When set, replace this existing run instead of creating a new
    # one. Used when a saved run is edited-and-resumed: the edited
    # (bundled) run replaces its pre-edit original so it is a single
    # Analytics row rather than two.
    run_id: Optional[str] = None
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
        return "GPU", _gpu_name()
    if device == "cpu":
        return "CPU", _cpu_name()
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
    model_type = (
        entry.capabilities.model_type if entry else "diffusion"
    )
    processor, processor_name = _describe_processor(provenance)
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
        "gpu": _gpu_name(),
        "git_commit": _git_commit(),
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
    )


def _save_run_blocking(body: SaveRunRequest) -> Dict[str, Any]:
    """Publish a run and describe it back to the client.

    Returns the id and revision as well as the display path, because
    an edited run has to be able to replace what it just saved, and
    doing that safely means quoting the revision it is replacing.

    Which run is written is the store's decision, not this one: it
    resolves the run token first and falls back to the id below. All
    that happens here is the older fallback's own check.

    Expanded first, on this thread, because rebuilding a long run's
    frames is real work and the event loop is not where it belongs.
    """
    body = body.normalized()
    replacing: Optional[str] = None
    if body.run_id:
        # Replace the pre-edit run; if it is gone (deleted from
        # another window, say), fall back to a fresh run rather than
        # failing a save the user has already paid for.
        try:
            run_store.resolve_run_dir(RESULTS_DIR, body.run_id)
            replacing = body.run_id
        except (
            run_store.InvalidRunIdError,
            run_store.RunNotFoundError,
        ):
            replacing = None

    bundle = _build_bundle(body)
    run_id, revision = run_store.save(
        RESULTS_DIR,
        bundle,
        model_id=body.model or DEFAULT_MODEL,
        run_id=replacing,
        expected_revision=(
            body.expected_revision if replacing else None
        ),
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

# The capability value marking a left-to-right model, which has no
# masked canvas and therefore no convergence curve.
MODEL_TYPE_AUTOREGRESSIVE = "autoregressive"


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
    if record.get("model_type") == MODEL_TYPE_AUTOREGRESSIVE:
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
            "gpu_name": _gpu_name(),
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
    """One UI-state value, stored verbatim as its localStorage string."""

    value: str


def _reconcile_new_runs(state: Dict[str, str]) -> Dict[str, str]:
    """Prune the "new run" cue to run IDs whose folders still exist.

    The cue accumulates IDs of saved-but-unviewed runs. A run deleted
    outside the app (or before per-delete clearing existed) would linger
    as an orphan and inflate the generator/menu count forever, since it
    no longer appears in Analytics to open or delete. Reconciling here,
    on the endpoint every page hydrates from, makes the count self-heal
    everywhere. A freshly saved run is never pruned: its folder
    exists before its ID is added to the cue.

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
            return None  # Corrupt: leave it for load_ui_state to drop.
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
    if active_id is None or not manager.is_serving(active_id):
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
