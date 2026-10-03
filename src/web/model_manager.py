"""Ownership of the resident model worker, outside the web app.

One supervisor runs one worker at a time, and everything about that
worker's life is decided here: which model may load and on which
device, the pre-flight that refuses one that cannot fit, the machine
lease that keeps a second launcher from loading alongside, the
download that fetches a checkpoint first, and the monitor that walks
a load to ready. The probes those decisions read, the GPU's name and
free memory and the host's CPU and RAM, sit beside it, and so do the
checks a downloaded artifact is judged by.

Its own module rather than the top of the web app (`A2-ORG-01`), and
free of any web framework on purpose: a test builds a manager on a
fake process and a scripted probe without starting an app, and the
routes that need the manager or its probes reach them through this
one boundary instead of through the module that defines every page.
A test holds the module to that.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)

import httpx

from src.backends.environments import (
    UnknownEnvironmentError,
    interpreter_for,
    lock_for,
)
from src.backends.protocol import (
    HubFiles,
    ModelInfo,
    is_hub_checkpoint,
)
from src.backends.registry import REGISTRY
from src.inference.download_main import (
    DOWNLOAD_EXIT_NO_SPACE,
    DOWNLOAD_EXIT_OK,
    DOWNLOAD_EXIT_UNREACHABLE,
)
from src.web.model_lease import (
    RUNTIME_DIR_ENV,
    LeaseUnavailable,
    PrimaryModelLease,
)
from src.web.worker_process import (
    WorkerHandle,
    download_command,
    spawn_worker,
    worker_command,
)

# The supervisor's own logger, by name, so a line logged here reads
# exactly as it did when this code lived in the server module.
logger = logging.getLogger("diffusion_supervisor")

# Workers are spawned from here, with this as their working directory
# and import path. The server module resolves the same directory for
# its static files and data root; a test holds the two to agreeing.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Disable the Xet download client before the first huggingface_hub
# import. Here huggingface_hub is imported lazily (in _is_downloaded /
# the download task), so setting the flag now, at module load, still
# precedes it. The flag is cached in hf constants at import time, so
# setting it any later is a no-op; Xet bypasses our tqdm progress
# hook, whereas the classic downloader routes through it, so the
# menu's download bar fills smoothly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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


def _residency_refusal(lease: PrimaryModelLease) -> str:
    """What to tell somebody whose machine is already busy.

    Written to be acted on rather than merely accurate: the
    instruction is to unload or close the other instance, because
    that is the only thing that frees the claim.

    Degrades when the note cannot be read. That happens if the holder
    was mid-rewrite, and naming a wrong process would be worse than
    naming none: it would send somebody to close a window that is not
    the one holding the model.
    """
    owner = lease.owner()
    if owner is None:
        return (
            "Another instance of this app already has a model"
            " loaded. Unload it there, or close it, and try again."
        )
    launcher = owner.get("launcher") or "another instance"
    model = owner.get("model")
    pid = owner.get("pid")
    who = f"{launcher} (pid {pid})" if pid else launcher
    holding = (
        f" has {model} loaded" if model else " has a model loaded"
    )
    return (
        f"{who}{holding}. Only one model can be resident on this"
        " machine at a time, so unload it there, or close it, and"
        " try again."
    )


def _lease_unavailable_refusal(error: LeaseUnavailable) -> str:
    """What to tell somebody with nowhere to keep a lease.

    A refusal rather than a warning (`A2-TRUST-01`): the lease is what
    keeps two launchers from loading two models into one card, so
    loading without it would break the promise it exists to keep.
    Names the path and the variable that moves it, because those are
    the two things a person can change.
    """
    return (
        f"Could not create the model lock at {error.path}"
        f" ({error.reason}), so this app cannot make sure only one"
        " model is loaded on this machine. Make"
        f" {error.path.parent} writable, or set {RUNTIME_DIR_ENV}"
        " to a directory that is, and try again."
    )


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
# unavailable. Resolve it explicitly with common fallbacks, cached,
# and log the outcome once so a missing binary is diagnosable.
_NVIDIA_SMI_FALLBACKS = (
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
)
_nvidia_smi_resolved = False
_nvidia_smi_path_cached: Optional[str] = None


def _nvidia_smi_path() -> Optional[str]:
    """The nvidia-smi binary, from PATH then common paths. Cached."""
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


# The card cannot be swapped under a running process, so its name is
# resolved once and kept. Free VRAM deliberately is not: it moves with
# every load and evict, and a stale reading is what made the menu
# promise headroom the activation then refused.
#
# Only a successful read is cached. A failure can be transient, and
# remembering it would turn one bad answer into a permanent claim of
# no GPU. An absent binary already costs nothing either way, since
# _nvidia_smi_path caches that and returns before any subprocess.
_gpu_name_cached: Optional[str] = None


def _gpu_name() -> Optional[str]:
    """Best-effort GPU name via nvidia-smi. Cached once it reads."""
    global _gpu_name_cached
    if _gpu_name_cached is not None:
        return _gpu_name_cached
    _gpu_name_cached = _nvidia_smi_query("name")
    return _gpu_name_cached


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
    version mismatch, e.g. after an NVIDIA update pending a reboot),
    or "error". Only called when the GPU name is unreadable, to
    explain why.
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
    """Best-effort CPU model name, from /proc/cpuinfo then platform.

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
    may still hold VRAM (the PDEATHSIG guard covers most cases, but
    not e.g. a supervisor that predates it). We match our worker
    command line and terminate only orphans (ppid == 1), never a
    worker still owned by a live supervisor, so a browser and desktop
    instance can coexist. Best-effort and Linux-only (/proc); a no-op
    elsewhere.
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


def _is_partial(checkpoint: str) -> bool:
    """Whether an interrupted fetch left parts of this one behind.

    Reported beside ``downloaded`` because that flag alone cannot
    tell a model never fetched from one stopped part way, and the
    two want different words on the row: offering to start a
    download over is wrong when clicking it will resume.

    Always false for a local path. Those are produced offline rather
    than fetched, so there is no partial state for them to be in.
    """
    if not is_hub_checkpoint(checkpoint):
        return False
    try:
        from src.inference.hf_download import (
            has_partial_download,
        )

        return has_partial_download(checkpoint)
    except Exception:  # noqa: BLE001 - probe failure: assume not
        return False


def _is_downloaded(
    checkpoint: str,
    revision: Optional[str] = None,
    companion: Optional[HubFiles] = None,
) -> bool:
    """Whether the checkpoint's files are fully present locally.

    A model that borrows files from another repository is downloaded
    only when those are present too; otherwise the first activation
    would have to fetch after all, and offline it could not.

    A partial cache (an interrupted download leaving ``*.incomplete``
    parts) counts as not-downloaded so the menu keeps its download
    veneer and a re-click resumes, rather than the model being
    marked ready and hanging on load. ``_is_partial`` above is what
    lets that veneer say "resume" rather than "download".

    The question is asked about the pinned commit, not the repository.
    A cache holding a different commit is not this model downloaded,
    and reporting it as ready would send the user into an activation
    that has to fetch after all.

    A local checkpoint has to carry a manifest. The existence of a
    directory used to be the whole test, which is why an interrupted
    quantization could leave a truncated 16 GB state dict that the
    menu offered as a ready model. The manifest is written last, after
    every file it names, so having one is the completion signal.
    """
    if is_hub_checkpoint(checkpoint):
        try:
            from src.inference.hf_download import (
                are_companion_files_cached,
                is_repo_cached,
            )

            if not is_repo_cached(checkpoint, revision=revision):
                return False
            if companion is None:
                return True
            return are_companion_files_cached(
                companion.repo,
                companion.files,
                revision=companion.revision,
            )
        except Exception:  # noqa: BLE001 - probe failure: treat as not cached
            return False
    try:
        from src.inference.artifact_manifest import (
            is_complete_artifact,
        )

        return is_complete_artifact(
            Path(checkpoint).expanduser()
        )
    except Exception:  # noqa: BLE001 - probe failure: treat as absent
        return False


def _download_argv(info: ModelInfo) -> List[str]:
    """The download child's argv, with the companion when the model
    borrows files, so one child fetches everything the model needs
    and one cancel ends all of it."""
    python = Path(sys.executable)
    companion = info.companion
    if companion is None:
        return download_command(
            python=python,
            repo_id=info.checkpoint,
            revision=info.revision,
        )
    return download_command(
        python=python,
        repo_id=info.checkpoint,
        revision=info.revision,
        companion_repo=companion.repo,
        companion_revision=companion.revision,
        companion_files=companion.files,
    )


def _validate_local_artifact(info: ModelInfo, path: Path) -> None:
    """Refuse a local checkpoint that cannot attest it is complete.

    Raised before any eviction, like every check around it, so
    discovering this costs nothing that was already loaded.

    The message names the command that fixes each case, because
    neither is guessable. A directory built before manifests existed
    is attested in place, with no rebuild and no GPU; a directory left
    behind by an interrupted build cannot be repaired and has to be
    built again. Telling the user only that something is wrong with a
    16 GB directory would leave the two indistinguishable.
    """
    from src.inference.artifact_manifest import (
        is_complete_artifact,
        read_manifest,
    )

    if is_complete_artifact(path):
        return
    if read_manifest(path) is None:
        raise ActivationRefused(
            f"{info.display_name} at {path} has no artifact"
            " manifest, so there is no way to tell a finished"
            " build from an interrupted one. If this checkpoint"
            " works, attest it in place with:"
            " .venv/bin/python"
            " scripts/quantize_diffusiongemma_nf4.py --adopt"
            f" --out {path}"
        )
    raise ActivationRefused(
        f"{info.display_name} at {path} is incomplete: its"
        " manifest does not match what is on disk, which is what"
        " an interrupted build leaves behind. Remove the"
        " directory and build it again."
    )


def _describe_no_space(
    checkpoint: str, revision: Optional[str] = None
) -> str:
    """The refusal message for a download that will not fit.

    Re-measured here because the child reported only a status, so its
    figures are gone. When they cannot be read back, this falls back
    to the situation without the numbers, which is still the one
    sentence that names the remedy.
    """
    try:
        from src.inference.hf_download import (
            describe_insufficient_space,
            repo_free_bytes,
            repo_total_bytes,
            space_needed_bytes,
        )

        needed = space_needed_bytes(
            checkpoint,
            total_bytes=repo_total_bytes(
                checkpoint, revision=revision
            ),
        )
        if needed > 0:
            return describe_insufficient_space(
                needed_bytes=needed,
                free_bytes_now=repo_free_bytes(checkpoint),
            )
    except Exception:  # noqa: BLE001 - the figures are a courtesy.
        logger.warning(
            "could not measure the shortfall for %s", checkpoint
        )
    return (
        "There is not enough disk space for this download. Free"
        " some space and try again."
    )


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
        # Drawn once per supervisor start. The activation number above
        # counts from zero again after a restart, so on its own it
        # would name a new worker the way it named an old one, and a
        # page left open across the restart could not tell them apart.
        self.instance: str = secrets.token_hex(4)
        self._proc: Optional[WorkerHandle] = None
        self._port: Optional[int] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # The machine-wide claim on having a model loaded. ``_lock``
        # above only orders this process's own activations; the two
        # launchers bind different ports on purpose, so without this a
        # browser and a desktop instance each hold their own manager
        # and each pass the VRAM pre-flight before the other's
        # allocation is visible. Held from the activation that claims
        # it until the worker is finalized.
        self._residency = PrimaryModelLease()
        # Download-only state (pre-fetch weights without loading into
        # VRAM). Independent of the worker, so it can run alongside a
        # resident model. States: idle | downloading | done | error.
        self.download_state: str = "idle"
        self.download_target: Optional[str] = None
        self.download_progress: Optional[Dict[str, Any]] = None
        self.download_error: Optional[str] = None
        # Held so the fire-and-forget download task is not collected
        # mid-run.
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

    def worker_identity(self) -> str:
        """Which worker this supervisor is serving, named so that no
        other activation, in this supervisor or a later start of it,
        is named the same way (`A2-LIFE-03`)."""
        assert self.activation_id > 0, "nothing has been activated"
        assert self.instance, "the supervisor start has no name"
        return f"{self.instance}:{self.activation_id}"

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
            # A switch, so the claim stays: this supervisor is
            # replacing its own worker, not giving the machine up.
            await self._stop_locked(keep_claim=True)
            # After the eviction, which looks wrong for a check that
            # can refuse, and is not. A refusal here can only happen
            # when this supervisor holds no claim, and holding no
            # claim means having no resident worker, so there was
            # nothing for the eviction to cost. A supervisor that does
            # have a model still owns the claim, kept through the
            # eviction above, and re-claiming it is a no-op that
            # records the model it is switching to.
            #
            # Before the pre-flight, though, which is the ordering
            # that matters: the pre-flight is precisely what cannot be
            # trusted here, because two supervisors both read free
            # VRAM before either allocation exists.
            self._claim_residency(model_id, device)
            try:
                await self._launch_locked(
                    info, model_id, device, python
                )
            except BaseException:
                # Nothing was left running, so nothing will finalize
                # and release on this supervisor's behalf. Holding a
                # claim with no worker behind it would lock the
                # machine out until this process exited.
                if self._proc is None:
                    self._residency.release()
                raise
            return self.activation_id

    async def _launch_locked(
        self,
        info: ModelInfo,
        model_id: str,
        device: str,
        python: Path,
    ) -> None:
        """Spawn the worker and record it. Called holding the lock.

        Split from ``activate`` so the claim taken just above it has
        one failure boundary rather than a release beside every raise
        between here and the spawn.
        """
        # CPU placement has no VRAM cost, so skip the GPU pre-flight
        # (which would otherwise block on nvidia-smi).
        if device != "cpu":
            await self._preflight_vram(info)
        port = self._free_port()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        lib_dirs = _venv_cuda_lib_dirs(python)
        if lib_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            parts = lib_dirs + ([existing] if existing else [])
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
        # Clears the previous failure, which `_finalize` keeps around
        # so the menu can explain a redirect. Trying again is the
        # moment it stops being news.
        self.load_error = None
        self.activation_id += 1
        self._monitor_task = asyncio.create_task(
            self._monitor_startup(proc, port)
        )

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
        try:
            relative = interpreter_for(info.environment)
        except UnknownEnvironmentError as exc:
            raise ActivationRefused(
                f"{info.display_name} runs in an environment this"
                f" build does not declare: {exc}"
            ) from exc
        python = REPO_ROOT / relative
        if not python.exists():
            raise ActivationRefused(
                f"{info.display_name} is not installed:"
                f" no interpreter at {relative}. Create it and"
                f" install {lock_for(info.environment)}."
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
        if not is_hub_checkpoint(info.checkpoint):
            path = Path(info.checkpoint).expanduser()
            if not path.is_dir():
                raise ActivationRefused(
                    f"{info.display_name} checkpoint not found"
                    f" at {path}."
                )
            _validate_local_artifact(info, path)
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
        once the worker is answering (loading/downloading), there is
        no wall-clock cap so long first-time downloads are not cut off
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
            await self._stop_locked(keep_claim=False)

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
        info = REGISTRY[model_id]
        checkpoint = info.checkpoint
        if not is_hub_checkpoint(checkpoint):
            raise ValueError(
                f"{model_id} is not downloadable from the Hub"
            )
        if self.download_state == "downloading":
            raise RuntimeError("a download is already running")
        handle = self._spawn(
            _download_argv(info),
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
            self._watch_download(
                checkpoint, handle, revision=info.revision
            )
        )
        return self.download_id

    async def _watch_download(
        self,
        checkpoint: str,
        handle: WorkerHandle,
        *,
        revision: Optional[str] = None,
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
            repo_total_bytes, checkpoint, revision=revision
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
        self._settle_download(
            checkpoint, code, revision=revision
        )

    def _settle_download(
        self,
        checkpoint: str,
        code: Optional[int],
        *,
        revision: Optional[str] = None,
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
        if code == DOWNLOAD_EXIT_NO_SPACE:
            # Rebuilt rather than relayed, like the message above: the
            # child reports an exit status and nothing else, so its
            # figures are gone. Measured again here, which is honest
            # anyway, since the answer may have changed since.
            self.download_error = _describe_no_space(
                checkpoint, revision
            )
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
        """Clear a finished pre-fetch, so its notice fires once.

        Resets only a terminal state (done/error) back to idle; a
        no-op while a download is still running. Called when the user
        acknowledges the veneer's "Ok" (or dismisses the toast), so
        the cross-page download toast and re-attach do not keep
        re-firing.
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
            await self._stop_locked(keep_claim=False)

    def _claim_residency(self, model_id: str, device: str) -> None:
        """Take the machine-wide claim, or refuse and say who has it.

        A supervisor that already holds it is switching models rather
        than competing with itself, which the lease treats as a no-op
        and a rewrite of the recorded model.

        The note names the launcher rather than only the pid, because
        "the desktop app has a model loaded" is something a person can
        act on and "pid 4001" is something they then have to look up.
        """
        owner = {
            "pid": os.getpid(),
            "launcher": Path(sys.argv[0]).name or "python",
            "model": model_id,
            "device": device,
        }
        try:
            granted = self._residency.acquire(owner)
        except LeaseUnavailable as exc:
            raise ActivationRefused(
                _lease_unavailable_refusal(exc)
            ) from exc
        if granted:
            return
        raise ActivationRefused(_residency_refusal(self._residency))

    async def _stop_locked(self, *, keep_claim: bool) -> None:
        """Stop the resident worker and prove it is gone.

        The monitor is cancelled first because this is not the
        monitor calling; a failure detected inside the monitor takes
        the same finalization without that step (see
        ``_monitor_startup``), since a task cannot await its own
        cancellation.

        ``keep_claim`` is for a switch, which replaces this
        supervisor's worker rather than giving the machine up. Every
        other stop leaves nothing resident, and releases.
        """
        await self._cancel_monitor()
        await self._finalize(
            self._proc, error=None, keep_claim=keep_claim
        )

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
        self,
        handle: Optional[WorkerHandle],
        *,
        error: Optional[str],
        keep_claim: bool = False,
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
        # Behind the supersede guard above, and for the same reason
        # the fields are: by the time a slow termination finishes, a
        # newer activation may already hold the claim, and releasing
        # here would hand the machine away while this supervisor still
        # has a worker coming up.
        #
        # Kept through a switch. Released here and taken back in
        # `activate`, the claim was free for the instant between, and
        # a peer arriving then got the machine after this supervisor
        # had already evicted its model for the one it was switching
        # to (`A2-LIFE-01`).
        if not keep_claim:
            self._residency.release()
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
