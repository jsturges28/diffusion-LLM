"""Desktop launcher for the Diffusion LLM Visualizer.

Wraps the existing FastAPI supervisor in a native window (pywebview)
instead of a browser tab, and owns the server lifecycle: it starts
uvicorn on a background thread bound to localhost on a stable port
(see ``DESKTOP_PORT``), opens the window against it (at ``/``, the Main
Menu, from which a model is selected before the generator page), and on
window close
signals a graceful shutdown so the model-worker subprocesses (and
their VRAM) are released through the supervisor's existing shutdown
hook.

Run ``.venv/bin/python desktop.py``. The browser path
(``python main.py``) is unchanged and wraps the same server, so there
is a single source of truth for the backend and frontend.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent

# Pin the working directory to the repo root so the launcher behaves
# the same from a .desktop entry with an arbitrary one. The data root
# no longer depends on this (the supervisor resolves it absolutely,
# see src/web/data_root.py), but the worker subprocesses are still
# spawned relative to the process cwd, so this stays.
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402  (imported after cwd/sys.path setup)
import webview  # noqa: E402

from src.web.server import APP_IDENTITY, app  # noqa: E402

WINDOW_TITLE = "LLM Visualizer"
APP_ID = "llm-xai-visualizer"
HOST = "127.0.0.1"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860
STARTUP_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 35.0
# Prefer the rasterized PNG for window-icon fidelity (some webview
# backends render an SVG window icon poorly), falling back to the SVG
# source until the PNG is generated. Regenerate the PNG after editing
# the icon with: .venv/bin/python scripts/render_icon.py
ICON_SVG_PATH = REPO_ROOT / "assets" / "icon.svg"
ICON_PNG_PATH = REPO_ROOT / "assets" / "icon.png"
ICON_PATH = ICON_PNG_PATH if ICON_PNG_PATH.is_file() else ICON_SVG_PATH

# A fixed localhost port keeps the desktop window's origin
# (scheme://host:port) stable across launches. Web storage (localStorage:
# Settings, prompt history, the analytics "new run" cue) is partitioned
# per origin, so an ephemeral port would hand each launch a fresh, empty
# partition and silently defeat persistence even with a persistent
# profile. Distinct from main.py's default 8000 so the browser and the
# desktop app can run side by side without colliding.
DESKTOP_PORT = 8760


def _free_port() -> int:
    """Reserve an ephemeral localhost port for the supervisor."""
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        port = int(probe.getsockname()[1])
    assert port > 0, "failed to acquire an ephemeral port"
    return port


def _port_available(port: int) -> bool:
    """Return whether `port` can be bound on the loopback host now."""
    assert 0 < port < 65536, "port must be in the valid range"
    with socket.socket() as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


# How long to wait for whatever holds the port to identify itself.
# Short on purpose: this runs while the user is waiting for a window,
# and something listening that will not answer promptly is, for this
# decision, indistinguishable from something that is not ours.
IDENTITY_PROBE_TIMEOUT_S = 1.5


def probe_supervisor(
    port: int, *, timeout_seconds: float = IDENTITY_PROBE_TIMEOUT_S
) -> Optional[Dict[str, Any]]:
    """Ask whatever is on ``port`` whether it is one of ours.

    Returns its identity payload, or None for a port that is empty,
    unresponsive, or held by something else. "Something else" is the
    case that makes this necessary: a failed bind alone cannot tell a
    second copy of this app from an unrelated process, and the two
    want opposite responses.
    """
    assert 0 < port < 65536, "port must be in the valid range"
    url = f"http://{HOST}:{port}/api/app"
    try:
        with urllib.request.urlopen(
            url, timeout=timeout_seconds
        ) as response:
            if response.status != 200:
                return None
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        # Refused, timed out, not HTTP, or not JSON. All of them mean
        # the same thing here: nothing of ours answered.
        return None
    if not isinstance(body, dict):
        return None
    if body.get("app") != APP_IDENTITY:
        return None
    return body


def find_running_instance(
    port: int = DESKTOP_PORT,
) -> Optional[Dict[str, Any]]:
    """The already-running copy of this app, if there is one.

    The bind test comes first because it is local and instant, and
    answers the common case (nothing is there) without a network
    round trip on every launch.
    """
    if _port_available(port):
        return None
    return probe_supervisor(port)


def focus_running_window() -> bool:
    """Best-effort raise of the window that is already open.

    Deliberately best-effort, and the caller must not depend on it.
    Activating another process's window is the window manager's to
    allow, and under Wayland these tools generally cannot. The
    guarantee this whole path provides is that a second supervisor is
    not started; raising the first one is a courtesy on top, and the
    printed message below is the fallback that always works.
    """
    attempts = (
        ["wmctrl", "-x", "-a", APP_ID],
        ["xdotool", "search", "--class", APP_ID,
         "windowactivate"],
    )
    for command in attempts:
        if shutil.which(command[0]) is None:
            continue
        try:
            done = subprocess.run(
                command,
                timeout=2.0,
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return True
    return False


def _resolve_port() -> int:
    """Prefer the stable desktop port; fall back to an ephemeral one.

    A stable origin is what lets localStorage survive restarts (see
    ``DESKTOP_PORT``). If the fixed port is already taken (a second
    instance, or an unrelated process), degrade to an ephemeral port so
    the app still launches. Web storage will not carry over for that one
    launch, which is a better failure than refusing to start.
    """
    if _port_available(DESKTOP_PORT):
        return DESKTOP_PORT
    fallback = _free_port()
    print(
        "[desktop] preferred port "
        + str(DESKTOP_PORT)
        + " is in use; falling back to ephemeral port "
        + str(fallback)
        + " (web storage will not persist this launch).",
        file=sys.stderr,
    )
    return fallback


def _wait_until_started(
    server: uvicorn.Server,
    timeout_seconds: float,
) -> None:
    """Block until uvicorn is serving, or raise on timeout."""
    assert timeout_seconds > 0, "timeout must be positive"
    deadline = time.monotonic() + timeout_seconds
    while not server.started:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "supervisor did not start within"
                f" {timeout_seconds:.0f}s"
            )
        time.sleep(0.05)


def _set_app_identity(gui: Optional[str]) -> None:
    """Tell the window manager which .desktop this window belongs to, so
    the dock shows the running indicator and re-activates the existing
    window instead of launching a second instance.

    GTK derives its Wayland app_id / WM_CLASS from the program name; Qt
    from ``QGuiApplication.desktopFileName``. Both are best-effort and
    no-op when that binding is absent.
    """
    try:
        from gi.repository import GLib
        GLib.set_prgname(APP_ID)
        GLib.set_application_name(WINDOW_TITLE)
    except ImportError:
        pass  # No GTK binding present.
    if gui == "qt":
        _set_qt_desktop_name()


def _set_qt_desktop_name() -> None:
    """Match the Qt window's Wayland app_id to the installed .desktop
    id (``llm-xai-visualizer``). Must run before the QApplication is
    created (i.e. before webview.start), which it does from main().
    """
    for binding in _QT_BINDINGS:
        try:
            qtgui = importlib.import_module(binding + ".QtGui")
        except ImportError:
            continue
        try:
            qtgui.QGuiApplication.setDesktopFileName(APP_ID)
        except Exception:  # noqa: BLE001 - best-effort identity hint
            pass
        return


# Qt bindings pywebview can drive its QtWebEngine (Chromium) backend
# through. Chromium scrolls/composites more smoothly than the GTK
# WebKit backend, so it is preferred when available.
_QT_BINDINGS = ("PySide6", "PyQt6", "PySide2", "PyQt5")


def _select_gui() -> Optional[str]:
    """Prefer the Qt (QtWebEngine/Chromium) backend when a Qt binding
    is installed; return None to let pywebview auto-detect (GTK).
    """
    for binding in _QT_BINDINGS:
        try:
            if importlib.util.find_spec(binding) is not None:
                return "qt"
        except (ImportError, ValueError):
            continue
    return None


def _persistent_storage_path() -> Path:
    """Per-user data dir for web storage that must survive restarts.

    Respects ``XDG_DATA_HOME`` on Linux, else ``~/.local/share``.
    """
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_ID


def _window_start_kwargs() -> dict:
    """Optional ``webview.start`` kwargs, each gated on this pywebview
    build's support (the arguments are backend- and version-dependent).

    - ``icon``: app window icon. The app-menu launcher icon comes from
      the .desktop entry regardless (see install_desktop_entry.sh).
    - ``private_mode=False`` + ``storage_path``: persist web storage
      (localStorage: Settings, prompt history, the analytics "new run"
      cue) across app restarts. pywebview otherwise defaults to a
      private, off-the-record profile that is cleared on close, so those
      would reset every launch.
    """
    params = inspect.signature(webview.start).parameters
    kwargs: dict = {}
    if "icon" in params and ICON_PATH.is_file():
        kwargs["icon"] = str(ICON_PATH)
    if "private_mode" in params:
        kwargs["private_mode"] = False
    if "storage_path" in params:
        storage = _persistent_storage_path()
        try:
            storage.mkdir(parents=True, exist_ok=True)
            kwargs["storage_path"] = str(storage)
        except OSError:
            # Fall back to pywebview's default persistent location
            # (still non-private since private_mode is False).
            pass
    return kwargs


# Chromium runs the page in a process of its own, and when that
# process dies QtWebEngine leaves the view blank. There is no error,
# no event the page can see, and nothing in any log: pywebview does
# not connect the signal that reports it (checked against 6.2.1), so
# the window simply sits white until the app is restarted.
#
# Observed here after the machine idles and the screen blanks or
# locks, which fits a GPU context lost to suspend that the renderer
# does not survive. Nothing the page does can prevent that, so the
# window recovers from it instead.
RENDERER_RELOAD_MAX = 3
RENDERER_LOG_NAME = "renderer-crashes.log"
# Set this to anything to launch without the watch. An escape hatch,
# because this hooks a private part of pywebview: if a future version
# moves what it hooks, the app must still be startable without an
# edit to this file.
RENDERER_WATCH_OFF_ENV = "LLM_VISUALIZER_NO_RENDERER_WATCH"

# QWebEnginePage.RenderProcessTerminationStatus. Spelled out rather
# than imported so this file stays free of a Qt import on the GTK
# path, where none of it applies.
_TERMINATION_NAMES = {
    0: "exited normally",
    1: "exited abnormally",
    2: "crashed",
    3: "was killed",
}


def _termination_reason(status: Any, exit_code: int) -> str:
    """A readable phrase for QtWebEngine's termination enum.

    The value is read off the member before falling back to the
    object itself, because PyQt6 hands over an enum member and
    ``int()`` on one raises rather than converting. Reading it wrong
    is only cosmetic, but a log nobody can read is the failure mode
    this whole file exists to avoid: the first version wrote
    "RenderProcessTerminationStatus.CrashedTerminationStatus".
    """
    try:
        code = int(getattr(status, "value", status))
    except (TypeError, ValueError):
        return f"ended ({status}, exit code {exit_code})"
    name = _TERMINATION_NAMES.get(code, f"ended with status {code}")
    return f"{name} (exit code {exit_code})"


def record_renderer_death(reason: str, action: str) -> str:
    """Write the crash somewhere it can be read later.

    A file rather than only stderr, because the app is normally
    launched from a desktop entry and anything printed then goes
    nowhere a user can find. Without this the sole evidence of a
    crash is a white window and a memory of roughly when.
    """
    assert reason, "a termination needs a reason"
    assert action, "say what was done about it"
    line = (
        time.strftime("%Y-%m-%d %H:%M:%S")
        + f"  renderer {reason}; {action}"
    )
    print("[desktop] " + line, file=sys.stderr)
    try:
        directory = _persistent_storage_path()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / RENDERER_LOG_NAME).open(
            "a", encoding="utf-8"
        ) as log:
            log.write(line + "\n")
    except OSError:
        pass  # Best effort. The stderr line above still happened.
    return line


def watch_renderer(browser: Any) -> bool:
    """Reload the view when Chromium's renderer dies under it.

    Takes pywebview's Qt ``BrowserView`` and returns whether the
    watch went on, which is false for any backend that cannot report
    this.

    **Must run on the GUI thread.** Qt objects belong to the thread
    that created them, and the caller below arranges for this to run
    where the view was made. An earlier version ran it on
    pywebview's post-start worker thread on the theory that
    connecting a signal across threads is harmless. It is not: the
    ``page()`` call alone parents a QWebEnginePage to a view owned by
    another thread, and Qt aborts the process over it, which turned
    a rare white window into a launch that never came up at all.
    """
    view = getattr(browser, "webview", None)
    page = view.page() if view is not None else None
    if page is None:
        return False
    if not hasattr(page, "renderProcessTerminated"):
        return False  # QtWebKit, which cannot report this.
    reloads = 0

    def on_terminated(status: Any, exit_code: int) -> None:
        nonlocal reloads
        reason = _termination_reason(status, exit_code)
        # Bounded on purpose. One death on resume is a fact of the
        # host and reviving it is invisible; a death that repeats is
        # a bug, and a window that reloads forever hides it. At the
        # cap the app degrades to what it did before this existed.
        if reloads >= RENDERER_RELOAD_MAX:
            record_renderer_death(
                reason, "giving up, restart the app"
            )
            return
        reloads += 1
        record_renderer_death(
            reason,
            f"reloading ({reloads} of {RENDERER_RELOAD_MAX})",
        )
        view.reload()

    page.renderProcessTerminated.connect(on_terminated)
    return True


def install_renderer_watch() -> bool:
    """Arrange for the watch to go on where Qt allows it.

    Wraps the Qt backend's own constructor, which is a private part
    of pywebview and hooked deliberately: ``BrowserView`` is built on
    the GUI thread and has set its page by the time ``__init__``
    returns, so a wrapper around it runs at the one moment that is
    both late enough to have a page and on the right thread to touch
    it. pywebview's documented post-start callback satisfies only the
    first of those, which is what made the first attempt abort at
    launch.

    Returns whether the wrap went on. Called before ``webview.start``
    and never after, since it only takes effect for windows built
    afterwards.
    """
    if os.environ.get(RENDERER_WATCH_OFF_ENV):
        print(
            "[desktop] renderer watch disabled by "
            + RENDERER_WATCH_OFF_ENV,
            file=sys.stderr,
        )
        return False
    try:
        from webview.platforms.qt import BrowserView
    except ImportError:
        return False  # GTK backend: no such signal to watch.
    if getattr(BrowserView, "_renderer_watch_installed", False):
        return True
    original = BrowserView.__init__

    def __init__(self: Any, window: Any) -> None:
        original(self, window)
        try:
            watch_renderer(self)
        except Exception as exc:  # noqa: BLE001 - never block launch
            print(
                f"[desktop] renderer watch unavailable ({exc});"
                " a crashed renderer will leave a blank window.",
                file=sys.stderr,
            )

    BrowserView.__init__ = __init__
    BrowserView._renderer_watch_installed = True
    return True


def _start_window(gui: Optional[str]) -> None:
    """Open the window with the given backend preference, gracefully
    falling back to GTK/auto if Qt is unavailable at runtime (e.g. a
    Qt binding is installed but QtWebEngine or its system libs are
    missing). Blocks on the main thread until the window closes.
    """
    start_kwargs = _window_start_kwargs()
    if gui is None:
        webview.start(**start_kwargs)
        return
    try:
        webview.start(gui=gui, **start_kwargs)
    except Exception as exc:  # noqa: BLE001 - optional-backend fallback
        # Qt was preferred but failed to initialize; retry with the
        # default (GTK/WebKit) backend so the app still opens instead
        # of crashing.
        print(
            f"desktop: Qt backend unavailable ({exc}); falling"
            " back to GTK/WebKit.",
            file=sys.stderr,
        )
        webview.start(**start_kwargs)


def main() -> None:
    # Before anything else, and before any GUI library is touched.
    # A second launch used to start a second supervisor on a fallback
    # port, and two supervisors each enforce "one resident model"
    # over a GPU neither knows it shares: both spawn a worker, and
    # the second one dies of CUDA out-of-memory after the user has
    # already waited out its load. Double-clicking a launcher twice
    # is not an exotic thing to do.
    running = find_running_instance()
    if running is not None:
        print(
            "desktop: already running (pid "
            + str(running.get("pid", "unknown"))
            + f") on port {DESKTOP_PORT}; focusing that window"
            " instead of starting a second copy.",
            file=sys.stderr,
        )
        focus_running_window()
        return
    gui = _select_gui()
    _set_app_identity(gui)
    port = _resolve_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app, host=HOST, port=port, log_level="info"
        )
    )
    # uvicorn only installs signal handlers on the main thread, so
    # running it off-thread is safe and leaves the main thread free
    # for the GUI-owning webview loop.
    thread = threading.Thread(
        target=server.run, name="supervisor", daemon=True
    )
    thread.start()
    try:
        _wait_until_started(server, STARTUP_TIMEOUT_SECONDS)
        # Before create_window, because the wrap only reaches windows
        # built after it goes on.
        if gui == "qt":
            install_renderer_watch()
        webview.create_window(
            WINDOW_TITLE,
            f"http://{HOST}:{port}/",
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
        )
        # Blocks on the main thread until the window is closed.
        _start_window(gui)
    finally:
        # Graceful stop -> supervisor shutdown hook -> workers freed.
        server.should_exit = True
        thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)


if __name__ == "__main__":
    main()
