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
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

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

from src.web.server import app  # noqa: E402

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
