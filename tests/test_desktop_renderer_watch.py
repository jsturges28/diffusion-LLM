"""A dead renderer revives the window instead of whitening it.

Strategy: drive the launcher's watch with a fake browser view that
carries the one signal that matters, so the recovery logic runs
without a display or a real crash, and check the thread-sensitive
wiring against a fake of pywebview's own class. The status decoder is
additionally checked against the real PyQt6 enum, because that is the
one place where a convincing fake was actively misleading.

Chromium runs the page in its own process. When that process dies,
QtWebEngine leaves the view blank and says nothing, and pywebview
does not connect the signal that reports it, so the window sits white
until the app is restarted. That is the defect this covers.

Two failures in the first attempt shaped this file, and both were
caused by testing a shape rather than the thing:

- The watch was installed from pywebview's post-start worker thread.
  Qt objects belong to the thread that made them, and reaching for
  ``page()`` from another one aborted the process at launch. No fake
  can show that, so the install now happens inside a wrapper around
  the backend's constructor, and what is tested here is that the
  wrapper is what goes on.
- The status decoder was tested with plain ints. PyQt6 hands over an
  enum member on which ``int()`` raises, so every test passed while
  the only line a user would read was the enum's repr.

Passing proves a termination reloads the view, that the reload is
bounded so a crash loop cannot spin forever, that every termination
is written where a desktop launch can still be read, that the status
reads as English for the type Qt actually sends, and that a backend
which cannot report this degrades to doing nothing rather than to
failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

import desktop


class _Signal:
    """The one Qt signal this depends on, and nothing else."""

    def __init__(self) -> None:
        self._slots: List[Callable[[Any, int], None]] = []

    def connect(self, slot: Callable[[Any, int], None]) -> None:
        self._slots.append(slot)

    def emit(self, status: Any, exit_code: int) -> None:
        assert self._slots, "nothing is watching the renderer"
        for slot in list(self._slots):
            slot(status, exit_code)


class _Page:
    def __init__(self) -> None:
        self.renderProcessTerminated = _Signal()


class _View:
    """A web view that counts how often it was revived."""

    def __init__(self) -> None:
        self._page = _Page()
        self.reloads = 0

    def page(self) -> _Page:
        return self._page

    def reload(self) -> None:
        self.reloads += 1


class _Browser:
    """Stands in for pywebview's Qt ``BrowserView``."""

    def __init__(self, view: Optional[_View] = None) -> None:
        self.webview = view if view is not None else _View()

    def crash(self, status: Any = 2, exit_code: int = 139) -> None:
        self.webview.page().renderProcessTerminated.emit(
            status, exit_code
        )

    @property
    def reloads(self) -> int:
        return self.webview.reloads


@pytest.fixture()
def storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the crash log at a temporary user data directory."""
    monkeypatch.setattr(
        desktop, "_persistent_storage_path", lambda: tmp_path
    )
    return tmp_path


@pytest.fixture()
def browser(storage: Path) -> _Browser:
    """A watched browser view, ready to be crashed."""
    view = _Browser()
    assert desktop.watch_renderer(view) is True
    return view


def _log_lines(root: Path) -> List[str]:
    path = root / desktop.RENDERER_LOG_NAME
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


# -- the recovery itself --


def test_a_dead_renderer_reloads_the_view(
    browser: _Browser
) -> None:
    browser.crash()

    assert browser.reloads == 1


def test_a_normal_exit_reloads_too(browser: _Browser) -> None:
    """Deliberately not filtered by status. Whatever the enum says,
    the user is looking at a blank window, and the only question
    worth asking is whether the page came back."""
    browser.crash(status=0, exit_code=0)

    assert browser.reloads == 1


def test_the_reload_is_bounded(browser: _Browser) -> None:
    """A death on resume is a fact of the host and reviving it is
    invisible. A death that repeats is a bug, and a window that
    reloads forever hides it."""
    for _ in range(desktop.RENDERER_RELOAD_MAX + 4):
        browser.crash()

    assert browser.reloads == desktop.RENDERER_RELOAD_MAX


def test_the_cap_degrades_to_the_old_behaviour(
    storage: Path, browser: _Browser
) -> None:
    """Past the cap the window is blank and stays blank, which is
    what it did before any of this existed. The log is what makes
    that a decision rather than a silence."""
    for _ in range(desktop.RENDERER_RELOAD_MAX + 1):
        browser.crash()

    assert "giving up" in _log_lines(storage)[-1]


def test_each_watch_counts_on_its_own(storage: Path) -> None:
    """The budget belongs to a watch, not to the process, so a
    second window does not open with the first one's already
    spent."""
    first = _Browser()
    desktop.watch_renderer(first)
    for _ in range(desktop.RENDERER_RELOAD_MAX + 1):
        first.crash()

    second = _Browser()
    desktop.watch_renderer(second)
    second.crash()

    assert first.reloads == desktop.RENDERER_RELOAD_MAX
    assert second.reloads == 1


# -- the evidence it leaves --


def test_a_termination_is_written_down(
    storage: Path, browser: _Browser
) -> None:
    """The app is launched from a desktop entry, so stderr goes
    nowhere a user can find. Without the file the only evidence of a
    crash is a white window and a memory of roughly when."""
    browser.crash()

    lines = _log_lines(storage)
    assert len(lines) == 1
    assert "crashed" in lines[0]
    assert "139" in lines[0]


def test_the_log_accumulates_rather_than_replacing(
    storage: Path, browser: _Browser
) -> None:
    """A single overwritten line would lose the pattern, and the
    pattern is the whole diagnostic value: once is the host, nightly
    is a bug."""
    browser.crash()
    browser.crash(status=3, exit_code=9)

    assert len(_log_lines(storage)) == 2


def test_a_reload_says_which_of_how_many(
    storage: Path, browser: _Browser
) -> None:
    browser.crash()

    assert "1 of" in _log_lines(storage)[0]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0, "exited normally"),
        (1, "exited abnormally"),
        (2, "crashed"),
        (3, "was killed"),
    ],
)
def test_every_status_reads_as_english(
    status: int, expected: str
) -> None:
    assert expected in desktop._termination_reason(status, 0)


def test_the_real_qt_enum_reads_as_english() -> None:
    """The shape that actually arrives, which the ints above do not
    stand in for.

    PyQt6 hands over an enum member and ``int()`` on one raises
    rather than converting, so the first version fell through to its
    own fallback and logged the repr. Every int-based test passed
    while the only line a user would ever read was unreadable, which
    is why this asserts against the real type.
    """
    core = pytest.importorskip("PyQt6.QtWebEngineCore")
    status = (
        core.QWebEnginePage.RenderProcessTerminationStatus
        .CrashedTerminationStatus
    )

    reason = desktop._termination_reason(status, 139)

    assert "crashed" in reason
    assert "RenderProcessTerminationStatus" not in reason


def test_an_unknown_status_still_says_something() -> None:
    """Qt may add one. A KeyError inside a crash handler would turn
    a recoverable blank window into an unrecoverable one."""
    reason = desktop._termination_reason(99, 7)

    assert "99" in reason
    assert "7" in reason


def test_a_status_that_is_not_a_number_survives() -> None:
    """Whatever a future binding hands over, a raise inside a crash
    handler would turn a recoverable blank window into a permanent
    one."""
    assert desktop._termination_reason(object(), 7)


def test_an_unwritable_log_does_not_raise(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording the crash is the least important thing happening at
    that moment. Reloading the window is the most."""
    monkeypatch.setattr(
        desktop,
        "_persistent_storage_path",
        lambda: Path("/proc/nonexistent/nope"),
    )
    browser = _Browser()
    desktop.watch_renderer(browser)

    browser.crash()

    assert browser.reloads == 1


# -- backends that cannot report it --


def test_a_backend_without_the_signal_is_declined() -> None:
    """QtWebKit and GTK/WebKit have no such signal. Saying so is the
    honest answer; pretending to watch would be worse."""

    class _Bare:
        def page(self) -> object:
            return object()

    class _Old:
        webview = _Bare()

    assert desktop.watch_renderer(_Old()) is False


def test_a_view_that_is_not_there_is_declined() -> None:
    class _Empty:
        webview = None

    assert desktop.watch_renderer(_Empty()) is False


# -- where it is installed, which is the thread-sensitive part --


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A stand-in for pywebview's Qt backend class.

    Faked rather than patched onto the real one, because installing
    the watch mutates the class for the whole process and a test
    that leaked that would change how the app behaves for every test
    after it.
    """

    class FakeBrowserView:
        built: List[Any] = []

        def __init__(self, window: Any) -> None:
            self.webview = _View()
            FakeBrowserView.built.append(window)

    monkeypatch.setattr(
        "webview.platforms.qt.BrowserView", FakeBrowserView
    )
    return FakeBrowserView


def test_the_watch_goes_on_the_backend_constructor(
    fake_backend: Any
) -> None:
    """Not on pywebview's post-start callback, which was the first
    attempt: that runs on a worker thread, and touching a Qt view
    from one aborts the process at launch. The constructor runs on
    the GUI thread and has set its page by the time it returns.
    """
    original = fake_backend.__init__

    assert desktop.install_renderer_watch() is True
    assert fake_backend.__init__ is not original


def test_a_built_window_is_watched(
    storage: Path, fake_backend: Any
) -> None:
    """The wrap has to actually connect, not merely replace."""
    desktop.install_renderer_watch()

    built = fake_backend(object())
    built.webview.page().renderProcessTerminated.emit(2, 139)

    assert built.webview.reloads == 1


def test_the_original_constructor_still_runs(
    storage: Path, fake_backend: Any
) -> None:
    """A wrapper that swallowed the real __init__ would leave a
    window with no view at all."""
    desktop.install_renderer_watch()
    window = object()

    built = fake_backend(window)

    assert fake_backend.built == [window]
    assert built.webview is not None


def test_installing_twice_wraps_once(fake_backend: Any) -> None:
    """main() calls this once, but a wrap on a wrap would file two
    watches per window and halve the reload budget."""
    desktop.install_renderer_watch()
    once = fake_backend.__init__

    assert desktop.install_renderer_watch() is True
    assert fake_backend.__init__ is once


def test_a_failing_watch_never_blocks_the_window(
    fake_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pywebview's internals are not a public API, so this can break
    on an upgrade. Trading a rare white screen for an app that will
    not open would be the worse bargain, and that is not
    hypothetical: it is what the first attempt did."""

    def explode(browser: Any) -> bool:
        raise RuntimeError("pywebview moved the view")

    monkeypatch.setattr(desktop, "watch_renderer", explode)
    desktop.install_renderer_watch()

    built = fake_backend(object())

    assert built.webview is not None


def test_the_environment_can_switch_it_off(
    fake_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An escape hatch, because this hooks a private part of
    pywebview: a future version that breaks it must still leave the
    app startable without editing the source."""
    monkeypatch.setenv(desktop.RENDERER_WATCH_OFF_ENV, "1")
    original = fake_backend.__init__

    assert desktop.install_renderer_watch() is False
    assert fake_backend.__init__ is original
