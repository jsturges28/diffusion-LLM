"""A dead renderer revives the window instead of whitening it.

Strategy: drive the launcher's watch with a fake page and view that
carry the one signal that matters, so the recovery logic is exercised
without Qt, a display, or a real crash. The sandbox has neither a
display nor a GPU, and the real trigger (a machine idling until the
screen blanks, taking the GPU context with it) cannot be staged in a
test at all, so the seam is drawn at the signal: everything above it
is checked here, and the wiring below it is a manual item.

Chromium runs the page in its own process. When that process dies,
QtWebEngine leaves the view blank and says nothing, and pywebview
does not connect the signal that reports it, so the window sits white
until the app is restarted. That is the defect this covers.

Passing proves a termination reloads the view, that the reload is
bounded so a crash loop cannot spin forever, that every termination
is written somewhere a user who launched from a desktop entry can
still read it, and that a backend which cannot report this degrades
to doing nothing rather than to failing.
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

    def __init__(self, page: Optional[_Page] = None) -> None:
        self._page = page if page is not None else _Page()
        self.reloads = 0

    def page(self) -> _Page:
        return self._page

    def reload(self) -> None:
        self.reloads += 1


class _Window:
    uid = "master"


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
def view(monkeypatch: pytest.MonkeyPatch) -> _View:
    """A window whose Qt view the launcher can find."""
    stub = _View()
    monkeypatch.setattr(
        desktop, "_renderer_view", lambda window: stub
    )
    return stub


def _log_lines(root: Path) -> List[str]:
    path = root / desktop.RENDERER_LOG_NAME
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


# -- the recovery itself --


def test_a_dead_renderer_reloads_the_view(
    storage: Path, view: _View
) -> None:
    window = _Window()
    assert desktop.watch_renderer(window) is True

    view.page().renderProcessTerminated.emit(2, 139)

    assert view.reloads == 1


def test_a_normal_exit_reloads_too(
    storage: Path, view: _View
) -> None:
    """Deliberately not filtered by status. Whatever the enum says,
    the user is looking at a blank window, and the only question
    worth asking is whether the page came back."""
    desktop.watch_renderer(_Window())

    view.page().renderProcessTerminated.emit(0, 0)

    assert view.reloads == 1


def test_the_reload_is_bounded(
    storage: Path, view: _View
) -> None:
    """A death on resume is a fact of the host and reviving it is
    invisible. A death that repeats is a bug, and a window that
    reloads forever hides it."""
    desktop.watch_renderer(_Window())

    for _ in range(desktop.RENDERER_RELOAD_MAX + 4):
        view.page().renderProcessTerminated.emit(2, 139)

    assert view.reloads == desktop.RENDERER_RELOAD_MAX


def test_the_cap_degrades_to_the_old_behaviour(
    storage: Path, view: _View
) -> None:
    """Past the cap the window is blank and stays blank, which is
    what it did before any of this existed. The log is what makes
    that a decision rather than a silence."""
    desktop.watch_renderer(_Window())
    for _ in range(desktop.RENDERER_RELOAD_MAX + 1):
        view.page().renderProcessTerminated.emit(2, 139)

    assert "giving up" in _log_lines(storage)[-1]


def test_each_watch_counts_on_its_own(
    storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget belongs to a watch, not to the process, so a
    second window does not open with the first one's already spent."""
    first = _View()
    monkeypatch.setattr(
        desktop, "_renderer_view", lambda window: first
    )
    desktop.watch_renderer(_Window())
    for _ in range(desktop.RENDERER_RELOAD_MAX + 1):
        first.page().renderProcessTerminated.emit(2, 139)

    second = _View()
    monkeypatch.setattr(
        desktop, "_renderer_view", lambda window: second
    )
    desktop.watch_renderer(_Window())
    second.page().renderProcessTerminated.emit(2, 139)

    assert first.reloads == desktop.RENDERER_RELOAD_MAX
    assert second.reloads == 1


# -- the evidence it leaves --


def test_a_termination_is_written_down(
    storage: Path, view: _View
) -> None:
    """The app is launched from a desktop entry, so stderr goes
    nowhere a user can find. Without the file the only evidence of a
    crash is a white window and a memory of roughly when."""
    desktop.watch_renderer(_Window())

    view.page().renderProcessTerminated.emit(2, 139)

    lines = _log_lines(storage)
    assert len(lines) == 1
    assert "crashed" in lines[0]
    assert "139" in lines[0]


def test_the_log_accumulates_rather_than_replacing(
    storage: Path, view: _View
) -> None:
    """A single overwritten line would lose the pattern, and the
    pattern is the whole diagnostic value: once is the host, nightly
    is a bug."""
    desktop.watch_renderer(_Window())

    view.page().renderProcessTerminated.emit(2, 139)
    view.page().renderProcessTerminated.emit(3, 9)

    assert len(_log_lines(storage)) == 2


def test_a_reload_says_which_of_how_many(
    storage: Path, view: _View
) -> None:
    desktop.watch_renderer(_Window())

    view.page().renderProcessTerminated.emit(2, 139)

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


def test_an_unknown_status_still_says_something() -> None:
    """Qt may add one. A KeyError inside a crash handler would turn
    a recoverable blank window into an unrecoverable one."""
    reason = desktop._termination_reason(99, 7)

    assert "99" in reason
    assert "7" in reason


def test_a_status_that_is_not_a_number_survives() -> None:
    """PyQt hands over an enum member, not an int, and a build that
    refuses int() must not take the handler down with it."""
    reason = desktop._termination_reason(object(), 7)

    assert reason


def test_an_unwritable_log_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, view: _View
) -> None:
    """Recording the crash is the least important thing happening at
    that moment. Reloading the window is the most."""
    monkeypatch.setattr(
        desktop,
        "_persistent_storage_path",
        lambda: Path("/proc/nonexistent/nope"),
    )
    desktop.watch_renderer(_Window())

    view.page().renderProcessTerminated.emit(2, 139)

    assert view.reloads == 1


# -- backends that cannot report it --


def test_a_backend_without_the_signal_is_declined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QtWebKit and GTK/WebKit have no such signal. Saying so is the
    honest answer; pretending to watch would be worse."""

    class _Bare:
        def page(self) -> object:
            return object()

    monkeypatch.setattr(
        desktop, "_renderer_view", lambda window: _Bare()
    )
    monkeypatch.setattr(desktop, "RENDERER_ATTACH_TIMEOUT_S", 0.2)

    assert desktop.watch_renderer(_Window()) is False


def test_a_window_that_never_appears_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded, because this runs on a thread that would otherwise
    poll for the life of the process."""
    monkeypatch.setattr(
        desktop, "_renderer_view", lambda window: None
    )
    monkeypatch.setattr(desktop, "RENDERER_ATTACH_TIMEOUT_S", 0.2)

    assert desktop.watch_renderer(_Window()) is False


def test_no_window_means_no_start_kwarg() -> None:
    """The browser path builds no window, and passing a watch for
    one that does not exist would run it against None."""
    assert desktop._renderer_watch_kwargs(None) == {}


def test_a_window_installs_the_watch_at_start() -> None:
    assert "func" in desktop._renderer_watch_kwargs(_Window())


def test_a_failing_watch_never_blocks_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pywebview's internals are not a public API, so this can break
    on an upgrade. Trading a rare white screen for an app that will
    not open would be the worse bargain."""

    def explode(window: Any) -> bool:
        raise RuntimeError("pywebview moved the view")

    monkeypatch.setattr(desktop, "watch_renderer", explode)
    install = desktop._renderer_watch_kwargs(_Window())["func"]

    install()
