"""A page whose worker was replaced says so and reloads.

Strategy: source inspection of `app.js` and `menu.js`, the approach
this repo uses for its two classic-script pages; giving them a
testable seam is `ORG-02` in stage 5. The two halves that can be
executed are: the supervisor's side in
`tests/web/test_activation_identity.py`, and the shared client's
operation filtering in
`tests/web/static/activation_client.test.js`.

What passing proves is the last clause of `LIFE-03`'s Verification,
the one about reconnecting. A generator caches its model, device,
capability gates and entire parameter form at boot and refreshes them
only by reloading. Another window can replace the resident worker
without that reload ever happening, and then a Generate from this
page is labelled and parameterised for one model while a different
one answers it, frequently accepted through defaults rather than
refused. That is a wrong answer that looks right, which is the worst
kind this app can produce.

The rescue is the other half. The run cannot be continued once its
worker is gone, since resume, What If and probe all read state that
died with the process, but it can still be saved, so it is.
"""

from __future__ import annotations

from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
MENU_JS = STATIC / "menu.js"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _region(path: Path, anchor: str, chars: int) -> str:
    source = _source(path)
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {path.name}; update this"
        " test rather than deleting it"
    )
    return source[start : start + chars]


# -- the page notices --


def test_the_resident_frame_is_handled() -> None:
    region = _region(APP_JS, "function handleMessage(data)", 700)

    assert '"resident"' in region
    assert "handleResident(data)" in region


def test_the_page_compares_both_model_and_device() -> None:
    """The same model on the other device is a different worker with
    its own output, which is why the switch path has always treated
    it as a switch."""
    region = _region(APP_JS, "function handleResident(data)", 1500)

    assert "data.model === activeModelId" in region
    assert "data.device === activeDevice" in region


def test_a_matching_resident_does_nothing() -> None:
    """The overwhelmingly common case. Every socket open sends this
    frame, so a page that acted on it every time would reload
    itself in a loop."""
    region = _region(APP_JS, "function handleResident(data)", 1500)

    guard = region.find("if (sameModel && sameDevice)")
    rescue = region.find("rescueRunThenReload()")

    assert guard != -1
    assert rescue != -1
    assert guard < rescue


def test_a_mismatch_stops_generation_being_possible() -> None:
    """Before the reload, not after it. The reload is asynchronous
    once a rescue save is in flight, and this page must not be able
    to send work to a model it was not built for in the meantime."""
    region = _region(APP_JS, "function handleResident(data)", 1500)

    assert "modelReady = false" in region
    assert "updateGenerateButton()" in region


def test_a_mismatch_stops_the_reconnect_loop() -> None:
    """Otherwise the socket's own retry pulls the page back onto the
    new worker as though it belonged there, racing the reload."""
    region = _region(APP_JS, "function handleResident(data)", 1500)

    assert "suppressReconnect = true" in region


def test_the_page_says_what_happened() -> None:
    """A page that reloads itself with no explanation is
    indistinguishable from a crash."""
    region = _region(APP_JS, "function handleResident(data)", 1500)

    assert "in another window" in region


# -- the rescue --


def test_an_unsaved_run_is_saved_before_the_reload() -> None:
    region = _region(
        APP_JS, "function rescueRunThenReload()", 1200
    )

    save = region.find("saveRun()")
    reload_call = region.find("location.reload()", save)

    assert save != -1
    assert reload_call != -1


def test_a_saved_run_reloads_without_saving_again() -> None:
    """`runSaved` is the guard, so a mismatch after a normal save
    does not file a duplicate."""
    region = _region(
        APP_JS, "function rescueRunThenReload()", 1200
    )

    guard = region.find("runSaved")
    save = region.find("saveRun()")

    assert guard != -1
    assert guard < save


def test_a_hung_save_cannot_strand_the_page() -> None:
    """The page is describing a worker that no longer exists, so it
    cannot wait indefinitely on the chance a request completes."""
    region = _region(
        APP_JS, "function rescueRunThenReload()", 1200
    )

    assert "Promise.race" in region
    assert "RESCUE_SAVE_TIMEOUT_MS" in region


def test_a_failed_save_still_reloads() -> None:
    """Both arms of the race's then(), because a rejected save must
    not leave the page pointed at a model that is gone."""
    region = _region(
        APP_JS, "function rescueRunThenReload()", 1200
    )

    assert region.count("location.reload()") >= 2


def test_save_hands_back_a_promise() -> None:
    """What makes the rescue orderable at all. Its other callers
    ignore the result, which is why returning it is safe."""
    region = _region(APP_JS, "function saveRun()", 900)

    assert "Promise.resolve()" in region
    assert "return fetch(\"/api/save\"" in _source(APP_JS)


# -- the menu's cancel --


def test_the_menu_cancels_through_the_client() -> None:
    """So the operation this window started travels with the
    request, instead of stopping whatever happens to be loading."""
    region = _region(MENU_JS, "function cancelSelection()", 1200)

    assert "watch.cancel()" in region


def test_the_menu_takes_the_watch_before_clearing_it() -> None:
    """`finishSelecting` drops the watch, and the watch is what
    knows which activation this window owns."""
    region = _region(MENU_JS, "function cancelSelection()", 1200)

    grab = region.find("var watch = activationWatch")
    finish = region.find("finishSelecting()")

    assert grab != -1
    assert finish != -1
    assert grab < finish


def test_a_refused_cancel_is_shown_not_swallowed() -> None:
    """The server refuses a cancel for another window's load. A
    button that silently does nothing reads as broken."""
    region = _region(MENU_JS, "function cancelSelection()", 1200)

    assert "showError(" in region
