"""One download watcher per page, and one place that can stop it.

Strategy: source inspection of the two consumers, the approach this
repo uses for its classic-script pages. The client's own behavior is
driven properly in `tests/web/static/download_client.test.js`; what
needs guarding here is that nobody goes back to talking to the
endpoint directly.

`ORG-02` owed "the remaining API clients with request epochs" and
downloads were one. Three readers of one URL: `menu.js` polled a
bound row every 500ms and re-read the status once at load, and
`download_toast.js` ran its own 1000ms loop from boot to unload.
Sitting on the downloading row ran two of them at once, on different
clocks, each with its own idea of what a terminal state meant.

Passing proves the transport has one home, that both consumers reach
it rather than around it, that the page which merely watches cannot
claim the download, and that a cancel exists where a download is
started.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
MENU_JS = STATIC / "menu.js"
TOAST_JS = STATIC / "download_toast.js"
CLIENT_JS = STATIC / "download_client.js"

STATUS_URL = "/api/models/download-status"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _region(text: str, anchor: str, chars: int) -> str:
    start = text.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone; update this test rather than"
        " deleting it"
    )
    return text[start : start + chars]


# -- the endpoint has one caller --


def test_only_the_client_reads_the_status_endpoint() -> None:
    """The duplication this replaced. Two loops on two clocks is
    what made a terminal state mean different things depending on
    which one saw it first."""
    assert STATUS_URL in _read(CLIENT_JS)
    assert STATUS_URL not in _read(MENU_JS)
    assert STATUS_URL not in _read(TOAST_JS)


def test_only_the_client_starts_or_cancels_a_download() -> None:
    for path in (MENU_JS, TOAST_JS):
        text = _read(path)
        assert "/download/cancel" not in text, path.name
        assert '"/api/models/" + encodeURIComponent(model.id)'
        assert "/download/ack" not in text, path.name


def test_the_page_creates_exactly_one_watcher() -> None:
    """A second would put the two loops back under a new name."""
    created = re.findall(
        r"downloadClientCreate\(", _read(TOAST_JS)
    )

    assert len(created) == 1
    assert "downloadClientCreate" not in _read(MENU_JS)


# -- both consumers reach it --


def test_the_toast_listens_rather_than_polls() -> None:
    body = _region(_read(TOAST_JS), "function init()", 500)

    assert "client.subscribe(onStatus)" in body
    assert "client.observe()" in body


def test_the_menu_borrows_the_same_watcher() -> None:
    """Not its own instance: the menu and the toast have to agree
    about which download is running and how far along it is."""
    body = _region(
        _read(MENU_JS), "function downloadClient()", 400
    )

    assert "downloadToastClient" in body


def test_the_menu_starts_through_the_client() -> None:
    body = _region(_read(MENU_JS), "function beginDownload(", 1200)

    assert "client.start(model.id)" in body


def test_the_row_is_drawn_from_the_shared_reading() -> None:
    """`renderDownloadRow` replaced a poll loop, so it must not have
    grown one back."""
    body = _region(
        _read(MENU_JS), "function renderDownloadRow(", 900
    )

    assert "fetch(" not in body
    assert "setTimeout" not in body


# -- watching is not owning --


def test_a_watching_page_claims_no_operation() -> None:
    """Every page runs the toast, and none of them started the
    fetch. Claiming it would let any tab cancel another's."""
    body = _region(_read(TOAST_JS), "function init()", 500)

    assert "client.start" not in body
    assert "client.observe()" in body


def test_the_client_forgets_its_claim_after_cancelling() -> None:
    body = _region(_read(CLIENT_JS), "function cancel()", 900)

    assert "operation = null" in body
    assert "JSON.stringify({ operation: operation })" in body


# -- the control exists where a download begins --


def test_the_progress_state_offers_a_cancel() -> None:
    body = _region(
        _read(MENU_JS), "function buildDownloadVeneer(", 2200
    )

    assert "menu-model-veneer-cancel" in body
    assert "prog.appendChild(cancel)" in body


def test_the_cancel_does_not_restart_the_download() -> None:
    """The whole row is the start-a-download target, so a click that
    reached it would cancel and immediately begin again."""
    body = _region(
        _read(MENU_JS), "function buildDownloadVeneer(", 2200
    )
    click = body.find('cancel.addEventListener("click"')

    assert click != -1
    assert "event.stopPropagation()" in body[click : click + 300]


def test_a_refused_cancel_is_surfaced() -> None:
    """It is refused when another window owns the download, which
    the user cannot guess from a button that did nothing."""
    body = _region(
        _read(MENU_JS), "function cancelDownload(", 1200
    )

    assert "showError" in body


def test_unbinding_a_row_does_not_stop_the_fetch() -> None:
    """A download belongs to the page, not to the row: paging away
    from it must leave it running for the toast to report. It used
    to clear a poll timer here, which is why the absence is worth
    asserting rather than assuming."""
    body = _region(_read(MENU_JS), "function stopPolling()", 600)
    statements = body.split("{", 1)[1]

    assert "activationWatch" in statements
    assert "pollTimer" not in statements
    assert "client" not in statements
