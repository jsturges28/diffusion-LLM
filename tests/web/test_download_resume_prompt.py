"""A stopped download offers to resume, and says so on any load.

Strategy: the `partial` flag over HTTP against the real app, plus
source inspection of the menu that reads it. The download lifecycle
itself is covered in `test_download_ownership.py`.

Two things went wrong once cancelling became possible, both found on
hardware rather than here.

The first was a regression. `syncDownloadBinding` treats "idle" as
not-active and used to drop the row binding without touching the
veneer, so a cancelled download left the last percentage frozen on
screen under a Cancel button with nothing to cancel. It hid until
now because idle-while-bound was previously only reachable through
the ack after done or error, by which point the veneer had already
been rewritten. Cancel made it the ordinary path.

The second was never right. `downloaded` is false both for a model
never fetched and for one stopped at 8%, so the row offered to
download something that would actually resume. The distinction
exists inside `is_repo_cached`, which checks for `*.incomplete`
parts, and was being discarded one layer down.

Passing proves the flag reaches the browser, that it is false for
the local-directory checkpoint where the idea does not apply, and
that the menu words the row from it rather than from a memory of
having just cancelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.web import server as server_module
from src.web.server import _is_partial, app

MENU_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "menu.js"
)


def _menu() -> str:
    return MENU_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    text = _menu()
    start = text.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from menu.js; update this test"
        " rather than deleting it"
    )
    return text[start : start + chars]


@pytest.fixture()
def client() -> Any:
    with TestClient(app) as test_client:
        yield test_client


def _models(client: Any) -> List[Dict[str, Any]]:
    return list(client.get("/api/models").json()["models"])


# -- the flag reaches the browser --


def test_every_model_reports_whether_it_is_partial(
    client: Any,
) -> None:
    models = _models(client)

    assert models
    for model in models:
        assert "partial" in model, model.get("id")
        assert isinstance(model["partial"], bool)


def test_a_partial_cache_is_reported(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.inference import hf_download

    monkeypatch.setattr(
        hf_download, "has_partial_download", lambda repo: True
    )

    partial = [m["partial"] for m in _models(client)]

    assert any(partial), "no model reported the partial cache"


def test_a_local_checkpoint_is_never_partial() -> None:
    """DiffusionGemma's weights are produced offline by the quantize
    script rather than fetched, so there is no partial state for
    them to be in and nothing to resume."""
    assert _is_partial("~/models/diffusiongemma-nf4") is False
    assert _is_partial("/opt/weights/thing") is False


def test_a_probe_failure_reports_not_partial(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same posture as `_is_downloaded`: a cache we cannot read is
    not a cache we should promise to resume."""
    from src.inference import hf_download

    def _boom(repo: str) -> bool:
        raise OSError("cache unreadable")

    monkeypatch.setattr(
        hf_download, "has_partial_download", _boom
    )

    assert _is_partial("org/model") is False


def test_the_registry_is_what_gets_asked(
    client: Any,
) -> None:
    """Guards the test above from passing on an empty list."""
    ids = {m["id"] for m in _models(client)}

    assert ids == set(server_module.REGISTRY)


# -- the menu words the row from it --


def test_the_prompt_depends_on_the_flag() -> None:
    body = _region("function downloadPrompt(", 400)

    assert "model.partial" in body
    assert "Click to Resume Download" in body
    assert "Click to Download" in body


def test_the_veneer_asks_rather_than_hardcoding() -> None:
    """Both construction sites, since a row is built at render and
    again on demand when a download re-attaches to it."""
    body = _region("function buildDownloadVeneer(", 700)

    assert "downloadPrompt(model)" in body
    assert '"Click to Download"' not in body


def test_a_reset_row_is_relabelled() -> None:
    """The row goes back to idle after a cancel, and idle now has
    two different things it can say."""
    body = _region("function resetDownload(", 1400)

    assert "downloadPrompt(modelForRow(li))" in body


def test_a_reset_row_keeps_no_bar() -> None:
    """The bar briefly survived a cancel, frozen at the percentage
    the fetch reached. It was the one thing here that could not be
    made true: only the window that pressed Cancel had it, it went
    on a reload, and it went stale in place if the menu was left
    open, so two windows showed one state two ways."""
    body = _region("function resetDownload(", 1400)

    assert "parts.prog.hidden = true" in body
    assert "Stopped at" not in body
    assert "keepBar" not in body


def test_every_idle_veneer_renders_the_same_way() -> None:
    """The point of dropping it: a partial cache looks identical
    whether you just cancelled, reloaded, or are watching from
    another window."""
    assert "Stopped at" not in _menu()
    assert "keepBar" not in _menu()


# -- the regression --


def test_an_ended_download_restores_the_veneer() -> None:
    """The bug: this branch dropped the binding and returned, so a
    cancelled download left its last percentage frozen under a
    Cancel button that no longer had anything to cancel."""
    body = _region("function syncDownloadBinding(", 1600)
    ended = body.find("if (!active)")
    reset = body.find("resetDownload(downloadRow)")

    assert ended != -1, "the inactive case is not distinguished"
    assert reset != -1
    assert ended < reset


def test_paging_away_does_not_reset_the_row() -> None:
    """The case the branch above must not swallow: still running,
    but its row is on another page. Resetting there would rewrite a
    row the user merely paged away from."""
    body = _region("function syncDownloadBinding(", 1600)
    absent = body.find("if (!row)")
    reset = body.find("resetDownload(downloadRow)")

    assert absent != -1
    assert reset < absent, "the two cases are in the wrong order"
    tail = body[absent : absent + 400]
    assert "resetDownload" not in tail
    assert "downloadRow = null" in tail


def test_the_cancel_button_goes_with_the_download() -> None:
    """It lives inside the progress area, so hiding that hides it
    too. An explicit hide here would have to be undone somewhere
    else when the fetch restarts, which is a second thing to keep
    in step for no gain."""
    veneer = _region("function buildDownloadVeneer(", 2200)
    reset = _region("function resetDownload(", 1400)

    assert "prog.appendChild(cancel)" in veneer
    assert "cancel.hidden" not in reset


# -- the flags come from the server, in every window --


def test_a_finished_fetch_re_reads_the_flags() -> None:
    """A second window's flags are from its own page load, so a
    download ending anywhere else leaves it describing a cache that
    has since changed."""
    body = _region("function onDownloadStatus(", 900)

    assert 'prevDownloadState === "downloading"' in body
    assert 'state !== "downloading"' in body
    assert "refreshModelFlags()" in body


def test_the_refresh_does_not_re_render_the_list() -> None:
    """`loadModels` is the tempting wrong version: it would tear
    down the very veneer this is correcting, and throw away the
    page the user is on."""
    body = _region("function refreshModelFlags(", 1200)

    assert "loadModels" not in body
    assert "renderModels" not in body
    assert "modelClientLoad()" in body


def test_the_refresh_copies_only_what_moved() -> None:
    body = _region("function refreshModelFlags(", 1200)

    assert "model.downloaded = fresh[i].downloaded" in body
    assert "model.partial = fresh[i].partial" in body


def test_a_failed_refresh_leaves_the_page_alone() -> None:
    """The flags stay as they were, which is what was on screen
    anyway. The next reading tries again."""
    body = _region("function refreshModelFlags(", 1200)

    assert ".catch(" in body


def test_the_relabel_only_touches_visible_prompts() -> None:
    """A hidden label belongs to a row mid-download or mid-message;
    rewriting it would put the idle wording under a progress bar."""
    body = _region("function relabelVeneers(", 700)

    assert "!label.hidden" in body
    assert "downloadPrompt(modelForRow(rows[i]))" in body


def test_a_cancelled_row_remembers_it_is_partial() -> None:
    """So the prompt is right immediately rather than a reload
    later. The server agrees on the next load; this is what makes
    the two say the same thing in between."""
    body = _region("function cancelDownload(", 1200)

    assert "model.partial = true" in body


def test_a_finished_download_stops_being_partial() -> None:
    body = _region("function completeDownload(", 700)

    assert "model.partial = false" in body
