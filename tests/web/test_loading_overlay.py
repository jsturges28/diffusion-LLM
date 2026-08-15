"""The loading overlay appears only when something is loading.

Strategy: read the shipped markup and `app.js`. The overlay's whole
behaviour is which class it carries at rest and which code paths take
that class off, so both are checkable without a browser.

What passing proves is a small honesty property that pass one made
possible. The overlay used to be in the markup with no `hidden`
class, so it painted on every arrival at the generator and said
"Loading model" while the page waited for a WebSocket handshake on a
model that was already resident: half a second of a message that was
not true, on the most ordinary navigation in the app. Since `LIFE-02`
the `/generate` gate turns away any model that is not actually
serving, so reaching the page is itself proof there is nothing to
wait for, and the overlay can start down.

It still has to come up for the three things that really are loads,
which is the other half of what is checked here. An overlay that
never appeared would be a worse bug than one that appeared too
often, because a model switch takes tens of seconds and a page that
looked idle through it would read as broken.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
INDEX_HTML = STATIC / "index.html"
APP_JS = STATIC / "app.js"


def _overlay_tag() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r'<div id="loading-overlay"[^>]*>', html)
    assert match is not None, (
        "the loading overlay is gone from index.html; update this"
        " test rather than deleting it"
    )
    return match.group(0)


def _region(anchor: str, chars: int) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- at rest --


def test_the_overlay_starts_hidden() -> None:
    """Arriving at the generator means a model is already serving,
    because the gate refuses anything else, so there is nothing to
    wait for and nothing to say."""
    assert 'class="hidden"' in _overlay_tag()


def test_the_progress_track_starts_hidden_too() -> None:
    """Unchanged, and for its own reason: a checkpoint whose size
    cannot be worked out has no bar to draw, so the track only
    appears once a poll reports something measurable."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    match = re.search(
        r'<div id="load-progress-container"[^>]*>', html
    )
    assert match is not None
    assert "hidden" in match.group(0)


# -- when it comes up --


def test_a_switch_raises_it() -> None:
    """The case it exists for: tens of seconds of worker spawn and
    weight reading, where a page that looked idle would read as
    broken."""
    region = _region("function switchModel(id, device)", 1200)

    assert 'loadingOverlay.classList.remove("hidden")' in region


def test_a_worker_reporting_loading_raises_it() -> None:
    """Covers the narrow race the gate cannot: the page was served
    while a model was serving, and the worker restarted before the
    socket opened."""
    region = _region("function handleModelStatus(data)", 1400)

    assert 'loadingOverlay.classList.remove("hidden")' in region


def test_a_model_swapped_from_another_window_raises_it() -> None:
    region = _region("function handleResident(data)", 1500)

    assert 'loadingOverlay.classList.remove("hidden")' in region


# -- when it goes down --


def test_a_ready_worker_lowers_it() -> None:
    region = _region("function handleModelStatus(data)", 1400)

    assert 'loadingOverlay.classList.add("hidden")' in region


def test_a_failed_switch_lowers_it() -> None:
    """Otherwise a refused switch leaves the page behind a curtain
    describing a load that is not happening."""
    region = _region("function switchFailed(err)", 900)

    assert 'loadingOverlay.classList.add("hidden")' in region
