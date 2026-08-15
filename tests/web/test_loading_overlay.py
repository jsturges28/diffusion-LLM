"""The generator is covered while it assembles itself.

Strategy: read the shipped markup, CSS and `app.js`. Every property
here is about which class an element carries at rest and which code
paths take it off, so all of them are checkable without a browser.

The overlay is deliberately up at boot, and the reason is not the one
its name suggests. Since `LIFE-02` the `/generate` gate turns away any
model that is not serving, so arriving at this page is proof no model
load is pending. What the curtain is actually over is the page
building itself: `buildParamPanel` empties a container and fills it
from `param_specs` once `/api/models` answers, and around twenty
elements sit `hidden` in the markup waiting on runtime state. First
paint is a skeleton and the second is the real page.

This file briefly asserted the opposite. The overlay was removed on
the gate argument above, which was correct about model loading and
missed everything else the curtain covered; the maintainer saw the
page visibly reassemble and asked for it back. That is recorded under
`ORG-02` in the ledger, along with the fix that would make the
curtain unnecessary rather than merely opaque: rendering boot state
into the HTML at serve time, which is `ORG-02`'s to do.

The scrubber reservation stays from that attempt, because it is worth
having either way. It removes one source of movement rather than
hiding it, which is what the whole page needs eventually.

What passing proves, then, is narrow: the curtain is up when the page
cannot yet be trusted to look right, it comes down when it can, and
one element that used to shove the canvas around no longer does.
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


def test_the_overlay_covers_the_page_being_built() -> None:
    """Up at boot on purpose. Not for a model load, which the gate
    already rules out, but for the two-phase render underneath: a
    skeleton first, the real page once the fetch answers."""
    assert "hidden" not in _overlay_tag()


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


# -- and nothing moves underneath it --


def test_the_scrubber_holds_its_place_when_idle() -> None:
    """It is a sibling of the output canvas in a flex column, so
    taking it out of the layout resizes the canvas every time a run
    appears. Reserved rather than removed."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    match = re.search(
        r'<section id="scrubber-section"[^>]*>', html
    )
    assert match is not None
    assert 'class="is-idle"' in match.group(0)
    assert "hidden" not in match.group(0)


def test_idle_means_invisible_not_absent() -> None:
    """`display: none` would reserve nothing and put the reflow
    straight back."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find("#scrubber-section.is-idle")

    assert start != -1
    rule = css[start : start + 120]
    assert "visibility: hidden" in rule
    assert "display: none" not in rule


def test_one_helper_owns_the_scrubber_s_visibility() -> None:
    """Four call sites toggled it directly before this, which is how
    one of them ends up using a mechanism the others do not."""
    source = APP_JS.read_text(encoding="utf-8")

    assert "function setScrubberVisible(visible)" in source
    assert "scrubberSection.hidden" not in source
