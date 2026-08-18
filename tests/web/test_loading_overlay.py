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

The reservations stay from that attempt, because they are worth
having either way. They remove sources of movement rather than hiding
them, which is what the whole page needs eventually, and the curtain
does not cover all of them: it lifts before the first run ends and
before a count of unopened runs arrives.

Reserving is not always right, though, and the entropy row is where
that shows. A row held for a model that will never fill it is a
permanent empty strip, which is worse than the shift it prevents, so
that one is reserved only when the resident model reports entropy.

What passing proves, then, is narrow: the curtain is up when the page
cannot yet be trusted to look right, it comes down when it can, and
the elements that used to shove their neighbours around hold their
places, each for as long as it is theirs to hold.
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


# -- nor beside it, nor above it --


def _is_hidden(tag: str) -> bool:
    """The bare `hidden` attribute, not `aria-hidden`, which several
    of these tags also carry and which reserves nothing either way."""
    return re.search(r"\shidden(?=[\s>=])", tag) is not None


def _rule(selector: str, chars: int = 200) -> str:
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find(selector)
    assert start != -1, (
        f"rule {selector!r} is gone from style.css; update this test"
        " rather than deleting it"
    )
    return css[start : start + chars]


def test_the_prompt_context_row_holds_its_line() -> None:
    """Only a ready worker can count a prompt's tokens, so this row
    is empty at first paint on every load. Removing it dropped the
    whole column below the prompt box a step when the count landed."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    match = re.search(r'<div id="prompt-context"[^>]*>', html)
    assert match is not None
    assert "is-empty" in match.group(0)
    assert not _is_hidden(match.group(0))


def test_an_empty_context_row_is_invisible_not_absent() -> None:
    rule = _rule(".prompt-context.is-empty")

    assert "visibility: hidden" in rule
    assert "display: none" not in rule


def test_the_analytics_badge_holds_a_slot_on_both_pages() -> None:
    """It lives inside a header link with more links to its right, so
    a badge that appears from nothing slides all of them across. The
    count comes from a fetch, so that is every page load."""
    for name in ("index.html", "menu.html"):
        markup = (STATIC / name).read_text(encoding="utf-8")

        match = re.search(
            r'<span id="[a-z-]*analytics-new[a-z-]*"[^>]*>', markup
        )
        assert match is not None, name
        assert "is-empty" in match.group(0), name
        assert not _is_hidden(match.group(0)), name


def test_the_badge_s_slot_fits_two_digits() -> None:
    """A slot sized for one digit still jumps at ten unopened runs,
    which is a count the maintainer passes routinely."""
    rule = _rule(".analytics-new-dot {")

    match = re.search(r"min-width:\s*(\d+)px", rule)
    assert match is not None
    assert int(match.group(1)) >= 20


def test_neither_badge_is_toggled_by_the_hidden_attribute() -> None:
    """`hidden` implies `display: none`, which un-reserves the slot
    the rule above reserves."""
    for name in ("app.js", "menu.js"):
        source = (STATIC / name).read_text(encoding="utf-8")

        assert ".hidden = true" not in _badge_body(source), name
        assert 'classList.toggle("is-empty"' in source, name


def _badge_body(source: str) -> str:
    for anchor in (
        "function refreshAnalyticsCue()",
        "function refreshAnalyticsNewBadge()",
    ):
        start = source.find(anchor)
        if start != -1:
            return source[start : start + 500]
    raise AssertionError(
        "neither badge refresher is in this file; update this test"
        " rather than deleting it"
    )


# -- except where holding a place would be the worse bargain --


def test_the_entropy_row_is_reserved_only_when_it_can_fill() -> None:
    """The conditional case. Diffusion models report no entropy
    today, and a row held for them is a permanent empty strip under
    every run rather than a shift avoided once."""
    region = _region(
        "function setEntropyProfileVisible(visible)", 400
    )

    assert (
        "entropyProfileRow.hidden = !visible && !isAutoregressive()"
        in region
    )
    assert 'classList.toggle("is-empty", !visible)' in region


def test_the_entropy_row_starts_absent_in_the_markup() -> None:
    """Before `/api/models` answers there is no resident model to ask,
    and absent is the reading that cannot leave a stray strip."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    match = re.search(r'<div id="entropy-profile-row"[^>]*>', html)
    assert match is not None
    assert _is_hidden(match.group(0))


def test_boot_settles_the_entropy_row_once_a_model_is_known() -> None:
    """Otherwise an autoregressive page keeps the markup's absent
    reading until its first run, which is the shift this prevents."""
    region = _region("      applyTokenBirthGlow();", 400)

    assert "setEntropyProfileVisible(false)" in region


def test_one_helper_owns_the_entropy_row_too() -> None:
    """Four call sites set `.hidden` directly before this, none of
    which knew about the conditional part."""
    source = APP_JS.read_text(encoding="utf-8")

    assert source.count("entropyProfileRow.hidden =") == 1
