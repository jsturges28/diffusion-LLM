"""What a mask says reaches every surface that draws one.

Strategy: source inspection of the shipped classic scripts, the
approach this repo uses for pages that cannot be imported. What the
builder does with these is tested properly in
`tests/web/static/overlays_span.test.js`, the settings round trip in
`tests/web/static/overlays_settings.test.js`, and the opacity curve
in `tests/web/static/overlays_mask_opacity.test.js`. What needs
guarding here is the wiring in between, which none of those can see:
a preference no caller passes is a toggle that does nothing, and a
curve no caller passes is a flat canvas.

There are more of those callers than either feature suggests. The
generator draws tokens live, scrubbed, crossfaded against a retained
pre-edit run, through the diff overlay, and as a faded preview of the
frame an edit will regenerate. Analytics draws them for a saved run
and for the same two comparison overlays, and had never read the
settings at all, which is what makes the reveal retroactive rather
than live-only.

Two features share this file because they are the same wiring
mistake twice. The reveal shipped first and reached every path; the
grading shipped earlier and reached two, leaving Analytics and both
diff overlays flat for months while the confidence sat in the saved
file unread. Pinning them together is what stops the next per-token
property from finding a third gap.

Passing proves each of those paths asks for the preference and for
the curve, that Analytics reads the settings, and that the Settings
page can set the one control.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

SETTING = "revealMaskCandidate"
FLAG = "revealMask"
HOOK = "opacityFor"


def _source(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _region(name: str, anchor: str, chars: int) -> str:
    source = _source(name)
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {name}; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- the generator --


def test_the_scrubbed_and_crossfaded_layers_ask_for_it() -> None:
    """One options builder serves the single scrubbed layer and both
    layers of the run crossfade, so they cannot disagree."""
    body = _region(
        "app.js", "function tokenLayerOptions(isOriginal)", 400
    )

    assert f"{FLAG}: appSettings.{SETTING}" in body


def test_the_diff_overlay_asks_for_it() -> None:
    """Its two layers are built by a different shared function, and
    it is the one view where a reveal on one layer and glyphs on the
    other would be actively misleading."""
    body = _region("app.js", "function renderDiffOverlay(", 900)

    assert f"{FLAG}: appSettings.{SETTING}" in body


def test_the_edit_preview_asks_for_it() -> None:
    """The faded still of the pre-edit frame. It used to build its
    spans by hand, with its own copy of the glyph decision, so the
    setting would have been silently absent from the one view you
    compare a branch against."""
    body = _region(
        "app.js", "function renderTargetPlaceholder(frameIndex)", 3000
    )

    assert f"{FLAG}: appSettings.{SETTING}" in body
    assert "overlaysBuildTokenSpan(" in body


def test_the_preview_no_longer_decides_the_glyph_itself() -> None:
    """The line the shared builder replaced. Leaving it would mean
    two owners of the same decision, which is how the first one
    drifted."""
    body = _region(
        "app.js", "function renderTargetPlaceholder(frameIndex)", 3000
    )

    assert "tok.m ? MASK_CHAR : tok.t" not in body


# -- the generator grades its masks --


def test_the_generators_live_and_scrubbed_paths_grade() -> None:
    """The two that always did. Pinned alongside the two that did
    not, so the set is readable in one place."""
    live = _region("app.js", "var LIVE_TOKEN_OPTIONS", 120)
    scrubbed = _region(
        "app.js", "function tokenLayerOptions(isOriginal)", 400
    )

    assert f"{HOOK}: tokenOpacityFn" in live
    assert f"{HOOK}: tokenOpacityFn" in scrubbed


def test_the_generators_diff_overlay_grades() -> None:
    """It did not, and its own colorFor says why that was wrong: it
    returns null for a masked position specifically to keep the mask
    identical to the single-layer paths. Without the curve it was
    the only view where a mask meant nothing."""
    body = _region("app.js", "function renderDiffOverlay(", 900)

    assert f"{HOOK}: tokenOpacityFn" in body


# -- analytics --


def test_analytics_reads_the_durable_settings() -> None:
    """The page had never read them. This one call is what makes a
    saved run answer to the setting, and a diffusion run saved before
    the feature existed carries the same per-position guess a live
    one does."""
    source = _source("analytics.js")

    assert "overlaysLoadSettings()" in source


def test_the_saved_run_view_asks_for_it() -> None:
    """Anchored on the options object rather than on the function,
    because a window wide enough to hold the whole function also
    holds the comparison layers below it, and then dropping the hook
    here still passes on the neighbour's copy of it."""
    body = _region("analytics.js", "var edited = {", 300)

    assert f"{FLAG}: analyticsSettings.{SETTING}" in body
    assert f"{HOOK}: overlayOpacityFn" in body


def test_the_analytics_comparison_layers_ask_for_it() -> None:
    """Both stacked layers, for the same reason the generator's do."""
    body = _region(
        "analytics.js", "function renderOverlayLayers(", 900
    )

    assert body.count(f"{FLAG}:") == 2
    assert body.count(f"{HOOK}:") == 2


def test_the_analytics_diff_overlay_asks_for_it() -> None:
    body = _region(
        "analytics.js", "overlayDiffOrigOpacity,\n", 300
    )

    assert f"{FLAG}: analyticsSettings.{SETTING}" in body
    assert f"{HOOK}: overlayOpacityFn" in body


def test_the_analytics_hook_has_no_selection_to_spare() -> None:
    """The generator holds a remask selection solid; Analytics has no
    selection, so its hook is the curve and nothing else. Stated here
    because the temptation is to copy the generator's version, which
    would read a `remaskedPositions` that does not exist on the
    page."""
    body = _region(
        "analytics.js",
        "function overlayOpacityFn(index, tok, masked)",
        300,
    )

    assert "overlaysMaskOpacity(" in body
    assert "remaskedPositions" not in body


def test_a_hole_is_faint_rather_than_solid_on_both_pages() -> None:
    """A position with no token is padding that exists so two stacked
    layers line up. It is not an unmeasured token, which draws solid,
    so both hooks send it to the floor before the curve sees it. The
    two are one line apart in the same function and easy to conflate,
    which is exactly what happened when the floor moved."""
    generator = _region(
        "app.js", "function tokenOpacityFn(index, tok, masked)", 400
    )
    analytics = _region(
        "analytics.js",
        "function overlayOpacityFn(index, tok, masked)",
        400,
    )

    for body in (generator, analytics):
        assert "if (!tok) {" in body
        assert "return MASK_OPACITY_FLOOR;" in body


def test_the_shared_diff_builder_forwards_the_hook() -> None:
    """Both pages reach their diff layers through this one function,
    so a hook it drops is two views flat at once."""
    body = _region(
        "overlays.js", "function overlaysBuildDiffLayers(", 1400
    )

    assert body.count(f"{HOOK}: opacityFor") == 2


# -- the settings page --


def test_the_settings_page_carries_the_key_through_a_save() -> None:
    """Save writes the whole blob, so a key the page drops on the way
    into its staged clone is a key the save deletes."""
    body = _region(
        "settings.js", "function cloneSettings(source)", 700
    )

    assert f"{SETTING}: source.{SETTING}" in body


def test_the_toggle_exists_on_the_page() -> None:
    source = _source("settings.js")
    markup = _source("settings.html")

    assert 'getElementById("setting-reveal-mask-candidate")' in source
    assert 'id="setting-reveal-mask-candidate"' in markup


def test_the_toggle_is_wired_both_ways() -> None:
    """Staged state into the checkbox on load, and the checkbox back
    into staged state on change. One direction alone is a control
    that either forgets or lies."""
    source = _source("settings.js")

    into_control = re.search(
        r"settingRevealMaskCb\.checked\s*=\s*"
        rf"stagedSettings\.{SETTING}",
        source,
    )
    out_of_control = re.search(
        rf"stagedSettings\.{SETTING}\s*=\s*"
        r"settingRevealMaskCb\.checked",
        source,
    )

    assert into_control is not None
    assert out_of_control is not None
