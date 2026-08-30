"""The mask reveal reaches every surface that draws a token.

Strategy: source inspection of the shipped classic scripts, the
approach this repo uses for pages that cannot be imported. What the
builder does with the flag is tested properly in
`tests/web/static/overlays_span.test.js`, and the settings round trip
in `tests/web/static/overlays_settings.test.js`. What needs guarding
here is the wiring in between, which those two cannot see: a setting
that no caller passes is a toggle that does nothing.

There are more of those callers than the feature suggests. The
generator draws tokens live, scrubbed, crossfaded against a retained
pre-edit run, and as a faded preview of the frame an edit will
regenerate. Analytics draws them for a saved run and for the same two
comparison overlays, and had never read the settings at all, which is
what makes the reveal retroactive rather than live-only.

Passing proves each of those paths asks for the preference, that
Analytics reads it, and that the Settings page can set it.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

SETTING = "revealMaskCandidate"
FLAG = "revealMask"


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


# -- analytics --


def test_analytics_reads_the_durable_settings() -> None:
    """The page had never read them. This one call is what makes a
    saved run answer to the setting, and a diffusion run saved before
    the feature existed carries the same per-position guess a live
    one does."""
    source = _source("analytics.js")

    assert "overlaysLoadSettings()" in source


def test_the_saved_run_view_asks_for_it() -> None:
    body = _region(
        "analytics.js", "function renderOverlayTokens(opts)", 1200
    )

    assert f"{FLAG}: analyticsSettings.{SETTING}" in body


def test_the_analytics_comparison_layers_ask_for_it() -> None:
    """Both stacked layers, for the same reason the generator's do."""
    body = _region(
        "analytics.js", "function renderOverlayLayers(", 900
    )

    assert body.count(f"{FLAG}:") == 2


def test_the_analytics_diff_overlay_asks_for_it() -> None:
    body = _region(
        "analytics.js", "overlayDiffOrigOpacity,\n", 300
    )

    assert f"{FLAG}: analyticsSettings.{SETTING}" in body


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
