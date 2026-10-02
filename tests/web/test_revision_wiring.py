"""The revision glow and the Revisions overlay reach the markup and
the styles.

Strategy: source inspection of the HTML and CSS, the approach this
repo uses for the parts of a classic-script page only a browser
reads. The behaviour is run in the JS suites (`revisions.test.js`,
`generator_revisions.test.js`, `analytics_revisions.test.js` and
`settings_glow_preview.test.js` under `tests/web/static/`), which
load the shipped scripts into a DOM stub. The stub never applies a
stylesheet, and it makes any element a script asks for by id, so a
toggle or a legend missing from the page would pass there unseen.

Passing proves the Settings page carries the toggle its script
reads, between the birth glow and the rows the two glows share;
that both pages carry the legend their script shows; that the
legend's swatches are the overlay's colours; and that the live and
preview flashes each have keyframes, the script's colour, and a
reduced-motion backstop.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _after(text: str, anchor: str, chars: int) -> str:
    start = text.find(anchor)
    assert start != -1, f"{anchor!r} is gone; update this test"
    return text[start : start + chars]


def _reduced_motion_blocks(css: str) -> list[str]:
    """Every reduced-motion block, as the text up to its close."""
    blocks: list[str] = []
    for match in re.finditer(
        r"@media \(prefers-reduced-motion: reduce\) \{", css
    ):
        depth = 1
        at = match.end()
        while depth > 0 and at < len(css):
            depth += {"{": 1, "}": -1}.get(css[at], 0)
            at += 1
        blocks.append(css[match.end() : at])
    return blocks


# -- the markup --


def test_the_toggle_sits_between_the_birth_glow_and_its_rows(
) -> None:
    html = _read("settings.html")

    birth = html.find('id="setting-token-birth-glow"')
    revision = html.find('id="setting-revision-glow"')
    shared = html.find('id="glow-class-row"')

    assert birth != -1
    assert revision != -1
    assert birth < revision < shared


def test_both_pages_carry_a_hidden_revision_legend() -> None:
    generator = _after(
        _read("index.html"), 'id="revision-legend"', 40
    )
    analytics = _after(
        _read("analytics.html"), 'id="overlay-revision-legend"', 80
    )

    assert "hidden" in generator
    assert "hidden" in analytics


def test_the_legend_swatches_are_the_overlay_colours() -> None:
    ramp = re.search(
        r"OVERLAYS_REVISION_COLORS = \[([^\]]+)\]",
        _read("overlays.js"),
    )
    assert ramp is not None
    colors = re.findall(r'"(#[0-9a-f]{6})"', ramp.group(1))
    css = _read("style.css")

    background = r" \{\s*background: (#[0-9a-f]{6})"
    swatches = [
        re.search(r"\.revision-" + step + background, css)
        for step in ("once", "twice", "more")
    ]

    assert len(colors) == 3
    assert all(swatch is not None for swatch in swatches)
    assert [s.group(1) for s in swatches if s] == colors


# -- the styles --


def test_the_live_flash_has_keyframes_in_the_scripts_colour() -> None:
    css = _read("style.css")
    keyframes = _after(css, "@keyframes token-revision", 400)
    rule = _after(
        css, "#output-area.live-tokens .token-span[data-revised]", 120
    )
    channels = re.search(
        r'var GLOW_REVISION_RGB = "([\d, ]+)";', _read("overlays.js")
    )

    assert channels is not None
    assert "--token-revision-shadow" in keyframes
    assert f"rgba({channels.group(1)}, 0.9)" in keyframes
    assert "animation: token-revision" in rule


def test_the_live_flash_stops_under_reduced_motion() -> None:
    blocks = _reduced_motion_blocks(_read("style.css"))

    stopping = [
        block for block in blocks
        if ".token-span[data-revised]" in block
    ]

    assert len(stopping) == 1
    assert "animation: none" in stopping[0]


def test_the_preview_flash_animates_and_holds_still() -> None:
    css = _read("settings.css")
    rule = _after(css, ".glow-preview-word[data-revised] {", 120)
    blocks = _reduced_motion_blocks(css)

    held = [
        block for block in blocks
        if ".glow-preview-word[data-revised]" in block
    ]

    assert "animation: token-revision" in rule
    assert len(held) == 1
    assert "var(--token-revision-shadow)" in held[0]
