"""The stopping readout reaches the markup and the styles.

Strategy: source inspection of the HTML and CSS, the approach this
repo uses for the parts of a classic-script page only a browser
reads. The behaviour is run in the JS suites (`stop_readout.test.js`,
`generator_stop_readout.test.js` and `analytics_stopping.test.js`
under `tests/web/static/`), which load the shipped scripts into a DOM
stub. The stub never applies a stylesheet, and it makes any element a
script asks for by id, so a readout missing from a page, or one whose
words never give way, would pass there unseen.

Passing proves each page sets the readout beside its metrics strip
rather than inside it, where the strip's idle dimming would reach it;
that the strip, not the readout, yields width; that the readout's
words give way while its trace stays; that a met condition takes the
accent; and that the trace is drawn at the size its box is styled.
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


def _rule(css: str, selector: str) -> str:
    """The body of the first rule for exactly ``selector``."""
    match = re.search(re.escape(selector) + r" \{([^}]*)\}", css)
    assert match is not None, f"{selector!r} is gone; update this"
    return match.group(1)


def _assert_readout_beside_strip(html: str) -> None:
    row = _after(html, 'class="token-metrics-row"', 400)
    strip = row.find('id="token-metrics"')
    readout = row.find('id="stop-readout"')

    assert strip != -1
    assert readout > strip
    # One close between them, the strip's own, so the readout is the
    # strip's sibling inside the row rather than after the row.
    assert row[strip:readout].count("</div>") == 1
    assert "hidden" in row[readout : readout + 80]


# -- the markup --


def test_the_generator_sets_the_readout_beside_the_strip() -> None:
    _assert_readout_beside_strip(_read("index.html"))


def test_the_readout_is_not_inside_the_strip() -> None:
    # The strip dims as a whole while idle; inside it, the readout
    # would dim too, whatever a hover was doing.
    html = _read("index.html")

    strip = _after(html, 'id="token-metrics"', 80)

    assert strip.startswith(
        'id="token-metrics" class="token-metrics"></div>'
    )


def test_both_pages_build_the_readout_at_boot() -> None:
    # Analytics builds both in its token viewer's wire, which the
    # page's boot calls. The generator delegates the same pair to its
    # readout controller.
    anchor = "overlaysBuildTokenMetrics(tokenMetricsStrip);"
    readout = "overlaysBuildStopReadout(stopReadout);"

    generator = _after(
        _read("generator_readouts.js"), anchor, 120
    )
    analytics = _after(_read("token_viewer.js"), anchor, 120)

    assert readout in generator
    assert readout in analytics
    assert "generatorReadouts.boot();" in _read("app.js")
    assert "\ntokenViewer.wire();\n" in _read("analytics.js")


def test_the_detail_modal_sets_the_readout_beside_its_strip() -> None:
    _assert_readout_beside_strip(_read("analytics.html"))


def _section(html: str, section_id: str) -> str:
    start = html.find(f'id="{section_id}"')
    assert start != -1, f"{section_id} is gone; update this test"
    end = html.find('class="chart-section"', start)
    return html[start : end if end != -1 else len(html)]


def test_the_stopping_chart_shares_the_confidence_slot() -> None:
    # Each section's own page is its disabled button, as Timing's
    # pair is, so the pager reads as "you are here".
    html = _read("analytics.html")
    confidence = _section(html, "confidence-section")
    stopping = _section(html, "stopping-section")
    own = 'data-confidence-page="{}" disabled'

    assert 'id="chart-stopping"' in stopping
    assert own.format("confidence") in confidence
    assert own.format("stopping") in stopping
    assert 'data-confidence-page="stopping"' in confidence
    assert 'data-confidence-page="confidence"' in stopping
    assert "hidden" in _after(html, 'id="stopping-section"', 60)


def test_the_stopping_chart_has_its_own_controls() -> None:
    stopping = _section(_read("analytics.html"), "stopping-section")

    for control in (
        r'class="zoom-btn tooltip-toggle-btn" data-chart="stopping"',
        r'class="compare-pins" data-chart="stopping"',
        r'data-chart="stopping"\s+data-action="in"',
    ):
        assert re.search(control, stopping), control


def test_the_modal_readout_takes_the_strips_padding() -> None:
    css = _read("analytics.css")

    rule = _rule(
        css, "#overlay-viewer .token-metrics,\n"
        "#overlay-viewer .stop-readout,\n"
        "#overlay-viewer .watermark-readout",
    )

    assert "padding-top: 0" in rule


# -- the styles --


def test_the_strip_yields_width_to_the_readout() -> None:
    css = _read("style.css")

    row = _rule(css, ".token-metrics-row")
    strip = _rule(css, ".token-metrics-row > .token-metrics")
    readout = _rule(css, ".stop-readout")

    assert "flex-shrink: 0" in row
    assert "min-width: 0" in strip
    assert "flex: 1 1 auto" in strip
    assert "flex: none" in readout


def test_the_words_give_way_but_the_trace_stays() -> None:
    css = _read("style.css")

    compact = _rule(
        css, ".stop-readout[data-compact] .stop-readout-text"
    )

    assert "display: none" in compact
    assert "[data-compact] .stop-readout-trace" not in css
    assert "[data-compact] .stop-readout-label" not in css


def test_a_met_condition_takes_the_accent() -> None:
    css = _read("style.css")

    met = _rule(
        css,
        ".stop-readout-clause.is-met,\n"
        ".stop-readout-clause.is-met .stop-readout-value",
    )

    assert "color: var(--accent)" in met


def test_the_trace_is_drawn_at_its_styled_size() -> None:
    # The script falls back to these numbers before layout has given
    # the canvas a size, so they have to be the box the CSS draws.
    css = _read("style.css")
    script = _read("overlays.js")

    trace = _rule(css, ".stop-readout-trace")
    meter = _rule(css, "#status-resource-spark")
    width = re.search(r"OVERLAYS_STOP_TRACE_WIDTH = (\d+);", script)
    height = re.search(r"OVERLAYS_STOP_TRACE_HEIGHT = (\d+);", script)
    assert width is not None
    assert height is not None

    assert f"width: {width.group(1)}px" in trace
    assert f"height: {height.group(1)}px" in trace
    assert f"width: {width.group(1)}px" in meter
