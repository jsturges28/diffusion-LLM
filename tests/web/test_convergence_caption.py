"""How the convergence chart says which measure it is showing.

Strategy: source inspection of `analytics.js` and `analytics.html`,
the approach this repo uses for its classic-script pages. Which
measure a run gets is decided server-side and tested in
`test_metrics_basis_choice.py`; this covers only how that decision
reaches the reader.

It reaches them through an icon beside the heading rather than a
paragraph above the chart, because the paragraph was noise on every
DiffusionGemma run. That trade has one condition: an explanation can
live behind a hover, but a caveat cannot, or a reader who never
hovers takes an approximate curve for an exact one. So the icon is
tinted when the measure is the weak one, which is visible without
interacting with anything.

Passing proves the icon exists and is hidden where the reading needs
no explaining, that the weak measure is marked as well as described,
that the model is named from the server's label rather than
hardcoded, and that an unrecognised basis says nothing at all.
"""

from __future__ import annotations

from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
ANALYTICS_JS = STATIC / "analytics.js"
ANALYTICS_HTML = STATIC / "analytics.html"
ANALYTICS_CSS = STATIC / "analytics.css"


def _js() -> str:
    return ANALYTICS_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _js()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from analytics.js; update this"
        " test rather than deleting it"
    )
    return source[start : start + chars]


# -- the icon replaces the paragraph --


def test_the_heading_carries_an_icon() -> None:
    markup = ANALYTICS_HTML.read_text(encoding="utf-8")

    assert 'id="convergence-basis-info"' in markup
    assert 'id="convergence-basis-tip"' in markup


def test_the_standalone_paragraph_is_gone() -> None:
    """It sat above the chart on every DiffusionGemma run."""
    markup = ANALYTICS_HTML.read_text(encoding="utf-8")

    assert "convergence-basis-note" not in markup
    assert "chart-basis-note" not in ANALYTICS_CSS.read_text(
        encoding="utf-8"
    )


def test_the_icon_reuses_the_shared_tooltip_styling() -> None:
    # Same "?" affordance as the hyperparameters, so it needs no
    # look of its own and reads as the same kind of thing.
    markup = ANALYTICS_HTML.read_text(encoding="utf-8")

    assert "info-icon info-icon-sm" in markup


def test_the_tooltip_cannot_overflow_the_column() -> None:
    """Beside the icon does not work here.

    The chart column scrolls, so it clips anything positioned
    outside it, and it is narrower than the shared tooltip's 280px.
    Opening either sideways left a sliver against the edge. Below
    the heading, spanning the row, is the one placement whose width
    is bounded by the column itself.
    """
    css = ANALYTICS_CSS.read_text(encoding="utf-8")

    assert ".info-icon.tooltip-below .tooltip" in css
    # Spanning the row, rather than sized and then placed.
    assert "left: 0;" in css
    assert "width: auto;" in css
    # Which only resolves against the row if the icon stops being
    # the containing block.
    assert ".info-icon.tooltip-below {\n  position: static;" in css
    assert ".chart-header-row {\n  position: relative;" in css


# -- what each measure says --


def test_an_exact_reading_shows_no_icon() -> None:
    """A run whose mask is a real token needs no explaining, and the
    heading stays clean."""
    body = _region("function convergenceBasisNote(", 1400)

    assert 'return "";' in body

    render = _region("function renderConvergenceBasis(data)", 900)
    assert "icon.hidden = !text;" in render


def test_the_settlement_note_names_the_model() -> None:
    body = _region("function convergenceBasisNote(", 1400)

    assert "modelLabel" in body
    assert "has no mask token" in body


def test_the_model_name_comes_off_the_response() -> None:
    """Hardcoding it would be wrong the moment a second maskless
    model exists, and the basis is not named for a model."""
    body = _region("function renderConvergenceBasis(data)", 900)

    assert "data.model_label" in body


def test_a_missing_label_still_reads_as_a_sentence() -> None:
    body = _region("function renderConvergenceBasis(data)", 900)

    assert '"This model"' in body


def test_the_weak_measure_is_marked_as_well_as_described() -> None:
    """The condition on hiding text behind a hover: a caveat has to
    be visible to someone who never hovers."""
    render = _region("function renderConvergenceBasis(data)", 900)

    assert '"is-approximate"' in render
    assert 'basis === "characters"' in render

    css = ANALYTICS_CSS.read_text(encoding="utf-8")
    assert ".info-icon.is-approximate" in css


def test_an_unknown_basis_says_nothing() -> None:
    # A newer server talking to an older page. Guessing which of the
    # three it meant would be worse than silence.
    body = _region("function convergenceBasisNote(", 1400)

    settlement = body.index('basis === "settlement"')
    characters = body.index('basis === "characters"')
    fallback = body.rindex('return "";')

    assert fallback > settlement
    assert fallback > characters
