"""Tests that saved-run metadata cannot inject markup into the page.

Strategy: read `analytics.js` and check the two places that decide
this. The detail panel's parameter rows must be built through
`metaRowHtml`, and `metaRowHtml` must escape its label as well as its
value. Source inspection rather than execution, because that file is
a 5,600-line classic script that reaches for the DOM at load; giving
it a testable seam is `ORG-02`'s job in stage 5.

The bug this pins: the loop rendering a run's parameters escaped each
*value* and then interpolated the *key* raw, one line apart. Keys come
from `metadata.json` on disk, so a run folder carrying markup in a
parameter name could execute script on this origin, which is the
origin that can activate models and delete runs. Nothing in the app
writes such a key, but "our own writer would not do that" is not a
property of a file sitting in a directory.
"""

from __future__ import annotations

from pathlib import Path

ANALYTICS_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "analytics.js"
)

# The comment that opens the parameter loop, used as an anchor so the
# check reads the right region rather than the whole file.
_PARAMS_ANCHOR = "Render whatever params this run recorded"

# Generous enough to cover the loop, short enough not to run into
# unrelated rendering below it.
_REGION_CHARS = 700


def _params_loop() -> str:
    source = ANALYTICS_JS.read_text(encoding="utf-8")
    start = source.find(_PARAMS_ANCHOR)
    assert start != -1, (
        "the parameter loop's anchor comment is gone; update this"
        " test rather than deleting it"
    )
    return source[start : start + _REGION_CHARS]


def test_param_rows_go_through_the_escaping_helper() -> None:
    loop = _params_loop()

    assert "metaRowHtml(" in loop


def test_the_param_loop_does_not_build_raw_markup() -> None:
    """The paired negative. Reaching for the class names directly is
    what the raw interpolation looked like, and it is how a future
    edit would reintroduce it while still calling escHtml on the
    value and feeling safe."""
    loop = _params_loop()

    assert "meta-label" not in loop
    assert "meta-row" not in loop


def test_the_helper_escapes_the_label_and_the_value() -> None:
    """Everything above rests on this, so it is checked rather than
    assumed."""
    source = ANALYTICS_JS.read_text(encoding="utf-8")
    start = source.find("function metaRowHtml(")
    assert start != -1, "metaRowHtml is gone"
    helper = source[start : start + 300]

    assert helper.count("escHtml(") == 2
    assert "escHtml(label)" in helper
    assert "escHtml(value)" in helper
