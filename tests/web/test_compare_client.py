"""The compare panel fences its own requests and hides nothing.

Strategy: source inspection of `analytics.js` and `analytics.html`,
the approach this repo uses for its classic-script pages. The
endpoint's half is covered in `test_compare_boundary.py`; the epoch
module's own behaviour is covered in
`tests/web/static/detail_requests.test.js`. What neither can check
is that the compare view actually uses them.

Two client defects sit behind this. Compare had no request fence at
all, so two comparisons could answer out of order and the loser
painted last, and closing the panel cancelled nothing, letting a
dismissed panel repopulate itself. Separately it dropped errored and
autoregressive selections with a bare `continue`, which is how three
ticked runs became one line with no explanation.

Passing proves the fence exists and is compare's own rather than the
detail panel's, that closing retires it, that omissions are rendered
instead of skipped, and that the LLaDA-only label builder is gone.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
ANALYTICS_JS = STATIC / "analytics.js"
ANALYTICS_HTML = STATIC / "analytics.html"


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


# -- compare has its own fence --


def test_compare_has_a_fence_of_its_own() -> None:
    """Sharing the detail panel's would have each cancel the other."""
    source = _js()

    assert "var compareRequests = detailRequestsCreate();" in source
    assert "var detailRequests = detailRequestsCreate();" in source


def test_a_comparison_begins_an_attempt() -> None:
    body = _region("function showComparison(ids)", 1200)

    assert "compareRequests.begin(" in body
    assert "compareRequests.accepts(token)" in body


def test_the_request_carries_the_abort_signal() -> None:
    # The epoch alone would let a superseded fetch run to completion
    # against a server that no longer needs to answer it.
    body = _region("function fetchCompare(ids, signal)", 600)

    assert "signal: signal" in body

    call = _region("fetchCompare(ids, token.signal)", 120)
    assert "token.signal" in call


def test_closing_the_panel_retires_the_request() -> None:
    body = _region("function hideComparison()", 400)

    assert "compareRequests.cancel()" in body


def test_a_failed_comparison_is_reported() -> None:
    # Without this an error is an empty chart, which reads as "these
    # runs are identical and flat" rather than as a failure.
    body = _region("function showComparison(ids)", 1400)

    assert ".catch(" in body
    assert "AbortError" in body


# -- nothing is dropped silently --


def test_omitted_selections_are_rendered() -> None:
    source = _js()

    assert "function renderCompareOmissions(" in source
    assert 'getElementById("compare-omitted")' in source


def test_the_chart_no_longer_skips_rows_silently() -> None:
    """The two bare continues are what this replaced."""
    body = _region("function renderComparison(results)", 1400)

    assert 'entry.status === "data"' in body
    assert "omitted.push(entry)" in body
    assert "renderCompareOmissions(omitted)" in body


def test_the_panel_has_somewhere_to_show_them() -> None:
    markup = ANALYTICS_HTML.read_text(encoding="utf-8")

    assert 'id="compare-omitted"' in markup


# -- labels are the server's job now --


def test_the_llada_only_label_builder_is_gone() -> None:
    """It read steps, gen_length and block_length off the run, so
    every other model rendered the word undefined."""
    source = _js()

    assert "buildCompareLabel" not in source
    assert "run.params.gen_length" not in source


def test_the_label_comes_off_the_response() -> None:
    body = _region("function renderComparison(results)", 2000)

    assert "res.label" in body


def test_the_client_no_longer_derives_the_model_type() -> None:
    # It used to scan the in-memory run list to find out whether a
    # compared run was autoregressive, coupling two endpoints for one
    # string. The compare response carries the answer now.
    source = _js()

    assert "runIdIsAutoregressive" not in source


def test_one_fetch_helper_still_owns_the_url() -> None:
    source = _js()
    calls = re.findall(r"/api/analytics/compare", source)

    assert len(calls) == 1
