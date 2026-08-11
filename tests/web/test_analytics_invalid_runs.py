"""A run that cannot be read is shown, not hidden, and not trusted.

Strategy: source inspection of `analytics.js`, the same approach
`test_analytics_escaping.py` takes and for the same reason. That file
is a 5,600-line classic script that touches the DOM at load, so it
cannot be imported into a test; giving it a testable seam is `ORG-02`
in stage 5. The server half of this behavior is tested properly, by
execution, in `tests/analytics/test_run_schema.py`.

What passing proves is narrow but worth pinning. The reason text a
broken run carries reaches the page as text, never as markup: it is
built from `metadata.json` on disk and from a server message about
it, and the detail panel builds HTML by string concatenation. And
opening such a run must not fetch: those two requests can only fail,
and a panel left on a spinner is how a damaged run looks like a
hung app.

The visual result, that the row lines up with its neighbors and the
delete button still works, is a manual check; see
`docs/MANUAL_VERIFICATION.md`.
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


def _source() -> str:
    return ANALYTICS_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone; update this test rather than"
        " deleting it"
    )
    return source[start : start + chars]


# -- the table row --


def test_the_row_puts_the_reason_in_as_text() -> None:
    """textContent, not innerHTML. The reason quotes a parser error
    on a file from disk, so it can contain anything."""
    region = _region("One spanning cell rather than per-column", 700)

    assert "tdWhy.textContent" in region
    assert "tdWhy.innerHTML" not in region


def test_the_row_does_not_offer_comparison() -> None:
    """A broken run has nothing to plot, so it must not be
    selectable into a comparison that would then fail."""
    region = _region("A run the server could not read", 900)

    assert "if (!run.invalid)" in region


def test_the_spanning_cell_replaces_the_data_cells() -> None:
    """Both halves of the arithmetic: the per-column loop is skipped
    and the edited column is not appended, because the spanning cell
    already covers both. Getting this wrong shifts every later cell
    in the row."""
    source = _source()

    assert "!run.invalid && k < TABLE_KEYS.length" in source
    assert "tdWhy.colSpan = TABLE_KEYS.length + 1" in source


# -- the detail panel --


def test_opening_a_broken_run_escapes_the_reason() -> None:
    region = _region("function showInvalidDetail", 900)

    assert "escHtml(reason)" in region


def test_opening_a_broken_run_fetches_nothing() -> None:
    """The early return has to come before the request token is
    taken, or the panel waits on two requests that cannot succeed."""
    region = _region("function showDetail(runId)", 700)

    early_return = region.find("showInvalidDetail(listed)")
    begins_fetch = region.find("detailRequests.begin")

    assert early_return != -1
    assert begins_fetch != -1
    assert early_return < begins_fetch


def test_opening_a_broken_run_clears_the_previous_one() -> None:
    """Otherwise the last run's charts sit under this run's title,
    which reads as this run's data."""
    region = _region("function showInvalidDetail", 1400)

    assert "clearRunCharts()" in region
    assert "clearOverlay()" in region


def test_the_panel_says_the_folder_is_still_there() -> None:
    """The one useful action is deleting it, and that is only an
    obvious choice if the user knows nothing was lost yet."""
    region = _region("function showInvalidDetail", 1400)

    assert "still on disk" in region
