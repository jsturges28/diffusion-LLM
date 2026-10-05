"""Analytics metadata keeps long opaque values readable.

Strategy: inspect the three shipped detail-metadata rules that divide
each row into a fixed label and a flexible value. Browser-module
tests exercise the generated markup; these checks cover CSS layout
properties that the synthetic DOM does not calculate.

Passing proves long identifiers wrap anywhere without truncation while
their labels stay intact.
"""

from __future__ import annotations

from pathlib import Path

ANALYTICS_CSS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "analytics.css"
)


def _rule(styles: str, selector: str) -> str:
    start = styles.find(selector)
    assert start != -1, selector
    end = styles.find("}", start)
    assert end != -1, selector
    return styles[start : end + 1]


def test_metadata_labels_stay_fixed() -> None:
    """Labels cannot surrender width to an opaque value."""
    styles = ANALYTICS_CSS.read_text(encoding="utf-8")
    label = _rule(styles, "#detail-meta .meta-label {")

    assert "flex: 0 0 auto" in label
    assert "white-space: nowrap" in label


def test_metadata_values_wrap_without_truncation() -> None:
    """IDs remain complete and may break at any available width."""
    styles = ANALYTICS_CSS.read_text(encoding="utf-8")
    value = _rule(styles, "#detail-meta .meta-value {")

    assert "flex: 1 1 auto" in value
    assert "min-width: 0" in value
    assert "overflow-wrap: anywhere" in value
    assert "word-break: break-word" in value
    assert "text-overflow" not in value
    assert "ellipsis" not in value


def test_metadata_rows_share_one_flexible_line() -> None:
    """The fixed and flexible children participate in flex layout."""
    styles = ANALYTICS_CSS.read_text(encoding="utf-8")
    row = _rule(styles, "#detail-meta .meta-row {")

    assert "display: flex" in row
    assert "align-items: baseline" in row
