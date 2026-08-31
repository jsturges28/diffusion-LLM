"""The page fetches what the catalog stopped sending.

Strategy: source inspection of `analytics.js`, the approach this
repo uses for its classic-script pages. The server half is covered in
`test_run_catalog.py`.

Two things had to move when the catalog became a summary. The detail
panel built its rows straight out of the list entry it already had,
which no longer carries the prompt in full, the hyperparameters, or
the tokenizer and context blocks; it fetches them now, behind the
same epoch as its charts and overlays, because a detail fetch that
outlives its panel is the exact failure `detail_requests.js` was
written for.

Collection membership also had to change shape. It was stored as a
list of runs per collection and asked the opposite question, "which
collections is this run in", once per row: up to 24 linear scans per
row per render. It is indexed by run now, which is also the shape
server-authoritative collections will want, so the index survives
that change where a client-owned copy of the array would not.

Passing proves the fetch exists and is fenced, that a failure costs
the extra rows rather than the panel, and that membership is read
through the index and rebuilt at every site that can change it.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
ANALYTICS_JS = STATIC / "analytics.js"


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


# -- the detail panel fetches its own metadata --


def test_the_panel_fetches_the_full_record() -> None:
    source = _js()

    assert "function fetchRunMeta(runId, signal)" in source
    assert "/metadata" in source


def test_the_fetch_is_behind_the_detail_epoch() -> None:
    """A slow answer must not land on a run already closed."""
    body = _region("function loadRunMeta(", 900)

    assert "detailRequests.accepts(token)" in body
    assert "token.signal" in body


def test_it_is_started_with_the_charts_and_overlays() -> None:
    # One token for all three, taken before any of them start, so
    # the panel paints as one run or not at all.
    body = _region("function showDetail(runId)", 1600)

    assert "loadRunMeta(runId, run, token)" in body
    assert "loadRunCharts(runId, run, token)" in body
    assert "loadRunOverlays(runId, run, token)" in body


def test_the_summary_is_shown_before_the_fetch_lands() -> None:
    # Otherwise the panel is blank for a round trip on every open.
    body = _region("function showDetail(runId)", 1600)

    assert "renderRunMeta(run)" in body


def test_a_failed_metadata_fetch_costs_only_the_extra_rows(
) -> None:
    body = _region("function loadRunMeta(", 900)

    assert ".catch(" in body
    assert "AbortError" in body
    assert "renderRunMeta(summary)" in body


def test_a_cut_prompt_is_marked_as_cut() -> None:
    """A sentence stopping mid-word should not read as typed."""
    body = _region("function promptWithEllipsis(run)", 400)

    assert "run.prompt_truncated" in body


# -- membership is indexed by run --


def test_membership_is_read_through_the_index() -> None:
    source = _js()

    assert "var membershipIndex" in source
    assert "function rebuildMembershipIndex()" in source

    collected = _region("function runIsCollected(runId)", 400)
    assert "membershipIndex[runId]" in collected

    has_run = _region(
        "function collectionHasRun(collection, runId)", 300
    )
    assert "membershipIndex[runId]" in has_run


def test_the_old_per_row_scans_are_gone() -> None:
    """Up to 24 indexOf calls per row per render, before this."""
    collected = _region("function runIsCollected(runId)", 400)
    has_run = _region(
        "function collectionHasRun(collection, runId)", 300
    )

    assert "collections[i].runs.indexOf" not in collected
    assert "collection.runs.indexOf" not in has_run


def test_one_place_changes_membership() -> None:
    """The index cannot go stale if only one path can move it.

    There used to be three: the boot read from localStorage, the
    shared after-change path, and a delete sweep that repainted
    itself and so bypassed the shared one. Each had to remember to
    reindex. Since collections became server-authoritative there is
    exactly one, `adoptCollections`, because the page no longer
    computes a new list at all: it is handed one and takes it.
    """
    source = _js()
    rebuilds = re.findall(r"rebuildMembershipIndex\(\)", source)

    # One definition plus the single call site.
    assert len(rebuilds) == 2, rebuilds
    assert "rebuildMembershipIndex();" in _region(
        "function adoptCollections(list)", 600
    )


def test_the_reindex_precedes_the_repaint() -> None:
    # Both repaints read through the index, so rebuilding after them
    # would draw one render's worth of stale stars.
    body = _region("function adoptCollections(list)", 600)

    assert body.index("rebuildMembershipIndex") < body.index(
        "renderTable"
    )
