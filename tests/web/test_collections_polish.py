"""The Analytics collections polish: copy, confirms, and filing.

Strategy: source inspection of the shipped classic scripts, the
approach this repo uses for pages that cannot be imported. The
operations behind bulk filing are tested for real in
`test_collection_ops.py`, the endpoint in
`test_collection_endpoints.py`, and the requests in the node test
beside the client module. What none of those can see is the page:
whether a refusal is reworded before it reaches a toast, whether a
confirmation fires when there is nothing to confirm, and whether the
Show all flag is consulted where it has to be and cleared where it
has to be.

The reason-coverage test is the one worth keeping honest. It reads
the reasons out of `src/web/collections.py` rather than listing them,
so adding a reason to the server without adding copy for it fails
here instead of shipping API text into a toast.

Passing proves every refusal the server can raise has page copy, that
an empty collection is deleted without a dialog while a populated one
still asks, and that Show all relaxes exactly one filter and does not
outlive the visit that turned it on.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


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


# -- refusals stop reading like API text --


def test_every_server_reason_has_page_copy() -> None:
    """Read out of the server rather than listed here, so a new
    reason cannot land without copy and reach a toast as the API
    string it was written to be."""
    server_source = (
        ROOT / "src" / "web" / "collections.py"
    ).read_text(encoding="utf-8")
    reasons = set(
        re.findall(r'^REASON_\w+ = "(\w+)"', server_source, re.M)
    )
    page = _region("analytics.js", "var COLLECTION_REFUSALS = {", 900)

    assert reasons, "no reasons found; the server file moved"
    for reason in sorted(reasons):
        assert f"{reason}:" in page, reason


def test_the_server_reasons_the_page_adds_are_real() -> None:
    """The other direction. Copy for a reason the server cannot
    raise is dead text that reads as coverage."""
    server_source = (
        ROOT / "src" / "web" / "collections.py"
    ).read_text(encoding="utf-8")
    endpoints = (ROOT / "src" / "web" / "server.py").read_text(
        encoding="utf-8"
    )
    page = _region("analytics.js", "var COLLECTION_REFUSALS = {", 900)
    mapped = set(re.findall(r"^  (\w+):", page, re.M))

    assert mapped
    for reason in sorted(mapped):
        found = (
            f'"{reason}"' in server_source
            or f'"{reason}"' in endpoints
        )
        assert found, reason


def test_a_refusal_goes_through_the_copy_map() -> None:
    """The regression this fixes: `runCollectionOp` had the reason
    available and printed the server's message anyway."""
    body = _region("analytics.js", "function runCollectionOp(", 400)

    assert "collectionRefusalText(error)" in body
    assert "error.message" not in body


def test_the_chooser_reuses_the_same_copy() -> None:
    """Two refusal paths reading differently for the same refusal
    would make the wording look accidental."""
    body = _region(
        "analytics.js", "function onCreateCollectionFromChooser(", 900
    )

    assert "collectionRefusalText(error)" in body


def test_the_cap_is_not_quoted_back_into_the_page() -> None:
    """The page stopped owning the collection limit when the server
    took it. A number here is a second copy that can drift."""
    source = _source("analytics.js")

    assert "COLLECTIONS_MAX" not in source


# -- an empty collection is not worth a dialog --


def test_an_empty_collection_is_deleted_without_asking() -> None:
    body = _region(
        "analytics.js", "function openCollectionDeleteModal(", 700
    )

    assert "collectionPresentCount(collection) === 0" in body
    assert "deleteCollection(collection.id)" in body


def test_a_populated_collection_still_asks() -> None:
    """The confirm is not removed, only skipped: deleting a filled
    collection throws away filing nothing on disk can rebuild."""
    body = _region(
        "analytics.js", "function openCollectionDeleteModal(", 700
    )

    assert "modalCollectionDelete.classList.remove" in body
    assert "pendingCollectionDelete = collection.id" in body


# -- filing several at once --


def test_the_bulk_star_files_in_one_call() -> None:
    """A loop here would be the shape `DATA-02` existed to remove:
    six writes that can stop at four."""
    body = _region("analytics.js", "function onBulkStar(", 300)

    assert "fileRunsInto(" in body
    assert "for (" not in body


def test_filing_a_selection_uses_the_bulk_endpoint() -> None:
    body = _region("analytics.js", "function fileRunsInto(", 600)

    assert "collectionsApi.addRuns(" in body


def test_naming_a_collection_for_a_selection_is_one_call() -> None:
    body = _region(
        "analytics.js",
        "function createCollectionForSelection(",
        700,
    )

    assert ".createWithRuns(name, runIds)" in body
    assert "collectionsApi" in body


def test_the_bulk_buttons_appear_with_the_delete() -> None:
    """One entry point, because a selection that can be deleted can
    always be filed and the two must not disagree about whether
    there is one."""
    body = _region("analytics.js", "function updateBulkActions(", 200)

    assert "updateBulkDeleteButton();" in body
    assert "updateBulkCollectButtons();" in body


def test_nothing_calls_the_halves_directly() -> None:
    source = _source("analytics.js")
    direct = re.findall(
        r"^\s+updateBulkDeleteButton\(\);", source, re.M
    )

    assert len(direct) == 1, "a call site skipped updateBulkActions"


def test_the_chooser_renders_targets_for_a_selection() -> None:
    """Add-only for several runs. A checkbox would have to be
    tri-state to answer "are these six in here" honestly."""
    body = _region(
        "analytics.js", "function renderCollectionChoices(", 700
    )

    assert "chooserIsBulk()" in body
    assert "buildCollectionTarget(" in body


def test_a_target_row_carries_no_checkbox() -> None:
    body = _region(
        "analytics.js", "function buildCollectionTarget(", 1200
    )

    assert 'type = "checkbox"' not in body
    assert "data-collection-id" in body


def test_the_chooser_hears_the_event_a_target_can_emit() -> None:
    """The bug that shipped, caught by pairing two facts the tests
    either side of this one hold separately.

    A target row is a button, and a button never fires `change`. The
    list listened for `change` only, so target mode rendered
    correctly, counted overlaps correctly, and could not be clicked:
    the rows were inert and nothing said so. Neither neighbouring
    test was wrong, and the defect lived in the seam between them.
    """
    target = _region(
        "analytics.js", "function buildCollectionTarget(", 400
    )
    wiring = _region("analytics.js", "if (collectionChoices) {", 300)

    assert 'createElement("button")' in target
    assert '"click", onCollectionTargetClick' in wiring
    assert '"change", onCollectionChoiceToggle' in wiring


def test_each_chooser_handler_ignores_the_other_mode() -> None:
    """Both listeners see every event on the list, and a label click
    fires alongside every checkbox change. Without the guard a
    single-run tick would also file the whole batch."""
    target = _region(
        "analytics.js", "function onCollectionTargetClick(", 400
    )
    toggle = _region(
        "analytics.js", "function onCollectionChoiceToggle(", 400
    )

    assert "if (!chooserIsBulk())" in target
    assert "chooserRunId === null" in toggle


def test_the_selection_is_captured_when_the_dialog_opens() -> None:
    """Read at click time instead, a selection cleared behind the
    dialog would turn a target click into a silent no-op."""
    body = _region(
        "analytics.js", "function openCollectionBulkChooser(", 500
    )

    assert "runIds.slice()" in body


def test_closing_the_chooser_forgets_the_selection() -> None:
    """Left set, the next single-run open would render targets."""
    body = _region(
        "analytics.js", "function closeCollectionChooser(", 250
    )

    assert "chooserRunIds = null" in body


# -- Show all --


def test_show_all_relaxes_the_membership_filter() -> None:
    body = _region("analytics.js", "function visibleRuns(", 900)

    assert "showAllInCollection" in body
    assert "return allRuns;" in body


def test_show_all_is_checked_after_the_deleted_fallback() -> None:
    """Checked first, a collection deleted in another window would
    leave its tab selected with the filter switched off."""
    body = _region("analytics.js", "function visibleRuns(", 900)
    fallback = body.find("activeCollectionId = null;")
    relaxed = body.find("if (showAllInCollection)")

    assert fallback != -1
    assert relaxed != -1
    assert fallback < relaxed


def test_switching_tabs_turns_show_all_off() -> None:
    """Per-visit, not a preference: a collection view that quietly
    showed non-members next time is no longer a collection view."""
    body = _region("analytics.js", "function selectCollection(", 600)

    assert "showAllInCollection = false;" in body


def test_toggling_show_all_clears_the_selection() -> None:
    """Same reason a tab change clears it: a bulk gesture must not
    reach a run that is no longer on screen."""
    body = _region("analytics.js", "function onShowAllToggle(", 600)

    assert "checkedIds = {};" in body


def test_the_toggle_is_hidden_outside_a_collection() -> None:
    """All is already every run, so there is nothing to relax."""
    body = _region(
        "analytics.js", "function updateShowAllToggle(", 700
    )

    assert "activeCollectionId === null" in body
    assert "btnShowAll.hidden = true;" in body


def test_the_star_files_into_the_collection_you_are_in() -> None:
    """Having just turned on Show all to file into this collection,
    being sent to Favorites would be the wrong answer.

    Asserts the star asks, not only that something can answer: a
    hardcoded FAVORITES_ID at the call site would leave
    `bulkFileTarget` correct and unused.
    """
    star = _region("analytics.js", "function onBulkStar(", 300)
    target = _region("analytics.js", "function bulkFileTarget(", 300)

    assert "fileRunsInto(bulkFileTarget(), runs)" in star
    assert "activeCollectionId" in target
    assert "FAVORITES_ID" in target


def test_the_star_names_where_it_files() -> None:
    """The target changes with the tab, and a label that always said
    Favorites would be wrong exactly when it mattered."""
    body = _region(
        "analytics.js", "function updateBulkCollectButtons(", 1200
    )

    assert "bulkFileTargetName()" in body


def test_already_filed_rows_are_marked() -> None:
    """Without it the table under Show all is a flat list with no
    way to see what the visit was for."""
    source = _source("analytics.js")

    assert "row-already-filed" in source
    assert "runIsInActiveCollection(" in source
    assert "row-already-filed" in _source("analytics.css")


def test_the_empty_collection_copy_points_at_the_toggle() -> None:
    """It used to send the reader to All, which meant leaving the
    collection they were trying to fill."""
    markup = _source("analytics.html")
    body = _region(
        "analytics.html", 'id="runs-empty-collection"', 400
    )

    assert "Show all runs" in body
    assert "Star a run under" not in markup


def test_show_all_takes_the_general_empty_message() -> None:
    """With the filter relaxed, the collection message would tell
    the reader to turn on a toggle that is already on."""
    body = _region("analytics.js", "var inCollection =", 200)

    assert "!showAllInCollection" in body
