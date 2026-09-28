"""Every hardware item has a recorded status.

Strategy: parse the numbered items out of
`docs/MANUAL_VERIFICATION.md` and the ranges out of its own "What has
been checked" ledger, and require the second to cover the first.
Passing proves the document can answer "was this verified" without
anyone reading a chat transcript.

**This exists because it could not.** The ledger stopped at item 216
while the items ran to 325, so 109 of them had no recorded state
anywhere in the file. A fresh session picking the work up had to ask,
and the answer was only reconstructible from conversation, which is
precisely the failure `META-01` moved this file out of the handoff to
avoid. Writing an item down and leaving its outcome in chat is worse
than not writing it down, because the item looks like a record.

"Not recorded" is an allowed status and is not the same as
passing. The point is that the gap is stated, not silent: a reader
who sees "217 to 266: status not recorded" knows not to trust those,
where a reader who sees nothing at all assumes someone checked.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "MANUAL_VERIFICATION.md"

LEDGER_HEADING = "## What has been checked"

# A numbered scenario, at the start of a line.
_ITEM = re.compile(r"^(\d+)\. ", re.M)

# A ledger entry: "- **216**", "- **217 to 266**",
# "- **142 and 145**".
_ENTRY = re.compile(
    r"^- \*\*(\d+)(?:\s+(?:to|and)\s+(\d+))?\*\*", re.M
)


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _items() -> List[int]:
    return [int(n) for n in _ITEM.findall(_text())]


def _ledger() -> str:
    text = _text()
    start = text.index(LEDGER_HEADING)
    rest = text[start + len(LEDGER_HEADING):]
    match = re.search(r"^## ", rest, re.M)
    finish = len(rest) if match is None else match.start()
    return rest[:finish]


def _covered() -> Set[int]:
    """Every item number the ledger accounts for.

    A range covers its endpoints inclusive, which is how these were
    written: "133 to 141" means all nine.
    """
    covered: Set[int] = set()
    for first, last in _ENTRY.findall(_ledger()):
        start = int(first)
        end = int(last) if last else start
        covered.update(range(start, end + 1))
    return covered


def test_there_are_items_and_a_ledger() -> None:
    """Guards the checks below, which pass on an empty file."""
    assert len(_items()) > 100
    assert len(_covered()) > 100


def test_every_item_has_a_recorded_status() -> None:
    """The gap this file exists for. An item with no entry is one
    nobody can say anything about."""
    missing = sorted(set(_items()) - _covered())

    assert missing == [], (
        f"{len(missing)} manual items have no status in the"
        f" '{LEDGER_HEADING}' ledger: {missing[:12]}."
        " Add a range, using 'status not recorded' if that is"
        " the truth. An item whose outcome lives only in chat looks"
        " like a record and is not one."
    )


def test_no_item_number_is_missing() -> None:
    """A gap means an item was deleted rather than marked, which loses
    the reason it was written and silently shrinks a ledger range."""
    items = set(_items())
    expected = set(range(min(items), max(items) + 1))

    assert sorted(expected - items) == [], (
        f"item numbers missing: {sorted(expected - items)[:8]}"
    )


def test_no_item_number_is_used_twice() -> None:
    """The half that matters for the ranges: two items numbered 240
    would make "240 to 250" cover eleven scenarios and twelve items.

    Document order is deliberately not checked. Item 239 sits after
    241, from an earlier pass, and reordering prose for tidiness is
    not worth the diff: the ledger indexes by number, so
    what has to hold is that each number means one scenario.
    """
    items = _items()

    duplicates = sorted(
        number for number in set(items) if items.count(number) > 1
    )
    assert duplicates == [], f"repeated item numbers: {duplicates}"


def test_the_ledger_reaches_the_last_item() -> None:
    """Stated separately from the coverage check because this is the
    way it failed: the ledger simply stopped being extended, and the
    items kept arriving."""
    items = _items()

    assert max(_covered()) >= max(items), (
        f"the ledger stops at {max(_covered())} and the items run to"
        f" {max(items)}"
    )


def _statuses() -> List[Tuple[str, str]]:
    """Each entry as (its range, the status it claims).

    Parsed per entry rather than searched across the section, which
    is the difference between this meaning something and not: the word
    "outstanding" also appears in the prose ("this is the outstanding
    debt"), so a whole-section search is satisfied by incidental text.
    Rewriting every status to "confirmed" passed until this was
    structural.
    """
    found: List[Tuple[str, str]] = []
    for entry in re.split(r"^- \*\*", _ledger(), flags=re.M)[1:]:
        label, _, body = entry.partition("**:")
        lowered = body.lower()
        # Ordered most specific first: "not yet validated" has to be
        # read before "validated", and "not reachable" before the
        # confirmations, or a negative reads as a pass.
        if "not recorded" in lowered:
            status = "unrecorded"
        elif ("not yet validated" in lowered
                or "outstanding" in lowered):
            status = "outstanding"
        elif any(phrase in lowered for phrase in (
            "not runnable", "not verifiable", "not reachable",
            "cannot be forced",
        )):
            status = "impossible"
        elif any(phrase in lowered for phrase in (
            "confirmed", "done", "validated", "judged",
        )):
            status = "confirmed"
        else:
            status = "unclear"
        found.append((label.strip(), status))
    return found


def test_every_entry_states_a_status_plainly() -> None:
    """An entry a reader cannot classify is the same problem as no
    entry, one step later."""
    unclear = [label for label, status in _statuses()
               if status == "unclear"]

    assert unclear == [], (
        f"these ledger entries do not say what happened: {unclear}"
    )


def test_the_ledger_distinguishes_states() -> None:
    """The ledger is only useful if it says different things about
    different items. One that claimed everything passed would satisfy
    every other check here and mislead every reader."""
    statuses = {status for _, status in _statuses()}

    assert "confirmed" in statuses, "nothing is recorded as confirmed"
    assert statuses & {"outstanding", "unrecorded", "impossible"}, (
        "every entry claims success, which is not the state this"
        " project has ever been in; check whether statuses were"
        " overwritten"
    )


def test_the_ledger_claims_no_item_that_does_not_exist() -> None:
    """The other direction. A range extended past the last item reads
    as verification of scenarios nobody wrote."""
    items = set(_items())

    invented = sorted(
        number for number in _covered()
        if number not in items and number >= min(items)
    )
    assert invented == [], (
        f"the ledger covers items that do not exist: {invented[:8]}"
    )
