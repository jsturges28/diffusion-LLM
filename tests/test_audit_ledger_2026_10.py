"""The second audit's ledger stays a ledger.

Strategy: parse the findings index out of
`docs/audit/AUDIT_REPORT_2026-10.md` and the findings table out of
`docs/audit/IMPLEMENTATION_LEDGER_2026-10.md`, then check that the two
agree, the way `test_manual_verification_ledger.py` holds its ledger
to the items it accounts for.

The first campaign's ledger reached 3,899 lines because every session
appended its story to it. Passing proves that every finding the report
raised has exactly one row and no row names a finding it did not
raise; that each row is whole and states a status the ledger defines;
that a row claiming a fix names the commits that made it, and a
blocked row what it waits on; and that the file still fits a budget
only a deliberate edit can raise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "docs" / "audit"
REPORT = AUDIT / "AUDIT_REPORT_2026-10.md"
LEDGER = AUDIT / "IMPLEMENTATION_LEDGER_2026-10.md"

# Set above the ledger's length on the day it was written, with room
# for what the last two stages will decide. Raise it on purpose and
# say why in the commit: a ledger that needs much more has started
# telling stories again.
LINE_BUDGET = 150

# The ledger's own vocabulary, defined under its Statuses heading.
STATUSES = frozenset(
    {"ready", "blocked", "needs hardware", "done", "deferred"}
)

# The statuses that claim code has landed.
LANDED = frozenset({"needs hardware", "done"})

# ID, stage, status, commits, manual items, waits on.
COLUMNS = 6

_COMMIT = re.compile(r"`[0-9a-f]{7,40}`")


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def _report_ids() -> List[str]:
    text = REPORT.read_text(encoding="utf-8")
    index = _section(text, "## Findings index")
    return re.findall(r"^\| (A2-[A-Z]+-\d{2}) \|", index, re.M)


def _rows() -> List[List[str]]:
    text = LEDGER.read_text(encoding="utf-8")
    table = _section(text, "## Findings")
    rows: List[List[str]] = []
    for line in table.splitlines():
        if line.startswith("| A2-"):
            cells = line.strip().strip("|").split("|")
            rows.append([cell.strip() for cell in cells])
    return rows


def test_the_report_has_findings_to_track() -> None:
    """Guards the checks below, which all pass on an empty table."""
    assert len(_report_ids()) == 21


def test_every_finding_has_exactly_one_row() -> None:
    ids = [row[0] for row in _rows()]

    assert sorted(ids) == sorted(_report_ids())
    assert len(ids) == len(set(ids))


def test_every_row_has_every_column() -> None:
    """A cell lost to a missing bar shifts every cell after it, and
    the status check would then read the wrong column."""
    ragged = [row[0] for row in _rows() if len(row) != COLUMNS]

    assert ragged == []


def test_every_row_states_a_known_status() -> None:
    unknown = [
        (row[0], row[2]) for row in _rows() if row[2] not in STATUSES
    ]

    assert unknown == []


def test_a_landed_finding_names_its_commits() -> None:
    """The record a reader checks the claim against."""
    bare = [
        row[0]
        for row in _rows()
        if row[2] in LANDED and not _COMMIT.search(row[3])
    ]

    assert bare == []


def test_a_blocked_finding_names_what_it_waits_on() -> None:
    silent = [
        row[0]
        for row in _rows()
        if row[2] == "blocked" and not row[5]
    ]

    assert silent == []


def test_the_ledger_stays_short() -> None:
    count = len(LEDGER.read_text(encoding="utf-8").splitlines())

    assert count <= LINE_BUDGET, (
        f"the ledger is {count} lines, over its {LINE_BUDGET}. Move"
        " reasoning to docs/ROADMAP.md and history to git rather than"
        " raising the budget to fit it."
    )
