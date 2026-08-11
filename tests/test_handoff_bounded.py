"""Tests that the cold-start page stays a page.

Strategy: count the lines in `HANDOFF.md` and look for the shapes that
made it unreadable before. Passing proves a cold reader still reaches
the current state cheaply.

This is a ratchet, not a style preference. The file reached 3,233
lines the way such files always do: every session appended what it had
just shipped, each addition was individually reasonable, and nobody
ever removed anything. The audit needed a special brief telling agents
to read only the first 73 lines of it. Nothing prevents that happening
again except a number that fails, so here is the number.

The bound is checked rather than the content, because the content will
and should change. What must not change is the cost of reading it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = REPO_ROOT / "HANDOFF.md"

# The finding asks for "roughly 200 lines". Taken literally, with no
# slack: slack is how the last one got to 3,233.
LINE_BUDGET = 200


def _lines() -> list[str]:
    return HANDOFF.read_text(encoding="utf-8").splitlines()


def test_the_handoff_fits_in_one_reading() -> None:
    count = len(_lines())

    assert count <= LINE_BUDGET, (
        f"HANDOFF.md is {count} lines, over the {LINE_BUDGET} line"
        " budget. Durable rationale belongs in ROADMAP.md, hardware"
        " scenarios in MANUAL_VERIFICATION.md, shipped features in"
        " README.md, and session narrative in git history."
    )


def test_it_still_carries_the_orientation() -> None:
    """The other half of the bound. A short file that dropped the
    architecture would pass the count and fail the reader."""
    text = HANDOFF.read_text(encoding="utf-8")

    for heading in (
        "## What it is",
        "## Models",
        "## Architecture",
        "## Where things stand",
    ):
        assert heading in text, f"missing section: {heading}"


def test_no_session_narrative_creeps_back() -> None:
    """"This session" is the phrase that grew the old file.

    It also dates the page the moment the session ends, which is what
    made stale claims read as current ones.
    """
    text = HANDOFF.read_text(encoding="utf-8").lower()

    for phrase in ("this session shipped", "recently shipped"):
        assert phrase not in text, (
            f"'{phrase}' belongs in git history, not the handoff"
        )


def test_no_duplicate_top_level_numbering() -> None:
    """The old file had two separate backlog entries numbered zero,
    which made "item 1" ambiguous across a 3,000-line document."""
    numbers = re.findall(r"^(\d+)\.\s", "\n".join(_lines()), re.M)

    assert len(numbers) == len(set(numbers)), (
        f"duplicate list numbers in HANDOFF.md: {sorted(numbers)}"
    )


def test_it_points_at_where_the_detail_went() -> None:
    """A bounded page only works if it says where the rest is."""
    text = HANDOFF.read_text(encoding="utf-8")

    for target in (
        "AGENTS.md",
        "ROADMAP.md",
        "MANUAL_VERIFICATION.md",
        "IMPLEMENTATION_LEDGER.md",
        "TIGERSTYLE.md",
    ):
        assert target in text, f"handoff never mentions {target}"
