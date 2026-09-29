"""Tests that the roadmap's orientation section stays an orientation.

Strategy: bound the one section whose heading makes a promise about
reading time, and check the things a short section can drop while
still passing a count. Passing proves somebody opening the roadmap
learns what this is and which document answers what, without reading
a build history first.

**The file is deliberately not bounded**, unlike `README.md` and
`docs/HANDOFF.md`. The roadmap is the project's memory and runs past
21,000 words on purpose; length is its job. What length destroys is a
section titled "Current status (orientation)", which reached 8,488
words and 39% of the file by accumulating twenty bullets that each
opened "Shipped (this session)". Twenty sessions all claiming to be
now is why it opened "Phase 1 is complete: both models" while three
models shipped.

So the bound goes on the section, the way the Help budget went on each
panel rather than on the modal. The history it used to hold is still
in the file, under a heading that says it is a record.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"

ORIENTATION = "## Current status (orientation)"

# Roughly a one-minute read, set just above where the cut landed (206
# words) with room for a model or a document. No more than that: slack
# is how the last one reached 8,488.
WORD_BUDGET = 400

# The phrase that grew it. It also dates the text the moment the
# session ends, which is what turned stale claims into current ones.
FORBIDDEN = ("this session", "latest session", "shipped (this")


def _text() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def _orientation() -> str:
    """The orientation section, heading to the next heading."""
    text = _text()
    assert ORIENTATION in text, "the orientation section is gone"
    start = text.index(ORIENTATION)
    rest = text[start + len(ORIENTATION):]
    match = re.search(r"^## ", rest, re.M)
    finish = start + len(ORIENTATION) + (
        match.start() if match else len(rest)
    )
    return text[start:finish]


def _words(section: str) -> list[str]:
    """Prose words, with the ownership table left out.

    The table is reference material a reader scans rather than reads,
    and it is the part that should grow when a document is added, so
    counting it would spend the budget on the wrong thing. Same
    reasoning as `test_readme_bounded.py`.
    """
    kept = [
        line for line in section.split("\n")
        if not line.startswith("|")
    ]
    return "\n".join(kept).split()


# -- the bound --


def test_the_orientation_orients_quickly() -> None:
    count = len(_words(_orientation()))

    assert count <= WORD_BUDGET, (
        f"the roadmap's orientation section is {count} words, over"
        f" the {WORD_BUDGET} budget. Durable reasoning belongs in"
        " Settled decisions, a pass write-up in Build record, what a"
        " feature does in GUIDE.md, and what shipped in git history."
    )


def test_the_rest_of_the_file_is_not_bounded() -> None:
    """Stated so nobody adds a file-wide budget later and starts
    deleting the reasoning this document exists to hold."""
    count = len(_text().split())

    assert count > 10_000, (
        f"ROADMAP.md is down to {count} words. It is the project's"
        " memory, not a summary; check what was deleted."
    )


# -- and the half a bound cannot check --


def test_it_still_says_what_runs_here() -> None:
    """A short section that dropped the models would pass the count
    and fail the reader."""
    section = _orientation()

    for model in ("LLaDA", "DiffusionGemma", "SmolLM3", "Mamba-3"):
        assert model in section, f"orientation omits {model}"


def test_it_still_explains_the_split_environments() -> None:
    """The one architectural fact that surprises everybody, and the
    reason a reader cannot just `pip install -r` one file."""
    section = _orientation().lower()

    assert "transformers" in section
    assert "resident" in section


def test_it_routes_each_question_to_one_document() -> None:
    """META-03's actual ask: define ownership. A reader who cannot
    tell which of five documents answers their question reads the
    wrong one, or writes into the wrong one, which is how this
    section grew."""
    section = _orientation()

    for target in (
        "docs/GUIDE.md",
        "docs/HANDOFF.md",
        "AGENTS.md",
        "git history",
    ):
        assert target in section, (
            f"the orientation does not route to {target}"
        )


def test_no_session_narrative_creeps_back() -> None:
    for phrase in FORBIDDEN:
        assert phrase not in _orientation().lower(), (
            f"the orientation section says {phrase!r}. That is what"
            " grew it to 8,488 words; a pass write-up belongs under"
            " Build record, which does not claim to be the present."
        )


def test_the_history_it_used_to_hold_is_still_in_the_file() -> None:
    """The cut moved words rather than deleting them, and this is the
    assertion that keeps that true. Without it the cheapest way to
    pass the budget above is to delete the record instead of filing
    it."""
    text = _text()

    assert "## Build record" in text, (
        "the Build record section is gone; the orientation's history"
        " was supposed to move there, not be deleted"
    )
