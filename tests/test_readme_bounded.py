"""Tests that the front page stays a front page.

Strategy: count the words in `README.md`, then check the things a
short file can drop while still passing a count. Passing proves
somebody arriving cold can decide whether they care, get it running,
and find the detail, without reading a manual first.

This is a ratchet, in the same spirit as `test_handoff_bounded.py`
and for the same reason. The README reached 16,240 words the way such
files do: each feature appended paragraphs under whichever heading was
nearest, every addition was individually reasonable, and nothing was
ever moved out. Three quarters of it sat in two sections, and the
larger of those was a 6,979-word "Quickstart", which is a user manual
wearing the wrong name.

**Words rather than lines**, unlike the handoff test. This file's
paragraphs are one per line and some ran past 1,400 characters, so a
line count passed a file nobody could read. Words are what the reader
spends.

The budget is checked rather than the prose, because the prose will
and should change. What must not change is the cost of the first five
minutes.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
GUIDE = REPO_ROOT / "docs" / "GUIDE.md"

# Roughly a five-minute read. Set just above where the split landed
# (1,249 words) with room for a feature or two, and no more: slack is
# how the last one reached 16,240.
WORD_BUDGET = 1_600


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _words() -> list[str]:
    """Prose words, with fenced code blocks and tables left out.

    A diagram and a dependency table are reference material a reader
    skips or scans; counting them would spend the budget on the parts
    that are not read linearly.
    """
    text = re.sub(r"```.*?```", " ", _text(), flags=re.S)
    text = "\n".join(
        line for line in text.split("\n")
        if not line.startswith("|")
    )
    return text.split()


def test_the_readme_fits_in_one_sitting() -> None:
    count = len(_words())

    assert count <= WORD_BUDGET, (
        f"README.md is {count} prose words, over the {WORD_BUDGET}"
        " budget. Feature detail belongs in docs/GUIDE.md, durable"
        " rationale in docs/ROADMAP.md, hardware scenarios in"
        " docs/MANUAL_VERIFICATION.md, and session narrative in git"
        " history."
    )


def test_it_still_says_what_this_is() -> None:
    """The first thing a stranger needs, and the easiest to lose while
    cutting: a sentence naming what the project actually is."""
    opening = " ".join(_words()[:80]).lower()

    assert "diffusion" in opening
    assert "language model" in opening or "llm" in opening.lower()


def test_it_still_carries_the_orientation() -> None:
    """The other half of the bound. A short file that dropped setup
    would pass the count and leave a reader unable to run it."""
    text = _text()

    for heading in (
        "## The models",
        "## Architecture",
        "## Setup",
        "## Quickstart",
        "## Documentation",
    ):
        assert heading in text, f"missing section: {heading}"


def test_it_points_at_where_the_detail_went() -> None:
    """A front page that cut the manual without saying where it went
    would be worse than the long one: the reader would conclude the
    detail does not exist."""
    text = _text()

    assert "docs/GUIDE.md" in text, "nothing points at the guide"
    assert "docs/ROADMAP.md" in text, "nothing points at the roadmap"


def test_the_guide_it_points_at_exists() -> None:
    """Paired with the assertion above, from the other side. A link to
    a file nobody wrote is the failure this whole split could produce.
    """
    assert GUIDE.is_file(), "docs/GUIDE.md is missing"

    assert len(GUIDE.read_text(encoding="utf-8").split()) > 3_000, (
        "docs/GUIDE.md is too small to be the manual the README"
        " promises; the detail probably did not survive the move"
    )


def test_the_manual_did_not_creep_back() -> None:
    """The specific shape that produced the 16,240 words: feature
    prose filed under a heading that promised brevity."""
    text = _text()

    for heading in (
        "#### Interactive remasking",
        "#### Visual overlays",
        "#### Analytics Suite",
        "#### Saving and reproducibility",
    ):
        assert heading not in text, (
            f"{heading} is back in README.md; it belongs in"
            " docs/GUIDE.md"
        )


def test_it_shows_the_app_rather_than_only_describing_it() -> None:
    """A tool whose whole point is what it draws should draw something
    on its front page. Checked because a screenshot is the first thing
    a prose edit drops."""
    text = _text()

    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)

    assert images, "README.md shows no screenshot"
    for path in images:
        assert (REPO_ROOT / path).is_file(), f"missing image: {path}"


def test_every_image_carries_alt_text() -> None:
    """The boundary of the assertion above: a screenshot nobody can
    describe is not an improvement for a reader using a screen
    reader."""
    for alt in re.findall(r"!\[([^\]]*)\]\([^)]+\)", _text()):
        assert alt.strip(), "an image has empty alt text"
