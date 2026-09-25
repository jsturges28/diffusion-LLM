"""Tests that Help stays navigable.

Strategy: parse the Help dialog out of `index.html` and check the
structure a reader depends on, the way `test_keyboard_reach.py` checks
the modals around it. Passing proves every tab leads somewhere, every
panel is reachable, and no one panel has grown back into a wall.

The defect this file exists to prevent is not length, it is drift
between a heading and what sits under it. Before the split, a section
titled "Stopping a run" held 21 paragraphs and three of them were
about stopping a run; the rest had accumulated there because each new
feature appended its prose under whichever heading happened to be
last, the resource meter's copy included. Nothing catches that except
a budget that fails when a panel becomes where things get dropped.

Structure only, never prose: the copy will and should change. What
must not change is that a reader can find the part they want.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX = REPO_ROOT / "src" / "web" / "static" / "index.html"
GUIDE = REPO_ROOT / "docs" / "GUIDE.md"

# A ratchet, set just above today's largest panel (1,855 words) with
# room for a feature or two. The whole modal was 11,282 words in one
# scrolling body; the point of bounding each panel rather than the
# total is that the total can hold steady while one panel quietly eats
# everything. Lower it when a split makes room, the way the lint
# baseline works. Do not raise it to make a paragraph fit.
PANEL_WORD_BUDGET = 2_200


def _help_markup() -> str:
    text = INDEX.read_text(encoding="utf-8")
    start = text.index('<dialog id="modal-help"')
    return text[start:text.index("</dialog>", start)]


def _tabs() -> list[str]:
    return re.findall(r'data-help-tab="([\w-]+)"', _help_markup())


def _labels() -> list[str]:
    return re.findall(
        r'data-help-tab="[\w-]+"\s*\n?\s*aria-selected="\w+">(.*?)<',
        _help_markup(),
    )


def _panels() -> dict[str, str]:
    found = re.findall(
        r'data-help-panel="([\w-]+)"(.*?)</section>',
        _help_markup(),
        re.S,
    )
    return {name: html for name, html in found}


def _words(html: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", html).split())


# -- every tab leads somewhere --


def test_there_are_tabs_at_all() -> None:
    assert _tabs(), "the Help modal has no tabs"


def test_every_tab_has_a_panel() -> None:
    """A tab whose panel is missing or misspelled looks right in the
    markup and does nothing when clicked, which is the failure a
    reviewer cannot see."""
    panels = _panels()

    for name in _tabs():
        assert name in panels, f"tab '{name}' leads to no panel"


def test_every_panel_has_a_tab() -> None:
    """The other direction. An unreachable panel is content nobody can
    get to, which is worse than not writing it."""
    tabs = set(_tabs())

    for name in _panels():
        assert name in tabs, f"panel '{name}' has no tab"


def test_the_tabs_are_unique() -> None:
    names = _tabs()

    assert len(names) == len(set(names)), f"duplicate tabs: {names}"


# -- and is labelled --


def test_every_panel_carries_a_heading() -> None:
    """The tab names it, but a panel also states its own title, for a
    reader who lands in one and for anyone reading the HTML."""
    for name, html in _panels().items():
        assert "<h3>" in html, f"panel '{name}' has no heading"


def test_every_panel_has_sections_under_it() -> None:
    """A panel is a grouping. One with no subsections is either a stub
    or a sign the grouping was abandoned."""
    for name, html in _panels().items():
        assert "<h4>" in html, f"panel '{name}' has no sections"


def test_the_active_tab_is_marked_for_a_screen_reader() -> None:
    """role="tab" without aria-selected leaves the active state purely
    visual. The Settings page this pattern comes from does exactly
    that, which is why it is asserted here rather than assumed."""
    markup = _help_markup()

    assert 'aria-selected="true"' in markup
    assert 'aria-selected="false"' in markup


def test_exactly_one_tab_starts_active() -> None:
    markup = _help_markup()

    assert markup.count('aria-selected="true"') == 1
    assert markup.count("is-active") == 1


def test_exactly_one_panel_starts_visible() -> None:
    """The others ship hidden. Two visible panels would stack their
    prose on first open, which is what the tabs exist to stop."""
    panels = _panels()

    visible = [
        name for name, html in panels.items()
        if "hidden" not in html.split(">", 1)[0]
    ]
    assert len(visible) == 1, f"visible panels: {visible}"


# -- and stays readable --


def test_no_panel_has_grown_into_a_wall() -> None:
    for name, html in _panels().items():
        count = _words(html)

        assert count <= PANEL_WORD_BUDGET, (
            f"Help panel '{name}' is {count} words, over the"
            f" {PANEL_WORD_BUDGET} budget. Split it, or move the"
            " reading-rather-than-doing part to docs/GUIDE.md."
        )


def test_the_panels_are_not_wildly_lopsided() -> None:
    """A grouping where one panel holds most of the words is a
    grouping that has stopped working, even if each panel is under
    budget on its own."""
    counts = sorted(_words(html) for html in _panels().values())

    assert counts[-1] <= sum(counts) * 0.45, (
        "one Help panel holds nearly half the modal; the grouping"
        f" needs revisiting (words per panel: {counts})"
    )


# Cross-references are written as "the <strong>Name</strong> tab", a
# shape chosen to be matchable: "under <strong>Commit Order</strong>"
# reads the same to a person but names an overlay, not a tab.
TAB_POINTER = re.compile(r"<strong>([^<]*?)</strong> tab")


def test_cross_references_name_a_tab_that_exists() -> None:
    """Grouping the copy into panels turned three "see the section
    above" pointers into lies, because the section had moved to
    another tab. They now name the tab, which only helps while the
    name is real, and a tab is renamed by editing a label sitting
    nowhere near the sentence that depends on it."""
    labels = set(_labels())

    for named in TAB_POINTER.findall(_help_markup()):
        assert named in labels, (
            f"a Help cross-reference points at a tab called"
            f" {named!r}, and there is no such tab: {sorted(labels)}"
        )


def test_the_tab_pointers_are_actually_used() -> None:
    """Paired with the test above, which passes on an empty set."""
    found = TAB_POINTER.findall(_help_markup())

    assert found, (
        "no Help section points at another tab, so the assertion"
        " that those pointers resolve is proving nothing"
    )


def test_help_says_where_the_deeper_material_went() -> None:
    """Help is the only documentation a user running the app has, so
    prose moved out of it has to leave a forwarding address."""
    markup = _help_markup()

    assert "docs/GUIDE.md" in markup, (
        "nothing in Help points at the guide, so the material moved"
        " out of it is unreachable from the app"
    )


def test_the_guide_it_names_exists() -> None:
    """Paired with the assertion above, from the other side."""
    assert GUIDE.is_file(), "docs/GUIDE.md is missing"
