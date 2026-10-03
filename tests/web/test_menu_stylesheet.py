"""The Main Menu's styles live in a stylesheet only the menu loads.

Strategy: read the pages and stylesheets as text, the way
`test_vision_page.py` reads the page it guards. Parse the class and id
selectors out of `menu.css`, and the names each page's markup and the
scripts it loads could build, then check where the rules live, who
loads them, and whom they can match.

What is being protected is a navigation cut, not a redesign. The last
679 lines of `style.css` styled one page and were parsed by all five.
Passing proves the menu loads its rules after the shared ones, which
keeps the cascade order they had as the tail of `style.css`; that no
other page loads them; that every rule names something the menu
builds and nothing another page builds alone; and that the rules left
behind by the generator dropdown's old headroom pill and the menu's
old status line did not come along.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "src" / "web" / "static"
MENU_CSS = STATIC / "menu.css"
SHARED_CSS = STATIC / "style.css"

# Every page other than the menu. None may load the menu's rules.
OTHER_PAGES = (
    "index.html",
    "analytics.html",
    "settings.html",
    "vision.html",
)


def _text(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _page_sources(page: str) -> str:
    """A page's markup and every local script it loads, as one text.

    What could put a class or an id on that page's elements. Vendored
    libraries are left out: none of them builds the app's classes.
    """
    html = _text(page)
    parts = [html]
    for script in re.findall(r'<script[^>]+src="/?([^"?]+)', html):
        if script.startswith("vendor/"):
            continue
        path = STATIC / script
        assert path.is_file(), f"{page} loads a missing {script}"
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _selectors(css: str) -> List[str]:
    """Each selector in a stylesheet, comma-separated ones split."""
    bare = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    found: List[str] = []
    for block in re.findall(r"([^{}]+)\{", bare):
        assert not block.strip().startswith("@"), (
            "an at-rule here needs this parser taught about it"
        )
        for selector in block.split(","):
            found.append(" ".join(selector.split()))
    return found


def _classes(selector: str) -> Set[str]:
    return set(re.findall(r"\.([a-zA-Z][\w-]*)", selector))


def _ids(selector: str) -> Set[str]:
    return set(re.findall(r"#([a-zA-Z][\w-]*)", selector))


def _builds_class(name: str, sources: str) -> bool:
    """Whether the name appears as a whole token anywhere in them.

    Class names here are hyphenated identifiers, which prose does not
    produce, so a token search is enough.
    """
    pattern = r"(?<![\w-])" + re.escape(name) + r"(?![\w-])"
    return re.search(pattern, sources) is not None


def _builds_id(name: str, sources: str) -> bool:
    """Whether markup carries the id or a script quotes it.

    Stricter than a class, because ids such as `menu` are plain words
    that turn up in every page's comments.
    """
    escaped = re.escape(name)
    attribute = r'id="' + escaped + r'"'
    quoted = r"[\"']#?" + escaped + r"[\"']"
    return (
        re.search(attribute, sources) is not None
        or re.search(quoted, sources) is not None
    )


def _built(selector: str, sources: str) -> List[bool]:
    """For each name the selector requires, whether sources build
    it."""
    found = [
        _builds_class(name, sources) for name in _classes(selector)
    ]
    found += [
        _builds_id(name, sources) for name in _ids(selector)
    ]
    return found


# -- who loads it --


def test_the_menu_loads_it_after_the_shared_stylesheet() -> None:
    """After, because these rules sat at the end of `style.css` and
    some of them restyle what the shared rules set up first."""
    html = _text("menu.html")
    shared = html.find('href="/style.css"')
    own = html.find('href="/menu.css"')

    assert shared != -1
    assert own != -1
    assert shared < own


@pytest.mark.parametrize("page", OTHER_PAGES)
def test_no_other_page_loads_it(page: str) -> None:
    assert "menu.css" not in _text(page)


def test_the_shared_stylesheet_no_longer_carries_the_menu() -> None:
    shared = SHARED_CSS.read_text(encoding="utf-8")

    assert "Main Menu (landing page)" not in shared
    assert ".menu-video" not in shared


# -- whom its rules can match --


def test_it_has_rules_to_check() -> None:
    """Guards the checks below, which pass on an empty file."""
    assert len(_selectors(MENU_CSS.read_text(encoding="utf-8"))) > 50


def test_every_name_it_styles_is_built_by_the_menu() -> None:
    """A rule naming something the menu never builds matches nothing:
    either a leftover or a typo, and both read as live styling."""
    menu = _page_sources("menu.html")
    css = MENU_CSS.read_text(encoding="utf-8")

    unbuilt = [
        selector
        for selector in _selectors(css)
        if not all(_built(selector, menu))
    ]

    assert unbuilt == []


def test_no_rule_can_match_on_another_page() -> None:
    """Each rule needs at least one name no other page builds. A rule
    made only of names the generator or Analytics also use, a state
    class such as `is-active` on its own, would style those pages
    once they loaded this file, and silently stop styling the menu
    the day the rule moved back."""
    others = "\n".join(_page_sources(page) for page in OTHER_PAGES)
    css = MENU_CSS.read_text(encoding="utf-8")

    shared = [
        selector
        for selector in _selectors(css)
        if all(_built(selector, others))
    ]

    assert shared == []


def test_the_dropdown_rule_did_not_come_along() -> None:
    """The generator dropdown dropped its headroom pill long ago, so
    the rule that shrank the pill there matched nothing."""
    css = MENU_CSS.read_text(encoding="utf-8")

    assert ".option-device" not in css
