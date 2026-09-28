"""The vision page ships, is reachable, and keeps its promises.

Strategy: source inspection of the shipped page plus the four it sits
beside, the approach this repo uses for pages that cannot be imported.
Passing proves the page is reachable from everywhere, that it declares
the elements its script reaches for, and that two promises printed on
it are structurally true rather than only written down.

The two promises are the interesting half. The page tells a reader
their image stays in the browser, and that nothing here costs GPU
memory. Both are architectural claims, and both would still read
correctly on a page that had quietly started uploading or activating,
so they are asserted against the code rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "src" / "web" / "static"
PAGE = STATIC / "vision.html"
SCRIPT = STATIC / "vision.js"
SERVER = REPO_ROOT / "src" / "web" / "server.py"

# Every page that carries the shared header. The new one has to be
# reachable from all of them, or it is a page nobody finds.
HEADER_PAGES = (
    "index.html",
    "menu.html",
    "analytics.html",
    "settings.html",
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


# -- it exists and is reachable --


def test_the_page_and_its_script_ship() -> None:
    assert PAGE.is_file()
    assert SCRIPT.is_file()
    assert (STATIC / "vision.css").is_file()


def test_the_server_serves_it() -> None:
    assert '@app.get("/vision.html")' in SERVER.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("name", HEADER_PAGES)
def test_every_other_page_links_to_it(name: str) -> None:
    text = (STATIC / name).read_text(encoding="utf-8")

    assert 'href="/vision.html"' in text, (
        f"{name} does not link the vision page, so a reader there"
        " cannot reach it"
    )


def test_the_page_marks_itself_as_the_active_tab() -> None:
    """Not a link to itself, matching what the other pages do for
    their own entry."""
    markup = _page()

    assert 'href="/vision.html"' not in markup
    assert "header-link-active" in markup


def _nav(name: str) -> str:
    """Just the header nav of one page.

    Scoped because a page names its own assets elsewhere: a whole-file
    search finds `analytics.css` and orders it by where it appears.
    Two markers because the Main Menu styles its nav differently from
    the three that share the header.
    """
    text = (STATIC / name).read_text(encoding="utf-8")
    for marker in ('<nav id="header-nav"', "<nav class="):
        if marker in text:
            start = text.index(marker)
            return text[start:text.index("</nav>", start)]
    raise AssertionError(f"{name} has no nav")


@pytest.mark.parametrize("name", HEADER_PAGES)
def test_the_link_sits_in_the_same_place_everywhere(
    name: str,
) -> None:
    """Directly after Analytics on every page, because a nav entry
    that moves between pages is one a reader hunts for.

    Only against Analytics, not against the Settings gear. Three pages
    keep the gear in the nav and the Main Menu pins it to a panel
    corner outside it, so a gear relationship would be asserting a
    coincidence of one layout rather than a rule.
    """
    nav = _nav(name)
    # Whitespace-tolerant: an active entry is written across lines,
    # as `<span ...>\n  Analytics\n</span>`.
    entries = re.findall(r">\s*([A-Z][a-z]+)\s*<", nav)

    assert "Vision" in entries, f"{name} nav has no Vision entry"
    assert "Analytics" in entries, f"{name} nav has no Analytics"
    after = entries.index("Analytics") + 1
    assert entries.index("Vision") == after, (
        f"{name} does not put Vision straight after Analytics:"
        f" {entries}"
    )


# -- the page declares what its script reaches for --


def _ids_used_by_script() -> List[str]:
    return sorted(set(
        re.findall(r'getElementById\("([\w-]+)"\)', _script())
    ))


def test_the_script_reaches_for_ids_the_page_declares() -> None:
    """The failure this prevents is silent: `getElementById` returns
    null, the page renders without that piece, and nothing says so."""
    markup = _page()

    missing = [
        name for name in _ids_used_by_script()
        if f'id="{name}"' not in markup
    ]
    assert missing == [], f"vision.html declares no {missing}"


def test_the_script_reaches_for_something() -> None:
    """The test above passes on a script that reads no elements."""
    assert len(_ids_used_by_script()) > 5


def test_the_three_canvases_are_declared() -> None:
    """One per pipeline step, which is the page's whole structure."""
    markup = _page()

    for name in ("resize", "tiles", "patches"):
        assert f'id="vision-canvas-{name}"' in markup


def test_the_page_loads_its_script_last() -> None:
    """`vision.js` calls `visionBoot` at the end of the file, so the
    elements have to exist by then."""
    markup = _page()

    assert markup.index("vision.js") > markup.index(
        'id="vision-canvas-patches"'
    )


# -- and keeps the two promises printed on it --


def test_the_image_never_leaves_the_browser() -> None:
    """The promise under the file picker, asserted against the code.

    The geometry needs only width and height, which is what makes this
    possible. A `FormData`, a body on a POST, or a data URL in a
    request would all break it while the sentence still read true.
    """
    script = _script()

    assert "FormData" not in script
    assert "readAsDataURL" not in script
    assert "createObjectURL" in script, (
        "the picture should be displayed from a local object URL"
    )
    # Every request it makes, and none may carry a body.
    for call in re.findall(r"fetch\((.*?)\)", script, re.S):
        assert "method" not in call, f"a fetch sends a body: {call}"
        assert "body" not in call, f"a fetch sends a body: {call}"


def test_only_dimensions_are_sent() -> None:
    """The other half: the query it builds names nothing else.

    Read from the whole assignment rather than one string fragment,
    since it is concatenated across several lines.
    """
    script = _script()

    query = re.search(
        r'var query = (.*?);\n', script, re.S
    )
    assert query, "the geometry query was not found"
    built = query.group(1)

    for allowed in ("encoder", "width", "height"):
        assert allowed in built, f"the query omits {allowed}"
    # Nothing about the picture itself travels.
    for forbidden in ("data", "base64", "blob", "name", "file"):
        assert forbidden not in built.lower(), (
            f"the query carries {forbidden!r}: {built}"
        )


def test_the_page_costs_no_gpu_memory() -> None:
    """The promise in the footer. This page must not activate a model,
    claim residency or evict anything, which is the reason it is a
    supervisor capability rather than a worker."""
    script = _script()

    for forbidden in ("activate", "/ws", "model_client", "download"):
        assert forbidden not in script, (
            f"vision.js reaches for {forbidden!r}, which would"
            " make an inspection cost somebody their model"
        )


def test_the_page_says_where_the_detail_is() -> None:
    """Help and the guide own the long form, and a page that explains
    itself completely is the README problem again."""
    assert "docs/GUIDE.md" in _page()


def test_the_page_works_without_a_third_party_origin() -> None:
    """TRUST-02. The font and any library come from this origin, so
    the page renders on a machine with no outbound network."""
    markup = _page()

    for external in ("http://", "https://", "//cdn", "fonts.google"):
        assert external not in markup, (
            f"vision.html references {external!r}"
        )


def test_the_page_states_the_lesson_before_a_reader_acts() -> None:
    """The lede exists so a reader knows what to look for. Checked
    structurally, since the wording will change."""
    markup = _page()

    assert 'id="vision-lede"' in markup
    lede = markup.split('id="vision-lede"', 1)[1][:600].lower()
    assert "token" in lede
    assert "tile" in lede or "patch" in lede
