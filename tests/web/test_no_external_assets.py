"""Tests that no page reaches a third-party origin to render.

Strategy: read every shipped HTML and CSS file and fail on any
absolute http(s) reference in a place the browser would fetch
automatically. What passing proves is the property behind the
finding: opening any page with the network unplugged asks for
nothing the repository does not contain.

This is a regression guard more than a discovery: the CDN and font
references were removed by hand, and the way they come back is
somebody pasting a snippet from documentation. A link in prose is
still fine; a link the browser follows on load is not, because it
decides whether the run browser exists offline, and it hands a
third-party origin the ability to call the model and deletion APIs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin

import pytest
from starlette.testclient import TestClient

import src.web.server as server

STATIC_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

# Anything the browser fetches without being asked. That is `src` on
# any element, and `href` only on `<link>`, which covers stylesheets,
# preconnect hints, and icons.
#
# `href` on an anchor is deliberately excluded: the About modal links
# out to the model cards and the LLaDA paper, and those are the
# user's to click. The distinction being drawn here is not "does this
# mention another origin" but "does opening the page contact one".
_HTML_SRC_ATTR = re.compile(
    r"""src\s*=\s*["'](https?://[^"']+)["']""",
    re.IGNORECASE,
)
_HTML_LINK_TAG = re.compile(
    r"<link\b[^>]*>", re.IGNORECASE | re.DOTALL
)
_HTML_HREF_ATTR = re.compile(
    r"""href\s*=\s*["'](https?://[^"']+)["']""",
    re.IGNORECASE,
)


def _auto_fetched_urls(html: str) -> List[str]:
    """Every URL the browser would request just by loading `html`."""
    found = _HTML_SRC_ATTR.findall(html)
    for tag in _HTML_LINK_TAG.findall(html):
        found.extend(_HTML_HREF_ATTR.findall(tag))
    return found
_CSS_FETCHED_URL = re.compile(
    r"""url\(\s*["']?(https?://[^"')]+)""", re.IGNORECASE
)
_CSS_IMPORT = re.compile(
    r"""@import\s+(?:url\()?\s*["'](https?://[^"']+)""",
    re.IGNORECASE,
)


def _static_files(suffix: str) -> List[Path]:
    found = sorted(STATIC_DIR.rglob("*" + suffix))
    assert len(found) > 0, f"no {suffix} files found to check"
    return found


@pytest.mark.parametrize(
    "page", _static_files(".html"), ids=lambda p: p.name
)
def test_no_page_fetches_from_another_origin(page: Path) -> None:
    text = page.read_text(encoding="utf-8")

    external = _auto_fetched_urls(text)

    assert external == [], (
        f"{page.name} loads {external} from the network;"
        " vendor it under static/vendor instead"
        " (see scripts/vendor_assets.py)"
    )


def test_the_guard_would_catch_a_reintroduced_cdn() -> None:
    """The check has to fail on the thing it was written for.

    An assertion that only ever passes proves nothing about what it
    would do to a pasted snippet, which is exactly how these come
    back. Both shapes that were removed are checked here.
    """
    reintroduced = (
        '<link rel="stylesheet"'
        ' href="https://fonts.googleapis.com/css2?family=X" />'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js">'
        "</script>"
    )

    found = _auto_fetched_urls(reintroduced)

    assert len(found) == 2


def test_the_guard_ignores_a_link_the_user_clicks() -> None:
    """The pair to the test above: an anchor in prose is not a
    fetch, and failing on one would push the About modal's model
    card links out of the page for no safety gain."""
    prose = (
        '<a href="https://arxiv.org/abs/2502.09992">paper</a>'
    )

    assert _auto_fetched_urls(prose) == []


@pytest.mark.parametrize(
    "sheet", _static_files(".css"), ids=lambda p: p.name
)
def test_no_stylesheet_fetches_from_another_origin(
    sheet: Path,
) -> None:
    text = sheet.read_text(encoding="utf-8")

    external = _CSS_FETCHED_URL.findall(
        text
    ) + _CSS_IMPORT.findall(text)

    assert external == [], (
        f"{sheet.name} loads {external} from the network"
    )


def test_the_vendored_assets_are_present() -> None:
    """The other half: proving nothing is fetched would also pass
    if the local files were missing and every page were broken."""
    vendor = STATIC_DIR / "vendor"
    expected = [
        "chart.umd.min.js",
        "hammer.min.js",
        "chartjs-plugin-zoom.min.js",
        "fonts/jetbrains-mono.css",
    ]

    for name in expected:
        path = vendor / name
        assert path.is_file(), f"missing vendored asset: {name}"
        assert path.stat().st_size > 0, f"empty asset: {name}"


def test_every_vendored_dependency_ships_its_license() -> None:
    """Vendoring copies someone else's work into this repository,
    so the terms travel with it."""
    vendor = STATIC_DIR / "vendor"
    expected = [
        "chart.js.LICENSE.md",
        "hammer.js.LICENSE.md",
        "chartjs-plugin-zoom.LICENSE.md",
        "fonts/OFL.txt",
    ]

    for name in expected:
        path = vendor / name
        assert path.is_file(), f"missing license: {name}"
        assert path.stat().st_size > 0, f"empty license: {name}"


def test_the_font_css_points_at_local_files() -> None:
    """The generated stylesheet is the one place a Google URL would
    survive unnoticed, since nothing else reads it."""
    css = (
        STATIC_DIR / "vendor" / "fonts" / "jetbrains-mono.css"
    ).read_text(encoding="utf-8")

    assert "fonts.gstatic.com" not in css
    assert "url(./jetbrains-mono-" in css


# -- and it renders without moving --
#
# Not an offline property, but it lives here because it is about the
# same generated file and would otherwise have nowhere to go.


def _font_css() -> str:
    return (
        STATIC_DIR / "vendor" / "fonts" / "jetbrains-mono.css"
    ).read_text(encoding="utf-8")


def test_no_face_swaps_the_font_in_after_first_paint() -> None:
    """`swap` paints in a fallback and re-lays-out when the real
    font lands. The fallback stack is not metric-matched, so that
    moved the header links and the hyperparameter column by a few
    pixels, on every navigation, because the font is served
    `no-store` and refetched each time.

    Checked across every face rather than by counting: one subset
    left on `swap` still shifts the page for anyone whose text
    reaches it."""
    css = _font_css()

    assert "font-display: swap" not in css
    assert css.count("font-display: block") == css.count(
        "@font-face"
    )


def test_the_generator_emits_what_the_file_carries() -> None:
    """The file says not to edit it by hand, so a fix applied only
    to the file lasts until the next `vendor_assets.py` run. This is
    the test that would have caught that."""
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "vendor_assets.py"
    ).read_text(encoding="utf-8")

    assert '"  font-display: block;\\n"' in source
    assert '"  font-display: swap;\\n"' not in source


@pytest.mark.parametrize(
    "page",
    ["index.html", "analytics.html", "settings.html", "menu.html"],
)
def test_every_page_preloads_the_font_it_blocks_on(
    page: str,
) -> None:
    """With `block`, first paint waits for the font, and without a
    preload that fetch does not start until layout asks for a glyph.
    The preload moves it to parse time, which is what keeps the wait
    to a frame instead of a visible pause."""
    html = (STATIC_DIR / page).read_text(encoding="utf-8")

    match = re.search(
        r'<link rel="preload"[^>]*jetbrains-mono-latin\.woff2[^>]*>',
        html,
        re.DOTALL,
    )
    assert match is not None, f"{page} blocks on an unpreloaded font"
    tag = match.group(0)
    assert 'as="font"' in tag, page
    # Required even same-origin: font requests are always CORS-mode,
    # and a preload without it is a second, separate download.
    assert "crossorigin" in tag, page


@pytest.mark.parametrize(
    "route", ["/", "/analytics.html", "/settings.html"]
)
def test_the_preload_names_the_url_the_css_asks_for(
    route: str,
) -> None:
    """A preload one character off the real URL is worse than none:
    the browser fetches the file twice and still waits for the
    second. The stamper is the thing that could introduce that, so
    this goes through the server rather than reading the file."""
    client = TestClient(server.app)

    html = client.get(route).text

    preload = re.search(
        r'<link rel="preload" href="([^"]+)"', html
    )
    assert preload is not None, route
    css_ref = re.search(
        r'href="(/vendor/fonts/jetbrains-mono\.css[^"]*)"', html
    )
    assert css_ref is not None, route

    css = client.get(css_ref.group(1)).text
    relative = re.search(r"url\((\./[^)]*latin\.woff2)\)", css)
    assert relative is not None, route
    resolved = urljoin(
        css_ref.group(1).split("?")[0], relative.group(1)
    )

    assert preload.group(1) == resolved, route


# -- through the server, not just on disk --
#
# The checks above read files. These ask the app, because the way
# this breaks in practice is a path that does not resolve or a
# vendor directory that never gets committed, and both of those look
# perfectly fine on the filesystem of the machine that made them.


@pytest.mark.parametrize(
    "path",
    [
        "/vendor/chart.umd.min.js",
        "/vendor/hammer.min.js",
        "/vendor/chartjs-plugin-zoom.min.js",
        "/vendor/fonts/jetbrains-mono.css",
        "/vendor/fonts/jetbrains-mono-latin.woff2",
    ],
)
def test_the_server_serves_each_vendored_asset(
    path: str,
) -> None:
    client = TestClient(server.app)

    response = client.get(path)

    assert response.status_code == 200, path
    assert len(response.content) > 0, path


def test_the_served_analytics_page_names_only_local_assets(
) -> None:
    """End to end, including the cache stamper, which rewrites the
    references and so is the last place they could go wrong."""
    client = TestClient(server.app)

    html = client.get("/analytics.html").text

    assert "jsdelivr" not in html
    assert "fonts.googleapis" not in html
    assert "/vendor/chart.umd.min.js?v=" in html
    assert "/vendor/fonts/jetbrains-mono.css?v=" in html
