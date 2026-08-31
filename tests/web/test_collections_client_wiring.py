"""The Analytics page no longer owns the collections list.

Strategy: source inspection of the shipped classic scripts, the
approach this repo uses for pages that cannot be imported. What the
operations do is tested in `test_collection_ops.py`, what the
endpoints answer in `test_collection_endpoints.py`, and what the
client module requests in the node test beside it. None of those can
see the thing that actually caused `DATA-02`, which is a page holding
its own copy and writing it back.

That is what this file guards. The fix is not that the writes are
safer, it is that the page has no way to name a successor state: it
sends the gesture, and takes whatever comes back. A single surviving
`persistSet` of the key, or a single surviving local mutation of the
array, would restore the lost update while every other test kept
passing.

Passing proves the blob write is gone from both the page and the
persist layer, that exactly one function replaces the page's copy, and
that the gestures go through the client module.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

KEY = "diffusion_collections"


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


# -- the write path that closed --


def test_the_page_never_serializes_the_list() -> None:
    """The whole defect in one line. Every gesture used to end in
    JSON.stringify(collections) going out as a whole value."""
    source = _source("analytics.js")

    assert "JSON.stringify(collections)" not in source


def test_the_page_never_persists_the_key() -> None:
    source = _source("analytics.js")

    assert f'persistSet("{KEY}"' not in source
    assert "COLLECTIONS_KEY" not in source


def test_the_key_is_off_the_persist_lists() -> None:
    """`persistSet` writes localStorage first and PUTs second, so a
    key still listed there is a key a page can still replace."""
    source = _source("overlays.js")
    listed = re.findall(rf'"{KEY}"\s*,', source)

    assert listed == [], listed


def test_the_page_does_not_read_the_key_either() -> None:
    """A local copy read at boot would be a second source of truth,
    and the stale one, since another window may have filed since."""
    source = _source("analytics.js")

    assert "localStorage.getItem" not in source


# -- what replaced it --


def test_one_function_adopts_the_server_answer() -> None:
    source = _source("analytics.js")
    adopts = re.findall(r"function adoptCollections\(", source)

    assert len(adopts) == 1


def test_the_page_holds_a_client() -> None:
    source = _source("analytics.js")

    assert "collectionsClientCreate(" in source


def test_the_module_loads_before_the_page() -> None:
    """A classic script, so order in the document is the dependency.
    Loaded after analytics.js it would be undefined at boot.

    Matched on the script tags rather than the bare filenames: the
    page's name appears earlier in the markup for other reasons, and
    comparing those positions would compare the wrong two things.
    """
    markup = _source("analytics.html")
    client = markup.find('<script src="/collections_client.js">')
    page = markup.find('<script src="/analytics.js">')

    assert client != -1, "the client is not on the page"
    assert page != -1, "the page script tag moved"
    assert client < page


# -- every gesture goes through it --


def test_the_star_is_one_call() -> None:
    """Clearing a filled star empties every collection the run is
    in. Looping here would be several writes that can stop half
    way, which is the shape being removed."""
    body = _region("analytics.js", "function toggleFavorite(", 400)

    assert "collectionsApi.toggleFavorite(" in body
    assert "for (" not in body


def test_filing_and_unfiling_go_through_the_client() -> None:
    body = _region(
        "analytics.js", "function setRunMembership(", 500
    )

    assert "collectionsApi.addRun(" in body
    assert "collectionsApi.removeRun(" in body


def test_create_rename_and_delete_go_through_the_client() -> None:
    source = _source("analytics.js")

    for call in (
        "collectionsApi.create(",
        "collectionsApi.rename(",
        "collectionsApi.destroy(",
    ):
        assert call in source, call


def test_refresh_asks_the_server() -> None:
    """Refresh used to re-read this window's own copy, so another
    window's filing appeared only after a full page load."""
    body = _region("analytics.js", "function loadAndRender(", 900)

    assert "refreshCollections()" in body
