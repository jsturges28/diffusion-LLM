"""The generator's run snapshot goes through its codec (`A2-ORG-03`).

Strategy: read the shipped `app.js`, `index.html` and
`run_snapshot.js`. The codec's rules are exercised in
`tests/web/static/run_snapshot.test.js`, and the page-level round
trips in `snapshot_budget.test.js` and its neighbours; what neither
can check is whether the page still builds or reads the snapshot
around the codec. That the DOM stub loads the page in its own order
is `tests/web/test_page_script_lists.py`'s to prove, for every page.

What passing proves is that the snapshot's format has one owner. The
page reads itself into a record and applies what comes back, and
everything between, the tiers, their order and every default an older
snapshot needs, is in a file with no page or storage to reach for. A
second place that serialised a family or parsed the stored text would
be a second format, and the two would drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
MODULE_JS = STATIC / "run_snapshot.js"

# What a codec with no page and no storage has no reason to name.
PAGE_NAMES = ("document", "window", "sessionStorage", "localStorage")

# What the page's two snapshot functions may no longer do themselves.
SERIALISERS = (
    "runFramesToJson(",
    "runCandidatesToSnapshot(",
    "originalRunToJson(",
)
READERS = (
    "JSON.parse(",
    "runFramesFromJson(",
    "runCandidatesFromSnapshot(",
    "originalRunFromJson(",
)

# The helpers that assembled and read the snapshot inside the page.
FORMER_HELPERS = (
    "candidateTiers",
    "originalCandidatesKeptApart",
    "restoredRunPrompt",
    "restoredOriginalCandidates",
)


def _app() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _function(name: str) -> str:
    """One top-level function of `app.js`, up to the next."""
    source = _app()
    start = source.find(f"\nfunction {name}(")
    assert start != -1, (
        f"{name} is gone from app.js; update this test rather than"
        " deleting it"
    )
    end = source.find("\nfunction ", start + 1)
    return source[start:end]


def _page_scripts() -> List[str]:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return re.findall(r'<script src="/([^"?]+)"', html)


def test_the_module_loads_after_what_it_reads() -> None:
    scripts = _page_scripts()
    at = scripts.index("run_snapshot.js")

    assert scripts.index("run_frames.js") < at
    assert scripts.index("run_candidates.js") < at
    assert at < scripts.index("app.js")


def test_the_module_reaches_for_no_page_and_no_storage() -> None:
    module = MODULE_JS.read_text(encoding="utf-8")

    for name in PAGE_NAMES:
        assert name not in module, name


def test_the_save_goes_through_the_codec() -> None:
    body = _function("saveSessionState")

    assert "runSnapshotTiers(" in body
    for serialiser in SERIALISERS:
        assert serialiser not in body, serialiser


def test_the_restore_goes_through_the_codec() -> None:
    restore = _function("restoreSessionState")
    apply = _function("restoreSessionStateApply")

    assert "runSnapshotDecode(" in restore
    for reader in READERS:
        assert reader not in restore, reader
        assert reader not in apply, reader


def test_the_page_no_longer_assembles_a_snapshot_itself() -> None:
    source = _app()

    for name in FORMER_HELPERS:
        assert f"function {name}(" not in source, name
