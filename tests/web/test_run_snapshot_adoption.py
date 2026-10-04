"""The generator's run snapshot goes through its codec (`A2-ORG-03`).

Strategy: read the shipped `app.js`, `generator_run.js`, `index.html`
and `run_snapshot.js`. The codec's rules are exercised in
`tests/web/static/run_snapshot.test.js`, and the page-level round
trips in `snapshot_budget.test.js` and its neighbours; what neither
can check is whether the run controller still builds or reads the
snapshot around the codec. The page-order test covers load order.

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
RUN_JS = STATIC / "generator_run.js"

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


def _function(path: Path, name: str) -> str:
    """One function of a classic script, up to the next."""
    source = path.read_text(encoding="utf-8")
    found = re.search(rf"\n\s*function {name}\(", source)
    assert found is not None, (
        f"{name} is gone from {path.name}; update this test"
        " rather than"
        " deleting it"
    )
    start = found.start()
    following = re.search(r"\n\s*function ", source[found.end() :])
    end = (
        found.end() + following.start()
        if following is not None
        else len(source)
    )
    return source[start:end]


def _page_scripts() -> List[str]:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return re.findall(r'<script src="/([^"?]+)"', html)


def test_the_module_loads_after_what_it_reads() -> None:
    scripts = _page_scripts()
    at = scripts.index("run_snapshot.js")

    assert scripts.index("run_frames.js") < at
    assert scripts.index("run_candidates.js") < at
    run = scripts.index("generator_run.js")
    assert at < run
    assert run < scripts.index("app.js")


def test_the_module_reaches_for_no_page_and_no_storage() -> None:
    module = MODULE_JS.read_text(encoding="utf-8")

    for name in PAGE_NAMES:
        assert name not in module, name


def test_the_save_goes_through_the_codec() -> None:
    body = _function(RUN_JS, "saveSession")

    assert "runSnapshotTiers(" in body
    for serialiser in SERIALISERS:
        assert serialiser not in body, serialiser


def test_the_restore_goes_through_the_codec() -> None:
    restore = _function(RUN_JS, "restoreSession")
    apply = _function(RUN_JS, "applyRestored")

    assert "runSnapshotDecode(" in restore
    for reader in READERS:
        assert reader not in restore, reader
        assert reader not in apply, reader


def test_the_page_no_longer_assembles_a_snapshot_itself() -> None:
    source = _app() + RUN_JS.read_text(encoding="utf-8")

    for name in FORMER_HELPERS:
        assert f"function {name}(" not in source, name
