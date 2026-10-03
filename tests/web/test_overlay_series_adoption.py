"""Analytics reads a run's frames and signals through its adapter
(`A2-ORG-04`).

Strategy: read the shipped `analytics.html`, `analytics.js` and
`overlay_series.js`, and the browser tests. What the adapter answers
is checked in `tests/web/static/overlay_series.test.js` against the
server's own payloads; what that cannot check is whether the page
still carries a copy of the adapter, or reaches it by a name it no
longer has.

Passing proves the adapter loads after `overlays.js`, which it reads,
and before the page that calls it; that it names no page, no page
state and no storage, so a reader is handed everything it reads; and
that `analytics.js` defines nothing that moved while no browser test
or page line uses a former name of the manifest helpers or the stop
source.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
ANALYTICS_HTML = STATIC / "analytics.html"
ANALYTICS_JS = STATIC / "analytics.js"
MODULE_JS = STATIC / "overlay_series.js"
TESTS_STATIC = Path(__file__).resolve().parent / "static"

# What a reader handed everything it reads has no reason to name: the
# page, its storage, and the viewer state analytics.js keeps.
PAGE_NAMES = (
    "document",
    "window",
    "sessionStorage",
    "localStorage",
    "overlayData",
    "overlayFrameIndex",
    "overlayCanvasOf",
    "overlayIsAutoregressive",
)

# Everything the adapter defines, under the names it defines them by.
MOVED = (
    "overlaySeries",
    "overlaySeriesLength",
    "overlaySeriesPresent",
    "overlaySeriesAt",
    "overlaySeriesFinalIndex",
    "overlaySeriesFinal",
    "overlaySeriesOf",
    "overlaySeriesCommitSteps",
    "overlaySeriesRevisions",
    "overlaySeriesStopSource",
    "overlaySeriesEntropyFrame",
    "overlaySeriesChannelShape",
    "overlaySeriesChannel",
    "overlaySeriesChannelFrame",
    "overlaySeriesEntropyAvailability",
    "overlaySeriesHasEntropy",
    "overlaySeriesHasTokenValue",
    "overlaySeriesSingleCanvas",
)

# What the manifest helpers and the stop source were called before
# they took the adapter's prefix.
FORMER_NAMES = (
    "ENTROPY_SHAPES",
    "channelShape",
    "signalChannel",
    "channelFrameIndex",
    "entropyAvailability",
    "framesHaveEntropy",
    "framesHaveTokenValue",
    "stopSourceOf",
)


def _module() -> str:
    return MODULE_JS.read_text(encoding="utf-8")


def _page() -> str:
    return ANALYTICS_JS.read_text(encoding="utf-8")


def _scripts() -> List[str]:
    html = ANALYTICS_HTML.read_text(encoding="utf-8")
    return re.findall(r'<script src="/([^"?]+)"', html)


def _uses(name: str, text: str) -> bool:
    """A whole-word use, so a name inside a longer one is not one."""
    pattern = r"(?<![\w$])" + re.escape(name) + r"(?![\w$])"
    return re.search(pattern, text) is not None


def test_the_adapter_loads_after_what_it_reads() -> None:
    scripts = _scripts()
    at = scripts.index("overlay_series.js")

    assert scripts.index("overlays.js") < at
    assert at < scripts.index("analytics.js")


def test_the_adapter_names_no_page_and_no_page_state() -> None:
    module = _module()

    for name in PAGE_NAMES:
        assert not _uses(name, module), name


def test_what_moved_is_defined_once_in_the_adapter() -> None:
    module = _module()
    page = _page()

    for name in MOVED:
        assert f"\nfunction {name}(" in module, name
        assert f"\nfunction {name}(" not in page, name
    assert "var OVERLAY_SERIES_ENTROPY_SHAPES = " in module


def test_no_former_name_survives() -> None:
    texts = [
        ("analytics.js", _page()),
        ("overlay_series.js", _module()),
    ]
    for path in sorted(TESTS_STATIC.glob("*.test.js")):
        texts.append((path.name, path.read_text(encoding="utf-8")))

    found = [
        (where, name)
        for where, text in texts
        for name in FORMER_NAMES
        if _uses(name, text)
    ]
    assert found == []


def test_the_page_hands_the_adapter_its_frame() -> None:
    """The one reader that needs the scrubbed frame is given it, where
    it used to read the page's own variable."""
    assert "overlayFrameIndex" in _page()
    assert re.search(
        r"overlaySeriesChannelFrame\(\s*channel,\s*series,"
        r"\s*overlayFrameIndex\s*\)",
        _page(),
    )
