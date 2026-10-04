"""Analytics' token viewer lives in its own controller (`A2-ORG-04`).

Strategy: read the shipped `analytics.html`, `token_viewer.js`,
`analytics.js`, the other shipped scripts and the browser tests. What
the viewer draws, and how its scrubber, crossfade and entropy chart
behave, is checked against the controller alone in
`tests/web/static/token_viewer.test.js` and through the page in the
`analytics_*` suites; what neither can check is how the controller
sits on the page: where it loads, how much of itself it shows, and
whether the page or a test still reaches past the object it returns.

Passing proves the viewer loads after everything it reads and before
the page that creates it; that it defines one global name and keeps
everything else inside its factory; that its code names none of the
page's catalog, fetches, fences or line charts, and no storage, so it
is handed what it reads; that the page creates it once and neither
defines nor reads a name the factory keeps; and that no browser test
reaches one of those as a page global.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Pattern, Set, Tuple

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
ANALYTICS_HTML = STATIC / "analytics.html"
ANALYTICS_JS = STATIC / "analytics.js"
MODULE_JS = STATIC / "token_viewer.js"
TESTS_STATIC = Path(__file__).resolve().parent / "static"

# What the viewer reads, each of which has to load before it.
READS = (
    "custom_select.js",
    "overlays.js",
    "overlay_series.js",
    "chart_support.js",
    "run_candidates.js",
    "candidate_flicker.js",
)

# What the viewer is handed or has no use for: the catalog and the
# open run, the page's fetches and their fences, the line charts and
# the tooltip eyes, and storage, which it reaches only through the
# settings and drawer helpers in overlays.js.
PAGE_NAMES = (
    "allRuns",
    "activeRunId",
    "activeRunTokenizer",
    "runTokenizer",
    "fetchFrames",
    "fetchMetrics",
    "detailRequests",
    "compareRequests",
    "lineCharts",
    "tooltipEnabled",
    "localStorage",
    "sessionStorage",
)

# A name a script defines at its top level, and a name the factory
# keeps, two spaces in.
_TOP_LEVEL = re.compile(
    r"^(?:function ([\w$]+)\(|var ([\w$]+) =)", re.MULTILINE
)
_INSIDE = re.compile(
    r"^  (?:function ([\w$]+)\(|var ([\w$]+) =)", re.MULTILINE
)


def _scripts() -> List[str]:
    html = ANALYTICS_HTML.read_text(encoding="utf-8")
    return re.findall(r'<script src="/([^"?]+)"', html)


def _code(path: Path) -> str:
    """A script without its line comments, so prose is not a use."""
    kept: List[str] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.lstrip().startswith("//"):
            continue
        kept.append(re.sub(r"\s//\s.*$", "", line))
    return "\n".join(kept)


def _names(pattern: Pattern[str], text: str) -> Set[str]:
    found = pattern.findall(text)
    return {function or variable for function, variable in found}


def _uses(name: str, text: str) -> bool:
    """A whole-word use, and not a property of some other object."""
    pattern = r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])"
    return re.search(pattern, text) is not None


def _kept() -> Set[str]:
    return _names(_INSIDE, MODULE_JS.read_text(encoding="utf-8"))


def _defined_elsewhere() -> Set[str]:
    """Top-level names of every other shipped script. The generator
    keeps some of the viewer's names, `setOverlayMode` among them, so
    a generator test reaching one is reaching its own page."""
    names: Set[str] = set()
    for path in sorted(STATIC.glob("*.js")):
        if path != MODULE_JS:
            text = path.read_text(encoding="utf-8")
            names |= _names(_TOP_LEVEL, text)
    return names


def test_the_viewer_loads_after_what_it_reads() -> None:
    scripts = _scripts()
    at = scripts.index("token_viewer.js")

    for name in READS:
        assert scripts.index(name) < at, name
    assert at < scripts.index("analytics.js")


def test_the_viewer_defines_one_global_name() -> None:
    module = MODULE_JS.read_text(encoding="utf-8")

    assert _names(_TOP_LEVEL, module) == {"tokenViewerCreate"}


def test_the_viewer_names_no_page_state() -> None:
    code = _code(MODULE_JS)

    for name in PAGE_NAMES:
        assert not _uses(name, code), name


def test_the_page_creates_the_viewer_once() -> None:
    page = _code(ANALYTICS_JS)

    assert page.count("tokenViewerCreate(") == 1
    assert "var tokenViewer = tokenViewerCreate({" in page


def test_the_page_defines_nothing_the_factory_keeps() -> None:
    kept = _kept()
    page_text = ANALYTICS_JS.read_text(encoding="utf-8")
    page = _names(_TOP_LEVEL, page_text)

    assert "renderRunOverlays" in kept
    assert "chartEntropy" in kept
    assert kept & page == set()


def test_the_page_reads_nothing_the_factory_keeps() -> None:
    # A key in the object the page creates the viewer with is a name
    # it hands over, not one it reads.
    kept = _kept()
    page = _code(ANALYTICS_JS)
    found = [
        name
        for name in sorted(kept)
        if re.search(
            r"(?<![\w$.])" + re.escape(name) + r"(?![\w$]|\s*:)", page
        )
    ]

    assert "compareBlend" in kept
    assert found == []


def test_no_browser_test_reaches_inside_the_factory() -> None:
    kept = _kept() - _defined_elsewhere()
    found: List[Tuple[str, str]] = []
    for path in sorted(TESTS_STATIC.glob("*.test.js")):
        text = path.read_text(encoding="utf-8")
        for name in sorted(kept):
            pattern = r"\bcontext\." + re.escape(name) + r"\b"
            if re.search(pattern, text):
                found.append((path.name, name))
    assert "renderRunOverlays" in kept
    assert found == []
