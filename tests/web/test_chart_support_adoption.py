"""The chart helpers live in one shared module (`A2-ORG-04`).

Strategy: read the shipped `chart_support.js`, `analytics.js`, every
other first-party script and the browser tests. What each helper
answers is checked in `tests/web/static/chart_support.test.js`, and
where the module loads in `test_page_services.py`; what neither can
check is whether a script still carries a copy of a helper, or calls
one by the name it had before it took the module's prefix.

Passing proves the module defines each helper once and the page none
of them; that no script or browser test names a former name, which
in a classic script would fail only once that chart was drawn; and
that the module names no page, DOM or storage, so the charts that
share it are handed everything it reads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
MODULE_JS = STATIC / "chart_support.js"
ANALYTICS_JS = STATIC / "analytics.js"
TESTS_STATIC = Path(__file__).resolve().parent / "static"

# Every function the module defines, under the names it defines them
# by. The availability flag is the one variable, checked beside them.
MOVED = (
    "chartSupportDestroy",
    "chartSupportGutterLayout",
    "chartSupportZoomOptions",
    "chartSupportTooltipTitle",
    "chartSupportLineLabelColor",
)

# What the helpers were called while they lived in analytics.js.
FORMER_NAMES = (
    "chartsAvailable",
    "destroyChart",
    "chartGutterLayout",
    "zoomPluginOptions",
    "tooltipTitle",
    "lineLabelColor",
)

# What a module shared between charts has no reason to name.
PAGE_NAMES = (
    "document",
    "window",
    "localStorage",
    "sessionStorage",
)


def _uses(name: str, text: str) -> bool:
    """A whole-word use, so a name inside a longer one is not one."""
    pattern = r"(?<![\w$])" + re.escape(name) + r"(?![\w$])"
    return re.search(pattern, text) is not None


def _texts() -> List[Tuple[str, str]]:
    """Every first-party script and every browser test, by name."""
    paths = sorted(STATIC.glob("*.js"))
    paths += sorted(TESTS_STATIC.glob("*.test.js"))
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in paths
    ]


def test_what_moved_is_defined_once_in_the_module() -> None:
    module = MODULE_JS.read_text(encoding="utf-8")
    page = ANALYTICS_JS.read_text(encoding="utf-8")

    for name in MOVED:
        assert f"\nfunction {name}(" in module, name
        assert f"\nfunction {name}(" not in page, name
    assert "\nvar chartSupportAvailable = " in module
    assert "\nvar chartSupportAvailable = " not in page


def test_no_former_name_survives() -> None:
    found = [
        (where, name)
        for where, text in _texts()
        for name in FORMER_NAMES
        if _uses(name, text)
    ]
    assert found == []


def test_the_module_names_no_page_and_no_storage() -> None:
    module = MODULE_JS.read_text(encoding="utf-8")

    for name in PAGE_NAMES:
        assert not _uses(name, module), name
