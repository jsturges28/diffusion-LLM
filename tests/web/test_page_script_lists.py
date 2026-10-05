"""Each page's script list has one copy, and it is the page's own
(`A2-ORG-04`).

Strategy: read the four pages the DOM stub loads, the stub's own
lists, and every browser test beside it. Classic scripts share one
scope, so a page loaded out of order fails on an undefined global,
and a test that loads its own copy of the order can go on passing
against a page that no longer ships that order.

Passing proves the stub holds one list per page, exported for the
tests to import, that each is exactly its page's first-party order
(the vendored chart libraries are stubbed rather than loaded), and
that no test keeps a list of its own. Fifteen tests used to copy
theirs, so adding one script to Analytics meant fourteen edits before
its behaviour was even considered.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pytest

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
TESTS_STATIC = Path(__file__).resolve().parent / "static"
DOM_STUB_JS = TESTS_STATIC / "dom_stub.js"

# Each list the stub holds, and the page whose order it is.
PAGES: Dict[str, str] = {
    "GENERATOR_SCRIPTS": "index.html",
    "ANALYTICS_SCRIPTS": "analytics.html",
    "MENU_SCRIPTS": "menu.html",
    "SETTINGS_SCRIPTS": "settings.html",
    "VISION_SCRIPTS": "vision.html",
}

# A script list's declaration, wherever it appears.
_LIST = re.compile(
    r"^const ([A-Z_]+_SCRIPTS) = \[(.*?)^\];",
    re.MULTILINE | re.DOTALL,
)


def _stub() -> str:
    return DOM_STUB_JS.read_text(encoding="utf-8")


def _stub_lists() -> Dict[str, List[str]]:
    lists: Dict[str, List[str]] = {}
    for name, body in _LIST.findall(_stub()):
        lists[name] = re.findall(r'"([^"]+\.js)"', body)
    return lists


def _page_scripts(page: str) -> List[str]:
    html = (STATIC / page).read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="/([^"?]+)"', html)
    return [
        name for name in scripts if not name.startswith("vendor/")
    ]


@pytest.mark.parametrize("name", sorted(PAGES))
def test_the_stub_holds_each_page_s_own_order(name: str) -> None:
    lists = _stub_lists()

    assert name in lists, f"dom_stub.js has no {name}"
    assert lists[name] == _page_scripts(PAGES[name]), PAGES[name]


def test_every_list_the_stub_holds_is_pinned_and_exported() -> None:
    stub = _stub()
    exports = stub[stub.index("module.exports = {") :]

    assert sorted(_stub_lists()) == sorted(PAGES)
    for name in PAGES:
        assert f"  {name},\n" in exports, name


def test_no_test_keeps_a_list_of_its_own() -> None:
    copies = [
        path.name
        for path in sorted(TESTS_STATIC.glob("*.test.js"))
        if _LIST.search(path.read_text(encoding="utf-8"))
    ]

    assert copies == []


def test_action_view_loads_before_its_controller() -> None:
    scripts = _page_scripts("index.html")

    assert scripts.index("conversation_view.js") < scripts.index(
        "conversation_action_view.js"
    )
    assert scripts.index(
        "conversation_action_view.js"
    ) < scripts.index("conversation_actions.js")


def test_action_view_exports_only_its_factory() -> None:
    source = (
        STATIC / "conversation_action_view.js"
    ).read_text(encoding="utf-8")
    globals_found = re.findall(
        r"^(?:var|function) ([A-Za-z0-9_$]+)",
        source,
        re.MULTILINE,
    )

    assert globals_found == ["conversationActionViewCreate"]


@pytest.mark.parametrize(
    ("script", "factory"),
    (
        ("run_settings_core.js", "runSettingsCoreCreate"),
        ("run_settings_panel.js", "runSettingsPanelCreate"),
    ),
)
def test_run_settings_scripts_export_one_factory(
    script: str, factory: str
) -> None:
    source = (STATIC / script).read_text(encoding="utf-8")
    globals_found = re.findall(
        r"^(?:var|function) ([A-Za-z0-9_$]+)",
        source,
        re.MULTILINE,
    )

    assert globals_found == [factory]
    assert "module.exports" not in source
