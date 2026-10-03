"""Each page loads exactly the shared modules it uses (`A2-ORG-05`).

Strategy: read every page's first-party scripts and the four modules
the pages' shared code lives in. A module is used on a page when one
of the page's other scripts names something the module defines,
comments aside. Then require that a page loads a module exactly when
it uses it, and before the first script that does.

`overlays.js` used to hold the services as well as the visual code,
so every page loaded both whatever it needed, and Vision loaded all
of it for nothing. Passing proves the split holds: no page carries a
module it never calls, and none calls one it has not loaded first.
Classic scripts share one scope, so a missing module fails only when
the call is made, which on most pages is a click away from boot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

PAGES = (
    "index.html",
    "analytics.html",
    "menu.html",
    "settings.html",
    "vision.html",
)
MODULES = (
    "persist.js",
    "reduced_motion.js",
    "activation_progress.js",
    "overlays.js",
)

# What the rule below comes to today, written out so a reader sees
# the split without running it: the menu and Vision draw no tokens.
VISUAL_PAGE = {"persist.js", "reduced_motion.js", "overlays.js"}
EXPECTED: Dict[str, Set[str]] = {
    "index.html": set(MODULES),
    "analytics.html": VISUAL_PAGE,
    "menu.html": {
        "persist.js",
        "reduced_motion.js",
        "activation_progress.js",
    },
    "settings.html": VISUAL_PAGE,
    "vision.html": set(),
}

# A name a module defines at its top level.
_DEFINED = re.compile(
    r"^(?:function ([A-Za-z_$][\w$]*)\(|var ([A-Za-z_$][\w$]*) =)",
    re.MULTILINE,
)


def _scripts(page: str) -> List[str]:
    html = (STATIC / page).read_text(encoding="utf-8")
    found = re.findall(r'<script src="/([^"?]+)"', html)
    return [name for name in found if not name.startswith("vendor/")]


def _code(script: str) -> str:
    """A script without its line comments, so a name mentioned only
    in prose is not a use."""
    kept: List[str] = []
    text = (STATIC / script).read_text(encoding="utf-8")
    for line in text.split("\n"):
        if line.lstrip().startswith("//"):
            continue
        kept.append(re.sub(r"\s//\s.*$", "", line))
    return "\n".join(kept)


def _defines(module: str) -> Set[str]:
    text = (STATIC / module).read_text(encoding="utf-8")
    return {
        function or variable
        for function, variable in _DEFINED.findall(text)
    }


def _uses(script: str, names: Set[str]) -> bool:
    """A whole-word use, and not a property of some other object."""
    code = _code(script)
    for name in names:
        word = r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])"
        if re.search(word, code):
            return True
    return False


@pytest.mark.parametrize("module", MODULES)
def test_every_module_exists_and_defines_something(
    module: str,
) -> None:
    assert (STATIC / module).is_file(), module
    assert _defines(module), module


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("module", MODULES)
def test_a_page_loads_a_module_exactly_when_it_uses_it(
    page: str, module: str
) -> None:
    scripts = _scripts(page)
    names = _defines(module)
    users = [
        script
        for script in scripts
        if script != module and _uses(script, names)
    ]

    if users:
        assert module in scripts, (
            f"{page}: {users[0]} uses {module}, which is not loaded"
        )
        assert scripts.index(module) < scripts.index(users[0]), (
            f"{page}: {module} loads after {users[0]}, which uses it"
        )
    else:
        assert module not in scripts, (
            f"{page} loads {module}, and nothing on it uses it"
        )


@pytest.mark.parametrize("page", PAGES)
def test_each_page_carries_the_modules_it_should(page: str) -> None:
    assert set(_scripts(page)) & set(MODULES) == EXPECTED[page]
