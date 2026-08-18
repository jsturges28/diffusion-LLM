"""The generator moves between editing phases only through the table.

Strategy: read the shipped `app.js` and `index.html`. The table and
every legal and illegal move are exercised in
`tests/web/static/run_phases.test.js`; what cannot be checked there is
whether the page still sets the phase behind its back. One direct
assignment would put the workflow back where it was, because the
table only constrains the moves that go through it.

What passing proves is narrow but load-bearing. The guided edit flow
needs a GPU to exercise, so this refactor cannot be run here at all.
What it can do is guarantee the shape: one held object, no free
assignments, and every phase named through a constant rather than a
string literal that a typo would silently turn into a phase nobody
enters.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
MODULE_JS = STATIC / "run_phases.js"

FORMER_NAMES = (
    "remaskMode",
    "substitutionMode",
    "guidedResumeAction",
    "guidedTargetFrame",
    "remaskModeEdits",
)


def _app() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_no_phase_field_is_a_variable_of_its_own() -> None:
    source = _app()

    for name in FORMER_NAMES:
        pattern = r"(?<![\w.$])" + name + r"(?![\w$])"
        assert re.search(pattern, source) is None, name


def test_the_phase_is_declared_once_and_held() -> None:
    source = _app()
    writes = re.findall(r"(?<![\w.$])runPhase\s*=(?!=)", source)

    assert source.count("var runPhase = runPhasesCreate()") == 1
    assert len(writes) == 1


def test_the_phase_is_never_assigned_directly() -> None:
    """The whole guarantee. A table that only constrains the moves
    routed through it constrains nothing."""
    source = _app()
    writes = re.findall(r"runPhase\.mode\s*=(?!=)", source)

    assert writes == []


def test_every_move_names_a_constant() -> None:
    """A string literal here would be a typo away from a phase
    nothing enters, and the table would never see it."""
    source = _app()
    calls = re.findall(r"runPhasesEnter\(runPhase, ([^)]+)\)", source)

    assert len(calls) == 9
    for target in calls:
        assert target.startswith("RUN_PHASE_"), target


def test_leaving_a_session_uses_the_module() -> None:
    source = _app()

    assert "runPhasesReset(runPhase)" in source


def test_session_open_checks_use_the_helper() -> None:
    """Rather than comparing the phase against null in two places,
    which is the comparison that goes stale when a phase is added."""
    source = _app()

    assert source.count("runPhasesEditing(runPhase)") == 2


def test_the_module_loads_first_and_has_no_dom() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    module = MODULE_JS.read_text(encoding="utf-8")

    assert html.index("/run_phases.js") < html.index("/app.js")
    assert "document" not in module
    assert "window" not in module


def test_a_resume_is_cleared_before_the_phase_moves() -> None:
    """The one ordering this refactor changed. The site that lands
    back in edit used to set the phase and then clear the resume it
    had just finished; the module checks consistency on arrival, so
    the clear has to come first."""
    source = _app()
    cleared = (
        "runPhase.guidedAction = null;\n"
        "    runPhase.targetFrame = null;"
    )
    start = source.find(cleared)

    assert start != -1
    after = source[start : start + 200]
    assert "runPhasesEnter(runPhase, RUN_PHASE_EDIT)" in after
