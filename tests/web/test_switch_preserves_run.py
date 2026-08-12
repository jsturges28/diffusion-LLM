"""A refused model switch keeps the run that is already on screen.

Strategy: source inspection of `app.js` and `menu.js`, the approach
this repo already uses for its two classic-script pages (see
`tests/web/test_analytics_escaping.py`); giving them a testable seam
is `ORG-02` in stage 5. The server half of `LIFE-06`, that a switch
which cannot work evicts nothing, is tested by execution in
`tests/web/test_activation_validation.py`.

What passing proves is the client's share of the same failure. Both
entry paths discarded the previous run *before* asking the server to
activate, so a switch the server refused, for a missing interpreter
or a model that could never fit, cost the user a completed run for an
error that freed nothing and changed nothing. The discard belongs in
the branch where a new worker is actually ready.

Also checks the other end of the redirect the honest `/generate` gate
introduced: a worker that failed to load no longer satisfies that
gate, so the browser lands back on the menu, and the menu has to say
why or the bounce reads as a bug.
"""

from __future__ import annotations

from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
MENU_JS = STATIC / "menu.js"


def _region(path: Path, anchor: str, chars: int) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {path.name}; update this"
        " test rather than deleting it"
    )
    return source[start : start + chars]


# -- the generator's switch --


def test_the_generator_does_not_discard_before_asking() -> None:
    """The bug, stated as an absence. `switchModel` runs from the
    click to the POST; the discard must not be in it."""
    region = _region(APP_JS, "function switchModel(id, device)", 1200)

    assert "clearSessionState()" not in region


def test_the_generator_discards_once_the_worker_is_ready() -> None:
    """The other half: it still has to happen, because a switch ends
    in a reload that the restore path cannot tell from a trip to
    Analytics and back."""
    region = _region(APP_JS, "function pollSwitch(name)", 1400)

    discard = region.find("clearSessionState()")
    reload_call = region.find("location.reload()")

    assert discard != -1
    assert reload_call != -1
    assert discard < reload_call


def test_the_generator_keeps_the_run_when_a_switch_fails() -> None:
    """`switchFailed` is where a refused activation lands, and it
    must not clean up what the request never replaced."""
    region = _region(APP_JS, "function switchFailed(err)", 700)

    assert "clearSessionState()" not in region


# -- the menu's selection --


def test_the_menu_does_not_discard_before_asking() -> None:
    region = _region(MENU_JS, "function selectModel(model, li)", 900)

    assert "overlaysClearLastRun()" not in region


def test_the_menu_discards_once_the_worker_is_ready() -> None:
    region = _region(MENU_JS, "function pollActivation()", 1400)

    discard = region.find("overlaysClearLastRun()")
    navigate = region.find("window.location.assign(GENERATE_URL)")

    assert discard != -1
    assert navigate != -1
    assert discard < navigate


def test_reselecting_the_resident_model_keeps_its_run() -> None:
    """Behavior that predates this change and has to survive it.
    Re-selecting the loaded model spawns nothing, so the run showing
    on the generator is still that model's own."""
    region = _region(MENU_JS, "function pollActivation()", 1400)

    assert "activeSelection.resident" in region


# -- the menu explains a redirect --


def test_the_menu_reports_a_load_that_failed_earlier() -> None:
    """The generator turns away a model that is not serving, so the
    reason has to surface here or the bounce looks like a bug."""
    source = MENU_JS.read_text(encoding="utf-8")

    assert "function showPriorLoadFailure()" in source
    assert "showPriorLoadFailure();" in source


def test_the_report_reads_the_activation_endpoint() -> None:
    region = _region(
        MENU_JS, "function showPriorLoadFailure()", 900
    )

    assert '"/api/models/activation"' in region
    assert 'status.state !== "error"' in region


def test_the_report_stays_quiet_during_a_live_selection() -> None:
    """A failure the user is watching is already reported by
    `pollActivation`; saying it twice would be worse than once."""
    region = _region(
        MENU_JS, "function showPriorLoadFailure()", 900
    )

    assert "if (selecting)" in region
