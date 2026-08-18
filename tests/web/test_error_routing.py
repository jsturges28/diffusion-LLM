"""The generator acts on an error's scope, not just its text.

Strategy: read the shipped `app.js` and `index.html`. The decision
itself is unit-tested in `tests/web/static/wire_errors.test.js`, which
loads the classifier into a vm and drives it directly; what cannot be
checked there is whether the page bothers to ask. A classifier nothing
consults is worse than no classifier, because it reads as solved.

What passing proves is that the teardown is behind the question. Every
worker error used to reach one handler that ended the run and, when an
edit session was open, restored the pre-edit snapshot and left guided
mode. Correct for a generation that died, and wrong for a probe
refused because a generation was already running, which used to close
What If and discard the edit being composed.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"


def _handle_error() -> str:
    """The body of `handleError`, up to the next declaration."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.find("function handleError(data)")
    assert start != -1, (
        "handleError is gone from app.js; update this test rather"
        " than deleting it"
    )
    end = source.find("\nfunction ", start + 1)
    assert end != -1
    return source[start:end]


# -- the page asks --


def test_the_handler_classifies_before_acting() -> None:
    assert "wireErrorsRoute(data)" in _handle_error()


def test_the_classifier_loads_before_the_page_that_uses_it() -> None:
    """Classic global scripts, so order in the document is the only
    thing making the function exist when app.js runs."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "/wire_errors.js" in html
    assert html.index("/wire_errors.js") < html.index("/app.js")


# -- and acts on the answer --


def test_the_teardown_is_behind_the_scope_check() -> None:
    """The whole behavioural change. Both of these unwind an edit the
    user is composing, so neither may run for a failure that never
    touched the run."""
    body = _handle_error()
    guard = body.find("if (routed.unwindsRun)")

    assert guard != -1
    assert body.find("restoreEditSnapshot()") > guard
    assert body.find("resetGuidedMode()") > guard


def test_the_run_indicators_are_behind_it_too() -> None:
    """A probe failing must not report the generation as finished."""
    body = _handle_error()
    guard = body.find("if (routed.unwindsRun)")

    assert body.find("setGenerating(false)") > guard
    assert body.find("endRunStatus()") > guard


def test_the_message_is_shown_either_way() -> None:
    """Scope decides what is undone, not what is said. An auxiliary
    failure the user cannot see is its own kind of wrong."""
    body = _handle_error()
    shown = body.find("statusMessage.textContent")
    guard = body.find("if (routed.unwindsRun)")
    closing = body.find("\n  }\n", guard)

    assert shown != -1
    assert shown > closing, "the message is inside the guard"


def test_the_handler_reads_the_routed_message() -> None:
    """Rather than `data.message`, so an absent one has already been
    turned into something printable."""
    assert "routed.message" in _handle_error()


# -- and nobody else re-reads the frame --


def test_scope_is_decided_in_one_place() -> None:
    """`app.js` must not grow its own idea of what a scope means; the
    strings live in wire_errors.js and are tested there."""
    source = APP_JS.read_text(encoding="utf-8")

    assert not re.search(r'data\.scope', source)
    assert not re.search(r'"request"\s*===', source)
