"""A mask reports its confidence while the run is being written.

Strategy: source inspection of `generator_canvas.js`, the approach
this repo uses for its classic-script pages. The span builder these
options reach is exercised properly in
`tests/web/static/overlays_span.test.js`; what needs guarding here is
that the live path asks for the grading at all, and that both of its
two entry points ask for the same thing.

The bug: when per-token spans replaced the character renderer in the
live view, the new path passed an empty options object, deliberately,
to keep that refactor visually neutral. The `opacityFor` hook existed
and only the scrubbed path used it. So a mask brightened toward its
reveal when you scrubbed back over a finished run and stayed flat
while the run was actually being written, which is the one moment the
reading says something. On DiffusionGemma, where the number then only
existed with the Entropy Signal on, that made a working feature look
like a broken one. That toggle has since been removed and the model
measures every position, so the confusion it caused cannot recur.

Passing proves the live view grades masks, that its two render paths
cannot drift apart on it, and that the three hooks which would be
meaningless mid-run are still left off.

The curve itself has since moved to `overlays.js`, so that both pages
draw a mask the same way, and it is tested by being run rather than
by being read: see `tests/web/static/overlays_mask_opacity.test.js`.
What stays here is the wiring, which source inspection is the only
tool for.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "app.js"
)
CANVAS_JS = APP_JS.with_name("generator_canvas.js")
EDIT_JS = APP_JS.with_name("generator_edit.js")


def _source() -> str:
    return CANVAS_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from generator_canvas.js;"
        " update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- the live view asks for it --


def test_the_live_options_grade_masks() -> None:
    body = _region("var liveTokenOptions", 120)

    assert "opacityFor: tokenOpacity" in body


def test_the_grading_is_the_same_one_the_scrubber_uses() -> None:
    """One function, so a mask cannot mean two different things
    depending on whether the run has finished."""
    body = _region("function tokenLayerOptions(isOriginal)", 700)

    assert "opacityFor:" in body
    assert "tokenOpacityWithEdit(" in body


def test_both_live_paths_pass_the_same_options() -> None:
    """A frame either reuses the spans already on the page or
    rebuilds them. Handing hooks to one and not the other is the
    exact shape of the bug this file was written after."""
    live = _region(
        "function renderLiveFrame(tokens, revealed, live)",
        1300,
    )
    rebuild = _region("function rebuildLiveTokens(tokens)", 700)

    assert "liveTokenOptions" in live
    assert "liveTokenOptions" in rebuild


def test_nothing_else_supplies_live_options() -> None:
    """One definition, the two callers above, and loadSettings
    writing the mask-reveal preference into it. A fifth would be a
    path that renders live tokens on its own terms."""
    uses = re.findall(r"\bliveTokenOptions\b", _source())

    assert len(uses) == 4


def test_the_reveal_preference_reaches_the_live_options() -> None:
    """The live options are one object built once, so the setting is
    copied in where the preferences load rather than read per token
    per frame inside the render loop."""
    body = _region("function applySettings()", 500)

    assert (
        "liveTokenOptions.revealMask =" in body
        and "overlaysDrawsGuess(settings)" in body
    )


# -- and only that one --


def test_the_hooks_with_nothing_to_do_stay_off() -> None:
    """`colorFor` has no work mid-run, because the overlay drawer is
    hidden until the scrubber activates. `maskedFor` and `classFor`
    serve remask selection, which is unreachable while a run is in
    flight. Adding them would cost a callback per token per frame to
    compute an answer nothing can display."""
    body = _region("var liveTokenOptions", 120)

    for hook in ("colorFor", "maskedFor", "classFor"):
        assert hook not in body, hook


def test_the_drawer_is_hidden_while_a_run_streams() -> None:
    """The premise of the test above, pinned so it cannot quietly
    stop being true and leave the reasoning stale."""
    source = EDIT_JS.read_text(encoding="utf-8")
    start = source.index("function deactivate()")
    body = source[start : start + 400]

    assert "canvas.deactivate()" in body


# -- what the hook reads --


def test_a_selected_remask_is_held_solid() -> None:
    """A position the user picked reads as a choice rather than as
    one more low-confidence mask, so it opts out of the grading."""
    body = _region(
        "function tokenOpacityWithEdit(", 400
    )

    assert "edit.remaskedPositions[index] === true" in body
    assert "return null" in body


def test_a_resolved_token_is_not_graded() -> None:
    body = _region(
        "function tokenOpacityWithEdit(", 400
    )

    assert "!masked" in body


def test_the_curve_is_the_shared_one() -> None:
    """The generator no longer owns it. The curve moved to
    `overlays.js` when Analytics needed the same mask to read the
    same way, and its behavior is tested by running it, in
    `tests/web/static/overlays_mask_opacity.test.js`, rather than by
    reading it here."""
    body = _region(
        "function tokenOpacityWithEdit(", 400
    )

    assert "overlaysMaskOpacity(" in body
    assert "function maskOpacity" not in _source()
