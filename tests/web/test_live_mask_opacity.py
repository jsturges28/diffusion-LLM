"""A mask reports its confidence while the run is being written.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages. The span builder these options reach
is exercised properly in `tests/web/static/overlays_span.test.js`;
what needs guarding here is that the live path asks for the grading
at all, and that both of its two entry points ask for the same
thing.

The bug: when per-token spans replaced the character renderer in the
live view, the new path passed an empty options object, deliberately,
to keep that refactor visually neutral. The `opacityFor` hook existed
and only the scrubbed path used it. So a mask brightened toward its
reveal when you scrubbed back over a finished run and stayed flat
while the run was actually being written, which is the one moment the
reading says something. On DiffusionGemma, where the number only
exists with the Entropy Signal on, that made a working feature look
like a broken one.

Passing proves the live view grades masks, that its two render paths
cannot drift apart on it, and that the three hooks which would be
meaningless mid-run are still left off.
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


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- the live view asks for it --


def test_the_live_options_grade_masks() -> None:
    source = _source()

    assert (
        "var LIVE_TOKEN_OPTIONS = { opacityFor: tokenOpacityFn };"
        in source
    )


def test_the_grading_is_the_same_one_the_scrubber_uses() -> None:
    """One function, so a mask cannot mean two different things
    depending on whether the run has finished."""
    body = _region("function tokenLayerOptions(isOriginal)", 400)

    assert "opacityFor: tokenOpacityFn" in body


def test_both_live_paths_pass_the_same_options() -> None:
    """A frame either reuses the spans already on the page or
    rebuilds them. Handing hooks to one and not the other is the
    exact shape of the bug this file was written after."""
    live = _region("function renderLiveFrame(tokens, revealed)", 900)
    rebuild = _region("function rebuildLiveTokens(tokens)", 700)

    assert "LIVE_TOKEN_OPTIONS" in live
    assert "LIVE_TOKEN_OPTIONS" in rebuild


def test_nothing_else_supplies_live_options() -> None:
    """One definition and the two callers above. A third would be a
    path that renders live tokens on its own terms."""
    uses = re.findall(r"\bLIVE_TOKEN_OPTIONS\b", _source())

    assert len(uses) == 3


# -- and only that one --


def test_the_hooks_with_nothing_to_do_stay_off() -> None:
    """`colorFor` has no work mid-run, because the overlay drawer is
    hidden until the scrubber activates. `maskedFor` and `classFor`
    serve remask selection, which is unreachable while a run is in
    flight. Adding them would cost a callback per token per frame to
    compute an answer nothing can display."""
    body = _region("var LIVE_TOKEN_OPTIONS", 120)

    for hook in ("colorFor", "maskedFor", "classFor"):
        assert hook not in body, hook


def test_the_drawer_is_hidden_while_a_run_streams() -> None:
    """The premise of the test above, pinned so it cannot quietly
    stop being true and leave the reasoning stale."""
    body = _region("function deactivateScrubber()", 400)

    assert "overlaySelectGroup.hidden = true" in body


# -- what the hook reads --


def test_a_selected_remask_is_held_solid() -> None:
    """A position the user picked reads as a choice rather than as
    one more low-confidence mask, so it opts out of the grading."""
    body = _region("function tokenOpacityFn(index, tok, masked)", 300)

    assert "remaskedPositions[index] === true" in body
    assert "return null" in body


def test_a_resolved_token_is_not_graded() -> None:
    body = _region("function tokenOpacityFn(index, tok, masked)", 300)

    assert "!masked" in body


def test_an_ungraded_mask_falls_to_the_floor() -> None:
    """Absent and zero mean the same thing to the renderer, which is
    what lets a run without the Entropy Signal look exactly as it
    did before any of this."""
    body = _region("function maskOpacity(c)", 300)

    assert 'typeof c !== "number" || c <= 0' in body
    assert "MASK_OPACITY_FLOOR" in body
