"""Confirm and Retry, and Continue on a branch that stopped, stay
reachable while an edit is reviewed.

Strategy: source inspection of `generator_edit.js`, plus a reading
of the two handlers to show that neither depends on where the
scrubber is.

The behavior this replaced: the review phase revealed both buttons
only on the last frame, and hid them again the moment you scrubbed
back to look at what the branch had done. The status line then read
"Return to the last frame to confirm or retry", so the only way back
to the actions was to find the end of the run again. Scrubbing back
is the point of the review phase, and it looked like it had cancelled
the edit.

Passing proves both buttons are revealed unconditionally in review,
and Continue on a stopped branch from any frame too, that the status
line still tells you which frame you are on, and that the two
actions are genuinely independent of the scrubber, which is what
makes revealing them safe rather than merely convenient.
"""

from __future__ import annotations

from pathlib import Path

EDIT_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "generator_edit.js"
)


def _source() -> str:
    return EDIT_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from generator_edit.js;"
        " update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


def _review_case() -> str:
    """The controller's review renderer."""
    body = _region("function renderReviewPhase()", 1400)
    end = body.find("function selectFrame()")
    assert end != -1, "the review renderer lost its boundary"
    return body[:end]


# -- the buttons --


def test_both_actions_are_revealed_in_review() -> None:
    body = _review_case()

    assert "btnConfirmEdit.hidden = false" in body
    assert "btnRetryEdit.hidden = false" in body


def test_neither_reveal_is_behind_the_frame_check() -> None:
    """The regression in one assertion: the reveals must come before
    the branch, not inside it. Both orders contain both lines, so
    position is the only thing that distinguishes them."""
    body = _review_case()
    gate = body.find(
        "currentFrame === run.frameCount()"
    )
    confirm = body.find("btnConfirmEdit.hidden = false")
    retry = body.find("btnRetryEdit.hidden = false")

    assert gate != -1, "the status line still needs the frame check"
    assert confirm < gate
    assert retry < gate


def test_continue_is_not_behind_the_frame_check_either() -> None:
    """Continue carries the branch on from its last frame whatever
    frame is on screen, so it is revealed with the other two, for a
    stopped branch only."""
    body = _review_case()
    gate = body.find(
        "currentFrame === run.frameCount()"
    )
    reveal = body.find(
        "btnContinueEdit.hidden = !reviewCanContinue()"
    )

    assert reveal != -1, "review no longer reveals Continue"
    assert reveal < gate


def test_the_status_line_still_names_the_frame() -> None:
    """Revealing the buttons everywhere costs the old wayfinding, so
    the text has to say where you are instead."""
    body = _review_case()

    assert "currentFrame" in body
    assert "Return to the last" not in body


# -- why revealing them is safe --


def test_confirm_does_not_read_the_scrubber() -> None:
    """It saves the whole run and then moves the scrubber itself, so
    the frame you are looking at cannot change what is written."""
    body = _region("function confirm()", 900)

    assert "requestCommit()" in body
    assert "requestSave()" in body
    assert "activate()" in body
    assert "currentFrame" not in body


def test_confirm_lands_on_the_last_frame_by_itself() -> None:
    """The premise of the test above. If this stopped being true,
    confirming from frame 12 would leave the scrubber at 12 over a
    saved run, which is the failure mode a snap-first workaround was
    going to defend against."""
    body = _region("function activate()", 900)

    assert "var frameCount = run.frameCount()" in body
    assert "currentFrame = frameCount - 1" in body


def test_retry_does_not_read_the_scrubber() -> None:
    """It restores the pre-edit arrays wholesale and re-enters the
    session, which navigates to the first editable frame."""
    body = _region("function retry()", 500)

    assert "restorePreEditCheckpoint()" in body
    assert "currentFrame" not in body
