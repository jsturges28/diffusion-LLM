"""What a keyboard can reach, and what it must not.

Strategy: read the shipped markup and CSS. Every claim here is about
which elements a browser puts in the tab order, or about whether a
focused control shows anything, and both are decided by static rules
and attributes, so none of it needs a browser. What it cannot check is
how the traversal feels, and that stays in
`docs/MANUAL_VERIFICATION.md`.

The bug that prompted this file: the maintainer counted eleven tab
stops between the last visible control on the generator and the first
link in its header, with nothing to show for them. They were the
controls inside the three modals, which sit in the document at all
times and were hidden with `opacity: 0` and `pointer-events: none`.
Neither of those touches the keyboard. Analytics was far worse, with
forty-eight, most of them inside the run detail modal.

`visibility: hidden` is what actually removes a subtree from the tab
order, from hit testing and from the accessibility tree, so that is
what the rule carries now. The tests below pin the rule, pin the fade
it must not break, and pin the reason it is load-bearing, because a
rule whose purpose is invisible is a rule someone simplifies away.

The rest of the file is about controls whose visible part is not the
element that takes focus, which turned out to be a recurring shape:
a toggle hides its real checkbox, and a table row is not focusable at
all while three things inside it are.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pytest

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
MODAL_PAGES = ("index.html", "analytics.html")

# Everything a browser tab-stops on by default, minus the ones
# explicitly opted out. `tabindex="-1"` and `disabled` are the two
# ways an author says "not this one".
_FOCUSABLE = re.compile(
    r"<(?:button|a|input|select|textarea)\b(?![^>]*\btabindex=\"-1\")"
    r"(?![^>]*\bdisabled\b)[^>]*>",
    re.IGNORECASE,
)
_MODAL_OPEN = re.compile(
    r'<div id="(?P<id>[a-z-]+)" class="modal-overlay hidden">'
)


def _rule(selector: str, chars: int = 1600) -> str:
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find(selector)
    assert start != -1, (
        f"rule {selector!r} is gone from style.css; update this test"
        " rather than deleting it"
    )
    body = css[start + len(selector) : start + chars]
    return body[: body.find("}")]


def _block(html: str, start: int) -> str:
    """The element beginning at `start`, by counting div tags."""
    depth = 0
    for tag in re.finditer(r"<(/?)div\b", html[start:]):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            return html[start : start + tag.end()]
    raise AssertionError("unclosed div; the markup changed shape")


def _hidden_modals(page: str) -> Dict[str, int]:
    """Focusable descendants per closed modal on a page."""
    html = (STATIC / page).read_text(encoding="utf-8")
    found: Dict[str, int] = {}
    for match in _MODAL_OPEN.finditer(html):
        block = _block(html, match.start())
        found[match.group("id")] = len(_FOCUSABLE.findall(block))
    return found


# -- the rule --


def test_a_closed_modal_is_hidden_from_the_keyboard() -> None:
    """The whole point. Without this the controls inside are
    invisible and still focusable, which is worse than either."""
    rule = _rule(".modal-overlay.hidden {")

    assert "visibility: hidden" in rule


def test_it_still_fades_rather_than_vanishing() -> None:
    """`visibility` does not interpolate, so it flips at the end of
    its own transition. Given no delay it flips immediately and takes
    the modal with it, losing the fade the opacity is there for."""
    rule = _rule(".modal-overlay.hidden {")

    match = re.search(
        r"transition:[^;]*visibility\s+0s\s+\w+\s+([\d.]+)s", rule
    )
    assert match is not None, (
        "visibility flips with no delay, so the fade-out is gone"
    )
    opacity = re.search(r"transition:\s*opacity\s+([\d.]+)s", rule)
    assert opacity is not None
    assert float(match.group(1)) >= float(opacity.group(1))


def test_showing_a_modal_is_not_delayed_too() -> None:
    """The delay belongs to the hidden state only. On the base rule
    it would postpone every open by a quarter second."""
    base = _rule(".modal-overlay {")

    assert "visibility" not in base


# -- and why it is load-bearing --


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_closed_modals_really_do_hold_controls(page: str) -> None:
    """The stakes, asserted so the rule above cannot be read as
    defensive tidying and simplified away. These are real buttons and
    links that were in the tab order while invisible."""
    counts = _hidden_modals(page)

    assert counts, f"no closed modals found in {page}"
    assert sum(counts.values()) >= 1, (
        f"{page} has closed modals with nothing focusable inside;"
        " if that is now true of all of them, this test is stale"
    )


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_every_modal_ships_closed(page: str) -> None:
    """The rule only applies at rest if the markup starts at rest. A
    modal shipped without the class is visible on load, which is a
    louder bug, but it also silently opts out of the fix above."""
    html = (STATIC / page).read_text(encoding="utf-8")

    tagged = re.finditer(
        r'<div id="[a-z-]+" class="([^"]*)"', html
    )
    for match in tagged:
        classes = match.group(1).split()
        if "modal-overlay" not in classes:
            continue
        assert "hidden" in classes, match.group(0)


# -- controls whose focus ring had nowhere to land --


def test_a_toggle_marks_itself_when_focused() -> None:
    """A `.toggle-switch` hides its real checkbox with `opacity: 0`
    and no width or height, so the platform's ring lands on nothing
    and tabbing onto Experimental, Thinking or Alternatives showed
    nothing at all. The visible slider has to wear it."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    assert (
        ".toggle-switch input:focus-visible + .toggle-slider" in css
    )


def test_the_hidden_input_is_still_the_focus_target() -> None:
    """The other half. The ring moves to the slider; the focus does
    not. Hiding the input with `display: none` instead would take the
    toggle out of the tab order altogether."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    rule = _rule(".toggle-switch input {", chars=200)

    assert "opacity: 0" in rule
    assert "display: none" not in rule
    assert css.count(".toggle-slider") > 1


def test_a_checkbox_shows_its_own_ring() -> None:
    """`appearance: none` takes the native control away, and with it
    the platform's ring on some engines, so the Analytics row
    checkboxes have to draw their own."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    assert (
        'input[type="checkbox"].app-checkbox:focus-visible' in css
    )


def test_a_focused_row_is_visible_on_analytics() -> None:
    """The rows themselves are not focusable, only their checkbox,
    star and caret are, so there is nothing to give a focus style to.
    `:focus-within` lights the row instead, whichever of the three
    Tab landed on."""
    css = (STATIC / "analytics.css").read_text(encoding="utf-8")

    assert "#runs-table tbody tr:focus-within" in css


# -- the model picker's resting semantics --
#
# Its behaviour is driven and asserted in
# tests/web/static/model_picker_keys.test.js. What lives here is the
# half that is markup, which that harness never reads.


def _model_select_tag() -> str:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<div id="model-select"[^>]*>', html)
    assert match is not None, (
        "the model picker is gone from index.html; update this test"
        " rather than deleting it"
    )
    return match.group(0)


def test_the_model_picker_is_a_combobox() -> None:
    """It declared `role="button"` while owning a listbox, which is
    the same mis-description `custom_select.js` carried: a control
    that expands is a combobox, and the list is what it controls."""
    tag = _model_select_tag()

    assert 'role="combobox"' in tag
    assert 'aria-controls="model-select-list"' in tag


def test_it_starts_collapsed_in_the_markup() -> None:
    """The JS flips this on open and close, but the resting value has
    to be right before anything has been opened, or the first thing a
    screen reader says about the control is wrong."""
    assert 'aria-expanded="false"' in _model_select_tag()


def test_the_closed_list_holds_no_tab_stops() -> None:
    """It uses the `hidden` attribute rather than the opacity trick
    the modals used, so its rows are out of the tab order at rest.
    Worth pinning next to that fix, since it is the same mistake one
    element over."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<ul id="model-select-list"[^>]*>', html)

    assert match is not None
    assert re.search(r"\shidden(?=[\s>])", match.group(0)) is not None


def test_the_loading_overlay_is_exempt_and_stays_earnable() -> None:
    """It hides the same way and deliberately lacks the visibility
    rule, which is only defensible while there is nothing inside it
    to focus. This is the test that notices when that changes."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    start = html.find('<div id="loading-overlay"')
    assert start != -1

    assert _FOCUSABLE.findall(_block(html, start)) == []
