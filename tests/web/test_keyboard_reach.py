"""What a keyboard can reach, and what it must not.

Strategy: read the shipped markup and CSS. Every claim here is about
which elements a browser puts in the tab order, which is decided by
static rules and attributes, so none of it needs a browser to check.
What it cannot check is how the traversal feels, and that stays in
`docs/MANUAL_VERIFICATION.md`.

The bug that prompted this file: the maintainer counted eleven tab
stops between the last visible control on the generator and the first
link in its header, with nothing to show for them. They were the
controls inside the three modals, which sit in the document at all
times and were hidden with `opacity: 0` and `pointer-events: none`.
Neither of those touches the keyboard. Analytics was far worse, with
forty-eight, most of them inside the run detail modal.

That was first fixed with `visibility: hidden`, and then fixed
properly: the modals are native `<dialog>` elements now, opened with
`showModal()`. A closed dialog is `display: none`, so it holds no tab
stops without anyone arranging it, and an open one traps focus and
makes the rest of the page inert, which the CSS fix could not do.

So the tests below moved from "the rule that hides them is present"
to "these are dialogs and nothing hides them by hand", which is a
weaker-looking claim about a stronger mechanism.
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
    r'<dialog id="(?P<id>[a-z-]+)" class="modal-overlay">'
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


def _block(html: str, start: int, tag_name: str = "div") -> str:
    """The element beginning at `start`, by counting its own tags."""
    depth = 0
    pattern = r"<(/?)" + tag_name + r"\b"
    for tag in re.finditer(pattern, html[start:]):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            return html[start : start + tag.end()]
    raise AssertionError(
        f"unclosed {tag_name}; the markup changed shape"
    )


def _modals(page: str) -> Dict[str, int]:
    """Focusable descendants per modal on a page."""
    html = (STATIC / page).read_text(encoding="utf-8")
    found: Dict[str, int] = {}
    for match in _MODAL_OPEN.finditer(html):
        block = _block(html, match.start(), "dialog")
        found[match.group("id")] = len(_FOCUSABLE.findall(block))
    return found


# -- they are dialogs, which is what does the work --


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_every_modal_is_a_dialog(page: str) -> None:
    """A plain div can be hidden but cannot trap focus or make the
    rest of the page inert, which is why the CSS-only fix that came
    before this one was only half of it."""
    html = (STATIC / page).read_text(encoding="utf-8")

    for match in re.finditer(r'class="[^"]*modal-overlay', html):
        line_start = html.rfind("<", 0, match.start())
        assert html.startswith("<dialog", line_start), (
            f"{page} has a modal-overlay that is not a dialog"
        )


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_no_modal_ships_open(page: str) -> None:
    """`open` in the markup would put it on screen at load, and would
    also be the non-modal kind that traps nothing."""
    html = (STATIC / page).read_text(encoding="utf-8")

    for match in _MODAL_OPEN.finditer(html):
        assert " open" not in match.group(0), match.group("id")


def test_a_closed_modal_is_not_laid_out() -> None:
    """The one that got away, and it broke the whole mouse.

    The UA closes a dialog with `dialog:not([open])
    { display: none }`, and an author rule beats the UA sheet whatever
    its specificity. So an unconditional `display: flex` on
    .modal-overlay left all seven closed dialogs laid out at
    `position: fixed; inset: 0; z-index: 90`, invisible only because
    their opacity was zero. Opacity does not stop a pointer: every
    click on either page hit a stack of invisible modals, and the
    maintainer navigated the whole app by keyboard for a session
    before mentioning it.

    Asserted as "display is scoped to [open]" rather than as a
    computed style, which nothing here can evaluate."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    base = _rule(".modal-overlay {")
    assert "display:" not in base, (
        "display on the unconditional rule overrides the UA's"
        " display:none and lays out every closed modal"
    )

    start = css.find(".modal-overlay[open] {")
    assert start != -1, "nothing gives an open modal a display"
    assert "display: flex" in css[start : start + 120]


def test_nothing_hides_a_modal_by_hand_any_more() -> None:
    """The `hidden` class and the `visibility` rule that went with it
    are both gone: closed is `display: none` from the UA sheet now.
    A leftover rule would fight the dialog rather than help it."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    assert ".modal-overlay.hidden" not in css


def test_the_backdrop_carries_the_dim() -> None:
    """It moved off the element, which is now only the box that
    centres the modal. Without this the page behind is undimmed."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find(".modal-overlay::backdrop")

    assert start != -1, "the backdrop lost its background"
    assert "rgba(0, 0, 0" in css[start : start + 120]


def test_it_still_fades_rather_than_vanishing() -> None:
    """`display` is what opens and closes a dialog and it does not
    interpolate, so an opacity transition alone animates nothing: the
    element is already gone. `allow-discrete` holds it through the
    fade and `@starting-style` gives the entry something to animate
    from."""
    rule = _rule(".modal-overlay {")
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    assert "display 0.25s allow-discrete" in rule
    assert "overlay 0.25s allow-discrete" in rule
    assert "@starting-style" in css


def test_the_fade_yields_to_reduced_motion() -> None:
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find("@media (prefers-reduced-motion: reduce)")
    found = False
    while start != -1 and not found:
        block = css[start : start + 260]
        found = ".modal-overlay" in block
        start = css.find(
            "@media (prefers-reduced-motion: reduce)", start + 1
        )

    assert found, "the modal fade ignores reduced motion"


# -- and why it is load-bearing --


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_modals_really_do_hold_controls(page: str) -> None:
    """The stakes, so none of the above reads as defensive tidying.
    These are real buttons and links, and every one of them was in
    the tab order while its modal was invisible: nine on the
    generator, forty-eight on Analytics."""
    counts = _modals(page)

    assert counts, f"no modals found in {page}"
    assert sum(counts.values()) >= 1, (
        f"{page} has modals with nothing focusable inside; if that is"
        " now true of all of them, this test is stale"
    )


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


@pytest.mark.parametrize("page", MODAL_PAGES)
def test_a_modal_opens_focused_on_its_box(page: str) -> None:
    """`showModal` focuses the first focusable descendant, which was
    the close button: it took a ring the moment a modal opened, and
    the arrow keys could not scroll a long modal because nothing
    scrollable held focus. The box takes the focus instead, which
    fixes both."""
    html = (STATIC / page).read_text(encoding="utf-8")

    for match in re.finditer(r'<div class="modal-box[^>]*>', html):
        tag = match.group(0)
        assert 'tabindex="-1"' in tag, tag
        assert "autofocus" in tag, tag


def test_the_box_shows_no_ring_of_its_own() -> None:
    """It is a container that happens to hold focus, not a control.
    A ring around the whole modal would be noise."""
    rule = _rule(".modal-box:focus {", chars=120)

    assert "outline: none" in rule


def test_the_close_button_rings_when_tabbed_to() -> None:
    """Not on open any more, but it still has to show something when
    Tab reaches it, and the platform's default hugs a box far wider
    than the glyph inside it."""
    rule = _rule(".modal-close:focus-visible {", chars=220)

    assert "outline:" in rule


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
