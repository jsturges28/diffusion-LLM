"""No page decides anything from a model id or from `model_type`.

Strategy: source inspection of the shipped classic scripts, the
approach this repo uses for pages it cannot import. What the axes mean
is tested in `test_model_axes.py` and what the supervisor does with
them in `test_activation_validation.py`; neither can see the thing
`ROADMAP-01` is actually about, which is a page inferring one property
of a model from an unrelated one.

The finding's Verification clause ends "no UI decision may inspect a
model ID", and one did: the renoise note in the edit UI compared
`activeModelId` against "diffusiongemma", so it would have gone quiet
for the next renoising model to arrive under another id. That is now a
declared capability, and this file is what stops the pattern coming
back.

Passing proves the conflated `capabilities.model_type` is gone from
every page, that no page compares against a registered model id, and
that each surviving decision reads the axis that actually answers it:
devices from the declaration, canvas affordances from the generation
shape, and per-class visuals from the family.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.backends.registry import REGISTRY

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
PAGE_SCRIPTS = (
    "app.js",
    "menu.js",
    "analytics.js",
    "settings.js",
    "overlays.js",
)


def _source(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _code(name: str) -> str:
    """Source with comments stripped.

    The rules below are about what the code does, and a comment
    naming a model to explain a behaviour is documentation rather
    than a decision. Without this, improving a comment could fail
    a test.
    """
    text = _source(name)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def _region(name: str, anchor: str, chars: int) -> str:
    source = _source(name)
    start = source.find(anchor)
    assert start != -1, f"{anchor!r} is gone from {name}"
    return source[start:start + chars]


# -- the conflated value is gone --


def test_no_page_reads_the_old_model_type() -> None:
    """One value drove the family glyph, the canvas affordances, the
    chart gating and CPU capability. A surviving read would be a
    decision taken on an axis that no longer answers it."""
    for name in PAGE_SCRIPTS:
        assert "capabilities.model_type" not in _code(name), name


def test_saved_runs_may_still_carry_it() -> None:
    """The other half of the decision, asserted so it is not quietly
    undone: the on-disk field stays, because the corpus is not
    migrated, and Analytics reads it off a run rather than off a live
    model."""
    assert 'run.model_type === "autoregressive"' in _code(
        "analytics.js"
    )


# -- no page names a model --


def test_no_page_compares_against_a_model_id() -> None:
    """The Verification clause's rule. A page that branches on an id
    has to be edited for every model added, and the edit is invisible
    until someone notices the missing behaviour."""
    for name in PAGE_SCRIPTS:
        code = _code(name)
        for model_id in REGISTRY:
            for quoted in (f'"{model_id}"', f"'{model_id}'"):
                assert quoted not in code, (
                    f"{name} names {model_id}"
                )


def test_the_renoise_note_reads_a_capability() -> None:
    """The decision that used to name DiffusionGemma. Kept as its own
    test because the rule above only proves the id is gone, not that
    what replaced it asks the right question."""
    region = _region("app.js", "function renoiseNote()", 400)

    assert "capabilities.remask_renoises" in region


# -- each decision reads its own axis --


def test_the_device_decision_reads_the_declaration() -> None:
    """Devices were inferred from the family, which is how a 17 GiB
    diffusion model came to be offered a CPU load."""
    region = _region(
        "app.js", "function defaultDeviceFor(model)", 400
    )

    assert "supportedDevices(model)" in region
    assert "capabilities.family" not in region


def test_the_menu_row_reads_the_declaration_too() -> None:
    """Two pages ask the same question, and the menu's answer is the
    one a user sees as a toggle or a static tag."""
    region = _region("menu.js", "function buildRow(model", 2400)

    assert "supportedDevices(model)" in region
    assert "devices.length > 1" in region


def test_the_canvas_affordances_read_the_shape() -> None:
    """Shape rather than family, so a state-space model is not offered
    remasking controls it has no masked positions for."""
    region = _region("app.js", "function isAppendOnly()", 400)

    assert "capabilities.generation_shape" in region
    assert '"append_only"' in region


def test_the_glow_reads_the_family() -> None:
    """Family rather than shape: the glow pairs are per model class,
    so a state-space model wants its own even though it appends."""
    region = _region(
        "app.js", "function applyTokenBirthGlow()", 500
    )

    assert "capabilities.family" in region
    assert "generation_shape" not in region


def test_the_prompt_copy_reads_the_input_mode() -> None:
    """A base model continues your text rather than answering it, and
    the box should say which. Read off the declared mode, so the next
    base checkpoint is described right without an edit here."""
    region = _region("app.js", "function applyPromptMode()", 500)

    assert "capabilities.input_mode" in region
    assert "PROMPT_MODE_COPY" in region


def test_the_prompt_count_shows_no_inequality() -> None:
    """A truncated count used to be prefixed with an inequality sign,
    which answered the wrong question. The number a user tunes against
    the window has to be the whole prompt's, and a floor is unusable
    for that."""
    code = _code("app.js")

    assert "\\u2265" not in code
    assert "\u2265" not in code


def test_a_truncated_count_still_warns() -> None:
    """The bug the inequality sign hid. A floor below the window read
    as "fits", so a 600,000 character prompt showed no warning at all
    while being three times over. Hitting the worker's cap is itself
    proof the prompt is over, because the cap sits far past any window
    here."""
    region = _region(
        "app.js", "function applyPromptContextWarning(", 700
    )

    assert "truncated ||" in region


def test_the_two_context_failures_look_different() -> None:
    """One is refused and one runs short, so they cannot both be
    amber. The refusal takes the danger colour."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")

    assert ".prompt-context.is-over" in css
    assert "is-over" in _code("app.js")


def test_a_clipped_status_message_carries_its_full_text() -> None:
    """The general answer to a row that truncates. Some messages are
    not ours to shorten: a CUDA out-of-memory report comes from torch
    and runs past the window on its own, and shortening ours one at a
    time loses whichever one is added next."""
    region = _region("app.js", "function watchStatusMessage()", 700)

    assert "MutationObserver" in region
    body = _region(
        "app.js", "function applyStatusMessageTitle()", 600
    )
    assert "statusMessage.title" in body
    assert "scrollWidth" in body


def test_the_tooltip_cannot_be_bypassed_by_a_new_message() -> None:
    """An observer rather than a helper, and this is the reason: there
    are already more than ten places that assign the row's text, and a
    helper is only as good as the next one remembering it."""
    code = _code("app.js")
    writes = code.count("statusMessage.textContent =")

    assert writes > 5, writes
    assert "observer.observe(statusMessage" in code


def test_the_reasoning_panel_can_scroll() -> None:
    """A long trace was unreachable: the panel and the canvas were
    plain blocks in a section that hides its overflow, so the panel
    took its natural height and pushed the canvas past the clip
    while neither of them scrolled."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    panel = css[css.index("#thinking-content {"):][:400]

    assert "overflow-y: auto" in panel
    assert "max-height" in panel
    section = css[css.index("#output-section {"):][:400]
    assert "flex-direction: column" in section


def test_the_reasoning_details_is_not_a_flex_container() -> None:
    """The first attempt at the cap made the `details` itself a flex
    column, which let the trace escape the panel's box and draw over
    the canvas: a `details` lays its disclosure content out through a
    slot. The cap belongs on the content, in units that need no
    resolved parent height."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    panel = css[css.index("#thinking-panel {"):][:400]

    assert "display: flex" not in panel
    assert "max-height" not in panel


def test_the_markup_ships_the_chat_wording() -> None:
    """Every model today is instruction-tuned, and a page with none
    loaded should read as chat rather than as a blank the script has
    to fill before the label makes sense."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert ">Prompt</label>" in html
    assert 'placeholder="Enter a prompt..."' in html


def test_the_family_glyph_is_keyed_by_family() -> None:
    """A table rather than a branch, so a new class is one entry. The
    label matters more than the glyph: falling back to the diffusion
    name would tell the user a state-space model was a diffusion
    one."""
    code = _code("menu.js")

    assert "_FAMILY_LABELS" in code
    assert "state_space:" in code
