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
