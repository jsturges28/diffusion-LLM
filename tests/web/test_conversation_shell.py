"""The generator's bounded conversation shell is wired whole.

Strategy: inspect the shipped page, its page-local styles, controller
lookups, and the two user-facing manuals. Browser-module tests drive
the behavior; these checks cover what their synthetic DOM cannot see:
containment, source order, native disclosure semantics, and asset
wiring.

Passing proves the fixed toolbar and one transcript scroller contain
the compact turns, rich active assistant, and Draft composer in
chronological and keyboard order, with every static controller id.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

import pytest

from src.web.server import _stamp_asset_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "src" / "web" / "static"
INDEX = STATIC / "index.html"
GUIDE = REPO_ROOT / "docs" / "GUIDE.md"
CONVERSATION_CSS = STATIC / "conversation.css"

OTHER_PAGES = (
    "menu.html",
    "analytics.html",
    "settings.html",
    "vision.html",
)
PAGE_SHEETS = ("conversation.css", "run_settings.css")


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _block(html: str, element_id: str, tag: str) -> str:
    """One nested element, found by id and balanced by tag name."""
    opening = re.search(
        rf'<{tag}\b[^>]*\bid="{re.escape(element_id)}"[^>]*>',
        html,
    )
    assert opening is not None, f"missing #{element_id}"
    depth = 0
    pattern = re.compile(rf"<(/?){tag}\b")
    for found in pattern.finditer(html, opening.start()):
        depth += -1 if found.group(1) else 1
        if depth == 0:
            return html[opening.start() : found.end()]
    raise AssertionError(f"unclosed {tag} #{element_id}")


def _ids(markup: str) -> Set[str]:
    return set(re.findall(r'\bid="([^"]+)"', markup))


def _rule(styles: str, selector: str, chars: int = 500) -> str:
    start = styles.find(selector)
    assert start != -1, selector
    return styles[start : start + chars]


def _page_scripts() -> List[Path]:
    names = re.findall(r'<script src="/([^"?]+)"', _html())
    return [
        STATIC / name
        for name in names
        if not name.startswith("vendor/")
    ]


def _static_controller_ids() -> Set[str]:
    """Literal ids the shipped generator scripts ask the DOM for."""
    found: Set[str] = set()
    patterns = (
        re.compile(
            r'document\.getElementById\(\s*"([^"]+)"\s*\)'
        ),
        re.compile(r'requiredElement\(\s*"([^"]+)"\s*\)'),
        re.compile(r'conversationActionsElement\(\s*"([^"]+)"\s*\)'),
    )
    for script in _page_scripts():
        source = script.read_text(encoding="utf-8")
        for pattern in patterns:
            found.update(pattern.findall(source))
    return found


def test_shell_regions_are_in_conversation_order() -> None:
    """Source order is also keyboard and reading order."""
    html = _html()
    positions = [
        html.index(f'id="{element_id}"')
        for element_id in (
            "conversation-toolbar",
            "conversation-transcript",
            "active-turn-workspace",
            "controls",
        )
    ]

    assert positions == sorted(positions)
    transcript = _block(
        html, "conversation-transcript", "section"
    )
    assert "active-assistant-card" in _ids(transcript)
    assert "active-turn-workspace" in _ids(transcript)
    assert "controls" in _ids(transcript)


def test_the_transcript_mount_has_bounded_native_controls() -> None:
    block = _block(
        _html(), "conversation-transcript", "section"
    )
    required = {
        "btn-load-older",
        "conversation-empty",
        "conversation-status",
        "conversation-action-status",
        "conversation-turns",
        "active-assistant-card",
        "active-assistant-actions",
        "active-turn-workspace",
        "controls",
    }

    assert required <= _ids(block)
    assert 'role="log"' in block
    assert 'aria-live="polite"' in block
    assert "Load older messages" in block


def test_the_active_workspace_owns_every_xai_surface() -> None:
    block = _block(_html(), "active-turn-workspace", "section")
    required = {
        "btn-save",
        "token-metrics",
        "stop-readout",
        "thinking-panel",
        "output-area",
        "overlay-select-group",
        "scrubber-section",
        "guided-edit-controls",
        "status-bar",
    }

    assert required <= _ids(block), required - _ids(block)
    assert "Save Run" in block


def test_the_rich_assistant_card_starts_hidden() -> None:
    html = _html()
    tag = re.search(
        r'<article id="active-assistant-card"[^>]*>',
        html,
    )

    assert tag is not None
    assert "conversation-active-assistant" in tag.group(0)
    assert " hidden" in tag.group(0)
    assert "Active response" in html


def test_composer_owns_prompt_settings_and_action() -> None:
    block = _block(_html(), "controls", "section")
    required = {
        "run-settings",
        "prompt-input",
        "btn-prompt-import",
        "prompt-history",
        "prompt-context",
        "btn-generate",
    }

    assert required <= _ids(block), required - _ids(block)
    assert "btn-save" not in _ids(block)
    assert "Draft" in block
    assert 'aria-label="Draft user message"' in block


def test_run_settings_uses_native_disclosure_semantics() -> None:
    html = _html()
    details = re.search(
        r'<details id="run-settings"[^>]*>', html
    )
    summary = re.search(
        r'<summary id="run-settings-summary"[^>]*>', html
    )

    assert details is not None
    assert summary is not None
    assert " open" not in details.group(0)
    assert 'aria-expanded="false"' in summary.group(0)
    assert 'aria-controls="run-settings-body"' in summary.group(0)
    assert "tabindex" not in summary.group(0)
    assert 'role="button"' not in summary.group(0)


def test_new_actions_are_native_buttons_with_names() -> None:
    html = _html()
    for element_id in (
        "btn-new-conversation",
        "btn-load-older",
        "btn-save",
        "btn-generate",
    ):
        tag = re.search(
            rf'<button id="{element_id}"[^>]*>', html
        )
        assert tag is not None, element_id
        assert 'type="button"' in tag.group(0), element_id

    assert "New Conversation" in html
    assert ">Send<" in re.sub(r"\s+", "", html)
    assert 'aria-label="Save Run"' in html


def test_message_action_dialogs_are_native_and_cancel_first() -> None:
    html = _html()
    for name in ("delete", "retry"):
        dialog = _block(
            html, f"conversation-{name}-dialog", "dialog"
        )
        cancel = re.search(
            rf'<button id="btn-conversation-{name}-cancel"[^>]*>',
            dialog,
        )
        confirm = re.search(
            rf'<button id="btn-conversation-{name}-confirm"[^>]*>',
            dialog,
        )

        assert cancel is not None
        assert confirm is not None
        assert 'type="button"' in cancel.group(0)
        assert 'type="button"' in confirm.group(0)
        assert "autofocus" in cancel.group(0)
        assert "autofocus" not in confirm.group(0)
        status = re.search(
            rf'<p id="conversation-{name}-status"[^>]*>',
            dialog,
        )
        assert status is not None
        assert 'role="alert"' in status.group(0)
        assert 'tabindex="-1"' in status.group(0)
        assert dialog.index(cancel.group(0)) < dialog.index(
            confirm.group(0)
        )


def test_action_mounts_and_live_feedback_ship_empty() -> None:
    html = _html()
    active = re.search(
        r'<div id="active-assistant-actions"[^>]*>',
        html,
    )
    feedback = re.search(
        r'<p id="conversation-action-status"[^>]*>',
        html,
    )

    assert active is not None
    assert " hidden" in active.group(0)
    assert 'role="group"' not in active.group(0)
    assert feedback is not None
    assert 'role="status"' in feedback.group(0)
    assert 'aria-live="polite"' in feedback.group(0)


def test_pending_cursor_is_scoped_to_conversation_dialogs() -> None:
    css = CONVERSATION_CSS.read_text(encoding="utf-8")

    assert ".conversation-confirmation-dialog.is-pending" in css
    assert ".modal-overlay.is-pending" not in css


def test_every_literal_controller_id_exists_once() -> None:
    """The stub creates missing ids, so check the shipped page."""
    html = _html()
    missing = [
        element_id
        for element_id in sorted(_static_controller_ids())
        if html.count(f'id="{element_id}"') != 1
    ]

    assert missing == []


def test_shell_controller_precedes_the_composition_root() -> None:
    html = _html()
    positions = [
        html.index(f'src="/{script}"')
        for script in (
            "conversation_view.js",
            "conversation_action_view.js",
            "conversation_actions.js",
            "conversation_shell.js",
            "app.js",
        )
    ]

    assert positions == sorted(positions)


def test_toolbar_is_fixed_over_one_transcript_scroller() -> None:
    styles = CONVERSATION_CSS.read_text(encoding="utf-8")
    shell = _rule(styles, "#conversation-shell {", 220)
    transcript = _rule(
        styles, "#conversation-transcript {", 350
    )

    assert "overflow: hidden" in shell
    assert "overflow-y: auto" in transcript
    assert "flex: 1 1 auto" in transcript


def test_rich_output_has_a_real_second_turn_minimum() -> None:
    styles = CONVERSATION_CSS.read_text(encoding="utf-8")
    card = _rule(
        styles, "#active-assistant-card {", 500
    )
    output = _rule(
        styles,
        "#active-turn-workspace > #output-section {",
        180,
    )
    canvas = _rule(
        styles, "#active-turn-workspace #output-area {", 100
    )

    assert "min-height: 410px" in card
    assert "flex: 0 0 auto" in card
    assert "flex: 1 0 auto" in output
    assert "min-height: 220px" in output
    assert "min-height: 190px" in canvas


def test_draft_and_sent_user_cards_remain_distinct() -> None:
    styles = CONVERSATION_CSS.read_text(encoding="utf-8")
    sent = _rule(styles, ".conversation-turn-user {", 220)
    draft = _rule(
        styles, "#controls.conversation-composer {", 420
    )

    assert "79, 195, 247" in sent
    assert "0, 255, 65" in draft
    assert "margin-top: auto" not in draft


def test_message_actions_keep_keyboard_and_coarse_pointer_reach(
) -> None:
    styles = CONVERSATION_CSS.read_text(encoding="utf-8")
    action = _rule(styles, ".conversation-action {", 420)
    focus = _rule(
        styles, ".conversation-action:focus-visible", 360
    )
    coarse = _rule(
        styles, "@media (hover: none), (pointer: coarse)", 420
    )
    reduced = _rule(
        styles, "@media (prefers-reduced-motion: reduce)", 300
    )

    assert "width: 28px" in action
    assert "height: 28px" in action
    assert "outline: 2px solid var(--accent)" in focus
    assert "opacity: 1" in coarse
    assert "width: 32px" in coarse
    assert "transition: none" in reduced


def test_generator_stylesheets_follow_the_shared_sheet() -> None:
    html = _html()
    positions = [
        html.index(f'href="/{sheet}"')
        for sheet in ("style.css", *PAGE_SHEETS)
    ]

    assert positions == sorted(positions)


@pytest.mark.parametrize("page", OTHER_PAGES)
def test_other_pages_do_not_load_generator_styles(
    page: str,
) -> None:
    html = (STATIC / page).read_text(encoding="utf-8")

    for sheet in PAGE_SHEETS:
        assert sheet not in html, page


def test_the_server_stamps_both_page_local_sheets() -> None:
    stamped = _stamp_asset_versions(_html())

    for sheet in PAGE_SHEETS:
        assert f"/{sheet}?v=" in stamped


def test_help_and_guide_name_the_moved_workflow() -> None:
    help_markup = _block(_html(), "modal-help", "dialog")
    guide = GUIDE.read_text(encoding="utf-8")

    for text in (help_markup, guide):
        assert "Run settings" in text
        assert "Save Run" in text
        assert "New Conversation" in text
        assert "Load older messages" in text
        assert "Text only" in text
