"""The generator's bounded conversation shell is wired whole.

Strategy: inspect the shipped page, its page-local styles, controller
lookups, and the two user-facing manuals. Browser-module tests drive
the behavior; these checks cover what their synthetic DOM cannot see:
containment, source order, native disclosure semantics, and asset
wiring.

Passing proves transcript, active XAI workspace, and composer have
separate owners, keep reading and keyboard order, and expose every
static controller id.
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


def test_the_transcript_mount_has_bounded_native_controls() -> None:
    block = _block(
        _html(), "conversation-transcript", "section"
    )
    required = {
        "btn-load-older",
        "conversation-empty",
        "conversation-status",
        "conversation-turns",
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


def test_every_literal_controller_id_exists_once() -> None:
    """The stub creates missing ids, so check the shipped page."""
    html = _html()
    missing = [
        element_id
        for element_id in sorted(_static_controller_ids())
        if html.count(f'id="{element_id}"') != 1
    ]

    assert missing == []


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
