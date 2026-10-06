"""Static contracts for the Saved Conversation product boundary.

Strategy: inspect shipped pages, scripts and durable prose. Passing
proves the generator action, separate Analytics artifact view, shared
run-detail module and explicit Text-only boundary ship together rather
than as an unreachable backend.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_generator_action_and_confirmation_ship_together() -> None:
    page = _text(STATIC / "index.html")

    assert 'id="btn-save-conversation"' in page
    assert 'id="conversation-save-dialog"' in page
    assert 'id="conversation-save-xai"' in page
    assert 'id="conversation-save-text-only"' in page
    assert 'src="/saved_conversation_client.js"' in page
    assert 'src="/conversation_save.js"' in page


def test_analytics_keeps_runs_and_conversations_separate() -> None:
    page = _text(STATIC / "analytics.html")

    assert 'id="tab-runs"' in page
    assert 'id="tab-conversations"' in page
    assert 'id="conversations-panel"' in page
    assert 'id="conversation-detail-modal"' in page
    assert 'aria-labelledby="saved-conversation-detail-title"' in page
    assert 'src="/analytics_run_detail.js"' in page
    assert 'src="/analytics_conversations.js"' in page

    script = _text(STATIC / "analytics.js")
    assert "analyticsViewTabKeyDown" in script
    assert 'event.key === "ArrowLeft"' in script
    assert 'event.key === "ArrowRight"' in script


def test_docs_state_the_immutable_selected_path_boundary() -> None:
    guide = _text(ROOT / "docs" / "GUIDE.md")
    roadmap = _text(ROOT / "docs" / "ROADMAP.md")
    readme = _text(ROOT / "README.md")

    for text in (guide, roadmap, readme):
        assert "Save Conversation" in text
        assert "selected path" in text
    assert "whole branch DAG" in roadmap
    assert "Text only" in guide
    assert "conversation-level watermark" in roadmap
