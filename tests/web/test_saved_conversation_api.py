"""HTTP contract for immutable saved conversation snapshots.

Strategy: seed the standard-library stores, then drive the real router
through a temporary supervisor data root. Passing proves exact heads,
idempotent publication, paged reads, pinned-run endpoints, rename CAS,
and deletion keep stable HTTP semantics.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from src.web import conversation_store
from src.web import run_store
from src.web import server


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _ready_path(
    root: Path, *, linked: bool = False
) -> tuple[
    conversation_store.ConversationMutation,
    conversation_store.AppendResult,
]:
    created = conversation_store.create(root)
    appended = conversation_store.append_user(
        root,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Why did this token change?",
        model_id="llada",
        input_mode="chat",
    )
    completed = conversation_store.update_assistant(
        root,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="The distribution changed.",
        partial=False,
        metadata={"status": "completed"},
    )
    if not linked:
        return completed, appended
    bundle = run_store.RunBundle(
        metadata={
            "backend": "llada",
            "model": "LLaDA",
            "model_type": "diffusion",
            "created_at": "2026-10-06T00:00:00Z",
            "prompt": "Why?",
        },
        final_text="The distribution changed.",
        frames=["The distribution changed."],
    )
    run_id, revision = run_store.save(
        root, bundle, model_id="llada"
    )
    linked_turn = conversation_store.set_run_link(
        root,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=conversation_store.RunLink(run_id, revision),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )
    return linked_turn, appended


def _head(
    mutation: conversation_store.ConversationMutation,
) -> dict[str, object]:
    manifest = mutation.manifest
    assert manifest.branch_id is not None
    assert manifest.tail_turn_id is not None
    assert manifest.tail_version is not None
    return {
        "conversation_id": manifest.id,
        "branch_id": manifest.branch_id,
        "branch_revision": manifest.revision,
        "turn_count": manifest.turn_count,
        "tail_turn_id": manifest.tail_turn_id,
        "tail_version": manifest.tail_version,
    }


def _save(
    client: TestClient,
    mutation: conversation_store.ConversationMutation,
    *,
    operation_id: str | None = None,
) -> dict[str, object]:
    body = dict(
        _head(mutation),
        operation_id=operation_id or uuid4().hex,
        title="Token investigation",
    )
    response = client.post(
        "/api/analytics/conversations", json=body
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_and_create_use_the_exact_server_path(
    client: TestClient, tmp_path: Path
) -> None:
    completed, _appended = _ready_path(tmp_path)

    preview = client.post(
        "/api/analytics/conversations/preview",
        json=_head(completed),
    )
    saved = _save(client, completed)

    assert preview.status_code == 200
    assert preview.json()["exchange_count"] == 1
    assert preview.json()["text_only_count"] == 1
    assert saved["analytics_url"].endswith(saved["snapshot_id"])


def test_list_metadata_turns_rename_and_delete(
    client: TestClient, tmp_path: Path
) -> None:
    completed, _appended = _ready_path(tmp_path)
    saved = _save(client, completed)
    snapshot_id = saved["snapshot_id"]

    listed = client.get("/api/analytics/conversations")
    metadata = client.get(
        f"/api/analytics/conversations/{snapshot_id}/metadata"
    )
    turns = client.get(
        f"/api/analytics/conversations/{snapshot_id}/turns"
    )
    renamed = client.patch(
        f"/api/analytics/conversations/{snapshot_id}",
        json={
            "title": "Renamed",
            "expected_title_revision": 1,
        },
    )
    deleted = client.delete(
        f"/api/analytics/conversations/{snapshot_id}"
    )

    assert listed.json()[0]["snapshot_id"] == snapshot_id
    assert metadata.json()["title"] == "Token investigation"
    assert len(turns.json()["turns"]) == 2
    assert renamed.json()["title_revision"] == 2
    assert deleted.json() == {"deleted": snapshot_id}
    assert client.get(
        f"/api/analytics/conversations/{snapshot_id}/metadata"
    ).status_code == 404


def test_lost_create_response_replays_one_snapshot(
    client: TestClient, tmp_path: Path
) -> None:
    completed, _appended = _ready_path(tmp_path)
    operation_id = uuid4().hex

    first = _save(client, completed, operation_id=operation_id)
    replay = _save(client, completed, operation_id=operation_id)

    assert first["snapshot_id"] == replay["snapshot_id"]
    assert replay["replayed"] is True
    assert len(client.get("/api/analytics/conversations").json()) == 1


def test_stale_head_and_title_are_conflicts(
    client: TestClient, tmp_path: Path
) -> None:
    completed, _appended = _ready_path(tmp_path)
    stale = _head(completed)
    stale["branch_revision"] = int(stale["branch_revision"]) - 1

    preview = client.post(
        "/api/analytics/conversations/preview", json=stale
    )
    saved = _save(client, completed)
    renamed = client.patch(
        (
            "/api/analytics/conversations/"
            + str(saved["snapshot_id"])
        ),
        json={
            "title": "Stale",
            "expected_title_revision": 9,
        },
    )

    assert preview.status_code == 409
    assert renamed.status_code == 409


def test_pinned_run_uses_snapshot_local_endpoints(
    client: TestClient, tmp_path: Path
) -> None:
    completed, appended = _ready_path(tmp_path, linked=True)
    saved = _save(client, completed)
    base = (
        "/api/analytics/conversations/"
        + str(saved["snapshot_id"])
        + "/turns/"
        + appended.assistant_turn.turn_id
        + "/run"
    )

    metadata = client.get(base + "/metadata")
    metrics = client.get(base + "/metrics")
    frames = client.get(base + "/frames")

    assert metadata.status_code == 200, metadata.text
    assert metrics.status_code == 200, metrics.text
    assert frames.status_code == 200, frames.text
    assert metadata.json()["revision"] == 1
    assert metrics.json()["run_id"] == appended.assistant_turn.turn_id


def test_malformed_snapshot_requests_are_bounded(
    client: TestClient, tmp_path: Path
) -> None:
    completed, _appended = _ready_path(tmp_path)
    malformed = dict(_head(completed), operation_id="short", title="")

    response = client.post(
        "/api/analytics/conversations", json=malformed
    )
    traversing = client.get(
        "/api/analytics/conversations/../escape/metadata"
    )

    assert response.status_code == 422
    assert traversing.status_code in (404, 405)
