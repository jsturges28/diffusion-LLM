"""HTTP ownership and responses for durable conversations.

Strategy: import the router in a fresh interpreter, inspect the real
supervisor's route owners, then drive every operation through a
``TestClient`` pointed at a temporary results root. Saved-run folders
are seeded directly because the API must validate links without
changing the legacy run format.

Passing proves ``server.py`` only includes the router, live
dependencies still follow the supervisor root, client faults map to
stable 4xx responses, corrupt storage maps to 500, and list/page
responses stay bounded and lightweight.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from src.web import _conversation_store_core as core_store
from src.web import (
    conversation_api,
    conversation_store,
    run_store,
    server,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

IMPORT_PROBE = """
import sys

import src.web.conversation_api

loaded = set(sys.modules)
forbidden = {"src.web.server", "torch", "transformers"}
print(",".join(sorted(loaded & forbidden)))
"""

CONVERSATION_ROUTES = {
    ("POST", "/api/conversations"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{conversation_id}/metadata"),
    ("GET", "/api/conversations/{conversation_id}/branches"),
    ("DELETE", "/api/conversations/{conversation_id}"),
    ("GET", "/api/conversations/{conversation_id}/turns"),
    ("POST", "/api/conversations/{conversation_id}/turns"),
    (
        "PUT",
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}"
        ),
    ),
    (
        "PUT",
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}/run"
        ),
    ),
    (
        "DELETE",
        (
            "/api/conversations/{conversation_id}/turns/"
            "{assistant_turn_id}/run"
        ),
    ),
    (
        "POST",
        (
            "/api/conversations/{conversation_id}/branches/"
            "edit-user/{user_turn_id}"
        ),
    ),
    (
        "POST",
        (
            "/api/conversations/{conversation_id}/branches/"
            "delete-from-path/{user_turn_id}"
        ),
    ),
    (
        "POST",
        (
            "/api/conversations/{conversation_id}/branches/"
            "retry-assistant/{assistant_turn_id}"
        ),
    ),
}


def _operation_id() -> str:
    return uuid4().hex


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _create(
    client: TestClient,
    *,
    title: str = "Example",
) -> Dict[str, object]:
    response = client.post(
        "/api/conversations", json={"title": title}
    )
    assert response.status_code == 201, response.text
    body: Dict[str, object] = response.json()
    conversation = body["conversation"]
    assert isinstance(conversation, dict)
    return conversation


def _append(
    client: TestClient,
    conversation: Dict[str, object],
    *,
    text: str = "Question",
    model_id: str = "llada",
    input_mode: str = "chat",
) -> Dict[str, object]:
    conversation_id = conversation["id"]
    response = client.post(
        f"/api/conversations/{conversation_id}/turns",
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "text": text,
            "model_id": model_id,
            "input_mode": input_mode,
            "metadata": {"client": "test"},
        },
    )
    assert response.status_code == 201, response.text
    body: Dict[str, object] = response.json()
    return body


def _complete(
    client: TestClient,
    appended: Dict[str, object],
    *,
    text: str = "Answer",
    partial: bool = False,
) -> Dict[str, object]:
    conversation = appended["conversation"]
    assistant = appended["assistant_turn"]
    assert isinstance(conversation, dict)
    assert isinstance(assistant, dict)
    response = client.put(
        (
            f"/api/conversations/{conversation['id']}/turns/"
            f"{assistant['turn_id']}"
        ),
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "text": text,
            "partial": partial,
            "context_pack": {
                "included_turn_ids": [
                    appended["user_turn"]["turn_id"]
                ],
                "omitted_turn_count": 0,
            },
            "metadata": {"worker": "terminal"},
        },
    )
    assert response.status_code == 200, response.text
    body: Dict[str, object] = response.json()
    return body


def _make_run(
    root: Path,
    run_id: str,
    revision: int,
    *,
    conversation: Dict[str, object],
    turn: Dict[str, object],
) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "backend": "llada",
                "conversation_id": conversation["id"],
                "branch_id": conversation["branch_id"],
                "assistant_turn_id": turn["turn_id"],
                "turn_index": turn["index"],
                "assistant_turn_version": turn["version"],
            }
        ),
        encoding="utf-8",
    )


def _conversation_from(
    body: Dict[str, object],
) -> Dict[str, object]:
    conversation = body["conversation"]
    assert isinstance(conversation, dict)
    return conversation


def _turn_from(body: Dict[str, object]) -> Dict[str, object]:
    turn = body["turn"]
    assert isinstance(turn, dict)
    return turn


def _complete_fork(
    client: TestClient,
    conversation: Dict[str, object],
    assistant: Dict[str, object],
    *,
    text: str,
) -> Dict[str, object]:
    response = client.put(
        (
            f"/api/conversations/{conversation['id']}/turns/"
            f"{assistant['turn_id']}"
        ),
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "text": text,
            "partial": False,
        },
    )
    assert response.status_code == 200, response.text
    body: Dict[str, object] = response.json()
    return body


def _retry_fork(
    client: TestClient,
    conversation: Dict[str, object],
    assistant: Dict[str, object],
) -> Dict[str, object]:
    response = client.post(
        (
            f"/api/conversations/{conversation['id']}/branches/"
            f"retry-assistant/{assistant['turn_id']}"
        ),
        json={
            "operation_id": _operation_id(),
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "catalog_revision": conversation["catalog_revision"],
            "model_id": "llada",
            "input_mode": "chat",
        },
    )
    assert response.status_code == 201, response.text
    body: Dict[str, object] = response.json()
    return body


def _two_exchange_path(
    client: TestClient,
) -> tuple[
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
]:
    created = _create(client)
    first = _append(client, created, text="Question 1")
    first_done = _complete(client, first, text="Answer 1")
    second = _append(
        client,
        _conversation_from(first_done),
        text="Question 2",
    )
    second_done = _complete(client, second, text="Answer 2")
    first_user = first["user_turn"]
    second_user = second["user_turn"]
    assert isinstance(first_user, dict)
    assert isinstance(second_user, dict)
    return (
        _conversation_from(second_done),
        first_user,
        second_user,
        _turn_from(second_done),
    )


# -- module and route ownership --


def test_api_imports_without_server_or_model_libraries() -> None:
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_conversation_routes_are_owned_by_the_api_module() -> None:
    actual: Dict[tuple[str, str], str] = {}
    for route in server.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/conversations"):
            continue
        endpoint = getattr(route, "endpoint", None)
        assert endpoint is not None, path
        methods = getattr(route, "methods", set())
        for method in methods:
            actual[(method, path)] = endpoint.__module__

    assert set(actual) == CONVERSATION_ROUTES
    assert set(actual.values()) == {conversation_api.__name__}


def test_api_dependencies_are_frozen(tmp_path: Path) -> None:
    dependencies = conversation_api.ConversationApiDependencies(
        results_dir=lambda: tmp_path
    )

    with pytest.raises(FrozenInstanceError):
        dependencies.results_dir = lambda: tmp_path  # type: ignore[misc]


def test_server_dependency_remains_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    client = TestClient(server.app)

    monkeypatch.setattr(server, "RESULTS_DIR", first)
    first_id = _create(client, title="First")["id"]
    monkeypatch.setattr(server, "RESULTS_DIR", second)
    second_id = _create(client, title="Second")["id"]

    listed = client.get("/api/conversations").json()
    assert [item["id"] for item in listed["conversations"]] == [
        second_id
    ]
    conversations = second / conversation_store.CONVERSATIONS_DIR_NAME
    assert not (conversations / str(first_id)).exists()


# -- successful lifecycle --


def test_create_list_and_metadata_are_lightweight(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    conversation_id = current["id"]

    listed = client.get("/api/conversations").json()
    metadata = client.get(
        f"/api/conversations/{conversation_id}/metadata"
    ).json()

    assert listed["conversations"] == [current]
    assert metadata["conversation"] == current
    assert current["schema_version"] == 2
    assert current["branch_revision"] == current["revision"]
    assert current["branch_id"] == current["default_branch_id"]
    assert current["catalog_revision"] == 1
    assert "turns" not in listed["conversations"][0]
    assert "text" not in listed["conversations"][0]
    assert "turns" not in metadata["conversation"]


def test_schema_v1_metadata_and_pages_remain_readable(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation_id = "a" * 32
    conversation_dir = (
        tmp_path
        / conversation_store.CONVERSATIONS_DIR_NAME
        / conversation_id
    )
    (conversation_dir / conversation_store.TURNS_DIR_NAME).mkdir(
        parents=True
    )
    timestamp = "2026-01-01T00:00:00.000Z"
    core_store.write_legacy_manifest(
        conversation_dir,
        conversation_store.ConversationManifest(
            id=conversation_id,
            title="Legacy",
            revision=1,
            created_at=timestamp,
            updated_at=timestamp,
            turn_count=0,
            tail_role=None,
            tail_turn_id=None,
            tail_version=None,
            pending_assistant_id=None,
        ),
    )

    metadata = client.get(
        f"/api/conversations/{conversation_id}/metadata"
    ).json()["conversation"]
    page = client.get(
        f"/api/conversations/{conversation_id}/turns"
    ).json()

    assert metadata["schema_version"] == 1
    assert metadata["branch_id"] == "b_" + conversation_id
    assert metadata["catalog_revision"] == 0
    assert page["schema_version"] == 1
    assert page["branch_id"] == metadata["branch_id"]
    assert page["turns"] == []


def test_lost_v1_fork_http_replay_needs_no_v2_branch_id(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The legacy request replays after its first call upgrades."""
    conversation_id = "b" * 32
    conversation_dir = (
        tmp_path
        / conversation_store.CONVERSATIONS_DIR_NAME
        / conversation_id
    )
    (conversation_dir / conversation_store.TURNS_DIR_NAME).mkdir(
        parents=True
    )
    timestamp = "2026-01-01T00:00:00.000Z"
    core_store.write_legacy_manifest(
        conversation_dir,
        conversation_store.ConversationManifest(
            id=conversation_id,
            title="Legacy replay",
            revision=1,
            created_at=timestamp,
            updated_at=timestamp,
            turn_count=0,
            tail_role=None,
            tail_turn_id=None,
            tail_version=None,
            pending_assistant_id=None,
        ),
    )
    appended = conversation_store.append_user(
        tmp_path,
        conversation_id,
        expected_revision=1,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = conversation_store.update_assistant(
        tmp_path,
        conversation_id,
        appended.assistant_turn.turn_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    route = (
        f"/api/conversations/{conversation_id}/branches/"
        f"retry-assistant/{appended.assistant_turn.turn_id}"
    )
    body = {
        "operation_id": "1" * 32,
        "branch_revision": completed.manifest.revision,
        "catalog_revision": 0,
        "model_id": "llada",
        "input_mode": "chat",
    }

    first = client.post(route, json=body)
    replayed = client.post(route, json=body)

    assert first.status_code == 201, first.text
    assert replayed.status_code == 201, replayed.text
    assert replayed.json() == first.json()


def test_append_returns_user_and_reserved_assistant(
    client: TestClient,
) -> None:
    conversation = _create(client)
    body = _append(client, conversation)
    updated = _conversation_from(body)
    user = body["user_turn"]
    assistant = body["assistant_turn"]
    assert isinstance(user, dict)
    assert isinstance(assistant, dict)

    assert updated["revision"] == 2
    assert updated["branch_revision"] == 2
    assert updated["pending_assistant_id"] == assistant["turn_id"]
    assert updated["branch_id"] == assistant["branch_id"]
    assert user["role"] == "user"
    assert user["text"] == "Question"
    assert user["branch_id"] == updated["branch_id"]
    assert assistant["role"] == "assistant"
    assert assistant["partial"] is True
    assert assistant["model_id"] == "llada"


def test_complete_preserves_partial_and_context_fields(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    body = _complete(
        client,
        appended,
        text="Interrupted",
        partial=True,
    )
    updated = _conversation_from(body)
    turn = _turn_from(body)

    assert updated["revision"] == 3
    assert updated["pending_assistant_id"] is None
    assert turn["text"] == "Interrupted"
    assert turn["partial"] is True
    assert turn["context_pack"]["omitted_turn_count"] == 0
    assert turn["metadata"] == {"worker": "terminal"}


def test_turn_page_is_chronological_and_lightweight(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)

    page = client.get(
        f"/api/conversations/{current['id']}/turns"
    ).json()

    assert page["conversation_id"] == current["id"]
    assert page["revision"] == current["revision"]
    assert page["branch_revision"] == current["branch_revision"]
    assert page["branch_id"] == current["branch_id"]
    assert page["catalog_revision"] == current["catalog_revision"]
    assert page["default_branch_id"] == current["default_branch_id"]
    assert [turn["role"] for turn in page["turns"]] == [
        "user",
        "assistant",
    ]
    assert all(
        turn["branch_id"] == current["branch_id"]
        for turn in page["turns"]
    )
    assert page["next_before"] is None
    assert page["has_more"] is False
    assert page["branch_points"] == []
    assert all("versions" not in turn for turn in page["turns"])


def test_retry_fork_lists_and_navigates_both_branches(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    assistant = _turn_from(completed)
    route = (
        f"/api/conversations/{current['id']}/branches/"
        f"retry-assistant/{assistant['turn_id']}"
    )

    response = client.post(
        route,
        json={
            "operation_id": _operation_id(),
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "catalog_revision": current["catalog_revision"],
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 201, response.text
    forked: Dict[str, object] = response.json()
    alternate = _conversation_from(forked)
    branches = client.get(
        f"/api/conversations/{current['id']}/branches"
    ).json()
    page = client.get(
        f"/api/conversations/{current['id']}/turns",
        params={"branch_id": alternate["branch_id"]},
    ).json()
    original = client.get(
        f"/api/conversations/{current['id']}/metadata",
        params={"branch_id": current["branch_id"]},
    ).json()["conversation"]

    assert branches["catalog_revision"] == 2
    assert len(branches["branches"]) == 2
    assert alternate["branch_id"] != current["branch_id"]
    assert alternate["default_branch_id"] == alternate["branch_id"]
    assert original["branch_id"] == current["branch_id"]
    assert page["branch_id"] == alternate["branch_id"]
    assert page["branch_points"] == [
        {
            "turn_index": 2,
            "source_branch_id": current["branch_id"],
            "selected_branch_id": alternate["branch_id"],
            "branch_ids": [
                current["branch_id"],
                alternate["branch_id"],
            ],
            "deleted_branch_ids": [],
        }
    ]


def test_retry_of_retry_returns_one_coalesced_control(
    client: TestClient,
) -> None:
    """The API emits one selector for one logical answer slot."""
    root, _first_user, _second_user, assistant = (
        _two_exchange_path(client)
    )
    first = _retry_fork(client, root, assistant)
    first_conversation = _conversation_from(first)
    first_assistant = first["assistant_turn"]
    assert isinstance(first_assistant, dict)
    first_done = _complete_fork(
        client,
        first_conversation,
        first_assistant,
        text="First retry",
    )
    second = _retry_fork(
        client,
        _conversation_from(first_done),
        first_assistant,
    )
    selected = _conversation_from(second)

    page = client.get(
        f"/api/conversations/{root['id']}/turns",
        params={"branch_id": selected["branch_id"]},
    ).json()

    assert len(page["branch_points"]) == 1
    point = page["branch_points"][0]
    assert point["turn_index"] == 4
    assert point["selected_branch_id"] == selected["branch_id"]
    assert point["branch_ids"] == [
        root["branch_id"],
        first_conversation["branch_id"],
        selected["branch_id"],
    ]


def test_early_descendant_edit_omits_stale_ancestor_control(
    client: TestClient,
) -> None:
    """An edit before an ancestor fork removes that fork from view."""
    root, first_user, _second_user, assistant = (
        _two_exchange_path(client)
    )
    retry = _retry_fork(client, root, assistant)
    retry_conversation = _conversation_from(retry)
    retry_assistant = retry["assistant_turn"]
    assert isinstance(retry_assistant, dict)
    retry_done = _complete_fork(
        client,
        retry_conversation,
        retry_assistant,
        text="Alternate answer 2",
    )
    selected_retry = _conversation_from(retry_done)
    response = client.post(
        (
            f"/api/conversations/{root['id']}/branches/"
            f"edit-user/{first_user['turn_id']}"
        ),
        json={
            "operation_id": _operation_id(),
            "branch_id": selected_retry["branch_id"],
            "branch_revision": selected_retry["branch_revision"],
            "catalog_revision": selected_retry["catalog_revision"],
            "text": "Edited question 1",
            "model_id": "llada",
            "input_mode": "chat",
        },
    )
    assert response.status_code == 201, response.text
    edited: Dict[str, object] = response.json()
    selected = _conversation_from(edited)

    page = client.get(
        f"/api/conversations/{root['id']}/turns",
        params={"branch_id": selected["branch_id"]},
    ).json()

    assert len(page["branch_points"]) == 1
    point = page["branch_points"][0]
    assert point["turn_index"] == 1
    assert point["source_branch_id"] == selected_retry["branch_id"]
    assert root["branch_id"] not in point["branch_ids"]


def test_nested_delete_returns_only_its_effective_marker(
    client: TestClient,
) -> None:
    """A same-slot delete stays marked after alternatives merge."""
    created = _create(client)
    appended = _append(client, created)
    completed = _complete(client, appended)
    root = _conversation_from(completed)
    user = appended["user_turn"]
    assert isinstance(user, dict)
    edit_response = client.post(
        (
            f"/api/conversations/{root['id']}/branches/"
            f"edit-user/{user['turn_id']}"
        ),
        json={
            "operation_id": _operation_id(),
            "branch_id": root["branch_id"],
            "branch_revision": root["branch_revision"],
            "catalog_revision": root["catalog_revision"],
            "text": "Edited question",
            "model_id": "llada",
            "input_mode": "chat",
        },
    )
    assert edit_response.status_code == 201, edit_response.text
    edited: Dict[str, object] = edit_response.json()
    edited_conversation = _conversation_from(edited)
    edited_user = edited["user_turn"]
    edited_assistant = edited["assistant_turn"]
    assert isinstance(edited_user, dict)
    assert isinstance(edited_assistant, dict)
    edited_done = _complete_fork(
        client,
        edited_conversation,
        edited_assistant,
        text="Edited answer",
    )
    selected_edit = _conversation_from(edited_done)
    response = client.post(
        (
            f"/api/conversations/{root['id']}/branches/"
            f"delete-from-path/{edited_user['turn_id']}"
        ),
        json={
            "operation_id": _operation_id(),
            "branch_id": selected_edit["branch_id"],
            "branch_revision": selected_edit["branch_revision"],
            "catalog_revision": selected_edit["catalog_revision"],
        },
    )
    assert response.status_code == 201, response.text
    deleted: Dict[str, object] = response.json()
    selected = _conversation_from(deleted)

    page = client.get(
        f"/api/conversations/{root['id']}/turns",
        params={"branch_id": selected["branch_id"]},
    ).json()

    assert len(page["branch_points"]) == 1
    point = page["branch_points"][0]
    assert point["turn_index"] == 1
    assert point["selected_branch_id"] == selected["branch_id"]
    assert point["branch_ids"] == [
        root["branch_id"],
        selected_edit["branch_id"],
        selected["branch_id"],
    ]
    assert point["deleted_branch_ids"] == [selected["branch_id"]]


def test_edit_user_fork_returns_replacement_pair(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    user = appended["user_turn"]
    assert isinstance(user, dict)
    route = (
        f"/api/conversations/{current['id']}/branches/"
        f"edit-user/{user['turn_id']}"
    )

    response = client.post(
        route,
        json={
            "operation_id": _operation_id(),
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "catalog_revision": current["catalog_revision"],
            "text": "Edited question",
            "model_id": "llada",
            "input_mode": "chat",
            "metadata": {"source": "edit"},
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["user_turn"]["text"] == "Edited question"
    assert body["assistant_turn"]["partial"] is True
    assert (
        body["user_turn"]["branch_id"]
        == body["branch"]["branch_id"]
    )
    assert body["catalog"]["catalog_revision"] == 2


def test_delete_from_path_returns_empty_branch_marker(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    user = appended["user_turn"]
    assert isinstance(user, dict)
    route = (
        f"/api/conversations/{current['id']}/branches/"
        f"delete-from-path/{user['turn_id']}"
    )

    response = client.post(
        route,
        json={
            "operation_id": _operation_id(),
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "catalog_revision": current["catalog_revision"],
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    branch_id = body["branch"]["branch_id"]
    page = client.get(
        f"/api/conversations/{current['id']}/turns",
        params={"branch_id": branch_id},
    ).json()
    assert body["removed_turn_count"] == 2
    assert page["turns"] == []
    assert page["branch_points"][0]["turn_index"] == 1
    assert page["branch_points"][0]["selected_branch_id"] == branch_id
    assert page["branch_points"][0]["deleted_branch_ids"] == [
        branch_id
    ]


def test_lost_edit_post_replays_same_created_payload(
    client: TestClient,
) -> None:
    """Repeating one edit POST returns its original 201 response."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    page = client.get(
        f"/api/conversations/{conversation['id']}/turns",
        params={"branch_id": conversation["branch_id"]},
    ).json()
    user = page["turns"][0]
    route = (
        f"/api/conversations/{conversation['id']}/branches/"
        f"edit-user/{user['turn_id']}"
    )
    request = {
        "operation_id": "1" * 32,
        "branch_id": conversation["branch_id"],
        "branch_revision": conversation["branch_revision"],
        "catalog_revision": conversation["catalog_revision"],
        "text": "Edited question",
        "model_id": "llada",
        "input_mode": "chat",
        "metadata": {"source": "replay"},
    }

    first = client.post(route, json=request)
    replayed = client.post(route, json=request)
    branches = client.get(
        f"/api/conversations/{conversation['id']}/branches"
    ).json()

    assert first.status_code == 201
    assert replayed.status_code == 201
    assert replayed.json() == first.json()
    assert len(branches["branch_ids"]) == 2


def test_lost_delete_post_replays_same_created_payload(
    client: TestClient,
) -> None:
    """Repeating one delete POST returns its original 201 response."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    page = client.get(
        f"/api/conversations/{conversation['id']}/turns",
        params={"branch_id": conversation["branch_id"]},
    ).json()
    user = page["turns"][0]
    route = (
        f"/api/conversations/{conversation['id']}/branches/"
        f"delete-from-path/{user['turn_id']}"
    )
    request = {
        "operation_id": "2" * 32,
        "branch_id": conversation["branch_id"],
        "branch_revision": conversation["branch_revision"],
        "catalog_revision": conversation["catalog_revision"],
    }

    first = client.post(route, json=request)
    replayed = client.post(route, json=request)
    branches = client.get(
        f"/api/conversations/{conversation['id']}/branches"
    ).json()

    assert first.status_code == 201
    assert replayed.status_code == 201
    assert replayed.json() == first.json()
    assert len(branches["branch_ids"]) == 2


def test_lost_retry_post_replays_same_created_payload(
    client: TestClient,
) -> None:
    """Repeating one retry POST returns its original 201 response."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    assistant = _turn_from(completed)
    route = (
        f"/api/conversations/{conversation['id']}/branches/"
        f"retry-assistant/{assistant['turn_id']}"
    )
    request = {
        "operation_id": "3" * 32,
        "branch_id": conversation["branch_id"],
        "branch_revision": conversation["branch_revision"],
        "catalog_revision": conversation["catalog_revision"],
        "model_id": "llada",
        "input_mode": "chat",
    }

    first = client.post(route, json=request)
    replayed = client.post(route, json=request)
    branches = client.get(
        f"/api/conversations/{conversation['id']}/branches"
    ).json()

    assert first.status_code == 201
    assert replayed.status_code == 201
    assert replayed.json() == first.json()
    assert len(branches["branch_ids"]) == 2


def test_changed_payload_operation_collision_is_a_conflict(
    client: TestClient,
) -> None:
    """A reused id with changed edit text maps to a stable 409."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    page = client.get(
        f"/api/conversations/{conversation['id']}/turns",
        params={"branch_id": conversation["branch_id"]},
    ).json()
    user = page["turns"][0]
    route = (
        f"/api/conversations/{conversation['id']}/branches/"
        f"edit-user/{user['turn_id']}"
    )
    request = {
        "operation_id": "4" * 32,
        "branch_id": conversation["branch_id"],
        "branch_revision": conversation["branch_revision"],
        "catalog_revision": conversation["catalog_revision"],
        "text": "First text",
        "model_id": "llada",
        "input_mode": "chat",
    }
    first = client.post(route, json=request)
    request["text"] = "Changed text"

    collided = client.post(route, json=request)

    assert first.status_code == 201
    assert collided.status_code == 409
    assert collided.json()["reason"] == "operation_id_conflict"
    assert collided.json()["operation_id"] == "4" * 32


def test_link_and_unlink_a_current_saved_run(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    run_id = "2026-01-01_00-00-00_llada"
    _make_run(
        tmp_path,
        run_id,
        7,
        conversation=current,
        turn=turn,
    )
    route = (
        f"/api/conversations/{current['id']}/turns/"
        f"{turn['turn_id']}/run"
    )

    linked_response = client.put(
        route,
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": turn["version"],
            "run_id": run_id,
            "run_revision": 7,
        },
    )
    assert linked_response.status_code == 200
    linked: Dict[str, object] = linked_response.json()
    linked_turn = _turn_from(linked)
    linked_conversation = _conversation_from(linked)

    unlinked_response = client.request(
        "DELETE",
        route,
        json={
            "branch_id": linked_conversation["branch_id"],
            "branch_revision": (
                linked_conversation["branch_revision"]
            ),
        },
    )
    assert unlinked_response.status_code == 200
    unlinked: Dict[str, object] = unlinked_response.json()

    assert linked_turn["run_link"] == {
        "run_id": run_id,
        "revision": 7,
    }
    assert linked_turn["version"] == int(turn["version"]) + 1
    assert linked_conversation["tail_version"] == (
        linked_turn["version"]
    )
    assert _turn_from(unlinked)["run_link"] is None


def test_run_replacement_finishes_before_link_validation(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement holding runs.lock makes a stale link conflict."""
    completed = _complete(client, _append(client, _create(client)))
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    run_id = "2026-01-01_replace-race"
    _make_run(
        tmp_path,
        run_id,
        1,
        conversation=current,
        turn=turn,
    )
    metadata = json.loads(
        (tmp_path / run_id / run_store.METADATA_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(metadata, dict)
    bundle = run_store.RunBundle(
        metadata=metadata,
        final_text="replacement",
        frames=["replacement"],
    )
    replacement_entered = threading.Event()
    release_replacement = threading.Event()
    link_helper_entered = threading.Event()
    validation_entered = threading.Event()
    real_publish = run_store._publish_replacement
    real_read_metadata = run_store.read_metadata
    real_set_link = conversation_api._set_validated_run_link

    def paused_publish(
        root: Path,
        replacement: run_store.RunBundle,
        selected_run_id: str,
        expected_revision: Optional[int],
        run_token: Optional[str],
    ) -> tuple[str, int]:
        replacement_entered.set()
        assert release_replacement.wait(timeout=5)
        return real_publish(
            root,
            replacement,
            selected_run_id,
            expected_revision,
            run_token,
        )

    def observed_read_metadata(
        root: Path,
        selected_run_id: str,
    ) -> Dict[str, object]:
        validation_entered.set()
        return real_read_metadata(root, selected_run_id)

    def observed_set_link(
        results_dir: Path,
        *,
        conversation_id: str,
        branch_id: str,
        assistant_turn_id: str,
        branch_revision: int,
        run_id: str,
        run_revision: int,
        assistant_turn_index: int,
        assistant_turn_version: int,
    ) -> conversation_store.ConversationMutation:
        link_helper_entered.set()
        return real_set_link(
            results_dir,
            conversation_id=conversation_id,
            branch_id=branch_id,
            assistant_turn_id=assistant_turn_id,
            branch_revision=branch_revision,
            run_id=run_id,
            run_revision=run_revision,
            assistant_turn_index=assistant_turn_index,
            assistant_turn_version=assistant_turn_version,
        )

    monkeypatch.setattr(
        run_store, "_publish_replacement", paused_publish
    )
    monkeypatch.setattr(
        run_store, "read_metadata", observed_read_metadata
    )
    monkeypatch.setattr(
        conversation_api, "_set_validated_run_link", observed_set_link
    )
    route = (
        f"/api/conversations/{current['id']}/turns/"
        f"{turn['turn_id']}/run"
    )
    body = {
        "branch_id": current["branch_id"],
        "branch_revision": current["branch_revision"],
        "assistant_turn_index": turn["index"],
        "assistant_turn_version": turn["version"],
        "run_id": run_id,
        "run_revision": 1,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        replacement = executor.submit(
            run_store.save,
            tmp_path,
            bundle,
            model_id="llada",
            run_id=run_id,
            expected_revision=1,
        )
        try:
            assert replacement_entered.wait(timeout=5)
            linking = executor.submit(client.put, route, json=body)
            assert link_helper_entered.wait(timeout=5)
            assert not validation_entered.is_set()
        finally:
            release_replacement.set()
        assert replacement.result(timeout=5) == (run_id, 2)
        response = linking.result(timeout=5)

    page = conversation_store.get_turns(
        tmp_path,
        str(current["id"]),
        branch_id=str(current["branch_id"]),
    )
    assert response.status_code == 409
    assert response.json()["reason"] == "run_revision_conflict"
    assert response.json()["revision"] == 2
    assert page.turns[-1].version == turn["version"]
    assert page.turns[-1].run_link is None


def test_run_deletion_finishes_before_link_validation(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deletion holding runs.lock makes the waiting link see 404."""
    completed = _complete(client, _append(client, _create(client)))
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    run_id = "2026-01-01_delete-race"
    _make_run(
        tmp_path,
        run_id,
        1,
        conversation=current,
        turn=turn,
    )
    deletion_entered = threading.Event()
    release_deletion = threading.Event()
    link_helper_entered = threading.Event()
    validation_entered = threading.Event()
    deletion_thread_id: Optional[int] = None
    real_resolve = run_store.resolve_run_dir
    real_read_metadata = run_store.read_metadata
    real_set_link = conversation_api._set_validated_run_link

    def paused_resolve(root: Path, selected_run_id: str) -> Path:
        if threading.get_ident() == deletion_thread_id:
            deletion_entered.set()
            assert release_deletion.wait(timeout=5)
        return real_resolve(root, selected_run_id)

    def observed_read_metadata(
        root: Path,
        selected_run_id: str,
    ) -> Dict[str, object]:
        validation_entered.set()
        return real_read_metadata(root, selected_run_id)

    def delete_run() -> None:
        nonlocal deletion_thread_id
        deletion_thread_id = threading.get_ident()
        run_store.delete(tmp_path, run_id)

    def observed_set_link(
        results_dir: Path,
        *,
        conversation_id: str,
        branch_id: str,
        assistant_turn_id: str,
        branch_revision: int,
        run_id: str,
        run_revision: int,
        assistant_turn_index: int,
        assistant_turn_version: int,
    ) -> conversation_store.ConversationMutation:
        link_helper_entered.set()
        return real_set_link(
            results_dir,
            conversation_id=conversation_id,
            branch_id=branch_id,
            assistant_turn_id=assistant_turn_id,
            branch_revision=branch_revision,
            run_id=run_id,
            run_revision=run_revision,
            assistant_turn_index=assistant_turn_index,
            assistant_turn_version=assistant_turn_version,
        )

    monkeypatch.setattr(run_store, "resolve_run_dir", paused_resolve)
    monkeypatch.setattr(
        run_store, "read_metadata", observed_read_metadata
    )
    monkeypatch.setattr(
        conversation_api, "_set_validated_run_link", observed_set_link
    )
    route = (
        f"/api/conversations/{current['id']}/turns/"
        f"{turn['turn_id']}/run"
    )
    body = {
        "branch_id": current["branch_id"],
        "branch_revision": current["branch_revision"],
        "assistant_turn_index": turn["index"],
        "assistant_turn_version": turn["version"],
        "run_id": run_id,
        "run_revision": 1,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        deletion = executor.submit(delete_run)
        try:
            assert deletion_entered.wait(timeout=5)
            linking = executor.submit(client.put, route, json=body)
            assert link_helper_entered.wait(timeout=5)
            assert not validation_entered.is_set()
        finally:
            release_deletion.set()
        deletion.result(timeout=5)
        response = linking.result(timeout=5)

    page = conversation_store.get_turns(
        tmp_path,
        str(current["id"]),
        branch_id=str(current["branch_id"]),
    )
    assert response.status_code == 404
    assert response.json()["reason"] == "not_found"
    assert page.turns[-1].version == turn["version"]
    assert page.turns[-1].run_link is None


@pytest.mark.parametrize(
    "missing",
    ["assistant_turn_index", "assistant_turn_version"],
)
def test_link_requires_the_exact_pre_link_turn_identity(
    client: TestClient,
    missing: str,
) -> None:
    conversation = _create(client)
    completed = _complete(client, _append(client, conversation))
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    body = {
        "branch_id": current["branch_id"],
        "branch_revision": current["branch_revision"],
        "assistant_turn_index": turn["index"],
        "assistant_turn_version": turn["version"],
        "run_id": "unused",
        "run_revision": 1,
    }
    del body[missing]

    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json=body,
    )

    assert response.status_code == 422


def test_delete_removes_conversation(
    client: TestClient,
) -> None:
    conversation = _create(client)
    conversation_id = conversation["id"]

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    missing = client.get(
        f"/api/conversations/{conversation_id}/metadata"
    )

    assert deleted.status_code == 200
    assert deleted.json() == {
        "success": True,
        "conversation_id": conversation_id,
    }
    assert missing.status_code == 404


# -- API-boundary model and run checks --


def test_unknown_model_is_a_bad_request(
    client: TestClient,
) -> None:
    conversation = _create(client)
    response = client.post(
        f"/api/conversations/{conversation['id']}/turns",
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "text": "Question",
            "model_id": "unknown",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_request"


@pytest.mark.parametrize(
    ("model_id", "input_mode"),
    [("llada", "completion"), ("mamba3", "chat")],
)
def test_model_input_mode_mismatch_is_refused(
    client: TestClient,
    model_id: str,
    input_mode: str,
) -> None:
    conversation = _create(client)
    response = client.post(
        f"/api/conversations/{conversation['id']}/turns",
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "text": "Question",
            "model_id": model_id,
            "input_mode": input_mode,
        },
    )

    assert response.status_code == 400
    assert "uses" in response.json()["error"]


def test_missing_run_link_is_not_found(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": turn["version"],
            "run_id": "2026-01-01_missing",
            "run_revision": 1,
        },
    )

    assert response.status_code == 404
    assert response.json()["reason"] == "not_found"


def test_traversing_run_link_is_a_bad_request(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": turn["version"],
            "run_id": "../escape",
            "run_revision": 1,
        },
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_request"


def test_stale_run_revision_is_a_conflict(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    run_id = "2026-01-01_llada"
    _make_run(
        tmp_path,
        run_id,
        4,
        conversation=current,
        turn=turn,
    )
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": turn["version"],
            "run_id": run_id,
            "run_revision": 3,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "run_revision_conflict"
    assert response.json()["revision"] == 4


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("conversation_id", "f" * 32),
        ("branch_id", "b_" + "f" * 32),
        ("assistant_turn_id", "00000004"),
        ("turn_index", 4),
        ("assistant_turn_version", 3),
    ],
)
def test_run_link_refuses_each_unrelated_owner_field(
    client: TestClient,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    """Every saved owner coordinate must match the current tail."""
    conversation = _create(client)
    completed = _complete(client, _append(client, conversation))
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    run_id = f"2026-01-01_{field}"
    _make_run(
        tmp_path,
        run_id,
        1,
        conversation=current,
        turn=turn,
    )
    metadata_path = tmp_path / run_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": turn["version"],
            "run_id": run_id,
            "run_revision": 1,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "run_ownership_conflict"


def test_run_link_refuses_a_stale_tail_version(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Matching stale metadata cannot bypass the live tail version."""
    conversation = _create(client)
    completed = _complete(client, _append(client, conversation))
    current = _conversation_from(completed)
    turn = _turn_from(completed)
    stale_version = int(turn["version"]) - 1
    run_id = "2026-01-01_stale-owner"
    _make_run(
        tmp_path,
        run_id,
        1,
        conversation=current,
        turn=turn,
    )
    metadata_path = tmp_path / run_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["assistant_turn_version"] = stale_version
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": turn["index"],
            "assistant_turn_version": stale_version,
            "run_id": run_id,
            "run_revision": 1,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "state_conflict"


def test_pending_assistant_cannot_link(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    current = appended["conversation"]
    assistant = appended["assistant_turn"]
    assert isinstance(current, dict)
    assert isinstance(assistant, dict)
    run_id = "2026-01-01_llada"
    _make_run(
        tmp_path,
        run_id,
        1,
        conversation=current,
        turn=assistant,
    )
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{assistant['turn_id']}/run"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "assistant_turn_index": assistant["index"],
            "assistant_turn_version": assistant["version"],
            "run_id": run_id,
            "run_revision": 1,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "state_conflict"


# -- conflicts, invalid input, and corrupt storage --


def test_stale_branch_revision_has_current_revision(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    assistant = appended["assistant_turn"]
    assert isinstance(assistant, dict)
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{assistant['turn_id']}"
        ),
        json={
            "branch_id": current["branch_id"],
            "branch_revision": 2,
            "text": "Stale",
            "partial": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "branch_revision_conflict"
    assert response.json()["branch_revision"] == 3
    assert response.json()["branch_id"] == current["branch_id"]


def test_stale_legacy_write_uses_branch_conflict_shape(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation_id = "a" * 32
    conversation_dir = (
        tmp_path
        / conversation_store.CONVERSATIONS_DIR_NAME
        / conversation_id
    )
    (conversation_dir / conversation_store.TURNS_DIR_NAME).mkdir(
        parents=True
    )
    timestamp = "2026-01-01T00:00:00.000Z"
    core_store.write_legacy_manifest(
        conversation_dir,
        conversation_store.ConversationManifest(
            id=conversation_id,
            title="Legacy",
            revision=1,
            created_at=timestamp,
            updated_at=timestamp,
            turn_count=0,
            tail_role=None,
            tail_turn_id=None,
            tail_version=None,
            pending_assistant_id=None,
        ),
    )
    branch_id = "b_" + conversation_id
    request = {
        "branch_id": branch_id,
        "branch_revision": 1,
        "text": "Question",
        "model_id": "llada",
        "input_mode": "chat",
    }
    route = f"/api/conversations/{conversation_id}/turns"
    created = client.post(route, json=request)
    assert created.status_code == 201, created.text

    stale = client.post(route, json=request)

    assert stale.status_code == 409
    assert stale.json()["reason"] == "branch_revision_conflict"
    assert stale.json()["branch_id"] == branch_id
    assert stale.json()["branch_revision"] == 2


def test_stale_catalog_revision_is_a_distinct_conflict(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    completed = _complete(client, appended)
    current = _conversation_from(completed)
    assistant = _turn_from(completed)
    route = (
        f"/api/conversations/{current['id']}/branches/"
        f"retry-assistant/{assistant['turn_id']}"
    )

    response = client.post(
        route,
        json={
            "operation_id": _operation_id(),
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "catalog_revision": 0,
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "catalog_revision_conflict"
    assert response.json()["catalog_revision"] == 1


def test_unknown_valid_branch_is_not_found(
    client: TestClient,
) -> None:
    conversation = _create(client)

    response = client.get(
        f"/api/conversations/{conversation['id']}/metadata",
        params={"branch_id": "b_" + "f" * 32},
    )

    assert response.status_code == 404
    assert response.json()["reason"] == "branch_not_found"


def test_malformed_branch_is_an_invalid_branch(
    client: TestClient,
) -> None:
    conversation = _create(client)

    response = client.get(
        f"/api/conversations/{conversation['id']}/metadata",
        params={"branch_id": "../branch"},
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_branch"


def test_pending_assistant_blocks_another_append(
    client: TestClient,
) -> None:
    conversation = _create(client)
    appended = _append(client, conversation)
    current = appended["conversation"]
    assert isinstance(current, dict)
    response = client.post(
        f"/api/conversations/{current['id']}/turns",
        json={
            "branch_id": current["branch_id"],
            "branch_revision": current["branch_revision"],
            "text": "Too soon",
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "state_conflict"


def test_invalid_conversation_id_is_a_bad_request(
    client: TestClient,
) -> None:
    response = client.get(f"/api/conversations/{'g' * 32}/metadata")

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_request"


def test_missing_conversation_is_not_found(
    client: TestClient,
) -> None:
    response = client.get(f"/api/conversations/{'a' * 32}/metadata")

    assert response.status_code == 404
    assert response.json()["reason"] == "not_found"


@pytest.mark.parametrize(
    "limit", [0, conversation_store.PAGE_SIZE_MAX + 1]
)
def test_invalid_page_limit_is_a_bad_request(
    client: TestClient,
    limit: int,
) -> None:
    conversation = _create(client)
    response = client.get(
        f"/api/conversations/{conversation['id']}/turns",
        params={"limit": limit},
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_request"


def test_malformed_query_type_uses_fastapi_validation(
    client: TestClient,
) -> None:
    conversation = _create(client)
    response = client.get(
        f"/api/conversations/{conversation['id']}/turns",
        params={"limit": "not-an-integer"},
    )

    assert response.status_code == 422


def test_extra_request_field_is_refused(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/conversations",
        json={"title": "Example", "turns": []},
    )

    assert response.status_code == 422


def test_schema_v2_mutation_requires_explicit_branch_identity(
    client: TestClient,
) -> None:
    conversation = _create(client)
    response = client.post(
        f"/api/conversations/{conversation['id']}/turns",
        json={
            "text": "Question",
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 422


def test_new_schema_v2_fork_requires_explicit_branch_identity(
    client: TestClient,
) -> None:
    """Only an authoritative v1 replay may omit its branch."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    assistant = _turn_from(completed)
    response = client.post(
        (
            f"/api/conversations/{conversation['id']}/branches/"
            f"retry-assistant/{assistant['turn_id']}"
        ),
        json={
            "operation_id": "2" * 32,
            "branch_revision": conversation["branch_revision"],
            "catalog_revision": conversation["catalog_revision"],
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_request"


def test_boolean_revision_is_not_coerced_to_one(
    client: TestClient,
) -> None:
    conversation = _create(client)
    response = client.post(
        f"/api/conversations/{conversation['id']}/turns",
        json={
            "branch_id": conversation["branch_id"],
            "branch_revision": True,
            "text": "Question",
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "operation_id",
    ["A" * 32, "a" * 31, "g" * 32, "../operation"],
)
def test_fork_operation_id_is_strict_lowercase_hex(
    client: TestClient,
    operation_id: str,
) -> None:
    """Pydantic rejects malformed ids before store access."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    assistant = _turn_from(completed)

    response = client.post(
        (
            f"/api/conversations/{conversation['id']}/branches/"
            f"retry-assistant/{assistant['turn_id']}"
        ),
        json={
            "operation_id": operation_id,
            "branch_id": conversation["branch_id"],
            "branch_revision": conversation["branch_revision"],
            "catalog_revision": conversation["catalog_revision"],
            "model_id": "llada",
            "input_mode": "chat",
        },
    )

    assert response.status_code == 422


def test_corrupt_manifest_is_a_store_error(
    client: TestClient,
    tmp_path: Path,
) -> None:
    conversation = _create(client)
    manifest_path = (
        tmp_path
        / conversation_store.CONVERSATIONS_DIR_NAME
        / str(conversation["id"])
        / conversation_store.MANIFEST_NAME
    )
    manifest_path.write_text("{not json", encoding="utf-8")

    response = client.get(
        f"/api/conversations/{conversation['id']}/metadata"
    )

    assert response.status_code == 500
    assert response.json()["reason"] == "store_error"


def test_corrupt_operation_receipt_is_a_store_error(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Receipt corruption maps to 500, not a client conflict."""
    completed = _complete(client, _append(client, _create(client)))
    conversation = _conversation_from(completed)
    assistant = _turn_from(completed)
    operation_id = "5" * 32
    route = (
        f"/api/conversations/{conversation['id']}/branches/"
        f"retry-assistant/{assistant['turn_id']}"
    )
    request = {
        "operation_id": operation_id,
        "branch_id": conversation["branch_id"],
        "branch_revision": conversation["branch_revision"],
        "catalog_revision": conversation["catalog_revision"],
        "model_id": "llada",
        "input_mode": "chat",
    }
    committed = client.post(route, json=request)
    assert committed.status_code == 201
    receipt_path = (
        tmp_path
        / conversation_store.CONVERSATIONS_DIR_NAME
        / str(conversation["id"])
        / conversation_store.OPERATIONS_DIR_NAME
        / f"{operation_id}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["unexpected"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    replayed = client.post(route, json=request)

    assert replayed.status_code == 500
    assert replayed.json()["reason"] == "store_error"
