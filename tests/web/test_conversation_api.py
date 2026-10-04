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
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Dict

import pytest
from starlette.testclient import TestClient

from src.web import conversation_api, conversation_store, server


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
}


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
            "expected_revision": conversation["revision"],
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
            "expected_revision": conversation["revision"],
            "text": text,
            "partial": partial,
            "context_pack": {
                "included_turn_ids": ["00000001"],
                "omitted_turn_count": 0,
            },
            "metadata": {"worker": "terminal"},
        },
    )
    assert response.status_code == 200, response.text
    body: Dict[str, object] = response.json()
    return body


def _make_run(root: Path, run_id: str, revision: int) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"revision": revision, "backend": "llada"}),
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
    assert "turns" not in listed["conversations"][0]
    assert "text" not in listed["conversations"][0]
    assert "turns" not in metadata["conversation"]


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
    assert updated["pending_assistant_id"] == "00000002"
    assert user["role"] == "user"
    assert user["text"] == "Question"
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
    assert [turn["role"] for turn in page["turns"]] == [
        "user",
        "assistant",
    ]
    assert page["next_before"] is None
    assert page["has_more"] is False
    assert all("versions" not in turn for turn in page["turns"])


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
    _make_run(tmp_path, run_id, 7)
    route = (
        f"/api/conversations/{current['id']}/turns/"
        f"{turn['turn_id']}/run"
    )

    linked_response = client.put(
        route,
        json={
            "expected_revision": current["revision"],
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
        json={"expected_revision": linked_conversation["revision"]},
    )
    assert unlinked_response.status_code == 200
    unlinked: Dict[str, object] = unlinked_response.json()

    assert linked_turn["run_link"] == {
        "run_id": run_id,
        "revision": 7,
    }
    assert _turn_from(unlinked)["run_link"] is None


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
            "expected_revision": conversation["revision"],
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
            "expected_revision": conversation["revision"],
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
            "expected_revision": current["revision"],
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
            "expected_revision": current["revision"],
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
    _make_run(tmp_path, run_id, 4)
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{turn['turn_id']}/run"
        ),
        json={
            "expected_revision": current["revision"],
            "run_id": run_id,
            "run_revision": 3,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "run_revision_conflict"
    assert response.json()["revision"] == 4


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
    _make_run(tmp_path, run_id, 1)
    response = client.put(
        (
            f"/api/conversations/{current['id']}/turns/"
            f"{assistant['turn_id']}/run"
        ),
        json={
            "expected_revision": current["revision"],
            "run_id": run_id,
            "run_revision": 1,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "state_conflict"


# -- conflicts, invalid input, and corrupt storage --


def test_stale_conversation_revision_has_current_revision(
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
            "expected_revision": 2,
            "text": "Stale",
            "partial": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "revision_conflict"
    assert response.json()["revision"] == 3


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
            "expected_revision": current["revision"],
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


def test_boolean_revision_is_not_coerced_to_one(
    client: TestClient,
) -> None:
    conversation = _create(client)
    response = client.post(
        f"/api/conversations/{conversation['id']}/turns",
        json={
            "expected_revision": True,
            "text": "Question",
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
