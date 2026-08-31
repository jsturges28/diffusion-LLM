"""The collections API, and the write path it closed.

Strategy: point the server's results dir at a tmp path, seed a few
real run folders, and drive the endpoints through a TestClient. The
transforms themselves are covered in `test_collection_ops.py`; what is
checked here is that each endpoint applies the right one, answers with
the authoritative list, and that the door the operations replaced is
shut.

The door matters as much as the operations. `DATA-02`'s lost update
was not that the old write path was slow or unlocked, it was that a
client could name the whole next state. Adding operations beside a
generic PUT that still accepts the array would leave the finding open
while looking closed, so the refusal below is part of the fix rather
than tidiness.

Passing proves every gesture round-trips through the store, that a
refusal carries a reason the browser can act on, and that no client
can replace the collections wholesale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

import src.web.server as server

_KEY = "diffusion_collections"
_RUN = "2026-01-01_00-00-00_llada"
_OTHER = "2026-01-02_00-00-00_smollm3"


def _make_run(root: Path, run_id: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"backend": "llada", "prompt": "p"}),
        encoding="utf-8",
    )


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    _make_run(tmp_path, _RUN)
    _make_run(tmp_path, _OTHER)
    return TestClient(server.app)


def _collections(response) -> List[Dict[str, Any]]:
    body = response.json()
    assert body["success"] is True, body
    return body["collections"]


def _stored(results_dir: Path) -> List[Dict[str, Any]]:
    """Read the file directly, so a response cannot pass by lying."""
    raw = json.loads(
        (results_dir / "ui_state.json").read_text(encoding="utf-8")
    )
    return json.loads(raw[_KEY])


# -- each gesture --


def test_a_new_collection_comes_back_in_the_list(
    client: TestClient,
) -> None:
    made = _collections(
        client.post("/api/collections", json={"name": "Papers"})
    )

    assert [c["id"] for c in made] == ["papers"]


def test_a_gesture_reaches_the_file(
    client: TestClient, tmp_path: Path
) -> None:
    client.post("/api/collections", json={"name": "Papers"})

    assert [c["id"] for c in _stored(tmp_path)] == ["papers"]


def test_a_run_can_be_filed_and_unfiled(
    client: TestClient,
) -> None:
    client.post("/api/collections", json={"name": "Papers"})

    filed = _collections(
        client.post(
            "/api/collections/papers/runs", json={"run_id": _RUN}
        )
    )
    assert filed[0]["runs"] == [_RUN]

    left = _collections(
        client.delete(f"/api/collections/papers/runs/{_RUN}")
    )
    assert left[0]["runs"] == []


def test_a_rename_answers_with_the_new_name(
    client: TestClient,
) -> None:
    client.post("/api/collections", json={"name": "Papers"})

    renamed = _collections(
        client.post(
            "/api/collections/papers/rename",
            json={"name": "Read Later"},
        )
    )

    assert renamed[0]["name"] == "Read Later"
    assert renamed[0]["id"] == "papers"


def test_a_delete_answers_with_what_is_left(
    client: TestClient,
) -> None:
    client.post("/api/collections", json={"name": "Papers"})
    client.post("/api/collections", json={"name": "Drafts"})

    left = _collections(client.delete("/api/collections/papers"))

    assert [c["id"] for c in left] == ["drafts"]


def test_the_star_files_and_then_clears(
    client: TestClient,
) -> None:
    """One request each way, which is the point: the second half
    touches every collection at once."""
    starred = _collections(
        client.post(
            "/api/collections/favorite", json={"run_id": _RUN}
        )
    )
    assert starred[0]["id"] == "favorites"
    assert starred[0]["runs"] == [_RUN]

    cleared = _collections(
        client.post(
            "/api/collections/favorite", json={"run_id": _RUN}
        )
    )
    assert cleared[0]["runs"] == []


def test_favorite_is_not_read_as_a_collection_id(
    client: TestClient,
) -> None:
    """Route order, pinned. `/favorite` and `/{collection_id}` both
    match one path segment, so declaring them the other way round
    would send the star to the rename handler."""
    response = client.post(
        "/api/collections/favorite", json={"run_id": _RUN}
    )

    assert response.status_code == 200
    assert _collections(response)[0]["id"] == "favorites"


def test_creating_with_a_run_files_it_in_one_request(
    client: TestClient,
) -> None:
    """Naming a collection from a run's own dialog is one gesture.
    Two requests could leave the collection made and empty."""
    made = _collections(
        client.post(
            "/api/collections",
            json={"name": "Papers", "run_id": _RUN},
        )
    )

    assert made[0]["runs"] == [_RUN]


def test_a_failed_pair_creates_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    """The other half of atomicity: if the run is refused, the
    collection must not survive the request."""
    response = client.post(
        "/api/collections",
        json={"name": "Papers", "run_id": "ghost"},
    )

    assert response.status_code == 409
    assert not (tmp_path / "ui_state.json").is_file()


# -- filing several at once --


def test_a_selection_is_filed_in_one_request(
    client: TestClient,
) -> None:
    client.post("/api/collections", json={"name": "Papers"})

    filed = _collections(
        client.post(
            "/api/collections/papers/runs",
            json={"run_ids": [_RUN, _OTHER]},
        )
    )

    assert filed[0]["runs"] == [_RUN, _OTHER]


def test_one_run_still_files_on_its_own(
    client: TestClient,
) -> None:
    """The single-run shape the row's own dialog sends did not
    change when the batch shape was added."""
    client.post("/api/collections", json={"name": "Papers"})

    filed = _collections(
        client.post(
            "/api/collections/papers/runs", json={"run_id": _RUN}
        )
    )

    assert filed[0]["runs"] == [_RUN]


def test_a_bad_id_in_a_batch_files_none_of_it(
    client: TestClient, tmp_path: Path
) -> None:
    """Atomic across the batch, which is the reason it is one
    request: six sequential adds can stop at four."""
    client.post("/api/collections", json={"name": "Papers"})

    response = client.post(
        "/api/collections/papers/runs",
        json={"run_ids": [_RUN, "ghost"]},
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "unknown_run"
    assert _stored(tmp_path)[0]["runs"] == []


def test_a_bulk_star_creates_favorites(
    client: TestClient,
) -> None:
    """The star's target may not exist yet, so the create and the
    file compose inside one lock rather than needing two requests."""
    filed = _collections(
        client.post(
            "/api/collections/favorites/runs",
            json={"run_ids": [_RUN, _OTHER]},
        )
    )

    assert filed[0]["id"] == "favorites"
    assert filed[0]["runs"] == [_RUN, _OTHER]


def test_a_bulk_star_into_favorites_is_idempotent(
    client: TestClient,
) -> None:
    client.post(
        "/api/collections/favorites/runs",
        json={"run_ids": [_RUN, _OTHER]},
    )

    again = _collections(
        client.post(
            "/api/collections/favorites/runs",
            json={"run_ids": [_RUN, _OTHER]},
        )
    )

    assert len(again) == 1
    assert again[0]["runs"] == [_RUN, _OTHER]


def test_a_refused_bulk_star_does_not_leave_favorites(
    client: TestClient, tmp_path: Path
) -> None:
    """Both halves or neither, the same contract create-with-a-run
    holds: a bad id must not leave an empty Favorites behind."""
    response = client.post(
        "/api/collections/favorites/runs",
        json={"run_ids": [_RUN, "ghost"]},
    )

    assert response.status_code == 409
    assert not (tmp_path / "ui_state.json").is_file()


def test_creating_with_a_selection_files_all_of_it(
    client: TestClient,
) -> None:
    made = _collections(
        client.post(
            "/api/collections",
            json={"name": "Papers", "run_ids": [_RUN, _OTHER]},
        )
    )

    assert made[0]["runs"] == [_RUN, _OTHER]


# -- refusals carry a reason --


def test_an_unknown_run_is_refused_with_a_reason(
    client: TestClient,
) -> None:
    client.post("/api/collections", json={"name": "Papers"})

    response = client.post(
        "/api/collections/papers/runs", json={"run_id": "ghost"}
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "unknown_run"


def test_an_unknown_collection_is_refused(
    client: TestClient,
) -> None:
    response = client.delete("/api/collections/ghost")

    assert response.status_code == 409
    assert response.json()["reason"] == "unknown_collection"


def test_an_oversized_name_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/collections", json={"name": "x" * 41}
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "invalid_name"


# -- the door that closed --


def test_the_collections_key_refuses_a_whole_value(
    client: TestClient,
) -> None:
    """The finding's lost update was a client naming the next state.
    Operations beside a PUT that still took the array would leave
    that possible while looking fixed."""
    response = client.put(
        f"/api/ui-state/{_KEY}", json={"value": "[]"}
    )

    assert response.status_code == 409
    assert response.json()["reason"] == "use_collection_operations"


def test_a_refused_write_changes_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    client.post(
        "/api/collections",
        json={"name": "Papers", "run_id": _RUN},
    )

    client.put(f"/api/ui-state/{_KEY}", json={"value": "[]"})

    assert _stored(tmp_path)[0]["runs"] == [_RUN]


def test_every_other_key_still_writes(client: TestClient) -> None:
    """Negative space: the refusal is for one key, not a seizure of
    the whole ui-state endpoint."""
    response = client.put(
        "/api/ui-state/diffusion_generate_teased",
        json={"value": "1"},
    )

    assert response.status_code == 200


# -- reading back --


def test_the_list_endpoint_prunes_deleted_runs(
    client: TestClient, tmp_path: Path
) -> None:
    """The same reconcile the hydrate does, on its own, so a window
    can resync without a page load."""
    client.post(
        "/api/collections",
        json={"name": "Papers", "run_id": _RUN},
    )
    for path in (tmp_path / _RUN).iterdir():
        path.unlink()
    (tmp_path / _RUN).rmdir()

    listed = _collections(client.get("/api/collections"))

    assert listed[0]["runs"] == []


def test_an_absent_file_lists_nothing(client: TestClient) -> None:
    assert _collections(client.get("/api/collections")) == []
