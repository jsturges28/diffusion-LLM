"""Tests for collection storage and its reconciliation on hydrate.

Strategy: collections live in ``ui_state.json`` under
``diffusion_collections``. These tests drive the store directly for
the key's contract (accepted, and bounded), and then point the
server's results dir at a tmp path to drive ``GET /api/ui-state`` for
the pruning, seeding collections that reference one real run folder
and one deleted.

This file predates the collections API and covers the storage half:
the key exists, it round-trips, it is size-bounded, and a deleted run
leaves no id behind on hydrate. Since ``DATA-02`` the server also owns
the *shape*, through the operations in ``src/web/collections.py``, and
the ui-state key is no longer writable by a client at all. Those live
in ``test_collection_ops.py``, ``test_collection_endpoints.py`` and
``test_collection_races.py``. The hydrate path tested here stayed, and
matters more than before: it is the reconcile that runs even when
Analytics is never opened.

Passing proves two things. A collection survives a restart under any
window origin, which is the reason it is stored server-side at all
rather than in localStorage alone. And a run deleted anywhere, from
the table, from another window, or from the filesystem, leaves no id
behind in a collection, so a tab's contents can always be opened and
its count cannot overstate what is in it. Set membership is preserved
throughout: a run filed into two collections stays in both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

import src.web.server as server
from src.web.ui_state import (
    UI_STATE_KEYS,
    load_ui_state,
    set_ui_state_key,
)

_KEY = "diffusion_collections"


def _make_run(root: Path, run_id: str) -> None:
    """A run folder the app would actually recognize.

    Metadata included, because that file is what makes a directory a
    run: the reconciliation asks the run store, and the store treats a
    folder without one as a half-written save rather than something a
    collection can point at.
    """
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"backend": "llada", "prompt": "p"}),
        encoding="utf-8",
    )

_REAL = "2026-01-01_00-00-00_llada"
_ALSO_REAL = "2026-01-02_00-00-00_smollm3"
_DELETED = "2026-01-01_09-99-99_llada"


@pytest.fixture()
def client_with_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _collection(
    name: str, runs: List[str]
) -> Dict[str, Any]:
    return {"id": name.lower(), "name": name, "runs": runs}


def _seed(results_dir: Path, collections: List[Any]) -> None:
    (results_dir / "ui_state.json").write_text(
        json.dumps({_KEY: json.dumps(collections)}),
        encoding="utf-8",
    )


def _read(client: TestClient) -> List[Any]:
    body = client.get("/api/ui-state").json()
    return json.loads(body[_KEY])


# -- the store --


def test_collections_are_a_known_key() -> None:
    """Unknown keys are rejected with a 404, so this is the whole
    difference between the feature persisting and silently not."""
    assert _KEY in UI_STATE_KEYS


def test_a_collection_survives_a_round_trip(
    tmp_path: Path,
) -> None:
    raw = json.dumps([_collection("Favorites", [_REAL])])

    set_ui_state_key(tmp_path, _KEY, raw)

    assert load_ui_state(tmp_path)[_KEY] == raw


def test_an_oversized_value_is_refused(
    tmp_path: Path,
) -> None:
    """The bound the other keys have. Without it a client looping on a
    write could grow the file until the disk filled."""
    oversized = "x" * (UI_STATE_KEYS[_KEY] + 1)

    with pytest.raises(ValueError):
        set_ui_state_key(tmp_path, _KEY, oversized)


# -- pruning on hydrate --


def test_a_deleted_run_leaves_no_id_behind(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    _make_run(tmp_path, _REAL)
    _seed(
        tmp_path,
        [_collection("Favorites", [_REAL, _DELETED])],
    )

    collections = _read(client_with_results)

    assert collections[0]["runs"] == [_REAL]


def test_the_pruned_list_is_written_back(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """The paired check: pruning only the response would re-prune on
    every hydrate and lose the fix the moment the client wrote."""
    _make_run(tmp_path, _REAL)
    _seed(
        tmp_path,
        [_collection("Favorites", [_REAL, _DELETED])],
    )

    client_with_results.get("/api/ui-state")

    on_disk = json.loads(
        (tmp_path / "ui_state.json").read_text(encoding="utf-8")
    )
    stored = json.loads(on_disk[_KEY])
    assert stored[0]["runs"] == [_REAL]


def test_a_run_in_two_collections_stays_in_both(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """Membership is a set, not an assignment. Pruning must not become
    the thing that quietly makes it exclusive."""
    _make_run(tmp_path, _REAL)
    _seed(
        tmp_path,
        [
            _collection("Favorites", [_REAL, _DELETED]),
            _collection("Papers", [_REAL]),
        ],
    )

    collections = _read(client_with_results)

    assert collections[0]["runs"] == [_REAL]
    assert collections[1]["runs"] == [_REAL]


def test_an_emptied_collection_is_kept(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """The user made it. Having its runs deleted is not the same as
    asking for the collection to go away."""
    _seed(tmp_path, [_collection("Favorites", [_DELETED])])

    collections = _read(client_with_results)

    assert len(collections) == 1
    assert collections[0]["runs"] == []
    assert collections[0]["name"] == "Favorites"


def test_nothing_is_rewritten_when_every_run_exists(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """The common case, and the one that must not write on every page
    load: hydrate happens on every navigation."""
    _make_run(tmp_path, _REAL)
    _make_run(tmp_path, _ALSO_REAL)
    _seed(
        tmp_path,
        [_collection("Favorites", [_REAL, _ALSO_REAL])],
    )
    before = (tmp_path / "ui_state.json").read_text(
        encoding="utf-8"
    )

    client_with_results.get("/api/ui-state")

    after = (tmp_path / "ui_state.json").read_text(
        encoding="utf-8"
    )
    assert after == before


def test_a_collection_of_only_deleted_runs_empties(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """The boundary: every id is an orphan, so the collection is
    emptied rather than left whole or removed."""
    _make_run(tmp_path, _REAL)
    _seed(
        tmp_path,
        [_collection("Stale", [_DELETED, "another-gone"])],
    )

    collections = _read(client_with_results)

    assert collections[0]["runs"] == []


def test_a_malformed_entry_passes_through(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    """Removing ids for missing runs is this endpoint's whole job.

    Hydrate is a mirror of the stored string, so it hands back what
    is on disk. The typed view is ``GET /api/collections``, which
    decodes and therefore does drop an entry it cannot address; the
    two differ on purpose, and only the second is what the page
    reads.
    """
    _seed(tmp_path, ["not a collection", {"name": "no runs"}])

    collections = _read(client_with_results)

    assert collections[0] == "not a collection"
    assert collections[1] == {"name": "no runs"}


def test_corrupt_json_is_left_alone(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    (tmp_path / "ui_state.json").write_text(
        json.dumps({_KEY: "{not json"}), encoding="utf-8"
    )

    body = client_with_results.get("/api/ui-state").json()

    assert body[_KEY] == "{not json"


def test_hydrate_without_collections_is_unaffected(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    (tmp_path / "ui_state.json").write_text(
        json.dumps({"diffusion_settings": "{}"}),
        encoding="utf-8",
    )

    body = client_with_results.get("/api/ui-state").json()

    assert body == {"diffusion_settings": "{}"}
