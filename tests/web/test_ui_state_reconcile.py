"""Tests for the "new run" cue reconciliation in GET /api/ui-state.

Strategy: the cue (``diffusion_new_runs``) accumulates saved-but-unviewed
run IDs. A run deleted outside the app leaves an orphan ID that would
inflate the generator/menu count forever. The GET endpoint prunes the
cue to run folders that still exist. These tests point the server's
results dir at a tmp path, seed a cue with one real and one orphan ID,
and assert the endpoint returns (and persists) only the real ID, while
leaving other keys untouched. Passing proves the count self-heals on
every page hydrate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import src.web.server as server


@pytest.fixture()
def client_with_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _make_run(root: Path, run_id: str) -> None:
    """A run folder the app would actually recognize.

    Metadata included, because that file is what makes a directory a
    run: the reconciliation asks the run store, and the store treats a
    folder without one as a half-written save rather than a run whose
    cue is worth keeping alive.
    """
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"backend": "llada", "prompt": "p"}),
        encoding="utf-8",
    )


def _seed_state(results_dir: Path, new_runs: list) -> None:
    (results_dir / "ui_state.json").write_text(
        json.dumps({"diffusion_new_runs": json.dumps(new_runs)}),
        encoding="utf-8",
    )


def test_get_prunes_orphan_run_ids(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    real = "2026-01-01_00-00-00_llada"
    orphan = "2026-01-01_09-99-99_llada"
    _make_run(tmp_path, real)
    _seed_state(tmp_path, [real, orphan])

    body = client_with_results.get("/api/ui-state").json()

    assert json.loads(body["diffusion_new_runs"]) == [real]
    on_disk = json.loads(
        (tmp_path / "ui_state.json").read_text(encoding="utf-8")
    )
    assert json.loads(on_disk["diffusion_new_runs"]) == [real]


def test_get_keeps_all_when_none_orphaned(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    real = "2026-01-01_00-00-00_llada"
    _make_run(tmp_path, real)
    _seed_state(tmp_path, [real])

    body = client_with_results.get("/api/ui-state").json()

    assert json.loads(body["diffusion_new_runs"]) == [real]


def test_get_without_cue_is_unaffected(
    tmp_path: Path, client_with_results: TestClient
) -> None:
    (tmp_path / "ui_state.json").write_text(
        json.dumps({"diffusion_settings": "{}"}), encoding="utf-8"
    )

    body = client_with_results.get("/api/ui-state").json()

    assert body == {"diffusion_settings": "{}"}
