"""A run the user stopped must not read as one that finished.

Strategy: drive `/api/save` against a temp results directory, then
read the metadata back off disk and through the Analytics catalog,
which is where the claim actually matters. No model is involved;
the flag travels from the terminal frame through the client to the
record, and this covers the server half of that.

Why it needs a record rather than only a live indicator: the page
knows a run was stopped while it is on screen, and forgets the
moment it is saved. Months later in Analytics, forty seconds of a
run cut short reads exactly like forty seconds of one that
finished, and the text stopping early looks like the model's
choice rather than the user's.

Passing proves a stopped run says so on disk and in the catalog,
that a finished run stays silent rather than saying "not stopped",
and that a run saved before this field existed still reads as
finished rather than as unknown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from starlette.testclient import TestClient

from src.analytics.metrics import list_runs
from src.web import run_store, server


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _payload(**overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": "llada",
        "prompt": "explain REST",
        "frames": ["frame one", "frame two"],
        "final_text": "hello",
    }
    body.update(overrides)
    return body


def _save(client: TestClient, **overrides: Any) -> str:
    response = client.post("/api/save", json=_payload(**overrides))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["success"] is True
    return str(result["run_id"])


def _metadata(root: Path, run_id: str) -> Dict[str, Any]:
    path = run_store.resolve_run_dir(root, run_id) / "metadata.json"
    return json.loads(path.read_text())


def test_a_stopped_run_records_that_it_was_stopped(
    client: TestClient, tmp_path: Path
) -> None:
    run_id = _save(client, partial=True)

    assert _metadata(tmp_path, run_id)["partial"] is True


def test_a_finished_run_stays_silent_about_it(
    client: TestClient, tmp_path: Path
) -> None:
    """Absent, not false.

    The negative space matters here: writing ``partial: false``
    would make every run saved before this field look unknown
    rather than finished, and there are two hundred of them.
    """
    run_id = _save(client)

    assert "partial" not in _metadata(tmp_path, run_id)


def test_the_flag_defaults_off_when_a_client_omits_it(
    client: TestClient, tmp_path: Path
) -> None:
    # An older page sends no such field, and its runs are finished
    # ones. Silence must not be read as a stopped run.
    run_id = _save(client, partial=False)

    assert "partial" not in _metadata(tmp_path, run_id)


def test_the_catalog_carries_it_to_analytics(
    client: TestClient, tmp_path: Path
) -> None:
    """The list is where a run is judged against its siblings."""
    stopped = _save(client, partial=True, run_token="a:1")
    whole = _save(client, run_token="b:1")

    catalog = {run["run_id"]: run for run in list_runs(tmp_path)}

    assert catalog[stopped].get("partial") is True
    assert "partial" not in catalog[whole]


def test_re_saving_a_stopped_run_keeps_it_stopped(
    client: TestClient, tmp_path: Path
) -> None:
    """The idempotent re-save must not quietly promote it.

    A save that lands on an existing run rebuilds its metadata, so
    a second delivery of the same stopped run has to arrive at the
    same answer rather than at a run that looks complete.
    """
    first = _save(client, partial=True, run_token="a3f9c1:1")
    second = _save(client, partial=True, run_token="a3f9c1:1")

    assert first == second
    assert _metadata(tmp_path, first)["partial"] is True
