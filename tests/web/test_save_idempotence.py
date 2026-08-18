"""One generation cannot become two saved runs.

Strategy: drive `/api/save` against a temp results directory, the way
the collections tests drive the UI-state endpoints. The store's own
resolution is unit-tested in `tests/web/test_run_store.py`; what this
adds is the endpoint carrying the token far enough to matter, and the
exact click sequence a user reported.

The sequence. Generate a run and save it. Navigate to another page
before the save's reply arrives. The server publishes the run; the
page never learns it did, because the handler that would have
recorded it belongs to a document that is gone. Come back, press Edit
Frames, and its `if (!runSaved)` guard fires and saves the run a
second time, leaving two rows in Analytics for one generation.

Flushing the request harder does not fix this, which is why the fix
is identity rather than delivery: the client needs the *response*, not
just for the bytes to arrive. So the run says which generation made
it, and a save for a generation already published lands on the run it
already made.

What passing proves is that the retry is harmless, whatever the client
believes it is asking for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.web import run_store, server

TOKEN = "a3f9c1:1"


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


def _save(client: TestClient, **overrides: Any) -> Dict[str, Any]:
    response = client.post("/api/save", json=_payload(**overrides))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["success"] is True
    return result


def _run_ids(root: Path) -> List[str]:
    return run_store.list_run_ids(root)


# -- the reported sequence --


def test_a_save_whose_reply_was_lost_does_not_duplicate(
    tmp_path: Path, client: TestClient
) -> None:
    """The bug as reported: save, navigate, come back, edit."""
    first = _save(client, run_token=TOKEN)

    # The page came back believing nothing was saved, so it asks
    # again with no run id, exactly as Edit Frames would.
    second = _save(client, run_token=TOKEN)

    assert second["run_id"] == first["run_id"]
    assert len(_run_ids(tmp_path)) == 1


def test_the_retry_answers_with_what_an_edit_needs_next(
    tmp_path: Path, client: TestClient
) -> None:
    """Not merely 'no duplicate'. The reply has to carry the id and
    revision, because the edit that follows quotes them."""
    _save(client, run_token=TOKEN)

    second = _save(client, run_token=TOKEN)

    assert second["run_id"]
    assert second["revision"] == 2


def test_the_second_save_writes_rather_than_shortcutting(
    tmp_path: Path, client: TestClient
) -> None:
    """Once entering an editor stops auto-saving, confirming an edit
    is a save that carries no run id. Treating a known token as
    already-done would silently drop the edit."""
    first = _save(client, run_token=TOKEN)

    _save(client, run_token=TOKEN, final_text="edited output")

    saved = (
        tmp_path / first["run_id"] / run_store.FINAL_TEXT_NAME
    )
    assert saved.read_text(encoding="utf-8") == "edited output"


# -- and the cases that must keep creating --


def test_two_generations_are_still_two_runs(
    tmp_path: Path, client: TestClient
) -> None:
    _save(client, run_token="a3f9c1:1")
    _save(client, run_token="a3f9c1:2")

    assert len(_run_ids(tmp_path)) == 2


def test_a_client_that_sends_no_token_is_unaffected(
    tmp_path: Path, client: TestClient
) -> None:
    """The upgrade path, and the whole existing corpus: no token
    means no identity, which creates exactly as it always did."""
    _save(client)
    _save(client)

    assert len(_run_ids(tmp_path)) == 2


def test_the_run_records_the_generation_that_made_it(
    tmp_path: Path, client: TestClient
) -> None:
    result = _save(client, run_token=TOKEN)

    stored = json.loads(
        (
            tmp_path
            / result["run_id"]
            / run_store.METADATA_NAME
        ).read_text(encoding="utf-8")
    )
    assert stored[run_store.RUN_TOKEN_KEY] == TOKEN


def test_an_edit_quoting_a_stale_revision_is_still_refused(
    tmp_path: Path, client: TestClient
) -> None:
    """The token decides *which* run is written; it does not excuse a
    writer from checking it is replacing what it thinks it is. That
    guard is DATA-01's and stays."""
    first = _save(client, run_token=TOKEN)
    _save(client, run_token=TOKEN)

    response = client.post(
        "/api/save",
        json=_payload(
            run_token=TOKEN,
            run_id=first["run_id"],
            expected_revision=1,
        ),
    )

    assert response.status_code == 409
    assert response.json()["success"] is False
