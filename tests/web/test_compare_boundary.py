"""A comparison accounts for every run it was asked about.

Strategy: drive `/api/analytics/compare` against a temp results
directory holding real saved runs, so the selection parsing, the
store's resolver, the per-run failure handling and the response
shape are all exercised together.

The defect this closes is quiet subtraction. The endpoint took an
unbounded, undeduplicated id list and answered with an array that
simply omitted whatever it could not read, collapsing "deleted",
"not a valid id", "written by a newer build" and "corrupt" into one
indistinguishable string. The chart then dropped autoregressive runs
on its own. Between them, a user could tick three runs, see one
line, and find nothing on the page explaining the other two.

The path traversal the report describes is already fixed: the
metrics path resolves through the run store like its siblings, which
`DATA-01` landed. There was never a compare-specific test for it, so
one lives here to keep it fixed.

Passing proves every selection produces exactly one record, that the
records distinguish their failure modes, that a corrupt run costs its
own row rather than the whole request, and that the selection itself
is bounded and deduplicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.web import server

MASK = "\u2591"


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _save(
    client: TestClient,
    *,
    model: str = "llada",
    params: Dict[str, Any] | None = None,
) -> str:
    body: Dict[str, Any] = {
        "model": model,
        "prompt": "explain REST",
        "frames": [MASK + MASK, "ab"],
        "final_text": "ab",
        "params": params or {},
    }
    response = client.post("/api/save", json=body)
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _compare(
    client: TestClient, ids: List[str]
) -> Any:
    return client.get(
        "/api/analytics/compare",
        params={"ids": ",".join(ids)},
    )


def _ok(client: TestClient, ids: List[str]) -> List[Dict[str, Any]]:
    response = _compare(client, ids)
    assert response.status_code == 200, response.text
    return list(response.json())


# -- every selection is accounted for --


def test_each_selection_gets_exactly_one_record(
    client: TestClient,
) -> None:
    first = _save(client)
    second = _save(client)

    results = _ok(client, [first, second, "gone"])

    assert len(results) == 3
    assert [r["run_id"] for r in results] == [
        first, second, "gone",
    ]
    for record in results:
        assert record["status"] in (
            "data", "unavailable", "error"
        )


def test_a_missing_run_says_it_is_missing(
    client: TestClient,
) -> None:
    good = _save(client)

    results = _ok(client, [good, "2020-01-01_00-00-00_gone"])

    assert results[0]["status"] == "data"
    assert results[1]["status"] == "error"
    assert results[1]["reason"] == "not_found"
    assert results[1]["message"]


def test_a_traversing_id_is_refused_and_named(
    client: TestClient, tmp_path: Path
) -> None:
    """The regression test compare never had.

    The metrics path used to join an id straight onto the results
    root while its siblings went through the guarded resolver. It no
    longer does, and this keeps it that way. A crafted id must come
    back as one refused row, not as data and not as a 500.
    """
    victim = tmp_path.parent / "outside"
    victim.mkdir(exist_ok=True)
    (victim / "metadata.json").write_text("{}", encoding="utf-8")

    results = _ok(client, ["../outside", "/etc", "a/../../b"])

    assert len(results) == 3
    for record in results:
        assert record["status"] == "error"
        assert record["reason"] == "invalid_id"


def test_an_autoregressive_run_is_explained_not_dropped(
    client: TestClient,
) -> None:
    """It has no convergence curve, and that is worth saying.

    The chart used to skip these with a bare continue, so selecting
    a diffusion run against an autoregressive one drew a single line
    with no hint that the second had been discarded.
    """
    diffusion = _save(client, model="llada")
    autoregressive = _save(client, model="smollm3")

    results = _ok(client, [diffusion, autoregressive])

    assert results[0]["status"] == "data"
    assert results[1]["status"] == "unavailable"
    assert results[1]["reason"] == "no_curve"
    assert "onvergence" in results[1]["message"]


def test_one_corrupt_run_does_not_sink_the_batch(
    client: TestClient, tmp_path: Path
) -> None:
    """It used to. Only two exception types were caught per run, so
    anything else escaped and failed the whole comparison, taking the
    healthy runs with it."""
    good = _save(client)
    broken = _save(client)
    broken_dir = tmp_path / broken
    (broken_dir / "frames.jsonl").write_text(
        "not json at all", encoding="utf-8"
    )

    results = _ok(client, [good, broken])

    assert results[0]["status"] == "data"
    assert results[1]["status"] == "error"
    assert results[1]["reason"] in ("unreadable", "invalid_id")


def test_a_future_run_says_so_rather_than_not_found(
    client: TestClient, tmp_path: Path
) -> None:
    run_id = _save(client)
    meta_path = tmp_path / run_id / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = 99
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    results = _ok(client, [run_id])

    assert results[0]["status"] == "error"
    assert results[0]["reason"] == "unsupported_version"


# -- the selection is bounded --


def test_duplicates_are_collapsed(client: TestClient) -> None:
    run_id = _save(client)

    results = _ok(client, [run_id, run_id, run_id])

    assert len(results) == 1


def test_too_many_runs_are_refused_with_a_reason(
    client: TestClient,
) -> None:
    ids = [f"run-{i}" for i in range(server.COMPARE_RUNS_MAX + 1)]

    response = _compare(client, ids)

    assert response.status_code == 400
    assert str(server.COMPARE_RUNS_MAX) in (
        response.json()["error"]
    )


def test_the_cap_itself_is_allowed(client: TestClient) -> None:
    # The boundary, from the inside: one fewer than the refusal.
    ids = [f"run-{i}" for i in range(server.COMPARE_RUNS_MAX)]

    results = _ok(client, ids)

    assert len(results) == server.COMPARE_RUNS_MAX


def test_an_empty_selection_is_refused(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/analytics/compare", params={"ids": ""}
    )

    assert response.status_code == 400


def test_whitespace_only_ids_are_not_a_selection(
    client: TestClient,
) -> None:
    response = _compare(client, [" ", "  ", ""])

    assert response.status_code == 400


# -- labels come from the model, not from LLaDA --


def test_a_label_names_the_model_that_ran(
    client: TestClient,
) -> None:
    """The browser built this from steps, gen_length and
    block_length, so anything but LLaDA read 'undefined'."""
    run_id = _save(
        client,
        model="smollm3",
        params={"max_new_tokens": 128, "temperature": 0.6},
    )

    results = _ok(client, [run_id])

    label = results[0]["label"]
    assert "undefined" not in label
    assert "SmolLM3" in label


def test_a_label_uses_the_run_s_own_parameters(
    client: TestClient,
) -> None:
    run_id = _save(
        client,
        model="llada",
        params={"steps": 64, "gen_length": 128},
    )

    label = _ok(client, [run_id])[0]["label"]

    assert "64" in label
    assert "128" in label


def test_a_refused_selection_still_carries_a_label(
    client: TestClient,
) -> None:
    # The omission list names what it could not draw, so a record
    # with no label would read as a blank line in that list.
    results = _ok(client, ["2020-01-01_00-00-00_gone"])

    assert results[0]["label"]
