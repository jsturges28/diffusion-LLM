"""The run list carries the table, not the whole archive.

Strategy: save real runs into a temp results directory, then read the
catalog and the new per-run metadata endpoint over HTTP, so the
projection, the detail fetch and their guards are exercised the way
the page uses them.

What was wrong: the catalog handed back each run's entire
`metadata.json`. That is the full prompt, the full output text, every
hyperparameter, the reproducibility block, and the per-frame arrays,
so a two-thousand-token run contributed two-thousand-element lists to
a row the table draws as a single line. Every run paid that on every
load, and the table reads six fields.

The trade this makes is explicit: the list pays for every run and the
metadata endpoint pays for the one the user opens. So the tests come
in pairs, one proving a field left the catalog and one proving it is
still reachable, because dropping a field the detail panel needs
would be a regression wearing this change's clothes.

Passing proves the catalog is bounded, that everything it drops is
still served per run, that a bounded prompt says when it was cut, and
that the new endpoint is guarded like every other run-id route.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.analytics.metrics import (
    SUMMARY_FIELDS,
    SUMMARY_PROMPT_MAX_CHARS,
    run_summary,
)
from src.web import server

MASK = "\u2591"

# Fields whose whole purpose is to be large. None may reach the list.
HEAVY_FIELDS = (
    "final_text",
    "params",
    "reproducibility",
    "per_frame_elapsed",
    "mean_conf",
    "canvas_index",
    "capture",
)


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _save(
    client: TestClient,
    *,
    prompt: str = "explain REST",
    frames: int = 40,
) -> str:
    body: Dict[str, Any] = {
        "model": "llada",
        "prompt": prompt,
        "frames": [MASK * 4 for _ in range(frames)],
        "final_text": "a rather long final text " * 20,
        "params": {"steps": 64, "gen_length": 128},
        "per_frame_elapsed": [
            float(i) for i in range(frames)
        ],
        "mean_conf": [0.5 for _ in range(frames)],
        "canvas_index": [0 for _ in range(frames)],
    }
    response = client.post("/api/save", json=body)
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _catalog(client: TestClient) -> List[Dict[str, Any]]:
    response = client.get("/api/analytics/runs")
    assert response.status_code == 200, response.text
    return list(response.json())


def _metadata(client: TestClient, run_id: str) -> Dict[str, Any]:
    response = client.get(
        f"/api/analytics/runs/{run_id}/metadata"
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- the catalog is bounded --


def test_the_catalog_drops_the_heavy_fields(
    client: TestClient,
) -> None:
    _save(client)

    row = _catalog(client)[0]

    for field in HEAVY_FIELDS:
        assert field not in row, field


def test_the_catalog_keeps_what_the_table_draws(
    client: TestClient,
) -> None:
    """Sort and group keys are fixed in the markup as created_at,
    model, processor, prompt, elapsed_seconds and has_diff. A row
    missing one of those is a column that stops working."""
    _save(client)

    row = _catalog(client)[0]

    for field in (
        "run_id",
        "created_at",
        "model",
        "processor",
        "prompt",
        "elapsed_seconds",
        "has_diff",
    ):
        assert field in row, field


def test_the_row_is_a_small_fraction_of_the_record(
    client: TestClient,
) -> None:
    """The number this finding is actually about."""
    run_id = _save(client, frames=200)

    row_bytes = len(json.dumps(_catalog(client)[0]))
    full_bytes = len(json.dumps(_metadata(client, run_id)))

    assert row_bytes < full_bytes / 4, (row_bytes, full_bytes)


def test_the_row_does_not_grow_with_the_run(
    client: TestClient,
) -> None:
    """A long run and a short one cost the list the same.

    The per-frame arrays were the reason they did not: one element
    per frame, in a row rendered as one line.
    """
    short = _save(client, frames=4)
    long = _save(client, frames=400)

    rows = {row["run_id"]: row for row in _catalog(client)}
    short_bytes = len(json.dumps(rows[short]))
    long_bytes = len(json.dumps(rows[long]))

    assert abs(long_bytes - short_bytes) < 40


def test_a_long_prompt_is_cut_and_says_so(
    client: TestClient,
) -> None:
    # Prompt import allows 200,000 characters, which is the case
    # this bound exists for.
    _save(client, prompt="x" * 5000)

    row = _catalog(client)[0]

    assert len(row["prompt"]) == SUMMARY_PROMPT_MAX_CHARS
    assert row["prompt_truncated"] is True


def test_a_short_prompt_is_not_marked_cut(
    client: TestClient,
) -> None:
    _save(client, prompt="short one")

    row = _catalog(client)[0]

    assert row["prompt"] == "short one"
    assert row["prompt_truncated"] is False


def test_a_stopped_run_still_shows_as_stopped(
    client: TestClient,
) -> None:
    # The duration column reads "(stopped)" off this, so it has to
    # survive the projection.
    response = client.post(
        "/api/save",
        json={
            "model": "llada",
            "prompt": "p",
            "frames": [MASK, "a"],
            "final_text": "a",
            "partial": True,
        },
    )
    assert response.status_code == 200

    assert _catalog(client)[0]["partial"] is True


# -- what it drops is still reachable --


def test_the_metadata_endpoint_serves_the_full_record(
    client: TestClient,
) -> None:
    run_id = _save(client)

    meta = _metadata(client, run_id)

    for field in HEAVY_FIELDS:
        assert field in meta, field
    assert meta["params"]["steps"] == 64


def test_the_full_prompt_is_reachable(
    client: TestClient,
) -> None:
    run_id = _save(client, prompt="y" * 5000)

    assert len(_metadata(client, run_id)["prompt"]) == 5000


def test_the_metadata_carries_the_computed_fields(
    client: TestClient,
) -> None:
    # has_diff and the repaired elapsed total are derived, not
    # stored, and the detail panel shows both.
    run_id = _save(client)

    meta = _metadata(client, run_id)

    assert "has_diff" in meta
    assert "elapsed_seconds" in meta


# -- the new route is guarded like its siblings --


def test_a_traversing_id_is_refused(
    client: TestClient, tmp_path: Path
) -> None:
    victim = tmp_path.parent / "outside-meta"
    victim.mkdir(exist_ok=True)
    (victim / "metadata.json").write_text("{}", encoding="utf-8")

    response = client.get(
        "/api/analytics/runs/..%2Foutside-meta/metadata"
    )

    assert response.status_code in (400, 404)


def test_a_missing_run_is_a_404(client: TestClient) -> None:
    response = client.get(
        "/api/analytics/runs/2020-01-01_00-00-00_gone/metadata"
    )

    assert response.status_code == 404


def test_a_future_run_is_refused_with_its_own_message(
    client: TestClient, tmp_path: Path
) -> None:
    run_id = _save(client)
    meta_path = tmp_path / run_id / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = 99
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    response = client.get(
        f"/api/analytics/runs/{run_id}/metadata"
    )

    assert response.status_code == 400
    assert response.json()["unsupported_version"] is True


# -- the projection itself --


def test_absent_fields_stay_absent(
) -> None:
    """Not filled with nulls.

    The rest of this reader works to keep "recorded nothing" and
    "never asked" distinguishable, and a projection that invented
    keys would erase that for every field it touched.
    """
    summary = run_summary({"run_id": "r", "created_at": "now"})

    assert "elapsed_seconds" not in summary
    assert "partial" not in summary
    assert summary["run_id"] == "r"


def test_the_projection_is_a_list_not_an_exclusion(
) -> None:
    # A field added to a saved run must not silently join the
    # catalog, which is the defect this whole change is about.
    summary = run_summary(
        {"run_id": "r", "some_future_field": "x" * 10000}
    )

    assert "some_future_field" not in summary


def test_every_named_field_survives_when_present(
) -> None:
    meta = {field: 1 for field in SUMMARY_FIELDS}
    meta["run_id"] = "r"

    summary = run_summary(meta)

    for field in SUMMARY_FIELDS:
        assert field in summary, field


def test_an_unreadable_run_still_lists(
    client: TestClient, tmp_path: Path
) -> None:
    """One broken run cannot empty the table, and the projection
    must not be the thing that breaks that."""
    good = _save(client)
    broken = tmp_path / "2020-01-01_00-00-00_broken"
    broken.mkdir()
    (broken / "metadata.json").write_text(
        "{ not json", encoding="utf-8"
    )

    rows = {row["run_id"]: row for row in _catalog(client)}

    assert good in rows
    assert rows["2020-01-01_00-00-00_broken"]["invalid"] is True
    assert rows["2020-01-01_00-00-00_broken"]["error"]
