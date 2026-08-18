"""The metrics endpoint says how it measured, and measures right.

Strategy: save runs through `/api/save` into a temp results
directory, then read them back through
`/api/analytics/runs/{id}/metrics`, so the whole path is exercised:
the store's writer, the schema dispatch, the choice of measure, and
the JSON the charts actually receive. The arithmetic itself is unit
tested in `tests/analytics/test_convergence_basis.py`.

The point of the endpoint half is the choice. Roughly a tenth of the
saved archive has no usable per-token records, either because it
predates them or because it stored bare ids without a mask flag.
Those runs still get a curve, counted in characters, and the response
has to say so, because a caption is the only thing separating an
approximate curve from the exact one beside it in another tab.

Passing proves a run with records is measured in positions, a run
without still answers rather than failing, each says which it was,
and the throughput numerator ships alongside so the browser stops
deriving it from a baseline that cannot survive a second canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _token(text: str, masked: bool) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "t": MASK if masked else text,
        "m": masked,
        "id": 7,
    }
    if not masked:
        record["c"] = 0.9
    return record


def _save(
    client: TestClient,
    *,
    schedule: List[List[bool]],
    text: str,
    with_tokens: bool = True,
    canvas_index: Optional[List[int]] = None,
) -> str:
    """Publish a run whose frames follow a mask schedule."""
    frames = [
        "".join(MASK if masked else text for masked in frame)
        for frame in schedule
    ]
    body: Dict[str, Any] = {
        "model": "llada",
        "prompt": "explain REST",
        "frames": frames,
        "final_text": frames[-1],
        "per_frame_elapsed": [
            float(i + 1) for i in range(len(frames))
        ],
    }
    if with_tokens:
        body["frame_tokens"] = [
            [_token(text, masked) for masked in frame]
            for frame in schedule
        ]
    if canvas_index is not None:
        body["canvas_index"] = canvas_index
    response = client.post("/api/save", json=body)
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _metrics(client: TestClient, run_id: str) -> Dict[str, Any]:
    response = client.get(
        f"/api/analytics/runs/{run_id}/metrics"
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


ONE_PER_FRAME = [
    [True, True, True],
    [False, True, True],
    [False, False, True],
    [False, False, False],
]


def test_a_run_with_records_is_measured_in_positions(
    client: TestClient,
) -> None:
    run_id = _save(
        client, schedule=ONE_PER_FRAME, text="extraordinarily"
    )

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "tokens"
    ratios = [
        point["resolved_ratio"] for point in metrics["convergence"]
    ]
    assert ratios == [0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0]


def test_the_curve_ignores_how_long_the_tokens_are(
    client: TestClient,
) -> None:
    """Two runs, one schedule, different spellings, one curve.

    This is the comparison the chart exists to support, and the one
    the character measure could reverse.
    """
    short = _save(client, schedule=ONE_PER_FRAME, text="a")
    long = _save(
        client, schedule=ONE_PER_FRAME, text="extraordinarily"
    )

    assert (
        _metrics(client, short)["convergence"]
        == _metrics(client, long)["convergence"]
    )


def test_a_run_without_records_still_answers(
    client: TestClient,
) -> None:
    """Degraded, not absent, and labelled as such."""
    run_id = _save(
        client,
        schedule=ONE_PER_FRAME,
        text="a",
        with_tokens=False,
    )

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "characters"
    assert len(metrics["convergence"]) == len(ONE_PER_FRAME)
    assert metrics["convergence"][-1]["resolved_ratio"] == 1.0


def test_the_basis_is_always_stated(
    client: TestClient,
) -> None:
    # The caption reads off this field, so a response without it
    # would silently present an approximate curve as an exact one.
    with_records = _save(
        client, schedule=ONE_PER_FRAME, text="a"
    )
    without = _save(
        client,
        schedule=ONE_PER_FRAME,
        text="a",
        with_tokens=False,
    )

    for run_id in (with_records, without):
        assert "convergence_basis" in _metrics(client, run_id)


def test_the_throughput_numerator_is_served(
    client: TestClient,
) -> None:
    run_id = _save(client, schedule=ONE_PER_FRAME, text="a")

    metrics = _metrics(client, run_id)

    assert metrics["tokens_produced"] == [0, 1, 2, 3]


def test_a_second_canvas_keeps_the_first_one_counted(
    client: TestClient,
) -> None:
    """End to end, the defect that moved this off the client.

    The browser subtracted each frame's mask count from the first
    frame's, so a committed canvas stopped being counted the moment
    the next one started from fresh noise.
    """
    schedule = [
        [True, True],
        [False, False],
        [True, True],
        [False, False],
    ]
    run_id = _save(
        client,
        schedule=schedule,
        text="a",
        canvas_index=[0, 0, 1, 1],
    )

    metrics = _metrics(client, run_id)

    assert metrics["tokens_produced"] == [0, 2, 2, 4]
    assert metrics["canvas_boundaries"] == [2]


def test_a_missing_run_is_still_a_404(
    client: TestClient,
) -> None:
    # The new branch sits inside the metrics path, so its error
    # behaviour has to be unchanged by it.
    response = client.get(
        "/api/analytics/runs/not-a-run/metrics"
    )

    assert response.status_code == 404
