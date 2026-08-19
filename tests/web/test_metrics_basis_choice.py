"""The endpoint picks a convergence measure per run.

Strategy: publish runs through `/api/save` whose token records are
shaped like each model's, then read them back through the metrics
endpoint. The arithmetic of each measure is unit tested in
`tests/analytics/test_settlement_basis.py`; what this pins is the
choice, which is the part that has to be right for a run nobody
thought about when it was saved.

Two shapes matter. A LLaDA-shaped run masks with one repeated vocab
id, so its flag is ground truth. A DiffusionGemma-shaped run marks
positions unresolved while they hold ordinary tokens with different
ids, because the sampler inferred the flag from a position holding
still rather than receiving a mask. Only the second needs replacing,
and replacing the first would make its curve wrong in the other
direction.

Passing proves each shape gets the measure that is exact for it, that
the basis travels so the caption can explain itself, that the
throughput series keeps counting what the sampler resolved even when
the chart no longer does, and that a run with no records still
answers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from starlette.testclient import TestClient

from src.web import server

MASK = "\u2591"
MASK_ID = 126336


@pytest.fixture()
def client(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _save(
    client: TestClient,
    *,
    frame_tokens: List[List[Dict[str, Any]]],
    canvas_index: Optional[List[int]] = None,
) -> str:
    """Publish a run built directly from token records."""
    frames = [
        "".join(
            MASK if t["m"] else t["t"] for t in frame
        )
        for frame in frame_tokens
    ]
    body: Dict[str, Any] = {
        "model": "llada",
        "prompt": "explain REST",
        "frames": frames,
        "final_text": frames[-1],
        "frame_tokens": frame_tokens,
        "per_frame_elapsed": [
            float(i + 1) for i in range(len(frames))
        ],
    }
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


def _llada_shaped() -> List[List[Dict[str, Any]]]:
    """One mask token, repeated. Three positions, one per frame."""
    frames = []
    for resolved in range(4):
        frame = []
        for position in range(3):
            if position < resolved:
                frame.append({
                    "t": "x", "m": False, "id": 500 + position,
                })
            else:
                frame.append({
                    "t": MASK, "m": True, "id": MASK_ID,
                })
        frames.append(frame)
    return frames


def _dgemma_shaped() -> List[List[Dict[str, Any]]]:
    """No mask token, and filler counted as resolved.

    Position 0 holds id 99 steadily at frame 1, so the sampler marks
    it resolved while the canvas goes on to commit id 500 there.
    That is the gap the settlement measure closes.
    """
    return [
        [
            {"t": "a", "m": True, "id": 11},
            {"t": "b", "m": True, "id": 12},
        ],
        [
            {"t": "the", "m": False, "id": 99},
            {"t": "c", "m": True, "id": 13},
        ],
        [
            {"t": "x", "m": False, "id": 500},
            {"t": "y", "m": False, "id": 501},
        ],
    ]


# -- each shape gets the measure that is exact for it --


def test_a_real_mask_keeps_the_mask_flag(
    client: TestClient,
) -> None:
    run_id = _save(client, frame_tokens=_llada_shaped())

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "tokens"
    assert [
        p["resolved_ratio"] for p in metrics["convergence"]
    ] == [0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0]


def test_an_inferred_mask_switches_to_settlement(
    client: TestClient,
) -> None:
    run_id = _save(client, frame_tokens=_dgemma_shaped())

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "settlement"


def test_settlement_refuses_to_count_the_filler(
    client: TestClient,
) -> None:
    """The defect, end to end.

    Frame 1 has one of two positions marked resolved, so the old
    measure reads 50%. That position holds filler the canvas later
    replaces, so nothing has settled and the honest answer is 0%.
    """
    run_id = _save(client, frame_tokens=_dgemma_shaped())

    ratios = [
        p["resolved_ratio"]
        for p in _metrics(client, run_id)["convergence"]
    ]

    assert ratios == [0.0, 0.0, 1.0]


def test_the_llada_curve_is_not_disturbed(
    client: TestClient,
) -> None:
    # Settlement would open this at above zero, because positions
    # still masked at the end agree with the end from frame zero.
    run_id = _save(client, frame_tokens=_llada_shaped())

    first = _metrics(client, run_id)["convergence"][0]

    assert first["resolved_ratio"] == 0.0


# -- the basis travels, so the caption can explain itself --


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (_llada_shaped, "tokens"),
        (_dgemma_shaped, "settlement"),
    ],
)
def test_the_basis_is_reported(
    client: TestClient, shape: Any, expected: str
) -> None:
    run_id = _save(client, frame_tokens=shape())

    assert _metrics(client, run_id)["convergence_basis"] == (
        expected
    )


def test_the_model_is_named_for_the_caption(
    client: TestClient,
) -> None:
    """The caption says which model has no mask token, and only the
    server can turn a registry id into something worth reading."""
    run_id = _save(client, frame_tokens=_dgemma_shaped())

    assert _metrics(client, run_id)["model_label"] == (
        "LLaDA-8B-Instruct"
    )


def test_an_unknown_model_is_named_by_its_id(
    client: TestClient, tmp_path
) -> None:
    # A run from a build that knew a model this one does not should
    # still read as itself rather than as an empty sentence.
    run_id = _save(client, frame_tokens=_dgemma_shaped())
    meta_path = tmp_path / run_id / "metadata.json"
    import json as _json

    meta = _json.loads(meta_path.read_text())
    meta["backend"] = "retired-model"
    meta_path.write_text(_json.dumps(meta), encoding="utf-8")

    assert _metrics(client, run_id)["model_label"] == (
        "retired-model"
    )


def test_a_run_without_records_still_answers(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/save",
        json={
            "model": "llada",
            "prompt": "p",
            "frames": [MASK + MASK, "ab"],
            "final_text": "ab",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "characters"
    assert len(metrics["convergence"]) == 2


# -- throughput keeps its own numerator --


def test_throughput_still_counts_what_the_sampler_resolved(
    client: TestClient,
) -> None:
    """The generator's live footer counts the sampler's reveals and
    cannot know settlement. If this followed the chart, the same run
    would read as two different speeds on two pages, which is the
    disagreement manual item 174 exists to catch."""
    run_id = _save(client, frame_tokens=_dgemma_shaped())

    metrics = _metrics(client, run_id)

    # One position marked resolved at frame 1, two at frame 2.
    assert metrics["tokens_produced"] == [0, 1, 2]
    # And the chart is on the other measure, so they differ here.
    assert metrics["convergence"][1]["resolved_ratio"] == 0.0


def test_a_second_canvas_is_measured_against_its_own_end(
    client: TestClient,
) -> None:
    frame_tokens = [
        [{"t": "a", "m": True, "id": 11}],
        [{"t": "x", "m": False, "id": 500}],
        [{"t": "b", "m": True, "id": 12}],
        [{"t": "y", "m": False, "id": 600}],
    ]
    run_id = _save(
        client,
        frame_tokens=frame_tokens,
        canvas_index=[0, 0, 1, 1],
    )

    metrics = _metrics(client, run_id)

    assert metrics["convergence_basis"] == "settlement"
    assert [
        p["resolved_ratio"] for p in metrics["convergence"]
    ] == [0.0, 1.0, 0.0, 1.0]
