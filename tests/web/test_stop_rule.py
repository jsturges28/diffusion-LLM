"""Tests for the stopping rule the Analytics frames endpoint reports.

Strategy: the whole server path. A run is saved through ``/api/save``
and read back from ``/api/analytics/runs/{id}/frames``, the payload
the Analytics readout and Stopping chart are drawn from. The rule a
run ran under has to come from the run itself: a saved run outlives
the defaults of the build that saved it.

Passing proves a DiffusionGemma run reports the rule it was given, a
run saved before the rule was a parameter reports the rule the
checkpoint applied to it, a malformed or out-of-range value cannot
reach the page as the rule, and a model that does not stop adaptively
reports no rule at all, which is what keeps the readout and the chart
off its runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.web import server


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _frame() -> List[Dict[str, Any]]:
    return [
        {"t": "a", "m": False, "id": 5, "c": 0.9, "e": 0.01},
        {"t": "b", "m": True, "id": 6, "c": 0.4, "e": 0.2},
    ]


def _save(
    client: TestClient, model: str, params: Dict[str, Any]
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": "p",
        "params": params,
        "frames": ["a", "b"],
        "final_text": "ab",
        "frame_tokens": [_frame(), _frame()],
    }
    response = client.post("/api/save", json=payload)
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _rule(client: TestClient, run_id: str) -> Any:
    response = client.get(f"/api/analytics/runs/{run_id}/frames")
    assert response.status_code == 200, response.text
    return response.json()["stop_rule"]


def test_a_diffusiongemma_run_reports_its_own_rule(
    client: TestClient,
) -> None:
    run_id = _save(
        client,
        "diffusiongemma",
        {
            "confidence_threshold": 0.02,
            "stability_threshold": 2,
            "max_denoising_steps": 32,
        },
    )

    assert _rule(client, run_id) == {
        "confidence_threshold": 0.02,
        "stability_threshold": 2,
        "max_denoising_steps": 32,
    }


def test_an_older_run_reports_the_checkpoints_rule(
    client: TestClient,
) -> None:
    """Saved before the rule was a parameter, so its params name
    neither threshold; the checkpoint applied 0.005 and 1."""
    run_id = _save(
        client, "diffusiongemma", {"max_denoising_steps": 48}
    )

    assert _rule(client, run_id) == {
        "confidence_threshold": 0.005,
        "stability_threshold": 1,
        "max_denoising_steps": 48,
    }


def test_a_malformed_value_falls_back_to_the_default(
    client: TestClient,
) -> None:
    run_id = _save(
        client,
        "diffusiongemma",
        {"confidence_threshold": "lots", "stability_threshold": None},
    )

    rule = _rule(client, run_id)

    assert rule["confidence_threshold"] == 0.005
    assert rule["stability_threshold"] == 1
    assert rule["max_denoising_steps"] == 48


def test_a_value_past_every_bound_is_held_to_the_widest(
    client: TestClient,
) -> None:
    """A run file edited by hand cannot hand the page a rule no run
    could have used."""
    run_id = _save(
        client,
        "diffusiongemma",
        {"confidence_threshold": 0.0, "stability_threshold": 99},
    )

    rule = _rule(client, run_id)

    assert rule["confidence_threshold"] == 0.0001
    assert rule["stability_threshold"] == 16


def test_a_model_without_adaptive_stopping_reports_none(
    client: TestClient,
) -> None:
    run_id = _save(client, "llada", {"steps": 64})

    assert _rule(client, run_id) is None
