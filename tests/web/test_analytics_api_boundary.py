"""The Analytics HTTP API stands apart from the supervisor.

Strategy: import the API in a fresh interpreter to expose its import
boundary, inspect the application's registered route owners, and
replace supervisor dependencies after the router has already been
built.

Passing proves the API does not reach back into ``server`` or import
model libraries, every Analytics URL is owned by the extracted
module, and the frozen dependency bundle still reads the live results
root and host probe.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.web import analytics_api, model_manager, server


REPO_ROOT = Path(__file__).resolve().parents[2]

IMPORT_PROBE = """
import sys

import src.web.analytics_api

loaded = set(sys.modules)
forbidden = {"src.web.server", "torch", "transformers"}
print(",".join(sorted(loaded & forbidden)))
"""

ANALYTICS_ROUTES = {
    ("GET", "/api/analytics/runs"),
    ("GET", "/api/analytics/runs/{run_id}/metrics"),
    ("GET", "/api/analytics/runs/{run_id}/metadata"),
    ("GET", "/api/analytics/runs/{run_id}/frames"),
    ("GET", "/api/analytics/compare"),
    ("GET", "/api/analytics/system"),
    ("DELETE", "/api/analytics/runs/{run_id}"),
}

MOVED_SERVER_NAMES = (
    "analytics_list_runs",
    "analytics_run_metrics",
    "analytics_run_metadata",
    "analytics_run_frames",
    "analytics_compare",
    "analytics_system_info",
    "analytics_delete_run",
    "_compute_run_metrics",
    "_compute_run_frames",
    "_unsupported_version_response",
    "_model_label",
    "_stop_rule",
)


def test_api_imports_without_server_or_model_libraries() -> None:
    """A fresh process makes transitive imports observable."""
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout


def test_analytics_routes_are_owned_by_the_api_module() -> None:
    """The supervisor includes routes but does not define them."""
    actual = {}
    for route in server.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/analytics"):
            continue
        methods = getattr(route, "methods", set())
        endpoint = getattr(route, "endpoint", None)
        assert endpoint is not None, path
        for method in methods:
            actual[(method, path)] = endpoint.__module__

    assert set(actual) == ANALYTICS_ROUTES
    assert set(actual.values()) == {analytics_api.__name__}


def test_server_does_not_reexport_moved_analytics_names() -> None:
    """Callers migrate to the owner instead of preserving aliases."""
    for name in MOVED_SERVER_NAMES:
        assert name not in vars(server), name


def test_api_dependencies_are_immutable(tmp_path: Path) -> None:
    """The router's control-plane inputs cannot drift in place."""

    def results_dir() -> Path:
        return tmp_path

    def gpu_name() -> str:
        return "Fake GPU"

    dependencies = analytics_api.AnalyticsApiDependencies(
        results_dir=results_dir,
        repo_root=REPO_ROOT,
        gpu_name=gpu_name,
    )

    with pytest.raises(FrozenInstanceError):
        dependencies.repo_root = tmp_path  # type: ignore[misc]


def test_server_dependencies_remain_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing globals after app construction reaches the route."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    client = TestClient(server.app)

    def first_gpu_name() -> str:
        return "First GPU"

    monkeypatch.setattr(server, "RESULTS_DIR", first)
    monkeypatch.setattr(model_manager, "gpu_name", first_gpu_name)
    first_body = client.get("/api/analytics/system").json()

    def second_gpu_name() -> str:
        return "Second GPU"

    monkeypatch.setattr(server, "RESULTS_DIR", second)
    monkeypatch.setattr(model_manager, "gpu_name", second_gpu_name)
    second_body = client.get("/api/analytics/system").json()

    assert first_body == {
        "gpu_name": "First GPU",
        "results_dir": str(first),
    }
    assert second_body == {
        "gpu_name": "Second GPU",
        "results_dir": str(second),
    }


def test_saved_watermark_records_expose_mismatch() -> None:
    """Analytics recomputes instead of trusting disk blindly."""
    payload = analytics_api._watermark_payload(
        {
            "watermark": {
                "green_count": 2,
                "scored_count": 2,
                "z_score": 2.0,
                "p0": 0.25,
            }
        },
        {
            "positions": [
                {"g": True, "we": False},
                {"g": True, "we": True},
                {"g": False, "we": True},
            ]
        },
    )

    assert payload is not None
    assert payload["record_consistency"] == "mismatch"
    assert payload["recomputed"]["green_count"] == 1
    assert payload["recomputed"]["scored_count"] == 2
