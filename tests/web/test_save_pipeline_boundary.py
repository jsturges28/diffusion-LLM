"""The save pipeline stands apart from the FastAPI supervisor.

Strategy: import each side in a fresh interpreter and report the
forbidden packages that came with it. Fresh processes make import
ownership observable even when another web test already loaded the
application. Then inspect the route's source to pin the narrow seam.

Passing proves save validation and publication need no FastAPI app or
model stack, importing the supervisor still loads no model stack, and
the route delegates persistence instead of rebuilding the pipeline.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.web import save_pipeline, server


REPO_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_FORBIDDEN = (
    "fastapi",
    "src.web.server",
    "torch",
    "transformers",
)

PIPELINE_PROBE = """
import sys

import src.web.save_pipeline

loaded = set(sys.modules)
forbidden = set(sys.argv[1:])
print(",".join(sorted(loaded & forbidden)))
"""

SUPERVISOR_PROBE = """
import sys

import src.web.server

loaded = {name.split(".")[0] for name in sys.modules}
print(",".join(sorted(loaded & {"torch", "transformers"})))
"""


def _run_probe(probe: str, *arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", probe, *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_pipeline_imports_without_http_or_model_stack() -> None:
    loaded = _run_probe(PIPELINE_PROBE, *PIPELINE_FORBIDDEN)

    assert loaded == "", f"save_pipeline loaded {loaded}"


def test_supervisor_imports_without_model_libraries() -> None:
    loaded = _run_probe(SUPERVISOR_PROBE)

    assert loaded == "", f"server loaded {loaded}"


def test_save_models_are_owned_by_the_pipeline() -> None:
    assert save_pipeline.SaveRunRequest.__module__ == (
        "src.web.save_pipeline"
    )
    assert "SaveRunRequest" not in vars(server)


def test_server_binds_current_fallback_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(server.manager, "active_device", "cpu")
    monkeypatch.setattr(
        server.manager, "active_versions", {"torch": "current"}
    )
    context = server._save_pipeline_context()
    facts = context.current_model_facts()

    assert context.results_dir == tmp_path
    assert facts.device == "cpu"
    assert facts.versions == {"torch": "current"}


def test_pipeline_dependencies_are_immutable() -> None:
    context = server._save_pipeline_context()
    facts = context.current_model_facts()

    # Assignments intentionally exercise the frozen runtime contract.
    with pytest.raises(FrozenInstanceError):
        context.results_dir = Path("/elsewhere")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        facts.device = "elsewhere"  # type: ignore[misc]


def test_route_delegates_publication_and_preview_scheduling() -> None:
    source = inspect.getsource(server.save_run)

    assert "save_pipeline.publish_run" in source
    assert "save_pipeline.schedule_preview" in source
    assert "run_store.save" not in source
    assert "_build_bundle" not in source
