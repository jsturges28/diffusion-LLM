"""A worker's model side stands apart from its socket shell
(`A2-ORG-02`).

Strategy: import each module in a fresh interpreter and report what
came along. A fresh process because this one has imported both
modules for other suites, so what one import brings with it cannot
be asked here. The launcher gets its own for a second reason: it sets
the Hub's download flag at import, which would leak into every test
after it.

``worker_base`` used to hold the backend contract and the FastAPI app
that serves it, so nothing could reach one without loading the other.
Passing proves the dependency runs one way, a backend importing with
no app, routes or generation slot, and that the launcher builds its
app from the module that now owns it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run in the fresh interpreter: import the model side, then report
# which worker modules are loaded.
MODEL_SIDE_PROBE = """
import sys

import src.backends.worker_base

loaded = [
    name for name in sys.modules
    if name.startswith("src.backends.worker_")
]
print(",".join(sorted(loaded)))
"""

# Run in the fresh interpreter: import the launcher, then report
# which module its app factory belongs to.
LAUNCHER_PROBE = """
from src.backends import run_worker

print(run_worker.create_worker_app.__module__)
"""


def _run_probe(probe: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_model_side_imports_without_the_socket_shell() -> None:
    loaded = _run_probe(MODEL_SIDE_PROBE).split(",")

    assert "src.backends.worker_base" in loaded, "nothing imported"
    assert "src.backends.worker_app" not in loaded, (
        "importing worker_base loaded worker_app"
    )


def test_the_launcher_builds_its_app_from_the_shell() -> None:
    owner = _run_probe(LAUNCHER_PROBE)

    assert owner == "src.backends.worker_app", owner
