"""The model manager stands apart from the web app (`A2-ORG-01`).

Strategy: import the module in a fresh interpreter, build a manager
there, and list which web or model packages came along. A fresh
process because this one has FastAPI loaded by every other suite, so
a check here would pass whatever the module imported.

The manager used to live at the top of the module that also defines
every route, page and save schema, so nothing could reach it without
building the whole app. Passing proves the manager, its probes and
its download checks import and construct with no web framework and
no model library loaded, which is what lets their tests run without
an app, and that workers start from the same repository root the
server serves its pages from.
"""

from __future__ import annotations

import subprocess
import sys

from src.web import model_manager, server

FORBIDDEN = ("fastapi", "starlette", "torch", "transformers")

# Run in the fresh interpreter: import, construct, report.
PROBE = """
import sys

from src.web.model_manager import ModelManager

manager = ModelManager()
assert manager.activation_id == 0, "nothing activated yet"
assert manager.active_id is None, "no model yet"
loaded = {name.split(".")[0] for name in sys.modules}
print(",".join(sorted(loaded & set(sys.argv[1:]))))
"""


def test_the_manager_needs_no_web_framework_or_model_code() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, *FORBIDDEN],
        cwd=server.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing the manager loaded {result.stdout.strip()}"
    )


def test_workers_start_where_the_server_serves_from() -> None:
    """The two modules each resolve the repository root, the server
    for its pages and data, the manager for the workers it spawns,
    and a worker started anywhere else would not find its code."""
    assert model_manager.REPO_ROOT == server.REPO_ROOT
    assert (model_manager.REPO_ROOT / "src" / "web").is_dir()
