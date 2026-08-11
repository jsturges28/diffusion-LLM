"""Where saved runs and durable UI state live.

Its own module, small as it is, because the supervisor is not the
only thing that has to know the answer before it runs: ``main.py``
reads the environment variable's name to write ``--results-dir``
into it, and it must do that *before* importing the server, since
the server resolves the root at import time. Importing the server
to learn the name would be too late.

The root used to be a relative ``Path("results")``, which made the
process working directory an undocumented configuration mechanism.
The desktop launcher chdir'd to compensate and the browser launcher
did not, so starting from another directory silently split saved
runs and UI state across two trees with no error to notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Read by the supervisor at import and written by ``main.py`` from
# ``--results-dir``. One mechanism, so the browser launcher, the
# desktop launcher, and a bare ``uvicorn src.web.server:app`` all
# agree on where the data is.
RESULTS_DIR_ENV = "DIFFUSION_LLM_RESULTS_DIR"

RESULTS_DIR_NAME = "results"


def resolve_results_dir(
    raw: Optional[str], *, repo_root: Path
) -> Path:
    """Pick the one absolute data root for this process.

    ``raw`` is the override, normally the environment variable's
    value; absent or blank selects ``<repo_root>/results``, which is
    where every existing run already sits.

    The result is always absolute, including for a relative
    override, because the whole point is that nothing downstream
    depends on the working directory. ``~`` is expanded so a home
    relative root can be given the way a user would type it.
    """
    assert repo_root.is_absolute(), (
        "the repository root must itself be absolute"
    )
    if raw is None or raw.strip() == "":
        resolved = (repo_root / RESULTS_DIR_NAME).resolve()
    else:
        resolved = Path(raw.strip()).expanduser().resolve()
    assert resolved.is_absolute(), (
        "the data root must not depend on the working directory"
    )
    return resolved
