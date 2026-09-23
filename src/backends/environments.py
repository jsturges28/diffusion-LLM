"""Resolve an environment name to the interpreter that runs it.

The registry used to carry `.venv-dgemma/bin/python` as a string per
model, which made it the fourth place an environment had to be spelled
out: once in the requirements file, once in the setup instructions,
once in the agent conventions, and once here. `DEPS-01` is about there
being one such place, so the registry names an environment instead and
this module answers where it lives, reading the same
`[tool.diffusion-llm]` table the locks are generated from.

Cached, because the registry checks every model against it at import
and the supervisor asks again per activation, and a TOML parse is not
worth repeating. `tomllib` is stdlib from 3.11, and the manifest pins
3.12, so reading it costs no dependency.

Resolving an interpreter stays separate from listing the names. The
registry only needs the names, and a `ModelInfo` built in a test can
name anything as long as nothing tries to launch it, which is what
keeps the stubs in `tests/backends/` constructible.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "pyproject.toml"


class UnknownEnvironmentError(KeyError):
    """A model names an environment the manifest does not declare.

    Its own type because the supervisor turns it into a refusal the
    user can act on, and because a bare ``KeyError`` on a dict lookup
    reads like an internal slip rather than a registry that names
    something nobody has set up.
    """


@lru_cache(maxsize=1)
def _environments() -> Dict[str, Dict[str, object]]:
    """The manifest's environments, parsed once per process."""
    data = tomllib.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    environments = data["tool"]["diffusion-llm"]["environments"]
    assert environments, "the manifest declares no environments"
    return environments


def environment_names() -> Tuple[str, ...]:
    """Every environment the manifest declares, in its own order."""
    return tuple(_environments())


def interpreter_for(name: str) -> Path:
    """The Python that runs ``name``, relative to the repo root.

    Relative rather than absolute because that is what the supervisor
    reports when an interpreter is missing, and an absolute path in
    that message would name a directory the reader has to strip back
    to a command they can run.

    An overlay resolves to the interpreter of whatever it extends,
    since an overlay is not a separate environment; the desktop
    packages live in core's.
    """
    assert isinstance(name, str) and name, "environment name required"
    environments = _environments()
    if name not in environments:
        raise UnknownEnvironmentError(
            f"{name} is not declared in pyproject.toml"
            f" (have: {', '.join(environments)})"
        )
    entry = environments[name]
    parent = entry.get("extends")
    if isinstance(parent, str):
        return interpreter_for(parent)
    path = entry.get("path")
    assert isinstance(path, str) and path, (
        f"{name} declares neither a path nor a parent"
    )
    return Path(path) / "bin" / "python"


def lock_for(name: str) -> str:
    """The lock file ``name`` is installed from.

    Here rather than only in the generator because the supervisor's
    missing-interpreter message names it: "no interpreter at
    .venv-ar/bin/python" tells the user what is wrong, and the lock
    tells them what to run.
    """
    assert isinstance(name, str) and name, "environment name required"
    environments = _environments()
    if name not in environments:
        raise UnknownEnvironmentError(
            f"{name} is not declared in pyproject.toml"
        )
    lock = environments[name].get("lock")
    assert isinstance(lock, str) and lock, f"{name} names no lock"
    return lock
