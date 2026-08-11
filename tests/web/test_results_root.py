"""Tests that saved runs resolve to one place, whatever the cwd.

Strategy: exercise ``resolve_results_dir`` over the ways a root can
be named (absent, blank, relative, home relative, absolute), then
check the two things that consume it: the module-level root the
supervisor resolves at import, and the path the Analytics delete
dialog is told to display.

The root used to be a relative ``Path("results")``, so the process
working directory silently decided where runs were written. The
handoff records the real incident: two result trees and split UI
state, with no error to notice, because the app happily read and
wrote a different folder than the one the user was looking at.
Passing proves that starting from anywhere lands on the same
directory, and that an explicit override is absolute and is named
rather than silently substituted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import src.web.server as server
from src.web.data_root import (
    RESULTS_DIR_ENV,
    resolve_results_dir,
)

REPO = Path("/srv/checkout")


# -- how a root is chosen --


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_an_absent_root_falls_back_to_the_repository(
    raw: str | None,
) -> None:
    """Blank in every form it can arrive in. An exported but empty
    environment variable is the likeliest of these, and treating it
    as a root would resolve to the working directory, which is the
    behavior being removed."""
    assert resolve_results_dir(raw, repo_root=REPO) == (
        REPO / "results"
    )


def test_a_relative_override_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the whole finding is about: nothing downstream
    may depend on where the process was started."""
    monkeypatch.chdir(tmp_path)

    resolved = resolve_results_dir("runs", repo_root=REPO)

    assert resolved.is_absolute()
    assert resolved == (tmp_path / "runs").resolve()


def test_an_absolute_override_is_kept() -> None:
    resolved = resolve_results_dir(
        "/var/lib/diffusion", repo_root=REPO
    )

    assert resolved == Path("/var/lib/diffusion")


def test_a_home_relative_override_expands() -> None:
    """Users type ~/runs, and a literal "~" directory would be a
    silent second tree of exactly the kind this replaces."""
    resolved = resolve_results_dir("~/runs", repo_root=REPO)

    assert resolved == Path.home() / "runs"
    assert "~" not in str(resolved)


def test_surrounding_whitespace_is_ignored() -> None:
    resolved = resolve_results_dir(
        "  /var/lib/diffusion  ", repo_root=REPO
    )

    assert resolved == Path("/var/lib/diffusion")


def test_every_answer_is_absolute() -> None:
    """Negative space, stated once over all the shapes: there is no
    input for which this returns something cwd-dependent."""
    for raw in [None, "", "runs", "~/runs", "/tmp/runs", "."]:
        assert resolve_results_dir(
            raw, repo_root=REPO
        ).is_absolute()


# -- what the supervisor actually resolved --


def test_the_servers_root_is_absolute_and_in_the_repo() -> None:
    """The default the maintainer's existing runs already sit in."""
    assert server.RESULTS_DIR.is_absolute()
    assert server.RESULTS_DIR == (
        server.REPO_ROOT / "results"
    )


def test_the_environment_override_reaches_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-resolved rather than re-imported: the module reads the
    environment once at import, and this checks the wiring between
    the two without reloading the supervisor mid-suite."""
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path))

    resolved = resolve_results_dir(
        os.environ.get(RESULTS_DIR_ENV),
        repo_root=server.REPO_ROOT,
    )

    assert resolved == tmp_path.resolve()


# -- what the user is shown --


def test_the_default_root_displays_as_a_short_path() -> None:
    """"results/..." is what the status line and the delete dialog
    have always said, and it should stay that way by default."""
    assert (
        server._display_run_path(server.RESULTS_DIR)
        == "results"
    )


def test_an_outside_root_displays_in_full(
    tmp_path: Path,
) -> None:
    """A --results-dir elsewhere must be named, not abbreviated
    into looking like the usual one. This is the UI half of "an
    explicit alternate data root must be isolated and named"."""
    outside = tmp_path / "elsewhere"

    shown = server._display_run_path(outside)

    assert shown == str(outside)
    assert shown.startswith("/")


def test_the_system_endpoint_reports_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How the delete dialog learns the directory it is about to
    permanently remove a run from."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    client = TestClient(server.app)

    body = client.get("/api/analytics/system").json()

    assert body["results_dir"] == str(tmp_path)
