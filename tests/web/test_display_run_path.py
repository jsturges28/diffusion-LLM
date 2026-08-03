"""Tests for the saved-run path shown in the generator's status line.

Strategy: ``/api/save`` reaches its run folder two ways. A fresh save
joins the relative ``RESULTS_DIR`` and stays relative; an edited save
updates the original in place via ``_existing_run_dir``, which must
``resolve()`` for its traversal guard and so hands back an absolute
path. Reporting either raw made one save read "results/..." and the
next "/home/you/.../results/...". ``_display_run_path`` normalizes
both at the single point they meet.

These tests feed it the shapes each branch produces, plus a path
outside the repo, and assert the two in-repo forms collapse to one
short string while the outsider degrades to its own full path rather
than raising. Passing proves the status line reads the same regardless
of which branch saved the run.
"""

from __future__ import annotations

from pathlib import Path

from src.web.server import REPO_ROOT, _display_run_path

RUN_NAME = "2026-08-03_02-45-12_HuggingFaceTB_SmolLM3-3B"


def test_relative_path_is_returned_unchanged() -> None:
    """The fresh-save branch's shape is already what we want."""
    relative = Path("results") / RUN_NAME

    assert _display_run_path(relative) == f"results/{RUN_NAME}"


def test_absolute_path_inside_repo_is_shortened() -> None:
    """The in-place-update branch's shape collapses to match it."""
    absolute = REPO_ROOT / "results" / RUN_NAME

    assert _display_run_path(absolute) == f"results/{RUN_NAME}"


def test_both_branches_agree() -> None:
    """The whole point: one run, one string, however it saved."""
    relative = Path("results") / RUN_NAME
    absolute = REPO_ROOT / "results" / RUN_NAME

    assert _display_run_path(relative) == _display_run_path(absolute)


def test_path_outside_repo_falls_back_to_full_path(
    tmp_path: Path,
) -> None:
    """An operating condition, not a broken invariant: no raise.

    A ``results`` directory symlinked elsewhere, or a server started
    from another working directory, lands here.
    """
    outside = tmp_path / "results" / RUN_NAME

    assert _display_run_path(outside) == str(outside)


def test_repo_root_itself_is_the_boundary() -> None:
    """The valid/invalid edge: the root is in, its parent is out."""
    outside = REPO_ROOT.parent

    assert _display_run_path(REPO_ROOT) == "."
    assert _display_run_path(outside) == str(outside)
