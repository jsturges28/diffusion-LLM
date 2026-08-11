"""Tests for the module that owns the saved-run directory.

Strategy: drive `run_store` directly against `tmp_path`. No FastAPI,
no app, no model, which is the point of the module existing: these
properties were previously only reachable through an HTTP endpoint in
a two-thousand-line application module, so they were never tested.

The first test enforces that isolation rather than trusting it. An
import of FastAPI or torch added here in a hurry would take away the
ability to run everything below, and it would do so silently.

What passing proves for this commit is narrower than it will be after
`DATA-01`: that one guarded resolver now answers for every caller,
including the metrics endpoint that had no guard, and that the
extraction moved the write path without changing what it writes.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.web import run_store
from src.web.run_store import (
    InvalidRunIdError,
    RunBundle,
    RunNotFoundError,
)

FORBIDDEN_IMPORTS = ("fastapi", "torch", "transformers", "pydantic")


def _bundle(**overrides: Any) -> RunBundle:
    base: Dict[str, Any] = {
        "metadata": {"backend": "llada", "prompt": "p"},
        "final_text": "hello",
        "frames": ["frame one", "frame two"],
    }
    base.update(overrides)
    return RunBundle(**base)


def _published(root: Path, run_id: str = "run-a") -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run_store.write_bundle(run_dir, _bundle())
    return run_dir


# -- the isolation the rest of this file depends on --


def test_the_store_needs_neither_a_framework_nor_a_model() -> None:
    """Reimport the module with the heavy packages poisoned.

    The finding asks for run-store tests that run "without importing
    FastAPI or model libraries". That is not a style note: it is what
    lets these tests race threads and inject write failures in
    milliseconds. Checking it here means the constraint survives
    somebody adding a convenient import later.
    """
    import importlib
    import sys

    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".", 1)[0]
        if root in FORBIDDEN_IMPORTS:
            raise AssertionError(
                f"run_store must not import {root}"
            )
        return real_import(name, *args, **kwargs)

    sys.modules.pop("src.web.run_store", None)
    try:
        builtins.__import__ = guarded
        importlib.import_module("src.web.run_store")
    finally:
        builtins.__import__ = real_import


# -- the one guarded resolver --


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "../../etc",
        "nested/deeper",
        "a/../../b",
        "/absolute",
    ],
)
def test_a_traversing_id_is_refused(
    tmp_path: Path, run_id: str
) -> None:
    """The guard three call sites had and a fourth did not.

    `_compute_run_metrics` joined the path unguarded, so the metrics
    endpoint would follow these out of the data root while delete and
    frames refused them.
    """
    (tmp_path / "escape").mkdir()

    with pytest.raises(InvalidRunIdError):
        run_store.resolve_run_dir(tmp_path, run_id)


def test_a_symlink_out_of_the_root_is_refused(
    tmp_path: Path,
) -> None:
    """Resolution happens before the containment check, so a link is
    judged by where it lands rather than where it sits."""
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "results"
    root.mkdir()
    (root / "sneaky").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidRunIdError):
        run_store.resolve_run_dir(root, "sneaky")


def test_a_missing_run_is_not_found(tmp_path: Path) -> None:
    """Distinct from a malformed id, because the routes answer them
    with different status codes."""
    with pytest.raises(RunNotFoundError):
        run_store.resolve_run_dir(tmp_path, "never-existed")


def test_the_two_failures_keep_their_builtin_meanings() -> None:
    """The routes catch FileNotFoundError and ValueError already, so
    the named types have to slot into those without a rewrite."""
    assert issubclass(RunNotFoundError, FileNotFoundError)
    assert issubclass(InvalidRunIdError, ValueError)


def test_a_real_run_resolves(tmp_path: Path) -> None:
    _published(tmp_path)

    resolved = run_store.resolve_run_dir(tmp_path, "run-a")

    assert resolved == (tmp_path / "run-a").resolve()


# -- writing a bundle --


def test_every_core_file_is_written(tmp_path: Path) -> None:
    run_dir = _published(tmp_path)

    names = {p.name for p in run_dir.iterdir()}

    assert names == {
        "metadata.json",
        "final.txt",
        "history.txt",
    }


def test_sidecars_are_written_only_when_present(
    tmp_path: Path,
) -> None:
    """Absent is meaningful: a run that captured no candidates must
    not be indistinguishable from one that captured an empty set."""
    run_dir = tmp_path / "run-b"
    run_dir.mkdir()

    run_store.write_bundle(
        run_dir, _bundle(alternatives=[[{"id": 1}]])
    )

    assert (run_dir / "alternatives.json").is_file()
    assert not (run_dir / "tokens.json").exists()
    assert not (run_dir / "original_alternatives.json").exists()


def test_an_empty_sidecar_is_still_written(
    tmp_path: Path,
) -> None:
    """The paired boundary. An empty list is a measurement."""
    run_dir = tmp_path / "run-c"
    run_dir.mkdir()

    run_store.write_bundle(run_dir, _bundle(alternatives=[]))

    assert (run_dir / "alternatives.json").is_file()


def test_metadata_round_trips(tmp_path: Path) -> None:
    run_dir = _published(tmp_path)

    loaded = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert loaded == {"backend": "llada", "prompt": "p"}


def test_history_keeps_the_delimiter_the_reader_expects(
    tmp_path: Path,
) -> None:
    """Byte-compatible with what the supervisor wrote before, because
    180 saved runs are parsed by the same regex."""
    run_dir = _published(tmp_path)

    text = (run_dir / "history.txt").read_text(encoding="utf-8")

    assert "===== FRAME 0 =====" in text
    assert "===== FRAME 1 =====" in text
    assert "frame one" in text


def test_history_framing_survives_the_reader(
    tmp_path: Path,
) -> None:
    """The paired check, against the real parser rather than a regex
    copied into the test."""
    from src.analytics.metrics import parse_history

    run_dir = _published(tmp_path)

    frames = parse_history(run_dir / "history.txt")

    assert [f.strip() for f in frames] == [
        "frame one",
        "frame two",
    ]


# -- allocation and deletion --


def test_a_new_run_directory_is_named_for_its_model(
    tmp_path: Path,
) -> None:
    run_dir = run_store.make_run_dir(tmp_path, "llada")

    assert run_dir.is_dir()
    assert run_dir.name.endswith("_llada")
    assert run_dir.parent == tmp_path


def test_a_model_id_with_a_slash_cannot_nest(
    tmp_path: Path,
) -> None:
    """Negative space: the id reaches a path, so a slash in it would
    otherwise create a subdirectory."""
    run_dir = run_store.make_run_dir(tmp_path, "org/model")

    assert run_dir.parent == tmp_path
    assert "/" not in run_dir.name.split("_", 2)[-1]


def test_deleting_removes_the_run(tmp_path: Path) -> None:
    _published(tmp_path)

    run_store.delete(tmp_path, "run-a")

    assert not (tmp_path / "run-a").exists()


def test_deleting_a_traversing_id_is_refused(
    tmp_path: Path,
) -> None:
    """The guard matters most here: this call ends in rmtree."""
    victim = tmp_path.parent / "not-a-run"
    victim.mkdir(exist_ok=True)
    root = tmp_path / "results"
    root.mkdir()

    with pytest.raises(InvalidRunIdError):
        run_store.delete(root, "../not-a-run")

    assert victim.is_dir()


# -- how the UI names a run --


def test_a_run_under_the_repo_displays_short(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "run-a"
    run_dir.mkdir(parents=True)

    shown = run_store.display_path(run_dir, tmp_path)

    assert shown == "results/run-a"


def test_a_run_outside_the_repo_displays_in_full(
    tmp_path: Path,
) -> None:
    """A --results-dir elsewhere must be named, not abbreviated into
    looking like the usual place."""
    outside = tmp_path / "elsewhere" / "run-a"
    outside.mkdir(parents=True)

    shown = run_store.display_path(outside, tmp_path / "repo")

    assert shown == str(outside)


def test_the_sidecar_table_matches_the_bundle(
    tmp_path: Path,
) -> None:
    """Every named sidecar has to be a real field, or a payload would
    be silently dropped on the way to disk."""
    bundle = _bundle()

    for attribute, filename in run_store.SIDECAR_NAMES:
        assert hasattr(bundle, attribute), attribute
        assert filename.endswith(".json"), filename


def test_writing_into_a_missing_directory_is_a_programmer_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError):
        run_store.write_bundle(tmp_path / "absent", _bundle())


def test_listing_ignores_things_that_are_not_runs(
    tmp_path: Path,
) -> None:
    """A file, and a directory with no metadata, are both not runs.

    The second case is what makes `DATA-01`'s staging invisible, so
    it is worth pinning before that lands."""
    _published(tmp_path, "run-a")
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")
    (tmp_path / "half-written").mkdir()

    found: List[str] = sorted(run_store.list_run_ids(tmp_path))

    assert found == ["run-a"]
