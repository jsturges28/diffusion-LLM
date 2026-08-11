"""Tests for the module that owns the saved-run directory.

Strategy: drive `run_store` directly against `tmp_path`, including
with real threads. No FastAPI, no app, no model, which is the point of
the module existing: these properties were previously only reachable
through an HTTP endpoint inside a two-thousand-line application
module, so none of them was ever tested. The first test enforces that
isolation rather than trusting it.

What passing proves is the finding's claim in reverse. Two saves of
one model in the same second used to share a directory and interleave
into a hybrid run. A failure at any file left a partial run that
Analytics listed as real. Two windows editing one run both committed,
last writer winning, and a sidecar the winner omitted stayed behind
advertising an overlay its metadata no longer described. Deletion
removed the visible folder in place, so a concurrent reader could
watch it disappear underneath itself.

The mechanism under all of it: a run is published by moving its
`metadata.json` in last, because that file is what every reader uses
to decide a directory is a run.
"""

from __future__ import annotations

import builtins
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from src.web import run_store
from src.web.run_store import (
    InvalidRunIdError,
    RevisionConflictError,
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


def _save(root: Path, **kwargs: Any) -> Tuple[str, int]:
    bundle = kwargs.pop("bundle", None) or _bundle()
    kwargs.setdefault("model_id", "llada")
    return run_store.save(root, bundle, **kwargs)


# -- the isolation the rest of this file depends on --


def test_the_store_needs_neither_a_framework_nor_a_model() -> None:
    """Reimport the module with the heavy packages poisoned.

    The finding asks for run-store tests that run "without importing
    FastAPI or model libraries". That is not a style note: it is what
    lets this file race threads and inject write failures in
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


# -- what a published run contains --


def test_a_saved_run_has_every_core_file(tmp_path: Path) -> None:
    run_id, _ = _save(tmp_path)

    names = {p.name for p in (tmp_path / run_id).iterdir()}

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
    run_id, _ = _save(
        tmp_path, bundle=_bundle(alternatives=[[{"id": 1}]])
    )

    run_dir = tmp_path / run_id
    assert (run_dir / "alternatives.json").is_file()
    assert not (run_dir / "tokens.json").exists()


def test_an_empty_sidecar_is_still_written(
    tmp_path: Path,
) -> None:
    """The paired boundary. An empty list is a measurement."""
    run_id, _ = _save(tmp_path, bundle=_bundle(alternatives=[]))

    assert (tmp_path / run_id / "alternatives.json").is_file()


def test_history_framing_survives_the_reader(
    tmp_path: Path,
) -> None:
    """Byte-compatible with what the supervisor wrote before, checked
    against the real parser rather than a regex copied into the test.
    180 saved runs are read by it."""
    from src.analytics.metrics import parse_history

    run_id, _ = _save(tmp_path)

    frames = parse_history(tmp_path / run_id / "history.txt")

    assert [f.strip() for f in frames] == [
        "frame one",
        "frame two",
    ]


# -- collision-proof identity --


def test_a_new_run_is_named_for_its_model(tmp_path: Path) -> None:
    run_id, _ = _save(tmp_path)

    assert run_id.endswith("_llada")
    assert (tmp_path / run_id).is_dir()


def test_a_model_id_with_a_slash_cannot_nest(
    tmp_path: Path,
) -> None:
    """Negative space: the id becomes a path segment, so a slash in
    it would otherwise create a subdirectory."""
    run_id, _ = _save(tmp_path, model_id="org/model")

    assert "/" not in run_id
    assert (tmp_path / run_id).parent == tmp_path


def test_same_second_saves_get_their_own_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collision the finding opens with, under a frozen clock.

    Two windows saving the same model within one second used to
    target one folder, opened with exist_ok, and write over each
    other file by file into a run that was neither of theirs.
    """
    _freeze_clock(monkeypatch)

    ids = [_save(tmp_path)[0] for _ in range(5)]

    assert len(set(ids)) == 5
    for run_id in ids:
        assert (tmp_path / run_id / "metadata.json").is_file()


def test_racing_saves_do_not_interleave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same collision from real threads, checking content rather
    than just names: every published run must be exactly one save's
    bundle, not a mixture of several."""
    _freeze_clock(monkeypatch)
    saved: List[Tuple[str, str]] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker(index: int) -> None:
        start.wait()
        marker = f"run-{index}"
        run_id, _ = run_store.save(
            tmp_path,
            _bundle(
                metadata={"backend": "llada", "marker": marker},
                final_text=marker,
                frames=[marker],
            ),
            model_id="llada",
        )
        with lock:
            saved.append((run_id, marker))

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({run_id for run_id, _ in saved}) == 8
    for run_id, marker in saved:
        run_dir = tmp_path / run_id
        meta = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert meta["marker"] == marker
        assert (
            run_dir / "final.txt"
        ).read_text(encoding="utf-8") == marker


# -- nothing partial ever becomes visible --


@pytest.mark.parametrize(
    "failing",
    [
        "metadata.json",
        "final.txt",
        "history.txt",
        "tokens.json",
    ],
)
def test_a_failure_at_any_file_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
) -> None:
    """A disk or encoding error used to leave a visible partial run,
    because files went straight into the folder Analytics reads."""
    before = set(run_store.list_run_ids(tmp_path))
    _install_write_failure(monkeypatch, failing)

    with pytest.raises(OSError):
        _save(
            tmp_path,
            bundle=_bundle(frame_tokens=[[{"t": "a"}]]),
        )

    assert set(run_store.list_run_ids(tmp_path)) == before


def test_a_failed_save_leaves_no_staging_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_write_failure(monkeypatch, "history.txt")

    with pytest.raises(OSError):
        _save(tmp_path)

    staging_root = tmp_path / run_store.STAGING_DIR_NAME
    assert not staging_root.exists() or not any(
        staging_root.iterdir()
    )


def test_a_staged_bundle_is_invisible_until_published(
    tmp_path: Path,
) -> None:
    """The property that stands in for "kill the process mid-save",
    which cannot be staged in-process. What matters after such a kill
    is that nothing half-written is listed as a run."""
    run_store.stage(tmp_path, "pending", _bundle())

    assert run_store.list_run_ids(tmp_path) == []
    with pytest.raises(RunNotFoundError):
        run_store.resolve_run_dir(tmp_path, "pending")


def test_a_reserved_directory_is_invisible(
    tmp_path: Path,
) -> None:
    """The other half of that: allocation creates the directory, and
    an empty directory is not a run until metadata lands in it."""
    run_id = run_store.allocate(tmp_path, "llada")

    assert (tmp_path / run_id).is_dir()
    assert run_store.list_run_ids(tmp_path) == []


# -- replacement is compare-and-swap --


def test_a_first_save_starts_at_revision_one(
    tmp_path: Path,
) -> None:
    _, revision = _save(tmp_path)

    assert revision == 1


def test_replacing_advances_the_revision(tmp_path: Path) -> None:
    run_id, first = _save(tmp_path)

    _, second = _save(
        tmp_path, run_id=run_id, expected_revision=first
    )

    assert second == first + 1
    assert run_store.read_revision(tmp_path, run_id) == second


def test_a_stale_replacement_is_refused(tmp_path: Path) -> None:
    """Two windows editing one run. The later writer used to win
    silently, erasing an edit its user never saw."""
    run_id, first = _save(tmp_path)
    _save(tmp_path, run_id=run_id, expected_revision=first)

    with pytest.raises(RevisionConflictError) as caught:
        _save(
            tmp_path,
            bundle=_bundle(final_text="stale"),
            run_id=run_id,
            expected_revision=first,
        )

    assert caught.value.expected == first
    assert caught.value.actual == first + 1


def test_a_refused_replacement_changes_nothing(
    tmp_path: Path,
) -> None:
    """The paired check: losing must not be destructive."""
    run_id, first = _save(tmp_path)
    _save(
        tmp_path,
        bundle=_bundle(final_text="winner"),
        run_id=run_id,
        expected_revision=first,
    )

    with pytest.raises(RevisionConflictError):
        _save(
            tmp_path,
            bundle=_bundle(final_text="loser"),
            run_id=run_id,
            expected_revision=first,
        )

    text = (tmp_path / run_id / "final.txt").read_text(
        encoding="utf-8"
    )
    assert text == "winner"


def test_only_one_of_two_racing_replacements_commits(
    tmp_path: Path,
) -> None:
    """Both from the same base revision, at the same moment."""
    run_id, base = _save(tmp_path)
    outcomes: List[str] = []
    lock = threading.Lock()
    start = threading.Barrier(2)

    def worker(name: str) -> None:
        start.wait()
        try:
            run_store.save(
                tmp_path,
                _bundle(final_text=name),
                model_id="llada",
                run_id=run_id,
                expected_revision=base,
            )
        except RevisionConflictError:
            with lock:
                outcomes.append("refused")
        else:
            with lock:
                outcomes.append("committed")

    threads = [
        threading.Thread(target=worker, args=(n,))
        for n in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["committed", "refused"]
    assert run_store.read_revision(tmp_path, run_id) == base + 1


def test_a_replacement_drops_sidecars_it_omits(
    tmp_path: Path,
) -> None:
    """A stale sidecar used to survive a replacement, advertising an
    overlay the new metadata no longer described."""
    run_id, first = _save(
        tmp_path,
        bundle=_bundle(alternatives=[[{"id": 1}]]),
    )
    assert (tmp_path / run_id / "alternatives.json").is_file()

    _save(tmp_path, run_id=run_id, expected_revision=first)

    assert not (
        tmp_path / run_id / "alternatives.json"
    ).exists()


def test_a_legacy_run_reads_as_revision_zero(
    tmp_path: Path,
) -> None:
    """None of the 180 existing runs has a revision, and they are not
    being rewritten, so absent has to mean zero rather than blocking
    the edit."""
    run_dir = tmp_path / "2026-01-01_00-00-00_llada"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"backend": "llada"}), encoding="utf-8"
    )

    assert run_store.read_revision(tmp_path, run_dir.name) == 0

    _, revision = _save(
        tmp_path, run_id=run_dir.name, expected_revision=0
    )

    assert revision == 1


def test_unreadable_metadata_reads_as_revision_zero(
    tmp_path: Path,
) -> None:
    """The caller is about to replace the file, so refusing to edit a
    run because its old metadata is corrupt would strand it."""
    run_dir = tmp_path / "2026-01-01_00-00-00_llada"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        "{not json", encoding="utf-8"
    )

    assert run_store.read_revision(tmp_path, run_dir.name) == 0


# -- deletion --


def test_deleting_removes_the_run(tmp_path: Path) -> None:
    run_id, _ = _save(tmp_path)

    run_store.delete(tmp_path, run_id)

    assert not (tmp_path / run_id).exists()
    assert run_store.list_run_ids(tmp_path) == []


def test_deleting_leaves_no_trash_behind(tmp_path: Path) -> None:
    """The rename out of the namespace is a step, not a graveyard."""
    run_id, _ = _save(tmp_path)

    run_store.delete(tmp_path, run_id)

    trash = tmp_path / run_store.TRASH_DIR_NAME
    assert not trash.exists() or not any(trash.iterdir())


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


# -- listing --


def test_listing_ignores_things_that_are_not_runs(
    tmp_path: Path,
) -> None:
    run_id, _ = _save(tmp_path)
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")
    (tmp_path / "half-written").mkdir()
    run_store.stage(tmp_path, "pending", _bundle())

    assert sorted(run_store.list_run_ids(tmp_path)) == [run_id]


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


# -- helpers --


def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the timestamp so every id collides on its base name."""

    class _Frozen:
        @staticmethod
        def now() -> Any:
            class _Stamp:
                @staticmethod
                def strftime(_fmt: str) -> str:
                    return "2026-01-01_00-00-00"

            return _Stamp()

    monkeypatch.setattr(run_store, "datetime", _Frozen)


def _install_write_failure(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """Make writing one named file raise, and nothing else."""
    real_write_text = Path.write_text

    def failing(
        self: Path, *args: Any, **kwargs: Any
    ) -> int:
        if self.name == filename:
            raise OSError(f"injected failure writing {filename}")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing)

    if filename == "history.txt":
        real_open = Path.open

        def failing_open(
            self: Path, *args: Any, **kwargs: Any
        ) -> Any:
            if self.name == filename:
                raise OSError("injected failure writing history")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failing_open)
