"""Tests for durable per-token frame loading (analytics overlays).

Strategy: ``load_run_frames`` reads ``tokens.json`` (and optional
``original_tokens.json``) from a run directory. These tests build tiny
run dirs in a tmp path and assert the loader distinguishes rich
per-token records from legacy id-only files, surfaces the original
snapshot when present, tolerates missing files, and rejects malformed
ones. Passing proves the analytics frames endpoint receives correctly
shaped, safe data (and that legacy runs degrade gracefully).

The second half covers ``total_elapsed_seconds`` and its use in
``list_runs``, which repairs the whole-run duration of edited runs
saved before the elapsed series was made cumulative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analytics.metrics import (
    list_runs,
    load_run_frames,
    total_elapsed_seconds,
)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _record_frame() -> List[Dict[str, Any]]:
    return [
        {"t": "he", "m": False, "id": 5, "c": 0.9},
        {"t": "llo", "m": False, "id": 6, "c": 0.8},
    ]


def test_load_run_frames_reads_rich_records(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frames = [_record_frame(), _record_frame()]
    _write_json(run_dir / "tokens.json", frames)

    result = load_run_frames(run_dir)

    assert result["records_available"] is True
    assert result["frames"] == frames
    assert result["original_frames"] is None


def test_load_run_frames_includes_original_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "tokens.json", [_record_frame()])
    original = [_record_frame()]
    _write_json(run_dir / "original_tokens.json", original)

    result = load_run_frames(run_dir)

    assert result["original_frames"] == original


def test_load_run_frames_flags_legacy_id_only_stream(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "tokens.json", [[1, 2, 3], [1, 2, 3]]
    )

    result = load_run_frames(run_dir)

    assert result["records_available"] is False
    assert result["frames"] == [[1, 2, 3], [1, 2, 3]]


def test_load_run_frames_skips_empty_leading_frames(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frames = [None, [], _record_frame()]
    _write_json(run_dir / "tokens.json", frames)

    result = load_run_frames(run_dir)

    assert result["records_available"] is True


def test_load_run_frames_missing_tokens_returns_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = load_run_frames(run_dir)

    assert result["frames"] is None
    assert result["original_frames"] is None
    assert result["records_available"] is False


def test_load_run_frames_rejects_malformed_tokens(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "tokens.json", {"not": "a list"})

    with pytest.raises(ValueError):
        load_run_frames(run_dir)


def _write_run(
    results_dir: Path, name: str, *, with_original: bool
) -> None:
    run_dir = results_dir / name
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {"created_at": "2026-01-01T00:00:00", "prompt": "hi"},
    )
    _write_json(run_dir / "tokens.json", [_record_frame()])
    if with_original:
        _write_json(
            run_dir / "original_tokens.json", [_record_frame()]
        )


def test_list_runs_flags_has_diff_from_original_snapshot(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "2026-01-01_edited", with_original=True)
    _write_run(tmp_path, "2026-01-02_plain", with_original=False)

    runs = list_runs(tmp_path)

    by_id = {run["run_id"]: run for run in runs}
    assert by_id["2026-01-01_edited"]["has_diff"] is True
    assert by_id["2026-01-02_plain"]["has_diff"] is False


# -- Whole-run elapsed across resumed segments --
#
# The worker restarts its clock per generate/resume/substitute
# segment. Runs edited before the client began carrying that offset
# forward have an elapsed series that drops at each branch, and stored
# an elapsed_seconds covering only the last segment. These pin both
# the repair and its no-op behaviour on already-cumulative series,
# since it is applied unconditionally to every run.


def test_total_elapsed_is_the_last_sample_when_monotonic() -> None:
    assert total_elapsed_seconds([1.0, 2.0, 3.25]) == pytest.approx(
        3.25
    )


def test_total_elapsed_sums_across_one_resume() -> None:
    """3.0s of original plus 1.5s of branch, not the 1.5s alone."""
    assert total_elapsed_seconds(
        [1.0, 2.0, 3.0, 0.5, 1.5]
    ) == pytest.approx(4.5)


def test_total_elapsed_sums_across_repeated_resumes() -> None:
    assert total_elapsed_seconds(
        [1.0, 2.0, 0.4, 0.9, 0.2, 0.7]
    ) == pytest.approx(3.6)


def test_total_elapsed_is_idempotent_on_a_repaired_series() -> None:
    """Re-running the repair on its own output must not inflate it."""
    once = total_elapsed_seconds([1.0, 2.0, 3.0, 0.5, 1.5])
    assert once is not None
    assert total_elapsed_seconds(
        [1.0, 2.0, 3.0, 3.5, once]
    ) == pytest.approx(once)


def test_total_elapsed_handles_a_single_frame() -> None:
    assert total_elapsed_seconds([2.5]) == pytest.approx(2.5)


def test_total_elapsed_reports_none_for_an_empty_series() -> None:
    assert total_elapsed_seconds([]) is None


@pytest.mark.parametrize(
    "series",
    [
        "not a list",
        None,
        [1.0, None],
        [1.0, "2.0"],
        [1.0, True],
    ],
)
def test_total_elapsed_refuses_non_numeric_series(
    series: Any,
) -> None:
    """None tells the caller to keep whatever was stored."""
    assert total_elapsed_seconds(series) is None


def test_list_runs_repairs_a_legacy_edited_elapsed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "2026-01-01_edited"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {
            "created_at": "2026-01-01T00:00:00",
            "prompt": "hi",
            # Stored by the pre-fix client: the branch's own timer.
            "elapsed_seconds": 1.5,
            "per_frame_elapsed": [1.0, 2.0, 3.0, 0.5, 1.5],
        },
    )

    runs = list_runs(tmp_path)

    assert runs[0]["elapsed_seconds"] == pytest.approx(4.5)


def test_list_runs_leaves_a_cumulative_elapsed_alone(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "2026-01-02_plain"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {
            "created_at": "2026-01-02T00:00:00",
            "prompt": "hi",
            "elapsed_seconds": 3.0,
            "per_frame_elapsed": [1.0, 2.0, 3.0],
        },
    )

    runs = list_runs(tmp_path)

    assert runs[0]["elapsed_seconds"] == pytest.approx(3.0)


def test_list_runs_keeps_elapsed_when_timing_is_missing(
    tmp_path: Path,
) -> None:
    """No timing array is not a reason to discard a stored total."""
    run_dir = tmp_path / "2026-01-03_legacy"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {
            "created_at": "2026-01-03T00:00:00",
            "prompt": "hi",
            "elapsed_seconds": 7.5,
        },
    )

    runs = list_runs(tmp_path)

    assert runs[0]["elapsed_seconds"] == pytest.approx(7.5)
