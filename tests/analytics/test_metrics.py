"""Tests for durable per-token frame loading (analytics overlays).

Strategy: ``load_run_frames`` reads ``tokens.json`` (and optional
``original_tokens.json``) from a run directory. These tests build tiny
run dirs in a tmp path and assert the loader distinguishes rich
per-token records from legacy id-only files, surfaces the original
snapshot when present, tolerates missing files, and rejects malformed
ones. Passing proves the analytics frames endpoint receives correctly
shaped, safe data (and that legacy runs degrade gracefully).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analytics.metrics import list_runs, load_run_frames


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
