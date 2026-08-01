"""Tests that the new xAI signals survive the save request models.

Strategy: ``TokenRecord`` is a strict pydantic model, so any field it
does not declare is silently dropped on the way to ``tokens.json``.
That is a quiet data-loss failure mode rather than an error, so these
tests pin the serialized shape directly: entropy is kept for resolved
tokens, omitted for masked ones, and per-position candidate sets keep
their index alignment (including the gaps).

Passing proves a saved run carries everything the durable Entropy
overlay and the candidate popover need to replay it post-hoc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.analytics.metrics import load_run_frames
from src.web.server import (
    SaveRunRequest,
    TokenAlternative,
    TokenRecord,
    _dump_alternatives,
    _dump_frame_tokens,
)


def _frame() -> List[Dict[str, Any]]:
    return [
        {"t": "he", "m": False, "id": 5, "c": 0.9, "e": 0.31},
        {"t": "\u2591", "m": True, "id": 2},
    ]


def _candidates() -> List[Any]:
    return [
        [
            {"id": 5, "t": "he", "p": 0.9},
            {"id": 7, "t": "she", "p": 0.05},
        ],
        None,
    ]


def _request() -> SaveRunRequest:
    return SaveRunRequest(
        prompt="p",
        frames=["frame"],
        final_text="hello",
        frame_tokens=[_frame()],
        alternatives=_candidates(),
    )


def test_entropy_survives_the_token_record() -> None:
    record = TokenRecord(t="he", m=False, id=5, c=0.9, e=0.31)
    assert record.e == pytest.approx(0.31)


def test_entropy_is_optional_for_models_without_it() -> None:
    record = TokenRecord(t="he", m=False, id=5, c=0.9)
    assert record.e is None


def test_dumped_records_keep_entropy() -> None:
    dumped = _dump_frame_tokens(_request().frame_tokens)
    assert dumped[0][0]["e"] == pytest.approx(0.31)


def test_dumped_masked_records_stay_compact() -> None:
    """Masked positions have no entropy, so none is written."""
    dumped = _dump_frame_tokens(_request().frame_tokens)
    assert "e" not in dumped[0][1]
    assert "c" not in dumped[0][1]


def test_alternatives_keep_position_alignment() -> None:
    dumped = _dump_alternatives(_request().alternatives)
    assert len(dumped) == 2
    assert dumped[1] is None
    assert [c["id"] for c in dumped[0]] == [5, 7]


def test_alternative_requires_all_three_fields() -> None:
    with pytest.raises(ValidationError):
        TokenAlternative(id=5, t="he")  # type: ignore[call-arg]


def test_alternatives_default_to_absent() -> None:
    body = SaveRunRequest(
        prompt="p", frames=["f"], final_text="hello"
    )
    assert body.alternatives is None


def test_saved_signals_reload_for_analytics(
    tmp_path: Path,
) -> None:
    """The loader's paired check on the serialized round-trip."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = _request()
    (run_dir / "tokens.json").write_text(
        json.dumps(_dump_frame_tokens(body.frame_tokens)),
        encoding="utf-8",
    )
    (run_dir / "alternatives.json").write_text(
        json.dumps(_dump_alternatives(body.alternatives)),
        encoding="utf-8",
    )

    loaded = load_run_frames(run_dir)

    assert loaded["records_available"] is True
    assert loaded["alternatives_available"] is True
    assert loaded["frames"][0][0]["e"] == pytest.approx(0.31)
    assert loaded["alternatives"][1] is None


def test_runs_without_candidates_report_unavailable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )

    loaded = load_run_frames(run_dir)

    assert loaded["alternatives"] is None
    assert loaded["alternatives_available"] is False


def test_all_empty_candidate_sets_report_unavailable(
    tmp_path: Path,
) -> None:
    """A file of only gaps is no more useful than no file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )
    (run_dir / "alternatives.json").write_text(
        json.dumps([None, []]), encoding="utf-8"
    )

    loaded = load_run_frames(run_dir)

    assert loaded["alternatives_available"] is False


def test_malformed_candidate_file_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )
    (run_dir / "alternatives.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_run_frames(run_dir)
