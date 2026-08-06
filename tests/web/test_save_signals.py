"""Tests that the new XAI signals survive the save request models.

Strategy: ``TokenRecord`` and ``SaveRunRequest`` are strict pydantic
models, so any field they do not declare is silently dropped on the
way to disk. That is a quiet data-loss failure mode rather than an
error, so these tests pin the serialized shape directly: entropy is
kept for resolved tokens, omitted for masked ones, per-position
candidate sets keep their index alignment (including the gaps), and
the pre-edit run's own signals survive alongside the edited run's.

Passing proves a saved run carries everything the durable Entropy
overlay, the candidate popover, and the original-versus-edited
comparison need to replay it post-hoc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.analytics.metrics import load_run_frames
from src.web.server import (
    RemaskEdit,
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


def test_a_captured_candidate_carries_no_rank() -> None:
    """The captured set's rank is its order in the list, so writing it
    would be five nulls and a repetition per position."""
    dumped = _dump_alternatives(_request().alternatives)
    assert all("rank" not in c for c in dumped[0])


def test_an_appended_candidate_keeps_its_rank() -> None:
    """The entry for a token chosen from outside the set: its place
    in the list says nothing, so the rank has to be stored."""
    positions = [
        [
            TokenAlternative(id=5, t="he", p=0.9),
            TokenAlternative(
                id=91, t="ec", p=0.0000042, rank=41203
            ),
        ]
    ]
    dumped = _dump_alternatives(positions)
    assert "rank" not in dumped[0][0]
    assert dumped[0][1]["rank"] == 41203
    assert dumped[0][1]["p"] == pytest.approx(0.0000042)


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


# -- The pre-edit run's own signals (edited runs only) --


def _edited_request() -> SaveRunRequest:
    """An edited run carrying both runs' signals."""
    return SaveRunRequest(
        prompt="p",
        frames=["frame"],
        final_text="hello",
        frame_tokens=[_frame()],
        alternatives=_candidates(),
        remask_edits=[
            RemaskEdit(frame_index=1, token_positions=[1])
        ],
        original_frame_tokens=[_frame()],
        original_per_frame_elapsed=[0.5, 1.25],
        original_elapsed_seconds=1.25,
        original_mean_conf=[0.8, None],
        original_alternatives=_candidates(),
    )


def test_original_run_signals_survive_the_request() -> None:
    body = _edited_request()
    assert body.original_per_frame_elapsed == [0.5, 1.25]
    assert body.original_elapsed_seconds == pytest.approx(1.25)
    assert body.original_mean_conf == [0.8, None]
    assert body.original_alternatives is not None


def test_original_signals_default_to_absent() -> None:
    """An unedited run sends none of them, so absent is meaningful."""
    body = SaveRunRequest(
        prompt="p", frames=["f"], final_text="hello"
    )
    assert body.original_per_frame_elapsed is None
    assert body.original_elapsed_seconds is None
    assert body.original_mean_conf is None
    assert body.original_alternatives is None


def test_original_candidates_keep_position_alignment() -> None:
    dumped = _dump_alternatives(
        _edited_request().original_alternatives
    )
    assert len(dumped) == 2
    assert dumped[1] is None
    assert [c["id"] for c in dumped[0]] == [5, 7]


def test_both_candidate_sets_reload_independently(
    tmp_path: Path,
) -> None:
    """The edited and pre-edit sets must not be confused."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )
    (run_dir / "alternatives.json").write_text(
        json.dumps([[{"id": 42, "t": "branch", "p": 1.0}]]),
        encoding="utf-8",
    )
    (run_dir / "original_alternatives.json").write_text(
        json.dumps(
            _dump_alternatives(
                _edited_request().original_alternatives
            )
        ),
        encoding="utf-8",
    )

    loaded = load_run_frames(run_dir)

    assert loaded["alternatives"][0][0]["id"] == 42
    assert loaded["original_alternatives"][0][0]["id"] == 5
    assert loaded["original_alternatives"][1] is None


def test_original_candidates_absent_on_unedited_runs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )

    loaded = load_run_frames(run_dir)

    assert loaded["original_alternatives"] is None


def test_malformed_original_candidate_file_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tokens.json").write_text(
        json.dumps([_frame()]), encoding="utf-8"
    )
    (run_dir / "original_alternatives.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_run_frames(run_dir)
