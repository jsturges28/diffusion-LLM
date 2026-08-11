"""Tests that the new XAI signals survive the save request models.

Strategy: pin the serialized shape of a save request directly.
Entropy is kept for resolved tokens and omitted for masked ones,
per-position candidate sets keep their index alignment including the
gaps, and the pre-edit run's own signals survive alongside the edited
run's. The context block gets the same treatment plus its own
boundary: a prompt of zero tokens is a measurement and must stay
distinguishable from a run that measured nothing.

These models used to drop any field they did not declare, so an
incomplete rollout returned HTTP 200 and a run missing the signal it
was supposed to be carrying. They now refuse unknown fields, and the
tests at the bottom of this file pin that, because a loud 422 naming
the key is the difference between a bug you find in a minute and one
you find when you open the run months later.

Passing proves a saved run carries everything the durable Entropy
overlay, the candidate popover, the original-versus-edited comparison,
and the context rows need to replay it post-hoc.
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
    _context_metadata,
    _dump_alternatives,
    _dump_frame_tokens,
    manager,
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


# -- The context block --


def test_the_prompt_length_survives_the_request() -> None:
    body = SaveRunRequest(
        prompt="p",
        frames=["f"],
        final_text="hello",
        prompt_len=1240,
    )
    assert body.prompt_len == 1240


def test_a_negative_prompt_length_is_rejected() -> None:
    """The boundary. A length is a count, and a negative one would
    mean the sampler reported something impossible."""
    with pytest.raises(ValidationError):
        SaveRunRequest(
            prompt="p",
            frames=["f"],
            final_text="hello",
            prompt_len=-1,
        )


def test_the_prompt_length_defaults_to_absent() -> None:
    """Runs whose sampler predates the field send none, so absent has
    to stay distinguishable from a prompt of zero tokens."""
    body = SaveRunRequest(
        prompt="p", frames=["f"], final_text="hello"
    )
    assert body.prompt_len is None


def test_the_context_block_pairs_the_prompt_with_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both figures, because either alone answers nothing: a length
    means one thing in a 4k window and another in a 128k one."""
    monkeypatch.setattr(
        manager, "active_context_length", 65_536
    )

    block = _context_metadata(1240, None)

    assert block == {
        "prompt_tokens": 1240,
        "context_length": 65_536,
    }


def test_the_window_is_omitted_when_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint that reported no window still records what its
    prompt cost; inventing a ceiling would be worse than none."""
    monkeypatch.setattr(
        manager, "active_context_length", None
    )

    block = _context_metadata(1240, None)

    assert block == {"prompt_tokens": 1240}


def test_no_context_block_without_a_measured_length() -> None:
    """An older run gets no block at all, which is what lets the
    Analytics rows stay absent rather than reading zero."""
    assert _context_metadata(None, None) == {}


def test_an_empty_prompt_still_records_its_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary that makes None and 0 different: zero tokens is a
    measurement, and it must not be mistaken for a missing one."""
    monkeypatch.setattr(
        manager, "active_context_length", None
    )

    assert _context_metadata(0, None) == {"prompt_tokens": 0}


# -- an undeclared field is an error, not a silent loss --
#
# The failure this replaces was the quiet kind. A client sending a
# signal the server did not know about got HTTP 200 and a run saved
# without it, and the gap only showed up later as an overlay that
# would not draw. Refusing the request names the field instead.


def test_an_unknown_request_field_is_refused() -> None:
    with pytest.raises(ValidationError) as caught:
        SaveRunRequest(
            prompt="p",
            frames=["f"],
            final_text="hello",
            brand_new_signal=[1, 2, 3],  # type: ignore[call-arg]
        )

    assert "brand_new_signal" in str(caught.value)


def test_an_unknown_token_field_is_refused() -> None:
    """The one the old docstring warned about: a per-token signal
    added to the protocol but not to this model used to vanish on the
    way to tokens.json."""
    with pytest.raises(ValidationError):
        TokenRecord(
            t="he", m=False, id=5, surprisal=0.4
        )  # type: ignore[call-arg]


def test_an_unknown_candidate_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        TokenAlternative(
            id=5, t="he", p=0.9, logit=1.2
        )  # type: ignore[call-arg]


def test_an_unknown_remask_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        RemaskEdit(
            frame_index=1, token_positions=[1], canvas=0
        )  # type: ignore[call-arg]


def test_run_parameters_stay_open() -> None:
    """The deliberate exception. ``params`` is a per-model bag whose
    keys come from each model's own registry, so forbidding unknown
    keys there would mean editing this file to add a hyperparameter.
    The strictness is about the envelope, not its cargo."""
    body = SaveRunRequest(
        prompt="p",
        frames=["f"],
        final_text="hello",
        params={"anything_a_model_declares": 7},
    )

    assert body.params["anything_a_model_declares"] == 7


def test_the_payload_the_browser_actually_sends_is_accepted() -> None:
    """The paired check on all of the above.

    Strictness is only safe if it accepts the real client. This is
    every field app.js can put in a save body, taken from the payload
    construction in ``saveRun``, including the edited-run additions
    and the revision that DATA-01 introduced.
    """
    body = SaveRunRequest(
        model="llada",
        prompt="p",
        params={"steps": 128, "seed": -1},
        frames=["f"],
        final_text="hello",
        elapsed_seconds=1.5,
        per_frame_elapsed=[0.5, 1.5],
        frame_tokens=[_frame()],
        mean_conf=[0.8, None],
        prompt_len=12,
        canvas_index=[0, 0],
        alternatives=_candidates(),
        remask_edits=[
            RemaskEdit(frame_index=1, token_positions=[1])
        ],
        original_frame_tokens=[_frame()],
        original_per_frame_elapsed=[0.5],
        original_elapsed_seconds=0.5,
        original_mean_conf=[0.8],
        original_alternatives=_candidates(),
        run_id="2026-01-01_00-00-00_llada",
        expected_revision=3,
    )

    assert body.run_id == "2026-01-01_00-00-00_llada"
    assert body.expected_revision == 3
