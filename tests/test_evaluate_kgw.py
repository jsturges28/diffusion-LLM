"""Offline KGW evaluation report tests.

Strategy: feed paired JSON and JSONL run records carrying known token
evidence, probabilities, repetition, and latency, then exercise a
wrong-key hook over the same ids. Passing proves both input formats
reach length bins and paired deltas while the report stays explicit
about the semantic measurements it does not make.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from scripts import evaluate_kgw
from src.inference.kgw_key import WatermarkKey, key_id

MATCHING_SECRET = bytes(range(32))
MATCHING_KEY_ID = key_id(MATCHING_SECRET)
FINGERPRINT = "ab" * 32


def _tokens(
    *, green: int, count: int = 61
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for index in range(count):
        records.append(
            {
                "id": index % 16,
                "t": str(index % 16),
                "m": False,
                "c": 0.5,
                "g": index > 0 and index <= green,
                "we": index > 0,
            }
        )
    return records


def _run(green: int, elapsed: float) -> Dict[str, object]:
    return {
        "backend": "smollm3",
        "frame_positions": _tokens(green=green),
        "elapsed_seconds": elapsed,
        "watermark": {
            "scheme": "kgw",
            "version": 1,
            "key_id": MATCHING_KEY_ID,
            "gamma": 0.25,
            "p0": 0.25,
            "model_id": "smollm3",
            "tokenizer_fingerprint": FINGERPRINT,
            "vocab_size": 16,
            "green_list_size": 4,
        },
    }


def test_paired_json_and_jsonl_reports_measured_fields(
    tmp_path: Path,
) -> None:
    marked = tmp_path / "marked.json"
    control = tmp_path / "control.jsonl"
    marked.write_text(
        json.dumps([_run(50, 2.0), _run(45, 3.0)]),
        encoding="utf-8",
    )
    control.write_text(
        "\n".join(
            json.dumps(record)
            for record in (_run(16, 1.5), _run(14, 2.5))
        ),
        encoding="utf-8",
    )
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(marked),
            "--control",
            str(control),
            "--length-bins",
            "50,100",
        ]
    )

    report = evaluate_kgw.evaluate(options)

    marked_summary = report["groups"]["watermarked"]
    assert marked_summary["run_count"] == 2
    assert marked_summary["overall"]["scored_tokens_total"] == 120
    assert (
        marked_summary["overall"]["base_chosen_probability_mean"]
        == 0.5
    )
    assert marked_summary["overall"]["elapsed_seconds_mean"] == 2.5
    assert marked_summary["length_bins"][1]["run_count"] == 2
    assert len(report["pairs"]) == 2
    assert report["pairs"][0]["z_score_delta"] > 0
    assert "semantic quality" in report["limitations"][1]
    assert "authorship" in report["limitations"][1]


def test_wrong_key_control_recomputes_from_ids(
    tmp_path: Path,
) -> None:
    marked = tmp_path / "marked.json"
    marked.write_text(json.dumps(_run(55, 2.0)), encoding="utf-8")
    wrong = bytes(reversed(range(32))).hex()
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(marked),
            "--wrong-key-hex",
            wrong,
        ]
    )

    report = evaluate_kgw.evaluate(options)

    controls = report["wrong_key_controls"]
    assert len(controls) == 1
    runs = controls[0]["watermarked"]["runs"]
    assert runs[0]["scored_count"] == 60
    assert runs[0]["p0"] == 0.25


def test_repetition_is_reported_without_quality_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokens.json"
    repeated = _run(20, 4.0)
    for token in repeated["frame_positions"]:
        token["id"] = 3
    path.write_text(json.dumps(repeated), encoding="utf-8")
    options = evaluate_kgw._arguments(["--watermarked", str(path)])

    report = evaluate_kgw.evaluate(options)
    run = report["groups"]["watermarked"]["runs"][0]

    assert run["repeated_token_rate"] > 0.9
    assert run["adjacent_repeat_rate"] == 1.0
    assert "quality" not in run


def test_pair_refuses_different_gamma_domain(
    tmp_path: Path,
) -> None:
    marked = tmp_path / "marked.json"
    control = tmp_path / "control.json"
    marked.write_text(json.dumps(_run(20, 2.0)), encoding="utf-8")
    changed = _run(20, 2.0)
    domain = changed["watermark"]
    assert isinstance(domain, dict)
    domain.update(
        {
            "gamma": 0.5,
            "p0": 0.5,
            "green_list_size": 8,
        }
    )
    control.write_text(json.dumps(changed), encoding="utf-8")
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(marked),
            "--control",
            str(control),
        ]
    )

    with pytest.raises(ValueError, match="gamma differs"):
        evaluate_kgw.evaluate(options)


def test_partial_token_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.json"
    record = _run(20, 2.0)
    tokens = record["frame_positions"]
    assert isinstance(tokens, list)
    del tokens[2]["we"]
    path.write_text(json.dumps(record), encoding="utf-8")
    options = evaluate_kgw._arguments(["--watermarked", str(path)])

    with pytest.raises(ValueError, match="we fields are partial"):
        evaluate_kgw.evaluate(options)


def test_membership_records_require_explicit_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-evidence.json"
    record = _run(20, 2.0)
    tokens = record["frame_positions"]
    assert isinstance(tokens, list)
    for token in tokens:
        token.pop("we")
    path.write_text(json.dumps(record), encoding="utf-8")
    options = evaluate_kgw._arguments(["--watermarked", str(path)])

    with pytest.raises(ValueError, match="require explicit we"):
        evaluate_kgw.evaluate(options)


def test_raw_ids_use_explicit_evidence_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = tmp_path / "ids.json"
    evidence = tmp_path / "evidence.json"
    ids.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    evidence.write_text(
        json.dumps([False, False, True]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate_kgw,
        "load_key",
        lambda: WatermarkKey(MATCHING_SECRET, MATCHING_KEY_ID),
    )
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(ids),
            "--watermarked-evidence",
            str(evidence),
            "--model-id",
            "smollm3",
            "--tokenizer-fingerprint",
            FINGERPRINT,
            "--vocab-size",
            "16",
        ]
    )

    report = evaluate_kgw.evaluate(options)
    run = report["groups"]["watermarked"]["runs"][0]

    assert run["evidence_source"] == "evidence_sidecar"
    assert run["scored_count"] == 1
    assert run["score_source"] == "recomputed_ids"


def test_cli_domain_must_match_saved_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "missing-memberships.json"
    record = _run(20, 2.0)
    tokens = record["frame_positions"]
    assert isinstance(tokens, list)
    for token in tokens:
        token.pop("g")
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        evaluate_kgw,
        "load_key",
        lambda: WatermarkKey(MATCHING_SECRET, MATCHING_KEY_ID),
    )
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(path),
            "--model-id",
            "smollm3",
            "--tokenizer-fingerprint",
            FINGERPRINT,
            "--vocab-size",
            "16",
            "--gamma",
            "0.5",
        ]
    )

    with pytest.raises(ValueError, match="gamma differs"):
        evaluate_kgw.evaluate(options)


def test_wrong_key_must_not_equal_matching_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "marked.json"
    path.write_text(json.dumps(_run(20, 2.0)), encoding="utf-8")
    options = evaluate_kgw._arguments(
        [
            "--watermarked",
            str(path),
            "--wrong-key-hex",
            MATCHING_SECRET.hex(),
        ]
    )

    with pytest.raises(ValueError, match="matches"):
        evaluate_kgw.evaluate(options)


def test_saved_run_directory_is_direct_input(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    record = _run(20, 2.0)
    metadata = {
        key: value
        for key, value in record.items()
        if key != "frame_positions"
    }
    metadata["frame_shape"] = "append"
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (run_dir / "tokens.json").write_text(
        json.dumps(record["frame_positions"]),
        encoding="utf-8",
    )
    options = evaluate_kgw._arguments(["--watermarked", str(run_dir)])

    report = evaluate_kgw.evaluate(options)
    run = report["groups"]["watermarked"]["runs"][0]

    assert run["token_count"] == 61
    assert run["domain"]["key_id"] == MATCHING_KEY_ID


def test_input_counts_and_bins_are_bounded() -> None:
    raw = evaluate_kgw._run_record([1, 2], "raw")
    assert raw.evidence == (False, True)
    assert raw.evidence_source == "first_token_excluded_default"

    with pytest.raises(ValueError, match="input paths"):
        evaluate_kgw._load_group(
            ["unused"] * (evaluate_kgw.INPUT_PATHS_MAX + 1),
            [],
        )
    with pytest.raises(ValueError, match="token limit"):
        evaluate_kgw._run_record(
            list(range(evaluate_kgw.TOKENS_PER_RUN_MAX + 1)),
            "too-many",
        )
    bins = ",".join(
        str(index + 1)
        for index in range(evaluate_kgw.LENGTH_BINS_MAX + 1)
    )
    with pytest.raises(ValueError, match="length bins"):
        evaluate_kgw._length_bins(bins)


def test_attested_token_ids_must_fit_vocabulary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outside.json"
    record = _run(20, 2.0)
    tokens = record["frame_positions"]
    assert isinstance(tokens, list)
    tokens[-1]["id"] = 16
    path.write_text(json.dumps(record), encoding="utf-8")
    options = evaluate_kgw._arguments(["--watermarked", str(path)])

    with pytest.raises(ValueError, match="outside"):
        evaluate_kgw.evaluate(options)
