"""Evaluate paired KGW run records without semantic-quality claims.

Inputs may be saved-run directories, JSON, or JSONL. Each record can
be a saved-run object, an Analytics frames payload, a flat
token-record array, or a token-id array. Strict bounds cover paths,
bytes, lines, runs, tokens, bins, and wrong keys. Reports retain each
complete watermark domain and evidence source, and paired deltas
require identical domains.

This script does not measure factuality, fluency, usefulness, or
authorship. Those are intentionally absent from its output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.kgw_key import (  # noqa: E402
    WatermarkKey,
    key_id,
    load_key,
)
from src.inference.kgw_watermark import (  # noqa: E402
    KGW_SCHEME,
    KGW_VERSION,
    DetectionResult,
    KgwConfig,
    detect_token_ids,
    detection_status,
    detection_z_score,
    green_list_size,
    null_probability,
)

LENGTH_BINS_DEFAULT = (50, 100, 200, 500)
WRONG_KEYS_MAX = 16
INPUT_PATHS_MAX = 64
INPUT_PATH_CHARS_MAX = 4_096
INPUT_FILE_BYTES_MAX = 64 * 1024 * 1024
JSONL_LINES_MAX = 4_096
JSONL_LINE_BYTES_MAX = 2 * 1024 * 1024
RUNS_MAX = 1_024
TOKENS_PER_RUN_MAX = 4_096
LENGTH_BINS_MAX = 64

assert INPUT_PATHS_MAX > 0
assert INPUT_FILE_BYTES_MAX > JSONL_LINE_BYTES_MAX
assert JSONL_LINES_MAX >= RUNS_MAX
assert max(LENGTH_BINS_DEFAULT) <= TOKENS_PER_RUN_MAX


@dataclass(frozen=True)
class WatermarkDomain:
    """Every value needed to interpret one membership trace."""

    scheme: str
    version: int
    key_id: str
    gamma: float
    p0: float
    model_id: str
    tokenizer_fingerprint: str
    vocab_size: int
    green_list_size: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "scheme": self.scheme,
            "version": self.version,
            "key_id": self.key_id,
            "gamma": self.gamma,
            "p0": self.p0,
            "model_id": self.model_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "vocab_size": self.vocab_size,
            "green_list_size": self.green_list_size,
        }


@dataclass(frozen=True)
class RunRecord:
    """One run's token-side measurements."""

    source: str
    token_ids: Tuple[int, ...]
    memberships: Tuple[Optional[bool], ...]
    evidence: Tuple[bool, ...]
    probabilities: Tuple[float, ...]
    elapsed_seconds: Optional[float]
    domain: Optional[WatermarkDomain]
    evidence_source: str


@dataclass(frozen=True)
class RunScore:
    """One run reduced to reportable measurements."""

    source: str
    token_count: int
    detection: DetectionResult
    repeated_token_rate: float
    adjacent_repeat_rate: float
    probability_mean: Optional[float]
    elapsed_seconds: Optional[float]
    tokens_per_second: Optional[float]
    domain: WatermarkDomain
    score_source: str
    evidence_source: str


def _arguments(
    arguments: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired KGW token records. No semantic quality"
            " or authorship claim is produced."
        )
    )
    parser.add_argument(
        "--watermarked",
        action="append",
        required=True,
        help=(
            "Saved-run directory, JSON, or JSONL watermarked"
            " run/token records."
        ),
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help=(
            "Paired saved-run directory, JSON, or JSONL control"
            " records."
        ),
    )
    parser.add_argument(
        "--watermarked-evidence",
        action="append",
        default=[],
        help=(
            "JSON or JSONL boolean arrays, one per watermarked"
            " raw-id run. Explicit token-record we cannot be"
            " overridden."
        ),
    )
    parser.add_argument(
        "--control-evidence",
        action="append",
        default=[],
        help=(
            "JSON or JSONL boolean arrays, one per control raw-id"
            " run. Explicit token-record we cannot be overridden."
        ),
    )
    parser.add_argument("--model-id")
    parser.add_argument("--tokenizer-fingerprint")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--expected-key-id")
    parser.add_argument(
        "--wrong-key-hex",
        action="append",
        default=[],
        help="Optional 64-hex wrong key for a control score.",
    )
    parser.add_argument(
        "--length-bins",
        default=",".join(str(value) for value in LENGTH_BINS_DEFAULT),
        help="Comma-separated scored-token upper boundaries.",
    )
    parser.add_argument(
        "--output",
        help="Write JSON here instead of stdout.",
    )
    return parser.parse_args(arguments)


def _read_path(path: Path) -> List[object]:
    """Read one bounded JSON/JSONL file or saved-run directory."""
    _validate_path(path)
    if path.is_dir():
        return [_read_saved_run(path)]
    text = _read_text_bounded(path)
    if path.suffix.lower() != ".jsonl":
        return _records_from_document(json.loads(text))
    return _jsonl_records(path, text)


def _validate_path(path: Path) -> None:
    encoded = str(path)
    if len(encoded) > INPUT_PATH_CHARS_MAX:
        raise ValueError(
            f"input path exceeds {INPUT_PATH_CHARS_MAX} characters"
        )
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(
            f"input path is not a file or directory: {path}"
        )


def _read_text_bounded(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    size = path.stat().st_size
    if size > INPUT_FILE_BYTES_MAX:
        raise ValueError(
            f"{path}: file is {size} bytes; limit is"
            f" {INPUT_FILE_BYTES_MAX}"
        )
    return path.read_text(encoding="utf-8")


def _jsonl_records(path: Path, text: str) -> List[object]:
    lines = text.splitlines()
    if len(lines) > JSONL_LINES_MAX:
        raise ValueError(
            f"{path}: JSONL holds {len(lines)} lines; limit is"
            f" {JSONL_LINES_MAX}"
        )
    values: List[object] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line.encode("utf-8")) > JSONL_LINE_BYTES_MAX:
            raise ValueError(
                f"{path}:{line_number}: line exceeds"
                f" {JSONL_LINE_BYTES_MAX} bytes"
            )
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
    return values


def _read_saved_run(path: Path) -> Dict[str, object]:
    """Merge one durable run's metadata and final token records."""
    metadata_path = path / "metadata.json"
    tokens_path = path / "tokens.json"
    metadata = json.loads(_read_text_bounded(metadata_path))
    stored = json.loads(_read_text_bounded(tokens_path))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    if not isinstance(stored, list):
        raise ValueError(f"{tokens_path}: expected a JSON array")
    tokens = _saved_run_tokens(metadata, stored, tokens_path)
    merged: Dict[str, object] = dict(metadata)
    merged["positions"] = tokens
    return merged


def _saved_run_tokens(
    metadata: Mapping[str, object],
    stored: List[object],
    path: Path,
) -> List[object]:
    if metadata.get("frame_shape") == "append":
        if not all(isinstance(item, dict) for item in stored):
            raise ValueError(
                f"{path}: append records must be objects"
            )
        return stored
    for frame in reversed(stored):
        if isinstance(frame, list) and all(
            isinstance(item, dict) for item in frame
        ):
            return frame
    raise ValueError(f"{path}: no final token frame found")


def _records_from_document(value: object) -> List[object]:
    if not isinstance(value, list):
        return [value]
    if not value:
        return [value]
    if all(isinstance(item, int) for item in value):
        return [value]
    if all(isinstance(item, dict) and "id" in item for item in value):
        return [value]
    return list(value)


def _load_group(
    paths: Sequence[str],
    evidence_paths: Sequence[str],
) -> List[RunRecord]:
    _check_path_count(paths, "input")
    _check_path_count(evidence_paths, "evidence")
    runs: List[RunRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        values = _read_path(path)
        for index, value in enumerate(values):
            label = str(path)
            if len(values) > 1:
                label = f"{path}#{index + 1}"
            runs.append(_run_record(value, label))
            if len(runs) > RUNS_MAX:
                raise ValueError(
                    f"a group may contain at most {RUNS_MAX} runs"
                )
    return _apply_evidence_sidecars(runs, evidence_paths)


def _check_path_count(paths: Sequence[str], kind: str) -> None:
    if len(paths) > INPUT_PATHS_MAX:
        raise ValueError(
            f"at most {INPUT_PATHS_MAX} {kind} paths are allowed"
        )


def _apply_evidence_sidecars(
    runs: List[RunRecord],
    paths: Sequence[str],
) -> List[RunRecord]:
    if not paths:
        return runs
    arrays: List[List[bool]] = []
    for raw_path in paths:
        arrays.extend(_evidence_documents(Path(raw_path)))
    if len(arrays) != len(runs):
        raise ValueError(
            "evidence sidecars must contain one array per run"
        )
    applied: List[RunRecord] = []
    for run, evidence in zip(runs, arrays, strict=True):
        if run.evidence_source == "token_records":
            raise ValueError(
                f"{run.source}: token-record we fields cannot be"
                " overridden by an evidence sidecar"
            )
        checked = _evidence_values(evidence, len(run.token_ids))
        applied.append(
            replace(
                run,
                evidence=tuple(checked),
                evidence_source="evidence_sidecar",
            )
        )
    return applied


def _evidence_documents(path: Path) -> List[List[bool]]:
    values = _read_path(path)
    if len(values) == 1 and _is_boolean_array(values[0]):
        only = values[0]
        assert isinstance(only, list)
        return [list(only)]
    arrays: List[List[bool]] = []
    for value in values:
        if not _is_boolean_array(value):
            raise ValueError(
                f"{path}: every evidence record must be a"
                " boolean array"
            )
        arrays.append(list(value))
    return arrays


def _is_boolean_array(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, bool) for item in value
    )


def _run_record(value: object, source: str) -> RunRecord:
    """Normalize one input shape into an immutable record."""
    owner = value if isinstance(value, dict) else {}
    tokens = _token_records(value)
    token_ids = _token_ids(value, tokens)
    if not token_ids:
        raise ValueError(f"{source}: no token ids found")
    if len(token_ids) > TOKENS_PER_RUN_MAX:
        raise ValueError(
            f"{source}: {len(token_ids)} tokens exceeds the"
            f" {TOKENS_PER_RUN_MAX}-token limit"
        )
    evidence, evidence_source = _evidence(
        owner, tokens, len(token_ids), source
    )
    memberships = _memberships(tokens, len(token_ids))
    probabilities = _probabilities(tokens)
    return RunRecord(
        source=source,
        token_ids=tuple(token_ids),
        memberships=tuple(memberships),
        evidence=tuple(evidence),
        probabilities=tuple(probabilities),
        elapsed_seconds=_elapsed_seconds(owner),
        domain=_record_domain(owner, source),
        evidence_source=evidence_source,
    )


def _token_records(value: object) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return list(value)
        return []
    if not isinstance(value, dict):
        raise ValueError("a run record must be an object or array")
    for name in (
        "frame_positions",
        "positions",
        "token_records",
        "tokens",
        "records",
    ):
        records = value.get(name)
        if _is_token_records(records):
            return list(records)
    for name in ("frame_tokens", "frames"):
        frames = value.get(name)
        final = _final_token_frame(frames)
        if final is not None:
            return final
    nested = value.get("run")
    if isinstance(nested, dict):
        return _token_records(nested)
    return []


def _is_token_records(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and "id" in item for item in value
    )


def _final_token_frame(
    value: object,
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return None
    for frame in reversed(value):
        if _is_token_records(frame):
            return list(frame)
    return None


def _token_ids(
    value: object, tokens: List[Dict[str, Any]]
) -> List[int]:
    if tokens:
        raw = [token.get("id") for token in tokens]
    elif isinstance(value, list):
        raw = value
    elif isinstance(value, dict):
        raw = []
        for name in ("ids", "token_ids", "tokens"):
            candidate = value.get(name)
            if isinstance(candidate, list):
                raw = candidate
                break
    else:
        raw = []
    ids: List[int] = []
    for token_id in raw:
        if isinstance(token_id, bool):
            raise ValueError("every token id must be an integer")
        if not isinstance(token_id, int):
            raise ValueError("every token id must be an integer")
        ids.append(token_id)
    return ids


def _evidence(
    owner: Mapping[str, object],
    tokens: List[Dict[str, Any]],
    count: int,
    source: str,
) -> Tuple[List[bool], str]:
    present = ["we" in token for token in tokens]
    if any(present):
        if not all(present):
            raise ValueError(
                f"{source}: token-record we fields are partial"
            )
        values = [token["we"] for token in tokens]
        return (
            _evidence_values(values, count),
            "token_records",
        )
    if any("g" in token for token in tokens):
        raise ValueError(
            f"{source}: watermarked token records require explicit we"
        )
    explicit = _explicit_evidence(owner)
    if explicit is not None:
        return (
            _evidence_values(explicit, count),
            "explicit_array",
        )
    return _default_evidence(count), "first_token_excluded_default"


def _explicit_evidence(
    owner: Mapping[str, object],
) -> Optional[List[object]]:
    for name in ("evidence", "watermark_evidence"):
        value = owner.get(name)
        if isinstance(value, list):
            return value
    return None


def _evidence_values(
    value: Sequence[object], count: int
) -> List[bool]:
    if len(value) != count:
        raise ValueError("evidence length differs from token ids")
    if not all(isinstance(item, bool) for item in value):
        raise ValueError("every evidence value must be boolean")
    return [bool(item) for item in value]


def _default_evidence(count: int) -> List[bool]:
    if count == 0:
        return []
    return [False, *([True] * (count - 1))]


def _memberships(
    tokens: List[Dict[str, Any]], count: int
) -> List[Optional[bool]]:
    if not tokens:
        return [None] * count
    present = ["g" in token for token in tokens]
    if any(present) and not all(present):
        raise ValueError("token-record g fields are partial")
    if not any(present):
        return [None] * count
    memberships: List[Optional[bool]] = []
    for token in tokens:
        value = token.get("g")
        if not isinstance(value, bool):
            raise ValueError(
                "every token-record g value must be boolean"
            )
        memberships.append(value)
    return memberships


def _probabilities(tokens: List[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for token in tokens:
        value = token.get("c")
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            values.append(number)
    return values


def _elapsed_seconds(record: Dict[str, Any]) -> Optional[float]:
    candidates: List[object] = [
        record.get("elapsed_seconds"),
        _nested(record, "metadata", "elapsed_seconds"),
    ]
    per_frame = record.get("per_frame_elapsed")
    if isinstance(per_frame, list) and per_frame:
        candidates.append(per_frame[-1])
    for value in candidates:
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        seconds = float(value)
        if math.isfinite(seconds) and seconds >= 0.0:
            return seconds
    return None


def _record_domain(
    record: Mapping[str, object],
    source: str,
) -> Optional[WatermarkDomain]:
    attestation = _domain_attestation(record)
    if attestation is None:
        return None
    model_id = _domain_model_id(record, attestation)
    if model_id is None:
        raise ValueError(
            f"{source}: watermark attestation has no model id"
        )
    try:
        domain = WatermarkDomain(
            scheme=_domain_string(attestation["scheme"], "scheme"),
            version=_domain_integer(
                attestation["version"], "version"
            ),
            key_id=_domain_string(attestation["key_id"], "key_id"),
            gamma=_domain_float(attestation["gamma"], "gamma"),
            p0=_domain_float(attestation["p0"], "p0"),
            model_id=model_id,
            tokenizer_fingerprint=_domain_string(
                attestation["tokenizer_fingerprint"],
                "tokenizer_fingerprint",
            ),
            vocab_size=_domain_integer(
                attestation["vocab_size"], "vocab_size"
            ),
            green_list_size=_domain_integer(
                attestation["green_list_size"],
                "green_list_size",
            ),
        )
    except KeyError as exc:
        raise ValueError(
            f"{source}: watermark domain lacks {exc.args[0]}"
        ) from None
    _validate_domain(domain, source)
    return domain


def _domain_attestation(
    record: Mapping[str, object],
) -> Optional[Mapping[str, object]]:
    candidates = (
        record.get("watermark"),
        _nested(record, "provenance", "watermark"),
        _nested(record, "metadata", "watermark"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("attested")
        if isinstance(nested, dict):
            return nested
        return candidate
    return None


def _domain_model_id(
    record: Mapping[str, object],
    attestation: Mapping[str, object],
) -> Optional[str]:
    candidates = (
        attestation.get("model_id"),
        record.get("model_id"),
        record.get("backend"),
        _nested(record, "provenance", "model_id"),
        _nested(record, "metadata", "backend"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _domain_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"watermark {name} must be an integer")
    return value


def _domain_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"watermark {name} must be a non-empty string"
        )
    return value


def _domain_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"watermark {name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"watermark {name} must be finite")
    return number


def _validate_domain(domain: WatermarkDomain, source: str) -> None:
    if domain.scheme != KGW_SCHEME:
        raise ValueError(f"{source}: unsupported watermark scheme")
    if domain.version != KGW_VERSION:
        raise ValueError(f"{source}: unsupported KGW version")
    if len(domain.key_id) != 16 or any(
        character not in "0123456789abcdef"
        for character in domain.key_id
    ):
        raise ValueError(f"{source}: invalid watermark key id")
    if not domain.model_id:
        raise ValueError(f"{source}: watermark model id is empty")
    if not domain.tokenizer_fingerprint:
        raise ValueError(f"{source}: tokenizer fingerprint is empty")
    expected_size = green_list_size(
        gamma=domain.gamma,
        vocab_size=domain.vocab_size,
    )
    if domain.green_list_size != expected_size:
        raise ValueError(
            f"{source}: green-list size disagrees with gamma"
        )
    expected_p0 = null_probability(
        green_list_size=expected_size,
        vocab_size=domain.vocab_size,
    )
    if not math.isclose(
        domain.p0,
        expected_p0,
        rel_tol=1e-15,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{source}: p0 disagrees with its domain")


def _nested(record: Mapping[str, object], *keys: str) -> object:
    value: object = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _config(
    options: argparse.Namespace, key: WatermarkKey
) -> KgwConfig:
    missing = [
        name
        for name in (
            "model_id",
            "tokenizer_fingerprint",
            "vocab_size",
        )
        if getattr(options, name) in (None, "")
    ]
    if missing:
        raise ValueError(
            "scoring ids requires "
            + ", ".join(
                "--" + name.replace("_", "-") for name in missing
            )
        )
    return KgwConfig(
        secret=key.secret,
        key_id=key.key_id,
        model_id=str(options.model_id),
        tokenizer_fingerprint=str(options.tokenizer_fingerprint),
        vocab_size=int(options.vocab_size),
        gamma=float(options.gamma),
        delta=0.0,
    )


def _domain_from_config(config: KgwConfig) -> WatermarkDomain:
    domain = WatermarkDomain(
        scheme=KGW_SCHEME,
        version=KGW_VERSION,
        key_id=config.key_id,
        gamma=config.gamma,
        p0=config.p0,
        model_id=config.model_id,
        tokenizer_fingerprint=config.tokenizer_fingerprint,
        vocab_size=config.vocab_size,
        green_list_size=config.green_list_size,
    )
    _validate_domain(domain, "CLI configuration")
    return domain


def _require_compatible_domains(
    left: WatermarkDomain,
    right: WatermarkDomain,
    *,
    context: str,
) -> None:
    fields = (
        "scheme",
        "version",
        "key_id",
        "model_id",
        "tokenizer_fingerprint",
        "vocab_size",
    )
    for name in fields:
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"{context}: watermark {name} differs")
    for name in ("gamma", "p0"):
        if not math.isclose(
            float(getattr(left, name)),
            float(getattr(right, name)),
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{context}: watermark {name} differs")
    if left.green_list_size != right.green_list_size:
        raise ValueError(
            f"{context}: watermark green_list_size differs"
        )


def _score_run(
    run: RunRecord,
    *,
    config: Optional[KgwConfig],
) -> RunScore:
    configured = (
        _domain_from_config(config) if config is not None else None
    )
    if run.domain is not None and configured is not None:
        _require_compatible_domains(
            run.domain,
            configured,
            context=run.source,
        )
    domain = run.domain or configured
    if domain is None:
        raise ValueError(
            f"{run.source}: a complete watermark domain is required"
        )
    _check_token_id_bounds(run, domain)
    detection = _record_detection(run, domain.p0)
    score_source = "record_membership"
    if detection is None:
        if config is None:
            raise ValueError(
                f"{run.source}: memberships absent;"
                " matching key configuration required"
            )
        detection = detect_token_ids(
            run.token_ids,
            run.evidence,
            config=config,
        )
        score_source = "recomputed_ids"
    token_count = len(run.token_ids)
    repeated = (
        1.0 - len(set(run.token_ids)) / token_count
        if token_count > 0
        else 0.0
    )
    adjacent = sum(
        left == right
        for left, right in zip(
            run.token_ids, run.token_ids[1:], strict=False
        )
    )
    adjacent_rate = adjacent / max(1, token_count - 1)
    probability_mean = (
        statistics.fmean(run.probabilities)
        if run.probabilities
        else None
    )
    elapsed = run.elapsed_seconds
    rate = (
        token_count / elapsed
        if elapsed is not None and elapsed > 0.0
        else None
    )
    return RunScore(
        source=run.source,
        token_count=token_count,
        detection=detection,
        repeated_token_rate=repeated,
        adjacent_repeat_rate=adjacent_rate,
        probability_mean=probability_mean,
        elapsed_seconds=elapsed,
        tokens_per_second=rate,
        domain=domain,
        score_source=score_source,
        evidence_source=run.evidence_source,
    )


def _check_token_id_bounds(
    run: RunRecord, domain: WatermarkDomain
) -> None:
    for token_id in run.token_ids:
        if token_id < 0 or token_id >= domain.vocab_size:
            raise ValueError(
                f"{run.source}: token id {token_id} is outside"
                f" [0, {domain.vocab_size})"
            )


def _record_detection(
    run: RunRecord, p0: float
) -> Optional[DetectionResult]:
    if any(
        membership is None
        for membership, scored in zip(
            run.memberships, run.evidence, strict=True
        )
        if scored
    ):
        return None
    scored = sum(run.evidence)
    green = sum(
        scored_flag and membership is True
        for membership, scored_flag in zip(
            run.memberships, run.evidence, strict=True
        )
    )
    return DetectionResult(
        status=detection_status(scored),
        green_count=green,
        scored_count=scored,
        z_score=detection_z_score(
            green_count=green,
            scored_count=scored,
            p0=p0,
        ),
        p0=p0,
    )


def _length_bins(value: str) -> Tuple[int, ...]:
    try:
        boundaries = tuple(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError:
        raise ValueError("length bins must be integers") from None
    if not boundaries:
        raise ValueError("at least one length bin is required")
    if len(boundaries) > LENGTH_BINS_MAX:
        raise ValueError(
            f"at most {LENGTH_BINS_MAX} length bins are allowed"
        )
    if any(boundary < 1 for boundary in boundaries):
        raise ValueError("length bins must be positive")
    if list(boundaries) != sorted(set(boundaries)):
        raise ValueError("length bins must be unique and ascending")
    return boundaries


def _summary(
    scores: Sequence[RunScore], boundaries: Tuple[int, ...]
) -> Dict[str, object]:
    return {
        "run_count": len(scores),
        "domains": _score_domains(scores),
        "overall": _score_summary(scores),
        "length_bins": [
            _bin_summary(scores, boundaries, index)
            for index in range(len(boundaries) + 1)
        ],
        "runs": [_score_record(score) for score in scores],
    }


def _score_domains(
    scores: Sequence[RunScore],
) -> List[Dict[str, object]]:
    domains: List[Dict[str, object]] = []
    seen = set()
    for score in scores:
        domain = score.domain.as_dict()
        encoded = json.dumps(domain, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        domains.append(domain)
    return domains


def _bin_summary(
    scores: Sequence[RunScore],
    boundaries: Tuple[int, ...],
    index: int,
) -> Dict[str, object]:
    low = 0 if index == 0 else boundaries[index - 1]
    high = boundaries[index] if index < len(boundaries) else None
    selected = [
        score
        for score in scores
        if score.detection.scored_count >= low
        and (high is None or score.detection.scored_count < high)
    ]
    return {
        "scored_tokens_min": low,
        "scored_tokens_max_exclusive": high,
        **_score_summary(selected),
    }


def _score_summary(scores: Sequence[RunScore]) -> Dict[str, object]:
    detections = [score.detection for score in scores]
    return {
        "run_count": len(scores),
        "scored_tokens_total": sum(
            result.scored_count for result in detections
        ),
        "green_rate_mean": _mean_optional(
            result.green_rate for result in detections
        ),
        "z_score_mean": _mean_optional(
            result.z_score for result in detections
        ),
        "repeated_token_rate_mean": _mean_optional(
            score.repeated_token_rate for score in scores
        ),
        "adjacent_repeat_rate_mean": _mean_optional(
            score.adjacent_repeat_rate for score in scores
        ),
        "base_chosen_probability_mean": _mean_optional(
            score.probability_mean for score in scores
        ),
        "elapsed_seconds_mean": _mean_optional(
            score.elapsed_seconds for score in scores
        ),
        "tokens_per_second_mean": _mean_optional(
            score.tokens_per_second for score in scores
        ),
    }


def _mean_optional(
    values: Iterable[Optional[float]],
) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _score_record(score: RunScore) -> Dict[str, object]:
    return {
        "source": score.source,
        "token_count": score.token_count,
        "domain": score.domain.as_dict(),
        "score_source": score.score_source,
        "evidence_source": score.evidence_source,
        **score.detection.as_dict(),
        "repeated_token_rate": score.repeated_token_rate,
        "adjacent_repeat_rate": score.adjacent_repeat_rate,
        "base_chosen_probability_mean": score.probability_mean,
        "elapsed_seconds": score.elapsed_seconds,
        "tokens_per_second": score.tokens_per_second,
    }


def _paired(
    watermarked: Sequence[RunScore],
    control: Sequence[RunScore],
) -> List[Dict[str, object]]:
    if not control:
        return []
    if len(watermarked) != len(control):
        raise ValueError(
            "watermarked and control inputs must contain equal runs"
        )
    pairs: List[Dict[str, object]] = []
    for index, (marked, plain) in enumerate(
        zip(watermarked, control, strict=True),
        start=1,
    ):
        _require_compatible_domains(
            marked.domain,
            plain.domain,
            context=f"pair {index}",
        )
        pairs.append(
            {
                "pair": index,
                "watermarked_source": marked.source,
                "control_source": plain.source,
                "z_score_delta": (
                    marked.detection.z_score - plain.detection.z_score
                ),
                "green_rate_delta": (
                    marked.detection.green_rate
                    - plain.detection.green_rate
                ),
                "elapsed_seconds_delta": _difference(
                    marked.elapsed_seconds,
                    plain.elapsed_seconds,
                ),
            }
        )
    return pairs


def _difference(
    left: Optional[float], right: Optional[float]
) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def _wrong_key_reports(
    options: argparse.Namespace,
    groups: Dict[str, Sequence[RunRecord]],
    boundaries: Tuple[int, ...],
    matching_config: Optional[KgwConfig],
) -> List[Dict[str, object]]:
    if len(options.wrong_key_hex) > WRONG_KEYS_MAX:
        raise ValueError(
            f"at most {WRONG_KEYS_MAX} wrong keys may be evaluated"
        )
    matching_ids = {
        run.domain.key_id
        for runs in groups.values()
        for run in runs
        if run.domain is not None
    }
    if matching_config is not None:
        matching_ids.add(matching_config.key_id)
    reports: List[Dict[str, object]] = []
    for encoded in options.wrong_key_hex:
        try:
            secret = bytes.fromhex(encoded)
        except ValueError:
            raise ValueError(
                "wrong keys must contain hexadecimal characters"
            ) from None
        wrong = WatermarkKey(secret=secret, key_id=key_id(secret))
        if wrong.key_id in matching_ids:
            raise ValueError(
                "a wrong-key control matches the scoring key"
            )
        config = _wrong_key_config(options, groups, wrong)
        report: Dict[str, object] = {"key_id": wrong.key_id}
        for name, runs in groups.items():
            # Force the wrong-key recomputation even when records hold
            # matching-key memberships.
            scores = [
                _score_run_wrong_key(run, config) for run in runs
            ]
            report[name] = _summary(scores, boundaries)
        reports.append(report)
    return reports


def _wrong_key_config(
    options: argparse.Namespace,
    groups: Mapping[str, Sequence[RunRecord]],
    key: WatermarkKey,
) -> KgwConfig:
    if all(
        getattr(options, name) not in (None, "")
        for name in (
            "model_id",
            "tokenizer_fingerprint",
            "vocab_size",
        )
    ):
        return _config(options, key)
    domains = [
        run.domain
        for runs in groups.values()
        for run in runs
        if run.domain is not None
    ]
    if not domains:
        return _config(options, key)
    reference = domains[0]
    assert reference is not None
    for domain in domains[1:]:
        assert domain is not None
        _require_compatible_domains(
            reference,
            domain,
            context="wrong-key controls",
        )
    return KgwConfig(
        secret=key.secret,
        key_id=key.key_id,
        model_id=reference.model_id,
        tokenizer_fingerprint=reference.tokenizer_fingerprint,
        vocab_size=reference.vocab_size,
        gamma=reference.gamma,
        delta=0.0,
    )


def _score_run_wrong_key(
    run: RunRecord, config: KgwConfig
) -> RunScore:
    stripped = RunRecord(
        source=run.source,
        token_ids=run.token_ids,
        memberships=tuple(None for _ in run.memberships),
        evidence=run.evidence,
        probabilities=run.probabilities,
        elapsed_seconds=run.elapsed_seconds,
        domain=None,
        evidence_source=run.evidence_source,
    )
    return _score_run(stripped, config=config)


def _matching_key_config(
    options: argparse.Namespace,
    groups: Sequence[RunRecord],
) -> Optional[KgwConfig]:
    needs_key = any(
        any(
            membership is None
            for membership, scored in zip(
                run.memberships, run.evidence, strict=True
            )
            if scored
        )
        for run in groups
    )
    needs_key = needs_key or any(run.domain is None for run in groups)
    supplied_identity = any(
        getattr(options, name) not in (None, "")
        for name in (
            "model_id",
            "tokenizer_fingerprint",
            "vocab_size",
        )
    )
    needs_key = needs_key or supplied_identity
    if not needs_key:
        _check_expected_key(options, groups)
        return None
    key = load_key()
    if (
        options.expected_key_id is not None
        and options.expected_key_id != key.key_id
    ):
        raise ValueError(
            f"loaded key id {key.key_id} does not match expected "
            f"{options.expected_key_id}"
        )
    return _config(options, key)


def _check_expected_key(
    options: argparse.Namespace,
    groups: Sequence[RunRecord],
) -> None:
    expected = options.expected_key_id
    if expected is None:
        return
    for run in groups:
        assert run.domain is not None
        if run.domain.key_id != expected:
            raise ValueError(
                f"{run.source}: saved key id {run.domain.key_id}"
                f" does not match expected {expected}"
            )
    loaded = load_key()
    if expected != loaded.key_id:
        raise ValueError(
            f"loaded key id {loaded.key_id} does not match"
            f" expected {expected}"
        )


def evaluate(options: argparse.Namespace) -> Dict[str, object]:
    """Load, score, bin, and pair every requested run."""
    boundaries = _length_bins(options.length_bins)
    watermarked = _load_group(
        options.watermarked,
        options.watermarked_evidence,
    )
    control = _load_group(
        options.control,
        options.control_evidence,
    )
    all_runs = [*watermarked, *control]
    if len(all_runs) > RUNS_MAX:
        raise ValueError(f"at most {RUNS_MAX} total runs are allowed")
    config = _matching_key_config(options, all_runs)
    groups: Dict[str, Sequence[RunRecord]] = {
        "watermarked": watermarked,
        "control": control,
    }
    scores = {
        name: [_score_run(run, config=config) for run in runs]
        for name, runs in groups.items()
    }
    return {
        "schema": "kgw-evaluation-v1",
        "cli_scoring_domain": (
            _domain_from_config(config).as_dict()
            if config is not None
            else None
        ),
        "input_limits": {
            "paths_per_group": INPUT_PATHS_MAX,
            "file_bytes": INPUT_FILE_BYTES_MAX,
            "jsonl_lines": JSONL_LINES_MAX,
            "jsonl_line_bytes": JSONL_LINE_BYTES_MAX,
            "runs_total": RUNS_MAX,
            "tokens_per_run": TOKENS_PER_RUN_MAX,
            "length_bins": LENGTH_BINS_MAX,
            "wrong_keys": WRONG_KEYS_MAX,
        },
        "limitations": [
            "Token statistics and recorded latency only.",
            "No semantic quality, factuality, or authorship claim.",
        ],
        "length_bins": list(boundaries),
        "groups": {
            name: _summary(group, boundaries)
            for name, group in scores.items()
        },
        "pairs": _paired(scores["watermarked"], scores["control"]),
        "wrong_key_controls": _wrong_key_reports(
            options, groups, boundaries, config
        ),
    }


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = _arguments(arguments)
    report = evaluate(options)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if options.output:
        Path(options.output).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
