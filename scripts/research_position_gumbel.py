"""Bounded research spike for position-seeded Gumbel-max.

This is a model-free experiment, not a production watermark. It uses
only the Python standard library and derives pseudorandom uniforms
from a research-only HMAC domain. The categorical score is exactly
``ln(r[token]) / probability[token]`` from Bagchi et al.

Run a deterministic synthetic report with:

```
.venv/bin/python scripts/research_position_gumbel.py \
  --key-hex 000102030405060708090a0b0c0d0e0f\
101112131415161718191a1b1c1d1e1f
```
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import struct
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

SCHEME = "position_gumbel_research"
VERSION = 1
HOST_KEY_BYTES = 32
VOCAB_SIZE_MIN = 2
VOCAB_SIZE_MAX = 262_144
SEQUENCE_LENGTH_MAX = 16_384
MODULUS_MAX = 4_096
DETECTOR_EVALUATIONS_MAX = 4_194_304
SYNTHETIC_EVALUATIONS_MAX = 4_194_304
CLI_PROBABILITIES_CHARS_MAX = 1_000_000

DEFAULT_PROBABILITIES = (0.1, 0.2, 0.3, 0.4)
DEFAULT_LENGTH = 1_024
DEFAULT_MODULUS = 256

HMAC_DOMAIN = b"diffusion-llm/research/position-gumbel/v1\x00"
_UNIFORM_DOMAIN = b"uniform-vector\x00"
_CONTROL_DOMAIN = (
    b"diffusion-llm/research/position-gumbel-control/v1\x00"
)
_UNIFORM_BITS = 52
_UNIFORM_SCALE = float(2**_UNIFORM_BITS)

LIMITATION = (
    "Finite synthetic results do not prove distortion-free "
    "production behavior."
)

assert VERSION >= 1, "the research scheme version is positive"
assert HOST_KEY_BYTES * 8 == 256, "the host key is 256 bits"
assert VOCAB_SIZE_MAX >= 128_256, "LLaDA-sized vocabularies fit"
assert SEQUENCE_LENGTH_MAX >= MODULUS_MAX
assert DEFAULT_LENGTH <= SEQUENCE_LENGTH_MAX
assert DEFAULT_MODULUS <= MODULUS_MAX
assert HMAC_DOMAIN != _CONTROL_DOMAIN
assert 0.0 < 1.0 / _UNIFORM_SCALE < 1.0


@dataclass(frozen=True, slots=True)
class DetectorStatistic:
    """Normalized detector scores for a bounded offset set."""

    token_count: int
    offsets: Tuple[int, ...]
    scores: Tuple[float, ...]
    best_offset: int
    best_score: float

    def __post_init__(self) -> None:
        assert self.token_count > 0
        assert len(self.offsets) == len(self.scores)
        assert len(self.offsets) > 0
        assert self.best_offset in self.offsets
        assert self.best_score == max(self.scores)

    def score_for(self, offset: int) -> float:
        """Return the normalized score for one evaluated offset."""
        if offset not in self.offsets:
            raise ValueError(f"offset {offset} was not evaluated")
        index = self.offsets.index(offset)
        score = self.scores[index]
        assert score >= 0.0
        return score


class PositionGumbel:
    """Research-only keyed uniforms, sampler, and detector."""

    __slots__ = (
        "_scheme_key",
        "modulus",
        "vocab_size",
    )

    def __init__(
        self,
        host_key: bytes,
        *,
        vocab_size: int,
        modulus: int,
    ) -> None:
        _validate_host_key(host_key)
        self.vocab_size = _bounded_integer(
            "vocab_size",
            vocab_size,
            minimum=VOCAB_SIZE_MIN,
            maximum=VOCAB_SIZE_MAX,
        )
        self.modulus = _bounded_integer(
            "modulus",
            modulus,
            minimum=1,
            maximum=MODULUS_MAX,
        )
        self._scheme_key = hmac.new(
            host_key,
            HMAC_DOMAIN,
            hashlib.sha256,
        ).digest()
        assert len(self._scheme_key) == hashlib.sha256().digest_size

    def seed_index(self, *, position: int, offset: int = 0) -> int:
        """The zero-based position/modulus/offset seed."""
        position = _bounded_integer(
            "position",
            position,
            minimum=0,
            maximum=SEQUENCE_LENGTH_MAX - 1,
        )
        offset = _bounded_integer(
            "offset",
            offset,
            minimum=0,
            maximum=self.modulus - 1,
        )
        seed = (position + offset) % self.modulus
        assert 0 <= seed < self.modulus
        return seed

    def uniform_vector(
        self,
        *,
        position: int,
        offset: int = 0,
    ) -> Tuple[float, ...]:
        """One deterministic open-interval uniform per token id."""
        seed = self.seed_index(position=position, offset=offset)
        uniforms = tuple(
            self._uniform(seed=seed, token_id=token_id)
            for token_id in range(self.vocab_size)
        )
        assert len(uniforms) == self.vocab_size
        return uniforms

    def uniform_for_token(
        self,
        *,
        position: int,
        token_id: int,
        offset: int = 0,
    ) -> float:
        """The detector's one needed element of a uniform vector."""
        _validate_token_id(token_id, self.vocab_size)
        seed = self.seed_index(position=position, offset=offset)
        uniform = self._uniform(seed=seed, token_id=token_id)
        assert 0.0 < uniform < 1.0
        return uniform

    def sample_token(
        self,
        probabilities: Sequence[float],
        *,
        position: int,
        offset: int = 0,
    ) -> int:
        """Maximize the paper's exact categorical score."""
        checked = _validate_probabilities(
            probabilities,
            expected_size=self.vocab_size,
        )
        return self._sample_checked(
            checked,
            position=position,
            offset=offset,
        )

    def detector_statistic(
        self,
        token_ids: Sequence[int],
        *,
        offsets: Optional[Sequence[int]] = None,
    ) -> DetectorStatistic:
        """Score token ids at each requested prefix alignment."""
        checked_tokens = _validate_token_ids(
            token_ids,
            vocab_size=self.vocab_size,
        )
        checked_offsets = _validate_offsets(
            offsets,
            modulus=self.modulus,
        )
        evaluations = len(checked_tokens) * len(checked_offsets)
        if evaluations > DETECTOR_EVALUATIONS_MAX:
            raise ValueError(
                "detector work exceeds "
                f"{DETECTOR_EVALUATIONS_MAX} evaluations"
            )
        scores = tuple(
            self._score_offset(checked_tokens, offset)
            for offset in checked_offsets
        )
        best_index = max(
            range(len(scores)),
            key=scores.__getitem__,
        )
        return DetectorStatistic(
            token_count=len(checked_tokens),
            offsets=checked_offsets,
            scores=scores,
            best_offset=checked_offsets[best_index],
            best_score=scores[best_index],
        )

    def _sample_checked(
        self,
        probabilities: Tuple[float, ...],
        *,
        position: int,
        offset: int,
    ) -> int:
        uniforms = self.uniform_vector(
            position=position,
            offset=offset,
        )
        scores = _paper_scores_checked(probabilities, uniforms)
        token_id = max(
            range(self.vocab_size),
            key=scores.__getitem__,
        )
        assert 0 <= token_id < self.vocab_size
        return token_id

    def _score_offset(
        self,
        token_ids: Tuple[int, ...],
        offset: int,
    ) -> float:
        terms = (
            -math.log1p(
                -self.uniform_for_token(
                    position=position,
                    token_id=token_id,
                    offset=offset,
                )
            )
            for position, token_id in enumerate(token_ids)
        )
        score = math.fsum(terms) / len(token_ids)
        assert score >= 0.0
        return score

    def _uniform(self, *, seed: int, token_id: int) -> float:
        message = struct.pack(
            ">IIIII",
            VERSION,
            self.vocab_size,
            self.modulus,
            seed,
            token_id,
        )
        digest = hmac.new(
            self._scheme_key,
            _UNIFORM_DOMAIN + message,
            hashlib.sha256,
        ).digest()
        return _unit_interval(digest)


def paper_scores(
    *,
    probabilities: Sequence[float],
    uniforms: Sequence[float],
) -> Tuple[float, ...]:
    """Return ``ln(r_x) / p_x`` exactly as the paper specifies."""
    checked_probabilities = _validate_probabilities(probabilities)
    checked_uniforms = _validate_uniforms(
        uniforms,
        expected_size=len(checked_probabilities),
    )
    return _paper_scores_checked(
        checked_probabilities,
        checked_uniforms,
    )


def synthetic_report(
    *,
    host_key: bytes,
    probabilities: Sequence[float] = DEFAULT_PROBABILITIES,
    length: int = DEFAULT_LENGTH,
    modulus: int = DEFAULT_MODULUS,
) -> dict[str, object]:
    """Build a deterministic, bounded empirical sanity report."""
    _validate_host_key(host_key)
    checked = _validate_probabilities(probabilities)
    length = _bounded_integer(
        "length",
        length,
        minimum=1,
        maximum=SEQUENCE_LENGTH_MAX,
    )
    primitive = PositionGumbel(
        host_key,
        vocab_size=len(checked),
        modulus=modulus,
    )
    evaluations = length * (len(checked) + (2 * primitive.modulus))
    if evaluations > SYNTHETIC_EVALUATIONS_MAX:
        raise ValueError(
            "synthetic report work exceeds "
            f"{SYNTHETIC_EVALUATIONS_MAX} evaluations"
        )

    watermarked = _watermarked_tokens(
        primitive,
        checked,
        length=length,
    )
    control = _control_tokens(
        host_key,
        checked,
        length=length,
    )
    offsets = tuple(range(primitive.modulus))
    marked_score = primitive.detector_statistic(
        watermarked,
        offsets=offsets,
    )
    control_score = primitive.detector_statistic(
        control,
        offsets=offsets,
    )
    marked_frequency = _token_frequencies(
        watermarked,
        vocab_size=len(checked),
    )
    control_frequency = _token_frequencies(
        control,
        vocab_size=len(checked),
    )
    return _build_report(
        primitive=primitive,
        probabilities=checked,
        length=length,
        marked_frequency=marked_frequency,
        control_frequency=control_frequency,
        marked_score=marked_score,
        control_score=control_score,
    )


def _paper_scores_checked(
    probabilities: Tuple[float, ...],
    uniforms: Tuple[float, ...],
) -> Tuple[float, ...]:
    assert len(probabilities) == len(uniforms)
    scores = tuple(
        -math.inf
        if probability == 0.0
        else math.log(uniform) / probability
        for probability, uniform in zip(
            probabilities,
            uniforms,
            strict=True,
        )
    )
    assert any(math.isfinite(score) for score in scores)
    return scores


def _watermarked_tokens(
    primitive: PositionGumbel,
    probabilities: Tuple[float, ...],
    *,
    length: int,
) -> Tuple[int, ...]:
    tokens = tuple(
        primitive._sample_checked(
            probabilities,
            position=position,
            offset=0,
        )
        for position in range(length)
    )
    assert len(tokens) == length
    return tokens


def _control_tokens(
    host_key: bytes,
    probabilities: Tuple[float, ...],
    *,
    length: int,
) -> Tuple[int, ...]:
    control_key = hmac.new(
        host_key,
        _CONTROL_DOMAIN,
        hashlib.sha256,
    ).digest()
    tokens = tuple(
        _control_token(
            control_key,
            probabilities,
            position=position,
        )
        for position in range(length)
    )
    assert len(tokens) == length
    return tokens


def _control_token(
    control_key: bytes,
    probabilities: Tuple[float, ...],
    *,
    position: int,
) -> int:
    digest = hmac.new(
        control_key,
        struct.pack(">II", VERSION, position),
        hashlib.sha256,
    ).digest()
    uniform = _unit_interval(digest)
    cumulative = 0.0
    selected = 0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if probability > 0.0:
            selected = token_id
        if uniform < cumulative:
            return token_id
    assert probabilities[selected] > 0.0
    return selected


def _token_frequencies(
    token_ids: Tuple[int, ...],
    *,
    vocab_size: int,
) -> Tuple[float, ...]:
    assert token_ids, "a frequency report needs tokens"
    counts = [0 for _ in range(vocab_size)]
    for token_id in token_ids:
        counts[token_id] += 1
    frequencies = tuple(count / len(token_ids) for count in counts)
    assert math.isclose(math.fsum(frequencies), 1.0)
    return frequencies


def _build_report(
    *,
    primitive: PositionGumbel,
    probabilities: Tuple[float, ...],
    length: int,
    marked_frequency: Tuple[float, ...],
    control_frequency: Tuple[float, ...],
    marked_score: DetectorStatistic,
    control_score: DetectorStatistic,
) -> dict[str, object]:
    marked_error = max(
        abs(observed - target)
        for observed, target in zip(
            marked_frequency,
            probabilities,
            strict=True,
        )
    )
    control_error = max(
        abs(observed - target)
        for observed, target in zip(
            control_frequency,
            probabilities,
            strict=True,
        )
    )
    return {
        "scheme": SCHEME,
        "version": VERSION,
        "hmac_domain": HMAC_DOMAIN.rstrip(b"\x00").decode("ascii"),
        "seed_formula": ("(zero_based_position + offset) % modulus"),
        "sampling_score": "ln(r[token_id]) / probability[token_id]",
        "vocab_size": primitive.vocab_size,
        "length": length,
        "modulus": primitive.modulus,
        "unique_seed_count": min(length, primitive.modulus),
        "distribution": {
            "target": list(probabilities),
            "watermarked": list(marked_frequency),
            "unwatermarked_control": list(control_frequency),
            "watermarked_max_absolute_error": marked_error,
            "control_max_absolute_error": control_error,
        },
        "detection": {
            "watermarked": _score_summary(marked_score),
            "unwatermarked_control": _score_summary(control_score),
        },
        "limitations": [
            LIMITATION,
            (
                "The control and keyed uniforms are deterministic "
                "synthetic pseudorandom draws, not model outputs."
            ),
            (
                "The maximum over offsets requires held-out false-"
                "positive calibration before any threshold is used."
            ),
        ],
    }


def _score_summary(
    statistic: DetectorStatistic,
) -> dict[str, object]:
    return {
        "token_count": statistic.token_count,
        "offset_count": len(statistic.offsets),
        "aligned_offset_zero_score": statistic.score_for(0),
        "best_offset": statistic.best_offset,
        "best_score": statistic.best_score,
    }


def _validate_probabilities(
    probabilities: Sequence[float],
    *,
    expected_size: Optional[int] = None,
) -> Tuple[float, ...]:
    size = len(probabilities)
    _bounded_integer(
        "vocab_size",
        size,
        minimum=VOCAB_SIZE_MIN,
        maximum=VOCAB_SIZE_MAX,
    )
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"probabilities have size {size}, "
            f"expected {expected_size}"
        )
    checked = tuple(
        _validate_probability(probability)
        for probability in probabilities
    )
    total = math.fsum(checked)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1")
    assert any(probability > 0.0 for probability in checked)
    return checked


def _validate_probability(probability: float) -> float:
    if isinstance(probability, bool):
        raise TypeError("probability must be numeric")
    if not isinstance(probability, (int, float)):
        raise TypeError("probability must be numeric")
    checked = float(probability)
    if not math.isfinite(checked):
        raise ValueError("probability must be finite")
    if checked < 0.0 or checked > 1.0:
        raise ValueError("probability must be within [0, 1]")
    return checked


def _validate_uniforms(
    uniforms: Sequence[float],
    *,
    expected_size: int,
) -> Tuple[float, ...]:
    if len(uniforms) != expected_size:
        raise ValueError(
            f"uniforms have size {len(uniforms)}, "
            f"expected {expected_size}"
        )
    checked = tuple(float(uniform) for uniform in uniforms)
    if not all(
        math.isfinite(uniform) and 0.0 < uniform < 1.0
        for uniform in checked
    ):
        raise ValueError("uniforms must be finite and within (0, 1)")
    return checked


def _validate_token_ids(
    token_ids: Sequence[int],
    *,
    vocab_size: int,
) -> Tuple[int, ...]:
    length = len(token_ids)
    _bounded_integer(
        "length",
        length,
        minimum=1,
        maximum=SEQUENCE_LENGTH_MAX,
    )
    checked = tuple(token_ids)
    for token_id in checked:
        _validate_token_id(token_id, vocab_size)
    return checked


def _validate_offsets(
    offsets: Optional[Sequence[int]],
    *,
    modulus: int,
) -> Tuple[int, ...]:
    checked = (
        tuple(range(modulus)) if offsets is None else tuple(offsets)
    )
    _bounded_integer(
        "offset count",
        len(checked),
        minimum=1,
        maximum=modulus,
    )
    for offset in checked:
        _bounded_integer(
            "offset",
            offset,
            minimum=0,
            maximum=modulus - 1,
        )
    if len(set(checked)) != len(checked):
        raise ValueError("offsets must be unique")
    return checked


def _validate_token_id(token_id: int, vocab_size: int) -> None:
    _bounded_integer(
        "token id",
        token_id,
        minimum=0,
        maximum=vocab_size - 1,
    )


def _validate_host_key(host_key: bytes) -> None:
    if not isinstance(host_key, bytes):
        raise TypeError("host key must be bytes")
    if len(host_key) != HOST_KEY_BYTES:
        raise ValueError(
            f"host key must contain exactly {HOST_KEY_BYTES} bytes"
        )


def _bounded_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be within [{minimum}, {maximum}]"
        )
    return value


def _unit_interval(digest: bytes) -> float:
    assert len(digest) >= 8
    integer = int.from_bytes(digest[:8], "big")
    integer >>= 64 - _UNIFORM_BITS
    uniform = (integer + 0.5) / _UNIFORM_SCALE
    assert 0.0 < uniform < 1.0
    return uniform


def _parse_host_key(value: str) -> bytes:
    try:
        host_key = bytes.fromhex(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "key must contain only hexadecimal characters"
        ) from error
    try:
        _validate_host_key(host_key)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return host_key


def _parse_probabilities(value: str) -> Tuple[float, ...]:
    if len(value) > CLI_PROBABILITIES_CHARS_MAX:
        raise argparse.ArgumentTypeError(
            "probability input exceeds the character limit"
        )
    try:
        probabilities = tuple(
            float(part.strip()) for part in value.split(",")
        )
        return _validate_probabilities(probabilities)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arguments(
    arguments: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded, model-free position-Gumbel experiment."
        ),
    )
    parser.add_argument(
        "--key-hex",
        required=True,
        type=_parse_host_key,
        help=(
            "64 hex characters for a synthetic host key. Do not use "
            "a production key."
        ),
    )
    parser.add_argument(
        "--probabilities",
        type=_parse_probabilities,
        default=DEFAULT_PROBABILITIES,
        help="Comma-separated categorical probabilities.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=DEFAULT_MODULUS,
    )
    return parser.parse_args(arguments)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = _arguments(arguments)
    report = synthetic_report(
        host_key=options.key_hex,
        probabilities=options.probabilities,
        length=options.length,
        modulus=options.modulus,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
