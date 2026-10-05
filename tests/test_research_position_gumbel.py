"""Position-seeded Gumbel-max research spike tests.

Strategy: lock the HMAC-derived uniform contract, compare the sampler
and detector directly with the paper's formulas, exercise every
resource boundary, and inspect the script's imports for model or RNG
dependencies. Passing proves the bounded standard-library primitive
is deterministic and internally faithful. It does not prove anything
about quality or distortion in a real diffusion model.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from scripts import research_position_gumbel as spike

HOST_KEY = bytes(range(spike.HOST_KEY_BYTES))
OTHER_KEY = bytes(reversed(range(spike.HOST_KEY_BYTES)))
PROBABILITIES = (0.1, 0.2, 0.3, 0.4)
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research_position_gumbel.py"
)


def _primitive(
    *,
    key: bytes = HOST_KEY,
    vocab_size: int = 4,
    modulus: int = 7,
) -> spike.PositionGumbel:
    return spike.PositionGumbel(
        key,
        vocab_size=vocab_size,
        modulus=modulus,
    )


def test_paper_score_is_log_uniform_divided_by_probability() -> None:
    probabilities = (0.2, 0.0, 0.8)
    uniforms = (0.25, 0.5, 0.75)

    scores = spike.paper_scores(
        probabilities=probabilities,
        uniforms=uniforms,
    )

    assert scores[0] == math.log(0.25) / 0.2
    assert scores[1] == -math.inf
    assert scores[2] == math.log(0.75) / 0.8


def test_position_modulus_and_offset_choose_the_same_vector() -> None:
    primitive = _primitive()

    shifted = primitive.uniform_vector(position=3, offset=2)

    assert shifted == primitive.uniform_vector(position=5, offset=0)
    assert shifted == primitive.uniform_vector(position=12, offset=0)
    assert all(0.0 < value < 1.0 for value in shifted)


def test_hmac_uniform_contract_is_stable() -> None:
    primitive = _primitive()

    vector = primitive.uniform_vector(position=3, offset=2)

    assert vector == (
        0.4350202301860321,
        0.3964893330114766,
        0.3665312104450772,
        0.8901993604759256,
    )


def test_host_key_separates_uniform_vectors() -> None:
    first = _primitive()
    second = _primitive(key=OTHER_KEY)

    assert first.uniform_vector(position=0) != second.uniform_vector(
        position=0
    )


def test_detector_matches_selected_uniform_formula() -> None:
    primitive = _primitive()
    token_ids = (0, 1, 2, 3, 0, 2)
    offsets = (0, 1, 4)
    expected = tuple(
        math.fsum(
            -math.log1p(
                -primitive.uniform_for_token(
                    position=position,
                    token_id=token_id,
                    offset=offset,
                )
            )
            for position, token_id in enumerate(token_ids)
        )
        / len(token_ids)
        for offset in offsets
    )

    statistic = primitive.detector_statistic(
        token_ids,
        offsets=offsets,
    )

    assert statistic.scores == pytest.approx(expected)
    assert statistic.best_score == pytest.approx(max(expected))
    assert (
        statistic.best_offset
        == offsets[expected.index(max(expected))]
    )


def test_synthetic_report_checks_marginals_and_scores() -> None:
    report = spike.synthetic_report(
        host_key=HOST_KEY,
        probabilities=PROBABILITIES,
        length=512,
        modulus=128,
    )
    distribution = report["distribution"]
    detection = report["detection"]
    limitations = report["limitations"]
    assert isinstance(distribution, dict)
    assert isinstance(detection, dict)
    assert isinstance(limitations, list)
    marked = detection["watermarked"]
    control = detection["unwatermarked_control"]
    assert isinstance(marked, dict)
    assert isinstance(control, dict)

    assert distribution["watermarked_max_absolute_error"] < 0.04
    assert distribution["control_max_absolute_error"] < 0.04
    assert marked["best_offset"] == 0
    assert marked["best_score"] > control["best_score"] + 0.5
    assert spike.LIMITATION in limitations


@pytest.mark.parametrize(
    ("vocab_size", "modulus", "error"),
    [
        (1, 1, ValueError),
        (spike.VOCAB_SIZE_MAX + 1, 1, ValueError),
        (True, 1, TypeError),
        (2, 0, ValueError),
        (2, spike.MODULUS_MAX + 1, ValueError),
        (2, True, TypeError),
    ],
)
def test_vocabulary_and_modulus_limits_are_strict(
    vocab_size: int,
    modulus: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        spike.PositionGumbel(
            HOST_KEY,
            vocab_size=vocab_size,
            modulus=modulus,
        )


@pytest.mark.parametrize(
    "host_key",
    [
        b"",
        bytes(spike.HOST_KEY_BYTES - 1),
        bytes(spike.HOST_KEY_BYTES + 1),
    ],
)
def test_host_key_length_is_strict(host_key: bytes) -> None:
    with pytest.raises(ValueError, match="exactly"):
        spike.PositionGumbel(
            host_key,
            vocab_size=2,
            modulus=1,
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.5,),
        (0.5, 0.4),
        (-0.1, 1.1),
        (math.nan, math.nan),
        (True, False),
    ],
)
def test_invalid_probability_domains_are_rejected(
    probabilities: tuple[float, ...],
) -> None:
    primitive = _primitive(
        vocab_size=max(2, len(probabilities)),
        modulus=1,
    )

    with pytest.raises((TypeError, ValueError)):
        primitive.sample_token(probabilities, position=0)


def test_sequence_length_limit_is_strict() -> None:
    primitive = _primitive(vocab_size=2, modulus=1)
    token_ids = (0,) * (spike.SEQUENCE_LENGTH_MAX + 1)

    with pytest.raises(ValueError, match="length"):
        primitive.detector_statistic(token_ids)


def test_detector_offset_work_is_bounded_before_hashing() -> None:
    primitive = _primitive(
        vocab_size=2,
        modulus=spike.MODULUS_MAX,
    )
    length = (spike.DETECTOR_EVALUATIONS_MAX // spike.MODULUS_MAX) + 1
    token_ids = (0,) * length

    with pytest.raises(ValueError, match="work exceeds"):
        primitive.detector_statistic(token_ids)


@pytest.mark.parametrize(
    "offsets",
    [
        (),
        (0, 0),
        (-1,),
        (7,),
    ],
)
def test_detector_offsets_are_bounded_and_unique(
    offsets: tuple[int, ...],
) -> None:
    primitive = _primitive()

    with pytest.raises((TypeError, ValueError)):
        primitive.detector_statistic((0, 1), offsets=offsets)


def test_synthetic_report_total_work_is_bounded() -> None:
    with pytest.raises(ValueError, match="synthetic report work"):
        spike.synthetic_report(
            host_key=HOST_KEY,
            probabilities=(0.5, 0.5),
            length=spike.SEQUENCE_LENGTH_MAX,
            modulus=spike.MODULUS_MAX,
        )


def test_script_imports_only_standard_library_modules() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
    )
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "math",
        "struct",
        "typing",
    }

    assert imported <= allowed
    assert "random" not in imported
    assert "torch" not in imported
    assert "transformers" not in imported


def test_cli_emits_report_without_echoing_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_hex = HOST_KEY.hex()

    result = spike.main(
        [
            "--key-hex",
            key_hex,
            "--probabilities",
            "0.25,0.75",
            "--length",
            "64",
            "--modulus",
            "8",
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(output)

    assert result == 0
    assert report["sampling_score"] == (
        "ln(r[token_id]) / probability[token_id]"
    )
    assert spike.LIMITATION in report["limitations"]
    assert key_hex not in output
