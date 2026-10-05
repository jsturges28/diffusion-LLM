"""Pure KGW green-list and detector contract tests.

Strategy: use small golden vocabularies for exact sets, a real 128k
synthetic vocabulary for the performance and cardinality boundary,
and sequences constructed from one key's green lists for detection.
Passing proves generation and post-hoc scoring share deterministic
sets, caches and global RNGs stay bounded and untouched, and online
scores equal a fresh batch score at every prefix.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest
import torch

from src.inference.ar_sampler import _watermark_bias
from src.inference.kgw_watermark import (
    KGW_CACHE_ENTRIES,
    KGW_DEVICE_CACHE_ENTRIES,
    KGW_DEVICE_CACHES_MAX,
    KGW_EVIDENCE_MIN,
    KgwAccumulator,
    KgwConfig,
    KgwOnlineDetector,
    KgwWatermark,
    detect_token_ids,
)

KEY = bytes(range(32))
OTHER_KEY = bytes(reversed(range(32)))
FINGERPRINT = "ab" * 32


def _config(
    *,
    secret: bytes = KEY,
    vocab_size: int = 16,
    gamma: float = 0.25,
    delta: float = 2.0,
) -> KgwConfig:
    return KgwConfig(
        secret=secret,
        key_id="0123456789abcdef",
        model_id="smollm3",
        tokenizer_fingerprint=FINGERPRINT,
        vocab_size=vocab_size,
        gamma=gamma,
        delta=delta,
    )


@pytest.mark.parametrize(
    ("config", "previous", "expected"),
    [
        (_config(), 3, [2, 5, 11, 12]),
        (
            _config(vocab_size=17, gamma=0.3),
            8,
            [2, 4, 13, 14, 16],
        ),
    ],
)
def test_green_lists_match_golden_vectors(
    config: KgwConfig,
    previous: int,
    expected: List[int],
) -> None:
    actual = KgwWatermark(config).green_ids(previous)

    assert actual.tolist() == expected
    assert len(actual) == config.green_list_size


def test_large_green_list_has_exact_cardinality() -> None:
    config = _config(vocab_size=128_256)
    green = KgwWatermark(config).green_ids(42)

    assert len(green) == 32_064
    assert len(np.unique(green)) == len(green)
    assert green[:12].tolist() == [
        1,
        6,
        9,
        14,
        22,
        27,
        29,
        31,
        32,
        35,
        36,
        37,
    ]


def test_green_list_cache_is_a_fixed_lru() -> None:
    watermark = KgwWatermark(_config(vocab_size=256))
    first = watermark.green_ids(0)
    assert watermark.green_ids(0) is first

    for previous in range(KGW_CACHE_ENTRIES + 5):
        watermark.green_ids(previous)

    assert watermark.cache.entry_count == KGW_CACHE_ENTRIES
    assert watermark.green_ids(0) is not first


def test_device_tensor_cache_reuses_and_bounds_entries() -> None:
    watermark = KgwWatermark(_config(vocab_size=256))
    logits = torch.zeros(256)

    for previous in range(KGW_DEVICE_CACHE_ENTRIES + 5):
        green = watermark.green_ids(previous)
        _watermark_bias(
            logits,
            green_ids=green,
            delta=2.0,
            watermark=watermark,
            previous_token=previous,
        )

    assert watermark.cache.device_entry_count("cpu") == (
        KGW_DEVICE_CACHE_ENTRIES
    )
    previous = KGW_DEVICE_CACHE_ENTRIES + 4
    first = watermark.cache.device_get(
        device_key="cpu",
        previous_token=previous,
    )
    green = watermark.green_ids(previous)
    _watermark_bias(
        logits,
        green_ids=green,
        delta=2.0,
        watermark=watermark,
        previous_token=previous,
    )
    second = watermark.cache.device_get(
        device_key="cpu",
        previous_token=previous,
    )
    assert first is second


def test_device_cache_bounds_the_number_of_devices() -> None:
    watermark = KgwWatermark(_config(vocab_size=256))

    for index in range(KGW_DEVICE_CACHES_MAX + 3):
        watermark.cache.device_put(
            device_key=f"device:{index}",
            previous_token=index,
            value=object(),
        )

    assert watermark.cache.device_count == KGW_DEVICE_CACHES_MAX


def test_green_list_consumes_no_numpy_or_torch_rng() -> None:
    np.random.seed(91)
    torch.manual_seed(91)
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    np.random.seed(91)
    torch.manual_seed(91)
    KgwWatermark(_config()).green_ids(3)
    actual_numpy = np.random.random(4)
    actual_torch = torch.rand(4)

    assert np.array_equal(actual_numpy, expected_numpy)
    assert torch.equal(actual_torch, expected_torch)


def test_every_domain_identity_field_changes_the_list() -> None:
    base = _config(vocab_size=256)
    variants = [
        KgwConfig(
            **{
                **base.__dict__,
                "model_id": "mamba3",
            }
        ),
        KgwConfig(
            **{
                **base.__dict__,
                "tokenizer_fingerprint": "cd" * 32,
            }
        ),
        _config(vocab_size=257),
    ]
    reference = KgwWatermark(base).green_ids(7).tolist()

    for variant in variants:
        changed = KgwWatermark(variant).green_ids(7).tolist()
        assert changed != reference


def _strong_sequence(
    watermark: KgwWatermark, count: int
) -> List[int]:
    ids = [3]
    for _ in range(count - 1):
        ids.append(int(watermark.green_ids(ids[-1])[0]))
    return ids


def test_online_equals_batch_at_every_prefix() -> None:
    config = _config(vocab_size=256)
    watermark = KgwWatermark(config)
    ids = _strong_sequence(watermark, 80)
    evidence = [False] + [True] * 9 + [False] + [True] * 69
    online = KgwOnlineDetector(KgwWatermark(config))

    for end, (token_id, scored) in enumerate(
        zip(ids, evidence, strict=True),
        start=1,
    ):
        current = online.add(token_id, evidence=scored)
        batch = detect_token_ids(
            ids[:end],
            evidence[:end],
            config=config,
        )
        assert current == batch


def test_wrong_key_does_not_reproduce_the_score() -> None:
    config = _config(vocab_size=256)
    ids = _strong_sequence(KgwWatermark(config), 100)
    evidence = [False] + [True] * 99

    matching = detect_token_ids(ids, evidence, config=config)
    wrong = detect_token_ids(
        ids,
        evidence,
        config=_config(secret=OTHER_KEY, vocab_size=256),
    )

    assert matching.green_count == 99
    assert matching.z_score > 10.0
    assert wrong.green_count < matching.green_count
    assert wrong.z_score < matching.z_score


def test_short_text_reports_insufficient_evidence() -> None:
    config = _config(vocab_size=256)
    ids = _strong_sequence(KgwWatermark(config), KGW_EVIDENCE_MIN)
    evidence = [False] + [True] * (KGW_EVIDENCE_MIN - 1)

    result = detect_token_ids(ids, evidence, config=config)

    assert result.scored_count == KGW_EVIDENCE_MIN - 1
    assert result.status == "insufficient_evidence"


def test_fifty_evidence_tokens_are_scored() -> None:
    config = _config(vocab_size=256)
    ids = _strong_sequence(
        KgwWatermark(config), KGW_EVIDENCE_MIN + 1
    )
    evidence = [False] + [True] * KGW_EVIDENCE_MIN

    result = detect_token_ids(ids, evidence, config=config)

    assert result.scored_count == KGW_EVIDENCE_MIN
    assert result.status == "scored"


def test_small_vocab_z_score_uses_exact_cardinality_p0() -> None:
    config = _config(vocab_size=3, gamma=0.4)
    accumulator = KgwAccumulator(config.p0)
    for index in range(100):
        accumulator.add(green=index < 33, evidence=True)

    result = accumulator.result()
    expected = (33 - (100 / 3)) / math.sqrt(200 / 9)
    detected = detect_token_ids(
        [0, 1],
        [False, True],
        config=config,
    )

    assert config.green_list_size == 1
    assert result.p0 == pytest.approx(1 / 3)
    assert result.z_score == pytest.approx(expected)
    assert detected.p0 == pytest.approx(1 / 3)


def test_detector_refuses_non_boolean_evidence() -> None:
    with pytest.raises(TypeError, match="evidence"):
        detect_token_ids(
            [1, 2],
            [False, 1],  # type: ignore[list-item]
            config=_config(),
        )


def test_128k_candidate_construction_meets_cpu_budget() -> None:
    """Synthetic hot-path budget, generous enough for shared CI."""
    watermark = KgwWatermark(_config(vocab_size=128_256))
    started = time.perf_counter()

    green = watermark.green_ids(42)

    elapsed = time.perf_counter() - started
    assert len(green) == 32_064
    assert elapsed < 0.1


def test_128k_cpu_logit_bias_meets_budget() -> None:
    watermark = KgwWatermark(_config(vocab_size=128_256))
    green = watermark.green_ids(42)
    logits = torch.zeros(128_256)
    started = time.perf_counter()

    biased = _watermark_bias(
        logits,
        green_ids=green,
        delta=2.0,
    )

    elapsed = time.perf_counter() - started
    assert int(torch.count_nonzero(biased).item()) == 32_064
    assert elapsed < 0.5


def test_provenance_contains_id_but_never_secret() -> None:
    block = KgwWatermark(_config()).provenance()

    assert block["key_id"] == "0123456789abcdef"
    assert block["p0"] == pytest.approx(0.25)
    assert "secret" not in block
    assert KEY.hex() not in repr(block)
