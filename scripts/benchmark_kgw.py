"""Profile KGW selection, bias, and pressure capture at 128k.

Run with ``.venv/bin/python scripts/benchmark_kgw.py`` for CPU and
add ``--cuda`` on the maintainer's hardware. The benchmark uses the
same green-list, bias, pressure, and sampler-candidate functions as
generation. Unique predecessors measure misses; a second pass measures
the bounded device-tensor hit.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.ar_sampler import (  # noqa: E402
    _green_probability,
    _sampler_candidate_set,
    _watermark_bias,
    _watermark_indices,
)
from src.inference.kgw_watermark import (  # noqa: E402
    KgwConfig,
    KgwWatermark,
    biased_green_mass,
)

VOCAB_SIZE = 128_256
GAMMA = 0.25
DELTA = 2.0
REPEATS_DEFAULT = 20
REPEATS_MAX = 100


class _SyntheticTokenizer:
    """Bounded decode stand-in for sampler candidate materialization."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert len(token_ids) == 1
        assert skip_special_tokens is False
        return f"<{token_ids[0]}>"


def _arguments(
    arguments: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KGW at a 128k vocabulary.",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS_DEFAULT,
    )
    return parser.parse_args(arguments)


def _milliseconds(values: List[float]) -> Dict[str, float]:
    assert values, "a profile has samples"
    return {
        "median_ms": statistics.median(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def _profile(device: str, repeats: int) -> Dict[str, object]:
    config = KgwConfig(
        secret=bytes(range(32)),
        key_id="0123456789abcdef",
        model_id="synthetic",
        tokenizer_fingerprint="ab" * 32,
        vocab_size=VOCAB_SIZE,
        gamma=GAMMA,
        delta=DELTA,
    )
    watermark = KgwWatermark(config)
    logits = torch.zeros(VOCAB_SIZE, device=device)
    candidate_times: List[float] = []
    bias_miss_times: List[float] = []
    for previous in range(repeats):
        started = time.perf_counter()
        green_ids = watermark.green_ids(previous)
        candidate_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        biased = _watermark_bias(
            logits,
            green_ids=green_ids,
            delta=DELTA,
            watermark=watermark,
            previous_token=previous,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        bias_miss_times.append(time.perf_counter() - started)
        assert int(torch.count_nonzero(biased).item()) == (
            config.green_list_size
        )
    repeated = repeats - 1
    green_ids = watermark.green_ids(repeated)
    bias_hit_times: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        biased = _watermark_bias(
            logits,
            green_ids=green_ids,
            delta=DELTA,
            watermark=watermark,
            previous_token=repeated,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        bias_hit_times.append(time.perf_counter() - started)
        assert int(torch.count_nonzero(biased).item()) == (
            config.green_list_size
        )
    indices = _watermark_indices(
        logits=logits,
        green_ids=green_ids,
        watermark=watermark,
        previous_token=repeated,
    )
    base_probs = torch.softmax(logits, dim=-1)
    sampler_probs = torch.softmax(biased / 0.6, dim=-1)
    tokenizer = _SyntheticTokenizer()
    next_id = int(torch.argmax(sampler_probs).item())
    pressure_times: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        base_mass = _green_probability(base_probs, indices)
        kgw_mass = biased_green_mass(
            base_mass=base_mass, delta=DELTA
        )
        sampler_mass = _green_probability(sampler_probs, indices)
        candidate_set = _sampler_candidate_set(
            sampler_probs,
            tokenizer=tokenizer,
            next_id=next_id,
            watermark=watermark,
            previous_token=repeated,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        pressure_times.append(time.perf_counter() - started)
        assert base_mass <= kgw_mass
        assert 0.0 <= sampler_mass <= 1.0
        assert candidate_set["support"] == VOCAB_SIZE
        assert len(candidate_set["candidates"]) == 5
    return {
        "device": device,
        "vocab_size": VOCAB_SIZE,
        "green_list_size": config.green_list_size,
        "candidate_construction": _milliseconds(candidate_times),
        "logit_bias_cache_miss": _milliseconds(bias_miss_times),
        "logit_bias_cache_hit": _milliseconds(bias_hit_times),
        "pressure_and_sampler_capture": _milliseconds(
            pressure_times
        ),
        "host_cache_entries": watermark.cache.entry_count,
        "device_cache_entries": (
            watermark.cache.device_entry_count(str(logits.device))
        ),
    }


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = _arguments(arguments)
    if options.repeats < 1 or options.repeats > REPEATS_MAX:
        raise ValueError(
            f"repeats must be in [1, {REPEATS_MAX}]"
        )
    device = "cuda" if options.cuda else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(_profile(device, options.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
