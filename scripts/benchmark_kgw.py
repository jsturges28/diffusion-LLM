"""Profile KGW selection and cached logit bias at 128k.

Run with ``.venv/bin/python scripts/benchmark_kgw.py`` for CPU and
add ``--cuda`` on the maintainer's hardware. The benchmark uses the
same green-list and bias functions as generation. Unique predecessors
measure misses; a second pass measures the bounded device-tensor hit.
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

from src.inference.ar_sampler import _watermark_bias  # noqa: E402
from src.inference.kgw_watermark import (  # noqa: E402
    KgwConfig,
    KgwWatermark,
)

VOCAB_SIZE = 128_256
GAMMA = 0.25
DELTA = 2.0
REPEATS_DEFAULT = 20
REPEATS_MAX = 100


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
    return {
        "device": device,
        "vocab_size": VOCAB_SIZE,
        "green_list_size": config.green_list_size,
        "candidate_construction": _milliseconds(candidate_times),
        "logit_bias_cache_miss": _milliseconds(bias_miss_times),
        "logit_bias_cache_hit": _milliseconds(bias_hit_times),
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
