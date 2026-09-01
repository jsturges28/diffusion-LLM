"""Measure what a run's frames cost, in both wire shapes.

`RUNTIME-01` asks for growth to be measured rather than asserted, at
128, 256, 1,024 and 2,048 tokens, and for it to be linear once the
frames append instead of restating. This is that measurement, kept in
the tree rather than run once, so the same table can be produced
again when stage two changes what reaches disk.

No model and no GPU. The numbers come from building the two frame
shapes directly and serializing them, which is exactly what the wire
and the save carry: the sampler decides *which* tokens, and that has
no bearing on how much a frame costs to describe.

Three figures per length:

- **wire**, the frames a worker streams for the whole run. This is
  the one that was quadratic, because every frame restated every
  position decoded so far.
- **payload**, what the browser posts when the run is saved.
- **disk**, what lands in ``tokens.json`` after the server expands
  the stream. Unchanged by design in stage one: the point of
  expanding server-side is that no reader downstream can tell.

Usage::

    .venv/bin/python scripts/measure_frame_payload.py
    .venv/bin/python scripts/measure_frame_payload.py --lengths 64,128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.web.server import (  # noqa: E402  (after sys.path setup)
    expand_position_text,
    expand_positions,
)

# The four the finding names. 2,048 is the experimental ceiling the
# registry allows, and the length the maintainer's ablation runs
# actually used, which is why the largest saved run is 282 MiB.
DEFAULT_LENGTHS = (128, 256, 1024, 2048)

# A token's display text has to be *something*, and its length moves
# every number here. Five characters is close to the mean for the
# English prose these runs generate.
SAMPLE_TEXT = " word"


def _positions(count: int) -> List[Dict[str, Any]]:
    """One record per token, as an append frame carries it."""
    assert count > 0, "a run has at least one token"
    return [
        {
            "t": SAMPLE_TEXT,
            "m": False,
            "id": 1000 + at,
            "c": round(0.5 + (at % 50) / 100, 4),
            "e": round(1.5 - (at % 50) / 100, 4),
        }
        for at in range(count)
    ]


def _append_wire(positions: List[Dict[str, Any]]) -> int:
    """Bytes for the whole run, one frame per token."""
    total = 0
    running = 0.0
    for index, token in enumerate(positions, start=1):
        running += token["c"]
        total += len(
            json.dumps(
                {
                    "type": "frame",
                    "shape": "append",
                    "index": index,
                    "total_steps": len(positions),
                    "canvas_index": 0,
                    "mean_conf": round(running / index, 4),
                    "token": token,
                    "revealed": [index - 1],
                }
            )
        )
    return total


def _snapshot_wire(positions: List[Dict[str, Any]]) -> int:
    """The same run the way it used to go out: every frame whole."""
    total = 0
    running = 0.0
    texts = expand_position_text_dicts(positions)
    for index, token in enumerate(positions, start=1):
        running += token["c"]
        total += len(
            json.dumps(
                {
                    "type": "frame",
                    "index": index,
                    "total_steps": len(positions),
                    "canvas_index": 0,
                    "mean_conf": round(running / index, 4),
                    "text": texts[index - 1],
                    "tokens": positions[:index],
                    "revealed": [index - 1],
                }
            )
        )
    return total


class _Record:
    """A token record shaped like the server's, without pydantic.

    The expansion helpers only read ``t``, so a small stand-in keeps
    this script from constructing validated models for two million
    positions to measure a length.
    """

    __slots__ = ("t",)

    def __init__(self, text: str) -> None:
        self.t = text


def expand_position_text_dicts(
    positions: List[Dict[str, Any]],
) -> List[str]:
    """``expand_position_text`` over plain dicts."""
    return expand_position_text(
        [_Record(p["t"]) for p in positions]
    )


def _measure(count: int) -> Dict[str, int]:
    positions = _positions(count)
    frames = expand_positions(positions)
    return {
        "tokens": count,
        "append_wire": _append_wire(positions),
        "snapshot_wire": _snapshot_wire(positions),
        "append_payload": len(json.dumps(positions)),
        "snapshot_payload": len(json.dumps(frames)),
        "disk": len(json.dumps(frames)),
    }


def _mib(value: int) -> str:
    return f"{value / 2 ** 20:.2f}"


def _report(rows: List[Dict[str, int]]) -> None:
    header = (
        f"{'tokens':>7}  {'append wire':>12}  {'was':>12}"
        f"  {'append post':>12}  {'was':>12}  {'disk':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['tokens']:>7}"
            f"  {_mib(row['append_wire']):>10} MiB"
            f"  {_mib(row['snapshot_wire']):>10} MiB"
            f"  {_mib(row['append_payload']):>10} MiB"
            f"  {_mib(row['snapshot_payload']):>10} MiB"
            f"  {_mib(row['disk']):>10} MiB"
        )
    print()
    _report_growth(rows)


def _report_growth(rows: List[Dict[str, int]]) -> None:
    """Growth per doubling, which is where linear shows itself.

    A linear series doubles when the run does; a quadratic one
    quadruples. Printing the ratio says which without anyone having
    to divide the column above in their head.
    """
    if len(rows) < 2:
        return
    print("growth per doubling of run length (2.0 is linear):")
    for earlier, later in zip(rows, rows[1:]):
        span = later["tokens"] / earlier["tokens"]
        append = later["append_wire"] / earlier["append_wire"]
        before = later["snapshot_wire"] / earlier["snapshot_wire"]
        print(
            f"  {earlier['tokens']:>5} -> {later['tokens']:<5}"
            f"  (x{span:.0f} tokens)"
            f"   append x{append:.2f}   snapshot x{before:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure AR frame payloads in both shapes."
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(n) for n in DEFAULT_LENGTHS),
        help="comma-separated token counts",
    )
    args = parser.parse_args()
    lengths = [
        int(part) for part in args.lengths.split(",") if part
    ]
    assert lengths, "measure at least one length"
    _report([_measure(count) for count in sorted(lengths)])


if __name__ == "__main__":
    main()
