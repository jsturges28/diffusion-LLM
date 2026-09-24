"""What the machine is doing right now, for the generator's meter.

One reading per sample, and which resource it describes depends on
where the model landed. On a card that is VRAM, because a diffusion
run's interesting cost is memory and the card is what runs out. On a
CPU-placed model it is how hard the worker is working, because there
is no VRAM to report and a slow CPU run's anxious question is whether
it is progressing at all. Showing both always would ship a flat line
in either configuration.

Both readings are fractions of an available resource, so they share
one 0-100% scale and the drawing on the other end needs no per-kind
branch. That is the reason the CPU figure is normalised against every
core rather than against one: a single saturated thread is 100% of a
core and 3% of a 32-core host, and only the second reading can be
drawn beside a VRAM fraction without misleading.

Nothing here is persisted. The per-run peak in ``worker_base`` is the
durable half of this and answers what a finished run cost; this
answers what is happening now, and a saved sparkline would be a
system monitor's log rather than an explanation of a run.

Separate from ``worker_base`` because none of it needs FastAPI, a
socket or a model, which keeps its tests to milliseconds.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How often the worker samples. Half a second is slow enough to cost
# nothing measurable (one driver call, one small file read) and fast
# enough that a line moves rather than steps.
SAMPLE_INTERVAL_SECONDS = 0.5

# The two things a sample can describe. The reader uses this to label
# the meter and to format the figures, so it travels with every
# sample rather than being inferred from which fields are present.
KIND_VRAM = "vram"
KIND_CPU = "cpu"

# Where this process's own accounting lives. ``self`` rather than a
# pid, because the process doing the sampling is the process holding
# the model: the supervisor never imports this.
PROC_SELF_STAT = Path("/proc/self/stat")


def vram_sample() -> Optional[Dict[str, Any]]:
    """How full the card is, or None when there is no card.

    The whole device rather than this process's allocator. The live
    question a meter answers is whether the card is about to run out,
    and that includes the CUDA context and anything else resident;
    what *this run* holds is already recorded as its peak.

    ``mem_get_info`` raises without CUDA rather than answering zero,
    unlike the peak readers beside it, so the availability check here
    is doing real work and not merely tidying.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return None
        used_bytes = total_bytes - free_bytes
        return {
            "kind": KIND_VRAM,
            "fraction": used_bytes / total_bytes,
            "used_bytes": int(used_bytes),
            "total_bytes": int(total_bytes),
        }
    except Exception:  # noqa: BLE001 - best-effort telemetry
        logger.warning("could not read VRAM", exc_info=True)
        return None


def read_process_cpu_seconds() -> Optional[float]:
    """CPU seconds this process has used, or None where unreadable.

    User and system time together, because a model does both and the
    split is not something a meter can act on.

    Parsed by finding the last ``)`` rather than by splitting on
    whitespace. The second field is the executable name in
    parentheses and it may itself contain spaces and parentheses, so
    a whitespace split shifts every field after it and silently reads
    the wrong numbers. This is the one real trap in the file format.
    """
    try:
        raw = PROC_SELF_STAT.read_text()
    except OSError:
        # No /proc: a platform this build has not been run on. Absent
        # is the honest answer, and the meter hides rather than
        # drawing a flat line at zero.
        return None
    try:
        fields = raw[raw.rindex(")") + 2:].split()
        # Fields 14 and 15 of the file, which are the third and
        # fourth of what is left once the name is out of the way.
        utime_ticks = int(fields[11])
        stime_ticks = int(fields[12])
    except (ValueError, IndexError):
        logger.warning("could not parse %s", PROC_SELF_STAT)
        return None
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    assert ticks_per_second > 0, "clock ticks must be positive"
    return (utime_ticks + stime_ticks) / ticks_per_second


def core_count() -> int:
    """Cores to measure busyness against, never below one.

    ``os.cpu_count`` can answer None, and a zero denominator here
    would turn a working meter into a crash on an unusual host.
    """
    return max(1, os.cpu_count() or 1)


class CpuSampler:
    """Turns the CPU-seconds counter into a busy fraction.

    Stateful because a rate needs two readings, and the first sample
    therefore has nothing to compare against. It answers None rather
    than zero: reporting an idle worker on the first tick would be a
    measurement claim about an interval that was never observed.

    One instance per socket, so two windows measure independently and
    neither resets the other's baseline.
    """

    def __init__(self) -> None:
        self._cpu_seconds: Optional[float] = None
        self._wall_seconds: Optional[float] = None

    def sample(self) -> Optional[Dict[str, Any]]:
        cpu_seconds = read_process_cpu_seconds()
        wall_seconds = time.monotonic()
        if cpu_seconds is None:
            return None
        previous_cpu = self._cpu_seconds
        previous_wall = self._wall_seconds
        self._cpu_seconds = cpu_seconds
        self._wall_seconds = wall_seconds
        if previous_cpu is None or previous_wall is None:
            return None
        elapsed = wall_seconds - previous_wall
        if elapsed <= 0:
            # Two samples inside one clock tick. Nothing to divide by,
            # and the next tick will have an interval worth reading.
            return None
        cores = core_count()
        busy_cores = (cpu_seconds - previous_cpu) / elapsed
        # Clamped because the two clocks are read a moment apart and a
        # fully loaded machine can round to slightly over its own core
        # count, which would draw off the top of the chart.
        fraction = min(1.0, max(0.0, busy_cores / cores))
        return {
            "kind": KIND_CPU,
            "fraction": fraction,
            "busy_cores": round(busy_cores, 3),
            "total_cores": cores,
        }
