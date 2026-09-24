"""What the meter measures, and when it declines to answer.

Strategy: three groups. The `/proc` parser is exercised against real
files, including the one shape that breaks the obvious implementation:
an executable whose name contains a space and a parenthesis, which a
whitespace split silently misreads as data. The rate arithmetic is
driven with both clocks substituted, so a known CPU delta over a known
wall interval has one right answer rather than a plausible range. And
VRAM is checked absent on this CUDA-less host and present with
``mem_get_info`` replaced.

The busy fraction is asserted against a stated core count throughout,
because "CPU percent" means two things that differ by the width of the
machine: one saturated thread is 100% of a core and about 3% of a
32-core host, and only the latter can be drawn on the same axis as a
VRAM fraction. A test that accepted either would not be testing the
decision.

Passing proves the parser survives a hostile name, a rate needs
two readings before it reports one, the fraction is of the machine
and stays in its axis, and every path answers None rather than zero
where there is nothing to measure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from src.backends import resource_sampler
from src.backends.resource_sampler import (
    KIND_CPU,
    KIND_VRAM,
    CpuSampler,
    core_count,
    read_process_cpu_seconds,
    vram_sample,
)

# A real /proc/<pid>/stat line, trimmed to the fields that matter.
# utime is the 14th field and stime the 15th; here 250 and 50 ticks,
# which at the usual 100 Hz is 3.0 CPU seconds together.
STAT_UTIME_TICKS = 250
STAT_STIME_TICKS = 50


# Fields 3 to 13 of the file, the ones between the process name and
# utime. Named rather than counted, because the arity is the whole
# point of the fixture: getting it wrong shifts what the parser reads
# and the test then disagrees with a parser that is correct, which is
# exactly what happened on the first attempt here.
STAT_FIELDS_BEFORE_UTIME = (
    "S",          # state
    "1",          # ppid
    "1",          # pgrp
    "1",          # session
    "0",          # tty_nr
    "-1",         # tpgid
    "4194304",    # flags
    "0",          # minflt
    "0",          # cminflt
    "0",          # majflt
    "0",          # cmajflt
)


def _stat_line(comm: str) -> str:
    """A stat file whose process is named *comm*."""
    fields = list(STAT_FIELDS_BEFORE_UTIME) + [
        str(STAT_UTIME_TICKS),
        str(STAT_STIME_TICKS),
    ]
    return f"1234 ({comm}) " + " ".join(fields) + " 0 0 0\n"


def _point_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, comm: str
) -> None:
    stat = tmp_path / "stat"
    stat.write_text(_stat_line(comm))
    monkeypatch.setattr(resource_sampler, "PROC_SELF_STAT", stat)


def _expected_seconds() -> float:
    ticks = os.sysconf("SC_CLK_TCK")
    return (STAT_UTIME_TICKS + STAT_STIME_TICKS) / ticks


# -- the parser --


def test_the_real_proc_file_reads(tmp_path: Path) -> None:
    """Unpatched, against this host.

    The synthetic files below only prove it is self-consistent.
    This proves it matches what the kernel actually writes, which
    is the half a fixture cannot check.
    """
    if not resource_sampler.PROC_SELF_STAT.exists():
        pytest.skip("no /proc on this platform")

    seconds = read_process_cpu_seconds()

    assert seconds is not None
    assert seconds > 0


def test_an_ordinary_name_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_at(monkeypatch, tmp_path, "python3.12")

    assert read_process_cpu_seconds() == _expected_seconds()


def test_a_name_with_a_space_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The trap. A whitespace split shifts each later field along
    and reads a neighbouring number as CPU time, which looks
    plausible and is wrong."""
    _point_at(monkeypatch, tmp_path, "my worker")

    assert read_process_cpu_seconds() == _expected_seconds()


def test_a_name_with_parentheses_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The trap's other half. Finding the *first* parenthesis is as
    wrong as splitting on spaces; only the last one is reliable."""
    _point_at(monkeypatch, tmp_path, "worker (llada)")

    assert read_process_cpu_seconds() == _expected_seconds()


def test_a_missing_proc_file_answers_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A platform without /proc. Absent, so the meter hides rather
    than drawing a flat line at zero."""
    monkeypatch.setattr(
        resource_sampler, "PROC_SELF_STAT", tmp_path / "absent"
    )

    assert read_process_cpu_seconds() is None


def test_a_corrupt_proc_file_answers_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative space. Something is there and it is not this format,
    which must not become a number."""
    stat = tmp_path / "stat"
    stat.write_text("not a stat file at all\n")
    monkeypatch.setattr(resource_sampler, "PROC_SELF_STAT", stat)

    assert read_process_cpu_seconds() is None


# -- the rate --


class _Clocks:
    """Both clocks the sampler reads, under the test's control.

    Substituted rather than slept through: a rate test that waits for
    real time either takes seconds or asserts a range wide enough to
    pass whatever the code does.
    """

    def __init__(self) -> None:
        self.cpu_seconds: Optional[float] = 0.0
        self.wall_seconds = 1000.0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            resource_sampler,
            "read_process_cpu_seconds",
            lambda: self.cpu_seconds,
        )
        monkeypatch.setattr(
            resource_sampler.time,
            "monotonic",
            lambda: self.wall_seconds,
        )

    def advance(self, *, cpu: float, wall: float) -> None:
        assert self.cpu_seconds is not None
        self.cpu_seconds += cpu
        self.wall_seconds += wall


def _with_cores(
    monkeypatch: pytest.MonkeyPatch, cores: int
) -> None:
    monkeypatch.setattr(
        resource_sampler, "core_count", lambda: cores
    )


def _rate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpu: float,
    wall: float,
    cores: int,
) -> Tuple[Optional[Dict[str, Any]], CpuSampler]:
    """One interval of *wall* seconds costing *cpu* seconds of CPU."""
    clocks = _Clocks()
    clocks.install(monkeypatch)
    _with_cores(monkeypatch, cores)
    sampler = CpuSampler()
    assert sampler.sample() is None, "the first tick has no interval"
    clocks.advance(cpu=cpu, wall=wall)
    return sampler.sample(), sampler


def test_the_first_sample_has_nothing_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None, not zero. Reporting an idle worker on the first tick
    would describe an interval nobody observed."""
    _Clocks().install(monkeypatch)

    assert CpuSampler().sample() is None


def test_one_saturated_core_is_one_core_of_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole second of CPU in a whole second of wall clock."""
    sample, _ = _rate(monkeypatch, cpu=1.0, wall=1.0, cores=8)

    assert sample is not None
    assert sample["kind"] == KIND_CPU
    assert sample["busy_cores"] == 1.0


def test_the_fraction_is_of_the_machine_not_of_one_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decision this file exists to pin. The same one busy core is
    an eighth of an 8-core host and a thirty-second of a 32-core one;
    read as a fraction of a core it would be 1.0 on both and would
    draw as a full meter on an almost idle machine."""
    on_eight, _ = _rate(monkeypatch, cpu=1.0, wall=1.0, cores=8)
    on_thirty_two, _ = _rate(monkeypatch, cpu=1.0, wall=1.0, cores=32)

    assert on_eight is not None
    assert on_thirty_two is not None
    assert on_eight["fraction"] == pytest.approx(0.125)
    assert on_thirty_two["fraction"] == pytest.approx(0.03125)


def test_a_fully_loaded_machine_reads_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every core busy for the whole interval."""
    sample, _ = _rate(monkeypatch, cpu=8.0, wall=1.0, cores=8)

    assert sample is not None
    assert sample["fraction"] == pytest.approx(1.0)


def test_an_overshoot_is_clamped_to_the_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two clocks are read a moment apart, so a saturated machine
    can measure slightly more CPU than wall time times cores. The
    chart has a top, so the figure does too."""
    sample, _ = _rate(monkeypatch, cpu=9.0, wall=1.0, cores=8)

    assert sample is not None
    assert sample["fraction"] == 1.0


def test_an_idle_worker_reads_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero is a real measurement once an interval exists to measure,
    unlike the None on the first tick."""
    sample, _ = _rate(monkeypatch, cpu=0.0, wall=1.0, cores=8)

    assert sample is not None
    assert sample["fraction"] == 0.0


def test_two_samples_in_one_instant_report_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero interval has nothing to divide by, and the next tick
    will. Guarded rather than risked, because it is a division."""
    sample, _ = _rate(monkeypatch, cpu=0.5, wall=0.0, cores=8)

    assert sample is None


def test_a_sampler_recovers_after_an_unreadable_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed read must not poison the series. The interval simply
    spans the gap, which is the honest reading of it."""
    clocks = _Clocks()
    clocks.install(monkeypatch)
    _with_cores(monkeypatch, 4)
    sampler = CpuSampler()
    sampler.sample()

    clocks.cpu_seconds = None
    assert sampler.sample() is None

    clocks.cpu_seconds = 2.0
    clocks.wall_seconds += 1.0

    assert sampler.sample() is not None


def test_the_core_count_never_divides_by_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os.cpu_count`` may answer None, and a zero denominator would
    turn a meter into a crash."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert core_count() == 1


# -- VRAM --


def test_vram_is_absent_without_a_card() -> None:
    """Unpatched, on this host. ``mem_get_info`` raises here rather
    than answering zero, which is why the guard is not decoration."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("this asserts the no-CUDA path")

    assert vram_sample() is None


def test_vram_reports_what_the_card_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 24 GiB card with 17 GiB resident, which is a LLaDA-shaped
    reading. Used is derived from free so the two cannot disagree."""
    import torch

    total = 24 * 1024**3
    used = 17 * 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (total - used, total)
    )

    sample = vram_sample()

    assert sample is not None
    assert sample["kind"] == KIND_VRAM
    assert sample["used_bytes"] == used
    assert sample["total_bytes"] == total
    assert sample["fraction"] == pytest.approx(used / total)


def test_a_card_that_reports_no_memory_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative space, and a guarded division: a total of zero would
    otherwise be the one arithmetic error a meter can make."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))

    assert vram_sample() is None


def test_a_raising_card_is_reported_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that falls over mid-run hides the meter rather than
    taking the socket down with it."""
    import torch

    def _explode() -> Tuple[int, int]:
        raise RuntimeError("device lost")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _explode)

    assert vram_sample() is None
