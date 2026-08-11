"""Report progress while model weights are read into memory.

Companion to :mod:`hf_download`, which covers the step before this
one. Once the weights are on disk the user still waits, often longer,
and until now the UI said only "Loading" with no way to tell a slow
load from a hung one.

There is no progress hook to borrow. ``from_pretrained`` emits a shard
counter through ``transformers``' own tqdm, but that is coarse,
differs across the three ``transformers`` versions this repo pins, and
says nothing about the host-to-device copy. So this samples an
external counter instead, exactly as :mod:`hf_download` samples the
cache directory: resident set size while weights land in RAM, and CUDA
allocated bytes while they land in VRAM.

Two details are what make the reading trustworthy.

**One fraction, not two phases.** LLaDA loads with
``device_map="auto"`` on CUDA, so accelerate streams shards straight
to the GPU and RSS barely moves; SmolLM3 memory-maps its shards, so
its RAM and VRAM counters climb over the same stretch rather than one
after the other. A sequential CPU-then-GPU bar would sit at zero
through half of one of them. Reporting ``max(rss_delta, device_delta)``
against a single target is correct either way, and the stage label
follows whichever counter is being reported.

**Except for one shape, which needs a reserved tail.** DiffusionGemma
is a pickled state dict, so ``torch.load`` materializes every byte in
anonymous RAM before a single one is copied to the GPU. Its RSS
reaches the target exactly while the copy has not begun, leaving the
bar at 100% for the whole second half of the wait. Such a load passes
a ``host_stage_ceiling``, which compresses the read into ``[0,
ceiling]`` and scales the copy into what is left. Clamping the
combined reading instead would only move the problem: the bar would
jump the reserved tail in one step the moment the copy started. The
parameter is opt-in and defaults to 1.0, at which the arithmetic is
exactly the single-counter one above, so the loads that already track
their wait are untouched by it.

**The target comes from the requested dtype, not the disk dtype.**
LLaDA on CPU passes ``torch_dtype=None``, which loads a BF16
checkpoint as fp32 and so takes twice its size on disk. Scaling by the
ratio of requested to on-disk dtype keeps the bar honest; when the
on-disk dtype cannot be established the target is 0, which callers
render as an indeterminate spinner rather than a wrong bar.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
)

ProgressSink = Callable[[Dict[str, Any]], None]

# Poll cadence and a ceiling on how long we keep sampling. The load
# finishing is the real bound; the ceiling only keeps the loop finite
# (TigerStyle: put a limit on everything) and never cuts a load short,
# since the sampler is a side channel and the caller does the work.
_POLL_INTERVAL_SECONDS: float = 0.25
_POLL_MAX_SECONDS: float = 60 * 60
_POLL_MAX_ITERATIONS: int = int(
    _POLL_MAX_SECONDS / _POLL_INTERVAL_SECONDS
)
assert _POLL_INTERVAL_SECONDS > 0.0, "poll interval must be positive"
assert _POLL_MAX_ITERATIONS > 0, "poll loop must run at least once"

# The ceiling a load passes when it does not stage through host RAM,
# i.e. no tail is reserved and the whole bar belongs to one counter.
NO_HOST_STAGE_CEILING: float = 1.0
# What a pickled state dict reserves for the host-to-device copy. The
# split is by expected time rather than by bytes: reading ~17 GiB off
# disk dominates the copy across PCIe, so an even split would park the
# bar at half and then sprint. Ten percent is the share of the wait
# the copy has been observed to take.
HOST_STAGE_CEILING_PICKLED: float = 0.9
assert HOST_STAGE_CEILING_PICKLED > 0.0, "ceiling must be positive"
assert (
    HOST_STAGE_CEILING_PICKLED < NO_HOST_STAGE_CEILING
), "a reserved tail must leave room for the copy"

# A safetensors file opens with a u64 length then a JSON header.
# Bounding the read stops a corrupt or hostile length from being
# turned straight into an allocation.
_HEADER_LENGTH_BYTES: int = 8
_HEADER_MAX_BYTES: int = 64 * 1024 * 1024

_SHARD_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_SAFETENSORS_NAME = "model.safetensors"

# Bytes per element for the dtype names safetensors writes in its
# header. Only the width matters here, not the semantics.
_DTYPE_BYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

# Fallback for torch builds without ``torch.dtype.itemsize`` (< 2.1),
# keyed on the repr torch gives its dtypes. Only the widths this repo
# can actually request need to be here.
_TORCH_DTYPE_BYTES: Dict[str, int] = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
}


def safetensors_dtype(path: Path) -> Optional[str]:
    """The single dtype every tensor in ``path`` uses, else ``None``.

    ``None`` for a mixed-dtype file as well as an unreadable one,
    because the caller's target is a whole-checkpoint scale factor and
    one ratio cannot describe a file with several widths. Both cases
    are reported the same way: no bar rather than a wrong one.
    """
    assert isinstance(path, Path), "path must be a Path"
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(_HEADER_LENGTH_BYTES)
            if len(raw_length) < _HEADER_LENGTH_BYTES:
                return None
            length = struct.unpack("<Q", raw_length)[0]
            if length == 0 or length > _HEADER_MAX_BYTES:
                return None
            header = json.loads(handle.read(length))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    found = set()
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        if isinstance(dtype, str):
            found.add(dtype)
    if len(found) != 1:
        return None
    return found.pop()


def dtype_bytes(dtype: Any) -> Optional[int]:
    """Element width in bytes for a torch dtype, or ``None``.

    Takes the torch object rather than a name so callers can hand over
    whatever they are about to pass to ``from_pretrained``. ``None``
    in means "no conversion requested", which is also ``None`` out:
    the caller then skips scaling instead of guessing a width.

    ``torch.dtype.itemsize`` only exists from torch 2.1, and this repo
    pins a different torch in each of its three virtualenvs, so the
    name table below is the fallback. Neither path imports torch,
    which keeps this module testable on its own.
    """
    if dtype is None:
        return None
    size = getattr(dtype, "itemsize", None)
    if isinstance(size, int) and size > 0:
        return size
    return _TORCH_DTYPE_BYTES.get(str(dtype))


def _disk_bytes_and_dtype(
    checkpoint: Path,
) -> Tuple[int, Optional[str]]:
    """On-disk weight bytes for ``checkpoint`` and their dtype name.

    Returns ``(0, None)`` when the layout is not one we can measure,
    which the caller turns into an indeterminate bar.
    """
    index = checkpoint / _SHARD_INDEX_NAME
    if index.is_file():
        return (_index_total_bytes(index), _index_dtype(checkpoint))
    single = checkpoint / _SINGLE_SAFETENSORS_NAME
    if single.is_file():
        return (
            _file_size(single),
            safetensors_dtype(single),
        )
    # A pickled state dict (the NF4 DiffusionGemma checkpoint). Its
    # bytes are already in the packed form torch.load reproduces, so
    # its size is the target and there is no dtype to scale by.
    for candidate in sorted(checkpoint.glob("*.pt")):
        return (_file_size(candidate), None)
    return (0, None)


def _index_total_bytes(index: Path) -> int:
    """``metadata.total_size`` from a safetensors shard index."""
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    total = metadata.get("total_size")
    return int(total) if isinstance(total, int) and total > 0 else 0


def _index_dtype(checkpoint: Path) -> Optional[str]:
    """Shared dtype across a sharded checkpoint, else ``None``.

    Reads every shard rather than sampling one: a checkpoint that
    keeps, say, fp32 embeddings in a separate shard would otherwise be
    scaled by the wrong ratio, and headers are small and few.
    """
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        return None
    found = set()
    for shard in shards:
        dtype = safetensors_dtype(shard)
        if dtype is None:
            return None
        found.add(dtype)
    if len(found) != 1:
        return None
    return found.pop()


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def load_target_bytes(
    checkpoint: Path, *, target_dtype: Any = None
) -> int:
    """Bytes the loaded weights are expected to occupy in memory.

    ``0`` means indeterminate, which every caller renders as a spinner
    with no bar. That is the deliberate outcome for an unrecognized
    layout, an unreadable header, or a mixed-dtype checkpoint: a bar
    that is confidently wrong is worse than no bar.
    """
    assert isinstance(checkpoint, Path), "checkpoint must be a Path"
    if not checkpoint.is_dir():
        return 0
    disk_total, disk_dtype = _disk_bytes_and_dtype(checkpoint)
    if disk_total <= 0:
        return 0
    wanted = dtype_bytes(target_dtype)
    if wanted is None or disk_dtype is None:
        return disk_total
    on_disk = _DTYPE_BYTES.get(disk_dtype)
    if on_disk is None or on_disk <= 0:
        return disk_total
    scaled = disk_total * wanted / on_disk
    return int(scaled)


def rss_bytes() -> int:
    """Resident set size of this process, or ``0`` if unavailable.

    Read straight from ``/proc`` rather than adding a psutil
    dependency for two numbers. Non-Linux hosts get ``0``, which
    degrades to the CUDA counter alone (and to no bar at all on a
    CPU-only load), rather than reporting something invented.
    """
    path = "/proc/self/statm"
    try:
        with Path(path).open("r", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            return 0
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def cuda_allocated_bytes() -> int:
    """Bytes torch has allocated on the GPU, or ``0`` if unused."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated())
    except Exception:  # noqa: BLE001 - a probe must not break a load.
        return 0


def progress_sample(
    *,
    baseline_rss: int,
    target_bytes: int,
    peak_fraction: float,
    baseline_device: int = 0,
    host_stage_ceiling: float = NO_HOST_STAGE_CEILING,
) -> Dict[str, Any]:
    """One progress reading, given the previous sample's fraction.

    Pure so the interesting part is testable without a model. The
    caller threads ``peak_fraction`` back in, which is what forces the
    reading upward only: the CPU allocator can hand pages back
    mid-load, and a bar that walks backwards reads as broken even
    though nothing is wrong.

    ``host_stage_ceiling`` below 1.0 splits the bar in two, and is for
    the one load shape that needs it (see the module docstring). At
    the default it does nothing, and the arithmetic is a plain
    ``max(rss_delta, device_delta) / target``.
    """
    assert baseline_rss >= 0, "baseline rss must be non-negative"
    assert baseline_device >= 0, "baseline device must be non-negative"
    assert target_bytes >= 0, "target bytes must be non-negative"
    assert 0.0 <= peak_fraction <= 1.0, "peak fraction must be in [0,1]"
    assert host_stage_ceiling > 0.0, "ceiling must be positive"
    assert host_stage_ceiling <= 1.0, "ceiling must not exceed 1.0"
    resident = rss_bytes() - baseline_rss
    if resident < 0:
        resident = 0
    on_device = cuda_allocated_bytes() - baseline_device
    if on_device < 0:
        on_device = 0
    loaded = resident if resident > on_device else on_device
    # The stage names the counter being reported, so it stays true
    # whether the weights route through RAM first or stream straight
    # to the GPU. Any device bytes at all mean the copy has started;
    # comparing the two counters instead would keep saying "weights"
    # through most of a copy that follows a full read into RAM.
    stage = "device" if on_device > 0 else "weights"
    fraction = _progress_fraction(
        resident=resident,
        on_device=on_device,
        target_bytes=target_bytes,
        host_stage_ceiling=host_stage_ceiling,
    )
    if fraction < peak_fraction:
        fraction = peak_fraction
    return {
        "phase": "load",
        "stage": stage,
        "fraction": round(fraction, 4),
        "loaded_bytes": int(loaded),
        "total_bytes": int(target_bytes),
    }


def _progress_fraction(
    *,
    resident: int,
    on_device: int,
    target_bytes: int,
    host_stage_ceiling: float,
) -> float:
    """Fraction of the load done, before the monotonic floor.

    Branchless-ish by design: the ceiling case is the only one that
    knows the load has two sequential halves, and every other load
    takes the single-counter path that predates it.
    """
    if target_bytes <= 0:
        return 0.0
    if host_stage_ceiling < NO_HOST_STAGE_CEILING and on_device > 0:
        # The read is over (device bytes exist), so the remaining band
        # belongs entirely to the copy. Scaling the copy into it,
        # rather than clamping the combined reading, is what keeps the
        # bar moving: the read already accounted for the whole target,
        # so a clamp would jump straight to full on the first copied
        # byte.
        copied = on_device / target_bytes
        if copied > 1.0:
            copied = 1.0
        span = NO_HOST_STAGE_CEILING - host_stage_ceiling
        return host_stage_ceiling + span * copied
    largest = resident if resident > on_device else on_device
    fraction = largest / target_bytes
    if fraction < 0.0:
        return 0.0
    if fraction > host_stage_ceiling:
        return host_stage_ceiling
    return fraction


@contextmanager
def sample_load_progress(
    *,
    target_bytes: int,
    sink: ProgressSink,
    host_stage_ceiling: float = NO_HOST_STAGE_CEILING,
) -> Iterator[None]:
    """Report load progress to ``sink`` for the body's duration.

    The sampler runs on the helper thread and the load stays on the
    caller's, the opposite of
    :func:`hf_download.download_with_progress`. That is deliberate:
    the load is the heavyweight, library-driven part, and moving it
    between threads to gain a progress bar would be trading real risk
    for a cosmetic one. Reading two counters is the safe thing to
    relocate.

    Pass ``host_stage_ceiling`` only for a load that fills host RAM
    with the whole checkpoint before copying any of it to the device
    (see the module docstring).

    A context manager so the thread is always joined and the sink
    always sees a terminal sample, including when the load raises.
    """
    assert target_bytes >= 0, "target bytes must be non-negative"
    assert host_stage_ceiling > 0.0, "ceiling must be positive"
    assert host_stage_ceiling <= 1.0, "ceiling must not exceed 1.0"
    baseline_rss = rss_bytes()
    # Baselined for the same reason RSS is: the worker is a fresh
    # process, so this should be zero, and reading it rather than
    # assuming it keeps a stray allocation from offsetting the bar.
    baseline_device = cuda_allocated_bytes()
    stop = threading.Event()
    state: Dict[str, Any] = {"peak": 0.0, "stage": "weights"}

    def _poll() -> None:
        iterations = 0
        while (
            not stop.is_set()
            and iterations < _POLL_MAX_ITERATIONS
        ):
            sample = progress_sample(
                baseline_rss=baseline_rss,
                baseline_device=baseline_device,
                target_bytes=target_bytes,
                peak_fraction=state["peak"],
                host_stage_ceiling=host_stage_ceiling,
            )
            state["peak"] = sample["fraction"]
            state["stage"] = sample["stage"]
            sink(sample)
            stop.wait(_POLL_INTERVAL_SECONDS)
            iterations += 1

    thread = threading.Thread(
        target=_poll, name="load-progress", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_POLL_INTERVAL_SECONDS * 4)
        # Land on a clean 100% so a small target/actual mismatch does
        # not leave the bar just shy of full as the overlay closes.
        # The stage carries over from the last real sample rather than
        # being named here: a CPU load never touched a device, and
        # this frame must not be the one that claims it did.
        if target_bytes > 0:
            sink(
                {
                    "phase": "load",
                    "stage": state["stage"],
                    "fraction": 1.0,
                    "loaded_bytes": int(target_bytes),
                    "total_bytes": int(target_bytes),
                }
            )
