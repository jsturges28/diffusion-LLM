"""Tests for the model-load progress sampler.

What is tested: the two halves of ``load_progress`` that decide
whether the bar is honest. First, target derivation, which reads a
checkpoint's layout off disk and scales it from the on-disk dtype to
the one the caller is about to request. Second, the per-sample
arithmetic, which turns two memory counters into a fraction.

Strategy: build real (tiny) safetensors and index files in
``tmp_path`` rather than mocking the readers, so the header parsing is
exercised for real; and drive ``progress_sample`` with monkeypatched
counters, since it is pure apart from those two reads.

A pass proves that a recognized checkpoint yields a target scaled to
the requested dtype, that every unmeasurable layout yields 0 (the
indeterminate signal) rather than a plausible-looking wrong number,
and that the fraction is clamped to [0, 1] and never decreases.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.inference import load_progress

GIB = 1024 * 1024 * 1024


def write_safetensors(
    path: Path, *, dtypes: List[str], payload: bytes = b""
) -> None:
    """Write a file with a valid safetensors header and no real data.

    Only the header is ever read by the code under test, so the tensor
    payload can be empty (or padded, when the test cares about size).
    """
    header: Dict[str, Any] = {}
    for index, dtype in enumerate(dtypes):
        header[f"tensor_{index}"] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, 0],
        }
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(blob)) + blob + payload
    )


def write_index(directory: Path, *, total_size: int) -> None:
    """Write a shard index declaring ``total_size`` weight bytes."""
    path = directory / "model.safetensors.index.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {"a": "model-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )


class FakeDtype:
    """Stand-in for a torch dtype, which the module never imports."""

    def __init__(self, itemsize: int, name: str) -> None:
        self.itemsize = itemsize
        self._name = name

    def __str__(self) -> str:
        return self._name


BF16 = FakeDtype(2, "torch.bfloat16")
FP32 = FakeDtype(4, "torch.float32")


# ---- Header reading ----


def test_safetensors_dtype_reads_a_uniform_header(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(shard, dtypes=["BF16", "BF16", "BF16"])
    assert load_progress.safetensors_dtype(shard) == "BF16"


def test_safetensors_dtype_is_none_for_mixed_widths(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(shard, dtypes=["BF16", "F32"])
    assert load_progress.safetensors_dtype(shard) is None


def test_safetensors_dtype_ignores_the_metadata_entry(
    tmp_path: Path,
) -> None:
    """__metadata__ is a string map, not a tensor; counting it as one
    would make every real checkpoint look mixed-dtype."""
    shard = tmp_path / "model.safetensors"
    blob = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "w": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 0],
            },
        }
    ).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob)
    assert load_progress.safetensors_dtype(shard) == "BF16"


def test_safetensors_dtype_is_none_for_a_truncated_file(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"\x00\x01\x02")
    assert load_progress.safetensors_dtype(shard) is None


def test_safetensors_dtype_rejects_an_absurd_header_length(
    tmp_path: Path,
) -> None:
    """A huge declared length must be refused, not allocated."""
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", 1 << 62) + b"{}")
    assert load_progress.safetensors_dtype(shard) is None


def test_safetensors_dtype_is_none_for_a_zero_length_header(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", 0))
    assert load_progress.safetensors_dtype(shard) is None


def test_safetensors_dtype_is_none_for_garbage_json(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    blob = b"not json at all"
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob)
    assert load_progress.safetensors_dtype(shard) is None


# ---- dtype widths ----


def test_dtype_bytes_prefers_itemsize() -> None:
    assert load_progress.dtype_bytes(BF16) == 2
    assert load_progress.dtype_bytes(FP32) == 4


def test_dtype_bytes_falls_back_to_the_name_table() -> None:
    """Torch below 2.1 has no dtype.itemsize; the name resolves."""

    class OldDtype:
        def __str__(self) -> str:
            return "torch.bfloat16"

    assert load_progress.dtype_bytes(OldDtype()) == 2


def test_dtype_bytes_is_none_when_no_dtype_is_requested() -> None:
    """None means 'load at the default', which is not a width."""
    assert load_progress.dtype_bytes(None) is None


def test_dtype_bytes_is_none_for_an_unknown_object() -> None:
    assert load_progress.dtype_bytes(object()) is None


# ---- Target derivation ----


def test_target_from_a_sharded_index_at_the_same_dtype(
    tmp_path: Path,
) -> None:
    write_index(tmp_path, total_size=16 * GIB)
    write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors", dtypes=["BF16"]
    )
    write_safetensors(
        tmp_path / "model-00002-of-00002.safetensors", dtypes=["BF16"]
    )
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=BF16
    )
    assert target == 16 * GIB


def test_target_doubles_when_bf16_is_loaded_as_fp32(
    tmp_path: Path,
) -> None:
    """LLaDA on CPU passes torch_dtype=None, which loads a BF16
    checkpoint as fp32 and occupies twice its size on disk."""
    write_index(tmp_path, total_size=8 * GIB)
    write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors", dtypes=["BF16"]
    )
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=FP32
    )
    assert target == 16 * GIB


def test_target_halves_when_fp32_is_loaded_as_bf16(
    tmp_path: Path,
) -> None:
    write_index(tmp_path, total_size=8 * GIB)
    write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors", dtypes=["F32"]
    )
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=BF16
    )
    assert target == 4 * GIB


def test_target_is_unscaled_when_no_dtype_is_requested(
    tmp_path: Path,
) -> None:
    write_index(tmp_path, total_size=8 * GIB)
    write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors", dtypes=["BF16"]
    )
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=None
    )
    assert target == 8 * GIB


def test_target_is_indeterminate_for_a_mixed_dtype_checkpoint(
    tmp_path: Path,
) -> None:
    """One ratio cannot describe several widths, so the honest
    answer is no scaling rather than a wrong scale."""
    write_index(tmp_path, total_size=8 * GIB)
    write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors", dtypes=["BF16"]
    )
    write_safetensors(
        tmp_path / "model-00002-of-00002.safetensors", dtypes=["F32"]
    )
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=BF16
    )
    assert target == 8 * GIB  # unscaled, not wrongly scaled


def test_target_from_a_single_safetensors_file(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(shard, dtypes=["BF16"], payload=b"x" * 4096)
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=BF16
    )
    assert target == shard.stat().st_size


def test_target_from_a_single_pt_file_is_never_scaled(
    tmp_path: Path,
) -> None:
    """The NF4 checkpoint is already packed; its bytes are the target
    and there is no on-disk dtype to scale from."""
    blob = tmp_path / "model_nf4.pt"
    blob.write_bytes(b"z" * 8192)
    target = load_progress.load_target_bytes(
        tmp_path, target_dtype=FP32
    )
    assert target == 8192


def test_target_is_zero_for_an_unrecognized_directory(
    tmp_path: Path,
) -> None:
    note = tmp_path / "README.md"
    note.write_text("nothing here", encoding="utf-8")
    assert load_progress.load_target_bytes(tmp_path) == 0


def test_target_is_zero_for_a_missing_directory(
    tmp_path: Path,
) -> None:
    assert load_progress.load_target_bytes(tmp_path / "nope") == 0


def test_target_is_zero_when_the_index_has_no_total_size(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )
    assert load_progress.load_target_bytes(tmp_path) == 0


def test_target_is_zero_when_the_index_is_corrupt(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        "{ truncated", encoding="utf-8"
    )
    assert load_progress.load_target_bytes(tmp_path) == 0


# ---- Per-sample arithmetic ----


def patch_counters(
    monkeypatch: pytest.MonkeyPatch, *, rss: int, cuda: int
) -> None:
    monkeypatch.setattr(load_progress, "rss_bytes", lambda: rss)
    monkeypatch.setattr(
        load_progress, "cuda_allocated_bytes", lambda: cuda
    )


def test_sample_subtracts_the_rss_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch and transformers are already resident before the load, so
    an unsubtracted RSS would start the bar well above zero."""
    patch_counters(monkeypatch, rss=1200, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["loaded_bytes"] == 200
    assert sample["fraction"] == pytest.approx(0.2)


def test_sample_reports_the_larger_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device_map='auto' streams to the GPU with RSS barely moving, so
    the reading has to follow whichever counter is climbing."""
    patch_counters(monkeypatch, rss=1010, cuda=800)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["loaded_bytes"] == 800
    assert sample["stage"] == "device"


def test_sample_stage_is_device_as_soon_as_the_copy_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any device bytes at all mean the copy is under way, even while
    RSS still leads. Comparing the two counters instead would keep
    saying 'weights' through nearly all of a copy that follows a full
    read into RAM, which is what made the label flash by at the end."""
    patch_counters(monkeypatch, rss=1900, cuda=100)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["stage"] == "device"


def test_sample_stage_is_weights_before_anything_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both counters are zero on the first sample of every load. That
    must not read as 'moving to GPU', least of all on a CPU load with
    no GPU to move to."""
    patch_counters(monkeypatch, rss=1000, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["stage"] == "weights"


def test_sample_stage_stays_weights_on_a_cpu_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_counters(monkeypatch, rss=1500, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["stage"] == "weights"


def test_sample_never_goes_backwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU allocator can return pages mid-load; a bar that walks
    back reads as broken even though nothing is wrong. The floor is on
    the fraction, which is the thing the user sees; the byte count
    stays honest about what is actually resident."""
    patch_counters(monkeypatch, rss=1300, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.7
    )
    assert sample["fraction"] == pytest.approx(0.7)
    assert sample["loaded_bytes"] == 300


def test_sample_clamps_a_shortfall_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RSS below the baseline is a negative delta, not negative
    progress."""
    patch_counters(monkeypatch, rss=800, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["loaded_bytes"] == 0
    assert sample["fraction"] == 0.0


def test_sample_clamps_an_overshoot_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace memory pushes RSS past the weight target."""
    patch_counters(monkeypatch, rss=4000, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["fraction"] == 1.0


def test_sample_reports_zero_fraction_when_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero target means the layout was unmeasurable. The bytes are
    still reported so the client can show them, but the fraction is
    not invented."""
    patch_counters(monkeypatch, rss=5000, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=0, peak_fraction=0.0
    )
    assert sample["fraction"] == 0.0
    assert sample["total_bytes"] == 0
    assert sample["loaded_bytes"] == 4000


def test_sample_is_tagged_as_the_load_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker_base tells a load apart from a download by this key."""
    patch_counters(monkeypatch, rss=1500, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["phase"] == "load"


def test_sample_subtracts_the_device_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh worker should have nothing on the GPU, but reading the
    counter beats assuming it: a stray allocation would otherwise
    offset the whole bar."""
    patch_counters(monkeypatch, rss=1000, cuda=1400)
    sample = load_progress.progress_sample(
        baseline_rss=1000,
        target_bytes=1000,
        peak_fraction=0.0,
        baseline_device=1000,
    )
    assert sample["loaded_bytes"] == 400
    assert sample["fraction"] == pytest.approx(0.4)


def test_sample_rejects_a_negative_baseline() -> None:
    with pytest.raises(AssertionError):
        load_progress.progress_sample(
            baseline_rss=-1, target_bytes=1000, peak_fraction=0.0
        )


def test_sample_rejects_a_negative_target() -> None:
    with pytest.raises(AssertionError):
        load_progress.progress_sample(
            baseline_rss=0, target_bytes=-1, peak_fraction=0.0
        )


# ---- The reserved tail (pickled state dicts) ----


def sample_with_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rss: int,
    cuda: int,
    peak_fraction: float = 0.0,
) -> Dict[str, Any]:
    """One sample from a load that stages through host RAM."""
    patch_counters(monkeypatch, rss=rss, cuda=cuda)
    return load_progress.progress_sample(
        baseline_rss=1000,
        target_bytes=1000,
        peak_fraction=peak_fraction,
        host_stage_ceiling=0.9,
    )


def test_default_ceiling_reserves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two loads that already track their wait must be untouched
    by this, so the default has to reduce to the plain arithmetic."""
    patch_counters(monkeypatch, rss=2000, cuda=0)
    sample = load_progress.progress_sample(
        baseline_rss=1000, target_bytes=1000, peak_fraction=0.0
    )
    assert sample["fraction"] == 1.0
    assert load_progress.NO_HOST_STAGE_CEILING == 1.0


def test_read_phase_is_capped_at_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported bug: unpickling fills RAM to the whole target
    before a byte is copied, which used to leave the bar at 100% with
    the copy still to come."""
    sample = sample_with_ceiling(monkeypatch, rss=2000, cuda=0)
    assert sample["fraction"] == pytest.approx(0.9)
    assert sample["stage"] == "weights"


def test_read_phase_below_the_ceiling_is_unscaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is a ceiling, not a rescale: a read that is genuinely
    40% done reports 40%, not 36%."""
    sample = sample_with_ceiling(monkeypatch, rss=1400, cuda=0)
    assert sample["fraction"] == pytest.approx(0.4)


def test_copy_phase_scales_into_the_reserved_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half the copy done puts the bar half way through the tail."""
    sample = sample_with_ceiling(
        monkeypatch, rss=2000, cuda=500, peak_fraction=0.9
    )
    assert sample["fraction"] == pytest.approx(0.95)
    assert sample["stage"] == "device"


def test_copy_phase_starts_where_the_read_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handoff must not jump. Clamping the combined reading rather
    than scaling the copy would leap the whole tail here, because RSS
    already accounts for the entire target."""
    sample = sample_with_ceiling(
        monkeypatch, rss=2000, cuda=1, peak_fraction=0.9
    )
    assert sample["fraction"] == pytest.approx(0.9, abs=1e-3)


def test_copy_phase_finishes_at_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = sample_with_ceiling(
        monkeypatch, rss=2000, cuda=1000, peak_fraction=0.9
    )
    assert sample["fraction"] == 1.0


def test_reserved_tail_never_decreases_across_the_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk a whole pickled load, read then copy, and prove the bar
    only ever moves forward and lands on full."""
    peak = 0.0
    seen: List[float] = []
    for resident in (0, 250, 500, 750, 1000):
        sample = sample_with_ceiling(
            monkeypatch,
            rss=1000 + resident,
            cuda=0,
            peak_fraction=peak,
        )
        peak = sample["fraction"]
        seen.append(peak)
    for copied in (100, 400, 700, 1000):
        sample = sample_with_ceiling(
            monkeypatch,
            rss=2000,
            cuda=copied,
            peak_fraction=peak,
        )
        peak = sample["fraction"]
        seen.append(peak)
    assert seen == sorted(seen), f"bar walked backwards: {seen}"
    assert seen[-1] == 1.0
    assert max(seen[:5]) == pytest.approx(0.9)


def test_sample_rejects_a_ceiling_above_one() -> None:
    with pytest.raises(AssertionError):
        load_progress.progress_sample(
            baseline_rss=0,
            target_bytes=1000,
            peak_fraction=0.0,
            host_stage_ceiling=1.5,
        )


def test_sample_rejects_a_zero_ceiling() -> None:
    with pytest.raises(AssertionError):
        load_progress.progress_sample(
            baseline_rss=0,
            target_bytes=1000,
            peak_fraction=0.0,
            host_stage_ceiling=0.0,
        )


def test_pickled_ceiling_leaves_room_for_the_copy() -> None:
    """A module-level relationship, asserted as documentation."""
    assert load_progress.HOST_STAGE_CEILING_PICKLED > 0.0
    assert (
        load_progress.HOST_STAGE_CEILING_PICKLED
        < load_progress.NO_HOST_STAGE_CEILING
    )


# ---- The context manager ----


def test_context_manager_emits_and_finishes_at_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_counters(monkeypatch, rss=1500, cuda=0)
    monkeypatch.setattr(
        load_progress, "_POLL_INTERVAL_SECONDS", 0.01
    )
    seen: List[Dict[str, Any]] = []
    with load_progress.sample_load_progress(
        target_bytes=1000, sink=seen.append
    ):
        pass
    assert seen, "the sampler emitted nothing"
    assert seen[-1]["fraction"] == 1.0
    assert seen[-1]["loaded_bytes"] == 1000


def test_context_manager_still_finishes_when_the_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thread must be joined and the bar completed even on the
    failure path, or the overlay would sit at a partial fill."""
    patch_counters(monkeypatch, rss=1500, cuda=0)
    monkeypatch.setattr(
        load_progress, "_POLL_INTERVAL_SECONDS", 0.01
    )
    seen: List[Dict[str, Any]] = []
    with pytest.raises(RuntimeError), (
        load_progress.sample_load_progress(
            target_bytes=1000, sink=seen.append
        )
    ):
        raise RuntimeError("load failed")
    assert seen[-1]["fraction"] == 1.0


def test_context_manager_emits_no_final_sample_when_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no target there is no bar to complete, so there is nothing
    to round up to."""
    patch_counters(monkeypatch, rss=1500, cuda=0)
    monkeypatch.setattr(
        load_progress, "_POLL_INTERVAL_SECONDS", 0.01
    )
    seen: List[Dict[str, Any]] = []
    with load_progress.sample_load_progress(
        target_bytes=0, sink=seen.append
    ):
        pass
    assert all(s["fraction"] == 0.0 for s in seen)


def test_final_sample_does_not_invent_a_device_on_a_cpu_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completing frame carries the stage the load actually
    reached. Naming the device here would put 'Moving to GPU' on the
    last frame of a load that never touched one."""
    patch_counters(monkeypatch, rss=1500, cuda=0)
    monkeypatch.setattr(
        load_progress, "_POLL_INTERVAL_SECONDS", 0.01
    )
    seen: List[Dict[str, Any]] = []
    with load_progress.sample_load_progress(
        target_bytes=1000, sink=seen.append
    ):
        pass
    assert seen[-1]["fraction"] == 1.0
    assert seen[-1]["stage"] == "weights"


def test_context_manager_rejects_a_negative_target() -> None:
    with pytest.raises(AssertionError), (
        load_progress.sample_load_progress(
            target_bytes=-1, sink=lambda _p: None
        )
    ):
        pass


def test_poll_loop_is_bounded() -> None:
    """TigerStyle: the loop has a ceiling even though the load
    finishing is what normally ends it."""
    assert load_progress._POLL_MAX_ITERATIONS > 0
    assert load_progress._POLL_MAX_ITERATIONS < 1_000_000


# ---- The real counters ----


def test_rss_bytes_reads_a_plausible_value() -> None:
    """Not a fixed number (it is this process's own footprint), but a
    live interpreter is always resident for more than nothing."""
    value = load_progress.rss_bytes()
    assert value > 0
    assert value < 1024 * GIB


def test_cuda_allocated_is_zero_without_a_gpu() -> None:
    """Must degrade quietly rather than raise: the sampler runs on
    every load, including CPU-only ones on machines without torch
    CUDA support at all."""
    value = load_progress.cuda_allocated_bytes()
    assert value >= 0
