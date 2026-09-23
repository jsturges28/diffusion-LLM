"""A download that cannot fit is refused before it starts.

Strategy: drive the arithmetic directly with the two measurements
stubbed, free space and bytes already on disk, so the boundary can be
walked a byte at a time. Then drive the fetch itself to prove the
check is actually on that path and runs before anything is written.
No network, no cache, no real filesystem.

What passing proves is that the failure arrives early and says
something. Today the library discovers a full disk by failing to write
to it: the fetch runs for twenty minutes, fills the partition the
user's home directory is on, and surfaces `[Errno 28] No space left on
device` from inside huggingface_hub. The size was known before the
first byte moved, so that outcome was avoidable for free.

The reserve is the part worth stating. Filling a disk to its last byte
takes the whole session down, not just the download, because the cache
shares a filesystem with the desktop, the shell, and this app's own
run store. So "fits exactly" is treated as not fitting.

The skip is deliberate and is tested as carefully as the refusal.
`repo_total_bytes` returns 0 when Hub metadata is unreachable, and a
check against 0 would pass every time while looking like it had run.
Refusing on a number nobody read would be worse: it would block a
download that fits, offline, where the user cannot see why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from src.inference import hf_download
from src.inference.hf_download import (
    DOWNLOAD_SPACE_RESERVE_BYTES,
    InsufficientSpaceError,
    check_space_for_download,
    describe_insufficient_space,
    download_with_progress,
    free_bytes,
    space_needed_bytes,
)

REPO = "GSAI-ML/LLaDA-8B-Instruct"
GIB = 1024**3
# Stands in for a checkpoint's weights.
TOTAL = 17 * GIB


def _measurements(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free: int,
    on_disk: int = 0,
) -> None:
    """Fix both sides of the comparison.

    The cache path is replaced too, so nothing here depends on what
    the machine running the tests happens to have cached or mounted.
    """
    monkeypatch.setattr(
        hf_download,
        "_repo_blobs_dir",
        lambda repo_id: Path("/fake-cache/blobs"),
    )
    monkeypatch.setattr(
        hf_download, "_downloaded_bytes", lambda blobs: on_disk
    )
    monkeypatch.setattr(
        hf_download, "free_bytes", lambda path: free
    )


# -- what the download needs --


def test_the_need_is_the_download_plus_the_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _measurements(monkeypatch, free=0, on_disk=0)

    needed = space_needed_bytes(REPO, total_bytes=TOTAL)

    assert needed == TOTAL + DOWNLOAD_SPACE_RESERVE_BYTES


def test_a_resumed_download_only_needs_what_is_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measuring the full repository would refuse a fetch that fits.

    An interrupted download leaves its parts on disk and the next
    attempt resumes them, so the bytes already there are not needed
    twice.
    """
    _measurements(monkeypatch, free=0, on_disk=10 * GIB)

    needed = space_needed_bytes(REPO, total_bytes=TOTAL)

    assert needed == 7 * GIB + DOWNLOAD_SPACE_RESERVE_BYTES


def test_a_cache_larger_than_the_repo_needs_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blobs directory holds every commit of a repository, so it
    can exceed one commit's size. The subtraction must not go
    negative and turn into a credit."""
    _measurements(monkeypatch, free=0, on_disk=40 * GIB)

    needed = space_needed_bytes(REPO, total_bytes=TOTAL)

    assert needed == DOWNLOAD_SPACE_RESERVE_BYTES


def test_an_unmeasurable_repo_skips_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero is the signal, not a size. See the module docstring."""
    _measurements(monkeypatch, free=0, on_disk=0)

    assert space_needed_bytes(REPO, total_bytes=0) == 0


# -- the boundary, a byte at a time --


def test_exactly_enough_including_the_reserve_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _measurements(
        monkeypatch, free=TOTAL + DOWNLOAD_SPACE_RESERVE_BYTES
    )

    check_space_for_download(REPO, total_bytes=TOTAL)


def test_one_byte_short_of_the_reserve_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair to the test above, and the whole point of the
    reserve: this download would succeed and leave the machine with
    nothing to write to."""
    _measurements(
        monkeypatch, free=TOTAL + DOWNLOAD_SPACE_RESERVE_BYTES - 1
    )

    with pytest.raises(InsufficientSpaceError):
        check_space_for_download(REPO, total_bytes=TOTAL)


def test_room_for_the_weights_but_not_the_reserve_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the old behaviour and the new one differ most visibly:
    the fetch itself would fit."""
    _measurements(monkeypatch, free=TOTAL)

    with pytest.raises(InsufficientSpaceError):
        check_space_for_download(REPO, total_bytes=TOTAL)


def test_an_unmeasurable_repo_is_never_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative space for the skip: no space at all, and it still
    proceeds, because nothing was measured."""
    _measurements(monkeypatch, free=0)

    check_space_for_download(REPO, total_bytes=0)


# -- what the refusal says --


def test_the_message_names_both_figures_and_the_remedy() -> None:
    """A shortfall alone does not tell the user what to do. Freeing
    200 MiB is a different afternoon from freeing 14 GiB."""
    message = describe_insufficient_space(
        needed_bytes=18 * GIB, free_bytes_now=3 * GIB
    )

    assert "18.0 GiB" in message
    assert "3.0 GiB" in message
    assert "reserve" in message
    assert "Free some space" in message


# -- free space, measured where the bytes will land --


def test_free_space_walks_up_to_a_directory_that_exists(
    tmp_path: Path,
) -> None:
    """A first download is exactly when this is asked, and the cache
    directory for a model never fetched does not exist yet."""
    missing = tmp_path / "models--org--name" / "blobs"

    assert free_bytes(missing) > 0


def test_free_space_of_an_existing_directory_is_read_directly(
    tmp_path: Path,
) -> None:
    assert free_bytes(tmp_path) > 0


# -- the check is on the fetch path, and runs first --


def _sink() -> Callable[[Dict[str, Any]], None]:
    seen: List[Dict[str, Any]] = []
    return seen.append


def test_the_fetch_refuses_before_writing_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property that makes this worth having. A check that ran
    after the download started would report the same problem and
    prevent none of it.
    """
    started: List[str] = []

    def fake_snapshot_download(
        repo_id: str, **kwargs: Any
    ) -> str:
        started.append(repo_id)
        return "/cache/snapshot"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        hf_download,
        "is_repo_cached",
        lambda repo_id, **kwargs: False,
    )
    monkeypatch.setattr(
        hf_download,
        "repo_total_bytes",
        lambda repo_id, **kwargs: TOTAL,
    )
    _measurements(monkeypatch, free=1 * GIB)

    with pytest.raises(InsufficientSpaceError):
        download_with_progress(REPO, sink=_sink())

    assert started == [], "the fetch should never have begun"


def test_a_cached_repo_is_not_asked_for_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model already on disk must keep activating on a full disk.

    Nothing is being fetched, so there is nothing to make room for,
    and refusing here would make a full disk look like a missing
    model.
    """
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, **kwargs: "/cache/snapshot",
    )
    monkeypatch.setattr(
        hf_download,
        "is_repo_cached",
        lambda repo_id, **kwargs: True,
    )
    _measurements(monkeypatch, free=0)

    path = download_with_progress(REPO, sink=_sink())

    assert path == "/cache/snapshot"
