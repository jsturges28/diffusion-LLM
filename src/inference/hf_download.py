"""Download HF Hub model weights with aggregate progress reporting.

Workers call :func:`download_with_progress` before ``from_pretrained``
so the supervisor can show a download progress bar (via the worker's
``/health`` ``downloading`` state) on a model's first activation. The
menu's "Click to Download" veneer calls it too. When the repo is already
cached this returns immediately with no progress, so the bar only appears
for genuine downloads.

Progress is sampled from the cache directory on disk rather than from a
tqdm hook: ``snapshot_download`` only routes its ``tqdm_class`` to the
outer "Fetching N files" bar, not to the per-file byte downloads inside
``hf_hub_download``/``http_get`` (the library documents this), so a custom
tqdm cannot observe byte-level progress. Polling the ``blobs`` directory
size against the repo's total size does, and is independent of whether
Xet or hf_transfer is in play.

Kept separate from any specific worker so both the LLaDA and SmolLM3
workers can share it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict

ProgressSink = Callable[[Dict[str, Any]], None]

# Poll cadence for the disk-size sampler and a generous ceiling on how
# long we keep sampling. The download completing is the real bound; the
# ceiling only keeps the poll loop finite (TigerStyle: bound every loop)
# and never cuts a legitimate download short (we block on join after).
_POLL_INTERVAL_SECONDS: float = 0.5
_POLL_MAX_SECONDS: float = 6 * 60 * 60
_POLL_MAX_ITERATIONS: int = int(_POLL_MAX_SECONDS / _POLL_INTERVAL_SECONDS)


def _repo_total_bytes(repo_id: str) -> int:
    """Total download size for ``repo_id`` from Hub file metadata.

    Sums the size of every sibling file. Returns 0 when the metadata is
    unavailable (offline / private without token); the caller then
    reports byte counts with an indeterminate percentage.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001 - metadata is best-effort.
        return 0
    siblings = getattr(info, "siblings", None) or []
    total = 0
    for sibling in siblings:
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size > 0:
            total += size
    assert total >= 0, "total bytes must be non-negative"
    return total


def _repo_blobs_dir(repo_id: str) -> Path:
    """Local cache ``blobs`` directory for ``repo_id`` (may not exist)."""
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = "models--" + repo_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / folder / "blobs"


def _downloaded_bytes(blobs_dir: Path) -> int:
    """Bytes on disk in ``blobs_dir`` (incl. ``*.incomplete`` parts)."""
    if not blobs_dir.is_dir():
        return 0
    total = 0
    for entry in blobs_dir.iterdir():
        try:
            total += entry.stat().st_size
        except OSError:
            # A blob can be renamed/removed mid-scan; skip it.
            continue
    return total


def _emit(sink: ProgressSink, done: int, total: int) -> None:
    """Report one progress sample in the shared sink shape."""
    assert done >= 0, "downloaded bytes must be non-negative"
    assert total >= 0, "total bytes must be non-negative"
    fraction = (done / total) if total > 0 else 0.0
    if fraction < 0.0:
        fraction = 0.0
    elif fraction > 1.0:
        fraction = 1.0
    sink(
        {
            "fraction": round(fraction, 4),
            "downloaded_bytes": int(done),
            "total_bytes": int(total),
        }
    )


def download_with_progress(
    repo_id: str, *, sink: ProgressSink
) -> str:
    """Ensure ``repo_id`` is fully cached, reporting progress to ``sink``.

    Returns the local snapshot path. On a cache hit this returns
    immediately without invoking ``sink`` (no download bar). Otherwise
    the fetch runs on a helper thread while this function polls the cache
    directory size and reports ``{fraction, downloaded_bytes,
    total_bytes}`` to ``sink`` roughly twice a second.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import snapshot_download

    # Fast path: everything is already cached locally.
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached; fall through to fetch.
        pass

    total_bytes = _repo_total_bytes(repo_id)
    blobs_dir = _repo_blobs_dir(repo_id)

    result: Dict[str, str] = {}
    failure: Dict[str, BaseException] = {}

    def _fetch() -> None:
        try:
            # Xet is disabled process-wide before the first
            # huggingface_hub import (see server.py / run_worker.py), so
            # bytes land in ``blobs`` as ``*.incomplete`` parts that the
            # poller below can measure as they grow.
            result["path"] = snapshot_download(repo_id)
        except BaseException as exc:  # noqa: BLE001 - reraised on join.
            failure["error"] = exc

    worker = threading.Thread(
        target=_fetch, name="hf-download", daemon=True
    )
    worker.start()

    _emit(sink, _downloaded_bytes(blobs_dir), total_bytes)
    iterations = 0
    while worker.is_alive() and iterations < _POLL_MAX_ITERATIONS:
        worker.join(timeout=_POLL_INTERVAL_SECONDS)
        _emit(sink, _downloaded_bytes(blobs_dir), total_bytes)
        iterations += 1

    # If the sampler ceiling was hit on a very long download, stop
    # sampling but still block for the fetch so the return is valid.
    if worker.is_alive():
        worker.join()

    if "error" in failure:
        raise failure["error"]

    # Land on a clean 100% once the snapshot is complete (guards against
    # a small total/disk mismatch leaving the bar just shy of full).
    if total_bytes > 0:
        _emit(sink, total_bytes, total_bytes)
    path = result.get("path")
    assert path is not None, "download finished without a path"
    return path
