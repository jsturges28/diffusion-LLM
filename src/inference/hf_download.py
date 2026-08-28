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
from typing import Any, Callable, Dict, Optional

ProgressSink = Callable[[Dict[str, Any]], None]

# Poll cadence for the disk-size sampler and a generous ceiling on how
# long we keep sampling. The download completing is the real bound; the
# ceiling only keeps the poll loop finite (TigerStyle: bound every loop)
# and never cuts a legitimate download short (we block on join after).
_POLL_INTERVAL_SECONDS: float = 0.5
_POLL_MAX_SECONDS: float = 6 * 60 * 60
_POLL_MAX_ITERATIONS: int = int(_POLL_MAX_SECONDS / _POLL_INTERVAL_SECONDS)


def repo_total_bytes(repo_id: str) -> int:
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


def _has_incomplete(blobs_dir: Path) -> bool:
    """Whether the blobs dir has any in-progress ``*.incomplete`` part."""
    if not blobs_dir.is_dir():
        return False
    for entry in blobs_dir.iterdir():
        if entry.name.endswith(".incomplete"):
            return True
    return False


def has_partial_download(repo_id: str) -> bool:
    """Whether an interrupted fetch left parts of ``repo_id`` behind.

    The same ``*.incomplete`` check ``is_repo_cached`` makes, exposed
    because the answer is worth more than the boolean it is folded
    into. "Not cached" covers both a model never fetched and one
    stopped at 8%, and the menu wants to say "resume" for the second
    rather than offering to start it over.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    return _has_incomplete(_repo_blobs_dir(repo_id))


def is_repo_cached(repo_id: str) -> bool:
    """Whether ``repo_id`` is *fully* cached, with no partial parts.

    Both the fast path here and the supervisor's ``_is_downloaded`` use
    this so an interrupted download (leaving ``*.incomplete`` blobs) is
    treated as not-downloaded rather than complete. Re-downloading then
    resumes the remaining parts instead of the cache being misread as
    ready and the model hanging on load.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import snapshot_download

    if _has_incomplete(_repo_blobs_dir(repo_id)):
        return False
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - not (fully) cached.
        return False


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


def progress_sample(done: int, total: int) -> Dict[str, Any]:
    """One progress reading, in the shape every consumer expects."""
    assert done >= 0, "downloaded bytes must be non-negative"
    assert total >= 0, "total bytes must be non-negative"
    fraction = (done / total) if total > 0 else 0.0
    if fraction < 0.0:
        fraction = 0.0
    elif fraction > 1.0:
        fraction = 1.0
    return {
        "fraction": round(fraction, 4),
        "downloaded_bytes": int(done),
        "total_bytes": int(total),
    }


def repo_progress(
    repo_id: str, total_bytes: int
) -> Dict[str, Any]:
    """Sample a fetch that some other process is performing.

    The supervisor runs its downloads as child processes so it can
    terminate one, and reads their progress from here. That costs no
    channel between the two, because progress was never coming from
    the downloader in the first place: it is the size of the cache
    directory on disk, which anyone can measure. ``total_bytes`` is
    passed in rather than looked up because it is one HTTP call and
    the caller samples this twice a second.
    """
    blobs = _repo_blobs_dir(repo_id)
    return progress_sample(_downloaded_bytes(blobs), total_bytes)


def _emit(sink: ProgressSink, done: int, total: int) -> None:
    """Report one progress sample in the shared sink shape."""
    sink(progress_sample(done, total))


class WeightsUnavailableError(RuntimeError):
    """Weights are neither cached nor reachable.

    Distinguished from every other download failure because it is an
    operating condition with an obvious remedy (connect, or download
    the model once from the menu), and because the underlying
    exception for it is a wall of urllib3 retry text that says
    "MaxRetryError" where it means "you are offline".
    """


def describe_unreachable(
    repo_id: str, cause: Optional[BaseException] = None
) -> str:
    """The offline sentence, with or without the exception to blame.

    A download running in a child process reports its outcome as an
    exit status, so the supervisor rebuilds this message from the
    repo alone and the parenthetical is simply left off. In process,
    the caller still has the exception and keeps it.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    message = (
        f"{repo_id} is not downloaded and the Hugging Face Hub"
        " could not be reached. Connect to the internet and try"
        " again, or download this model once while online; after"
        " that it loads from the local cache with no network."
    )
    if cause is None:
        return message
    return (
        message
        + f" (underlying error: {type(cause).__name__})"
    )


# Exception types that mean "the network is not there", across the
# requests, urllib3, and huggingface_hub layers a fetch passes
# through. Matched by name rather than by class so this does not have
# to import three libraries to ask one question, and so it keeps
# working when a library moves an error between modules.
_UNREACHABLE_ERROR_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ConnectTimeoutError",
        "LocalEntryNotFoundError",
        "MaxRetryError",
        "NameResolutionError",
        "NewConnectionError",
        "OfflineModeIsEnabled",
        "ReadTimeout",
        "ReadTimeoutError",
    }
)

# Any real chain is a few links; this only keeps the walk finite.
_CAUSE_CHAIN_MAX = 20


def _is_unreachable(exc: BaseException) -> bool:
    """Whether a failed fetch failed for want of a network.

    Walks the cause chain because the interesting type is usually
    wrapped: a DNS failure surfaces as a urllib3 NameResolutionError
    inside a MaxRetryError inside a requests ConnectionError. Anything
    unrecognized is reported as False so its own message survives
    rather than being relabelled as an offline problem.
    """
    current: Optional[BaseException] = exc
    for _ in range(_CAUSE_CHAIN_MAX):
        if current is None:
            return False
        if type(current).__name__ in _UNREACHABLE_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def download_with_progress(
    repo_id: str, *, sink: ProgressSink
) -> str:
    """Ensure ``repo_id`` is fully cached, reporting progress to ``sink``.

    Returns the local snapshot path. On a cache hit this returns
    immediately without invoking ``sink`` (no download bar) and without
    touching the network. Otherwise the fetch runs on a helper thread
    while this function polls the cache directory size and reports
    ``{fraction, downloaded_bytes, total_bytes}`` to ``sink`` roughly
    twice a second.

    Callers may treat a successful return as proof that every file is
    on disk, which is what lets the workers load with
    ``local_files_only=True`` and never revalidate against the Hub.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import snapshot_download

    # Fast path: fully cached (and not partial) already. A partial cache
    # falls through so the fetch below resumes the ``*.incomplete`` parts
    # and the poller continues from the on-disk size.
    if is_repo_cached(repo_id):
        return snapshot_download(repo_id, local_files_only=True)

    total_bytes = repo_total_bytes(repo_id)
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
        cause = failure["error"]
        # Being offline with nothing cached is the one failure here a
        # user can act on, and it arrives as a wall of urllib3 retry
        # text. Everything else (a full disk, permissions, a 403) is
        # reraised untouched so its own message survives.
        if _is_unreachable(cause):
            raise WeightsUnavailableError(
                describe_unreachable(repo_id, cause)
            ) from cause
        raise cause

    # Land on a clean 100% once the snapshot is complete (guards against
    # a small total/disk mismatch leaving the bar just shy of full).
    if total_bytes > 0:
        _emit(sink, total_bytes, total_bytes)
    path = result.get("path")
    assert path is not None, "download finished without a path"
    return path
