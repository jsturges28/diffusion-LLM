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

import shutil
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


def repo_total_bytes(
    repo_id: str, *, revision: Optional[str] = None
) -> int:
    """Total download size for ``repo_id`` from Hub file metadata.

    Sums the size of every sibling file. Returns 0 when the metadata is
    unavailable (offline / private without token); the caller then
    reports byte counts with an indeterminate percentage.

    ``revision`` sizes the commit that will actually be fetched, which
    matters for the progress bar: asking about the branch tip while
    downloading an older commit reports a percentage of the wrong
    total. ``None`` means the repository default.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(
            repo_id, revision=revision, files_metadata=True
        )
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


def revision_from_snapshot(snapshot: str) -> Optional[str]:
    """The commit a cached snapshot path names, if it names one.

    The cache lays a repository out as
    ``models--org--name/snapshots/<commit>/``, so the commit that was
    actually resolved is already in the path the loaders were given.
    Reading it there costs nothing and needs no network, which is what
    makes it usable for attestation: asking the Hub what a branch
    points at would fail offline, and would answer about now rather
    than about the run.

    Returns ``None`` for a path that is not a Hub snapshot, which is
    the local quantized directory's case. Deliberately does not check
    that the value looks like a sha: a snapshot directory can be named
    for a tag or a branch, and reporting what was resolved beats
    reporting nothing because it was not the shape expected.
    """
    assert isinstance(snapshot, str), "snapshot path required"
    parts = Path(snapshot).parts
    if len(parts) < 2:
        return None
    if parts[-2] != "snapshots":
        return None
    return parts[-1] or None


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

    Takes no revision because the cache does not keep one: every
    commit of a repository shares a single ``blobs`` directory, so a
    partial part cannot be attributed to the commit that was being
    fetched when it was left behind.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    return _has_incomplete(_repo_blobs_dir(repo_id))


def is_repo_cached(
    repo_id: str, *, revision: Optional[str] = None
) -> bool:
    """Whether ``revision`` of ``repo_id`` is fully cached.

    Both the fast path here and the supervisor's ``_is_downloaded`` use
    this so an interrupted download (leaving ``*.incomplete`` blobs) is
    treated as not-downloaded rather than complete. Re-downloading then
    resumes the remaining parts instead of the cache being misread as
    ready and the model hanging on load.

    The revision has to be part of the question. A cache holding some
    other commit of the same repository would otherwise answer "yes"
    to a pinned load that then finds its own files missing, and it
    would find that out offline, where it cannot fix it.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import snapshot_download

    if _has_incomplete(_repo_blobs_dir(repo_id)):
        return False
    try:
        snapshot_download(
            repo_id, revision=revision, local_files_only=True
        )
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


# How much room to leave free beyond the download itself. A disk
# filled to the last byte takes the whole host down with it, not just
# the fetch: the cache lives under the user's home directory, where
# the desktop session, the shell and this app's own run store are all
# writing. One gibibyte is small next to a 17 GiB checkpoint and large
# enough to leave the machine usable if the estimate is off.
DOWNLOAD_SPACE_RESERVE_BYTES: int = 1 * 1024**3

# Below this, a "total" is not a measurement. ``repo_total_bytes``
# returns 0 when Hub metadata is unavailable, and a check against 0
# would pass every time while looking like it had run.
_SPACE_CHECK_MIN_TOTAL_BYTES: int = 1

# A cache path is a handful of components deep. This only keeps the
# ancestor walk in ``free_bytes`` finite.
_ANCESTOR_WALK_MAX: int = 64

assert DOWNLOAD_SPACE_RESERVE_BYTES > _SPACE_CHECK_MIN_TOTAL_BYTES, (
    "the reserve must exceed the floor it is compared past"
)
assert _ANCESTOR_WALK_MAX > 0, "the walk must take a step"


class InsufficientSpaceError(RuntimeError):
    """The download will not fit, found before starting it.

    Its own type rather than an OSError because it is an operating
    condition with a specific remedy, and because the alternative is
    what happens today: the fetch runs for twenty minutes, fills the
    disk, and fails with ``[Errno 28] No space left on device`` from
    inside a library, having left partial blobs behind and possibly
    taken the desktop session with it.
    """


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding ``path``.

    Walks up to the nearest existing ancestor, because the cache
    directory for a model that has never been fetched does not exist
    yet, and a first download is exactly when this question matters.
    """
    assert isinstance(path, Path), "path required"
    current = path.expanduser()
    for _ in range(_ANCESTOR_WALK_MAX):
        if current.exists():
            usage = shutil.disk_usage(current)
            assert usage.free >= 0, "free space must be non-negative"
            return int(usage.free)
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Root itself did not exist, which cannot happen on a running
    # system; reported as no space rather than crashing the caller.
    return 0


def describe_insufficient_space(
    *, needed_bytes: int, free_bytes_now: int
) -> str:
    """The refusal, in numbers the user can act on.

    Both figures, not just the shortfall, because the action depends
    on the gap: freeing 200 MiB is a different afternoon from freeing
    14 GiB, and a message saying only "not enough space" leaves the
    user to work that out by trial.
    """
    assert needed_bytes >= 0, "needed bytes must be non-negative"
    assert free_bytes_now >= 0, "free bytes must be non-negative"
    return (
        f"This download needs about {_gib(needed_bytes)} GiB free,"
        f" including a {_gib(DOWNLOAD_SPACE_RESERVE_BYTES)} GiB"
        f" reserve, and {_gib(free_bytes_now)} GiB is available."
        " Free some space and try again."
    )


def _gib(value: int) -> str:
    """Bytes as gibibytes, at one decimal place."""
    return f"{value / 1024**3:.1f}"


def space_needed_bytes(repo_id: str, *, total_bytes: int) -> int:
    """Room this download still requires, including the reserve.

    What is left to fetch rather than the repository's full size,
    because a resumed download has already put some of it on disk and
    refusing on the full figure would block a fetch that fits.

    Returns 0 when ``total_bytes`` is not a measurement, which is the
    signal to skip the check: Hub metadata is unavailable offline and
    for a private repo without a token, and a fetch that might fit is
    better than one refused on a number nobody read.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    assert total_bytes >= 0, "total bytes must be non-negative"
    if total_bytes < _SPACE_CHECK_MIN_TOTAL_BYTES:
        return 0
    remaining = total_bytes - _downloaded_bytes(
        _repo_blobs_dir(repo_id)
    )
    if remaining < 0:
        remaining = 0
    return remaining + DOWNLOAD_SPACE_RESERVE_BYTES


def repo_free_bytes(repo_id: str) -> int:
    """Free space where this repository's blobs would land.

    Named per repository rather than taken as a path so callers do not
    have to know the cache layout to ask the question, and so they
    cannot accidentally measure a different filesystem than the one
    the download will write to.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    return free_bytes(_repo_blobs_dir(repo_id))


def check_space_for_download(
    repo_id: str, *, total_bytes: int
) -> None:
    """Raise if this download cannot fit, before it starts."""
    needed = space_needed_bytes(repo_id, total_bytes=total_bytes)
    if needed == 0:
        return
    available = repo_free_bytes(repo_id)
    if available < needed:
        raise InsufficientSpaceError(
            describe_insufficient_space(
                needed_bytes=needed, free_bytes_now=available
            )
        )


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
    repo_id: str,
    *,
    revision: Optional[str] = None,
    sink: ProgressSink,
) -> str:
    """Ensure ``revision`` of ``repo_id`` is cached, with progress.

    Returns the local snapshot path. On a cache hit this returns
    immediately without invoking ``sink`` (no download bar) and without
    touching the network. Otherwise the fetch runs on a helper thread
    while this function polls the cache directory size and reports
    ``{fraction, downloaded_bytes, total_bytes}`` to ``sink`` roughly
    twice a second.

    Callers may treat a successful return as proof that every file of
    that commit is on disk, which is what lets the workers load with
    ``local_files_only=True`` and never revalidate against the Hub.

    ``revision`` reaches every path here, the cache check and the fast
    return as much as the fetch. Pinning only the slow path would pin
    nothing in practice, because after the first download every
    activation takes the fast one.
    """
    assert isinstance(repo_id, str) and repo_id, "repo_id required"
    from huggingface_hub import snapshot_download

    # Fast path: fully cached (and not partial) already. A partial cache
    # falls through so the fetch below resumes the ``*.incomplete`` parts
    # and the poller continues from the on-disk size.
    if is_repo_cached(repo_id, revision=revision):
        return snapshot_download(
            repo_id, revision=revision, local_files_only=True
        )

    total_bytes = repo_total_bytes(repo_id, revision=revision)
    # Before the fetch, not during it. The library discovers a full
    # disk by failing to write to it, which happens partway through a
    # long download and after the damage is done; the size is already
    # known here, so the answer is available for free.
    check_space_for_download(repo_id, total_bytes=total_bytes)
    blobs_dir = _repo_blobs_dir(repo_id)

    result: Dict[str, str] = {}
    failure: Dict[str, BaseException] = {}

    def _fetch() -> None:
        try:
            # Xet is disabled process-wide before the first
            # huggingface_hub import (see server.py / run_worker.py), so
            # bytes land in ``blobs`` as ``*.incomplete`` parts that the
            # poller below can measure as they grow.
            result["path"] = snapshot_download(
                repo_id, revision=revision
            )
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
