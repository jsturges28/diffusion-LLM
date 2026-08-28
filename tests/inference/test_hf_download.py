"""Tests that an unreachable Hub is reported as such, not as a stack.

Strategy: classify synthetic exception chains shaped the way the real
layers stack them, then drive ``download_with_progress`` with a fetch
that raises, checking which failures get translated and which are
passed through untouched. No network and no Hub are involved.

What passing proves is that the one failure a user can act on says so.
Being offline with nothing cached surfaced as a urllib3 retry dump
reading "MaxRetryError ... Max retries exceeded", which describes the
mechanism rather than the situation and offers no remedy. Every other
failure keeps its own message, because relabelling a 403 or a full
disk as a connectivity problem would send the user to fix the wrong
thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from src.inference import hf_download
from src.inference.hf_download import (
    WeightsUnavailableError,
    _is_unreachable,
    download_with_progress,
)

REPO = "GSAI-ML/LLaDA-8B-Instruct"


class _NameResolutionError(Exception):
    """Stands in for urllib3's, which is what a DNS failure raises."""


class _MaxRetryError(Exception):
    """Stands in for urllib3's retry wrapper."""


class _RequestsConnectionError(Exception):
    """Stands in for requests.exceptions.ConnectionError."""


# The chain names have to match the real ones, since that is what the
# classifier reads.
_NameResolutionError.__name__ = "NameResolutionError"
_MaxRetryError.__name__ = "MaxRetryError"
_RequestsConnectionError.__name__ = "ConnectionError"


def _offline_chain() -> BaseException:
    """The exact nesting the maintainer's offline run produced."""
    try:
        try:
            try:
                raise _NameResolutionError(
                    "Failed to resolve 'huggingface.co'"
                )
            except _NameResolutionError as dns:
                raise _MaxRetryError(
                    "Max retries exceeded"
                ) from dns
        except _MaxRetryError as retries:
            raise _RequestsConnectionError(
                "connection failed"
            ) from retries
    except _RequestsConnectionError as outer:
        return outer
    raise AssertionError("unreachable")


# -- classifying a failure --


def test_a_wrapped_dns_failure_is_recognized() -> None:
    """Three layers deep, which is where the real one sits."""
    assert _is_unreachable(_offline_chain())


def test_a_bare_connectivity_error_is_recognized() -> None:
    assert _is_unreachable(_RequestsConnectionError("no route"))


def test_an_unrelated_failure_is_not_recognized() -> None:
    """Negative space, and the reason this classifies rather than
    assuming: a 403 or a full disk is not an offline problem, and
    saying so would send the user to fix the wrong thing."""
    assert not _is_unreachable(
        PermissionError("403 Forbidden")
    )
    assert not _is_unreachable(OSError("No space left on device"))
    assert not _is_unreachable(ValueError("bad repo id"))


def test_the_walk_survives_a_cyclic_chain() -> None:
    """A bounded walk, so a self-referential chain cannot hang the
    worker's load thread."""
    first = ValueError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert not _is_unreachable(first)


# -- what download_with_progress does with it --


def _install_fetch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cached: bool,
    error: BaseException,
) -> None:
    """Point the fetch at a failure of a chosen kind.

    Only ``snapshot_download`` is replaced, on the real
    ``huggingface_hub``, because the module also reads
    ``huggingface_hub.constants`` for the cache location. The two
    disk helpers are stubbed so nothing touches a real cache.
    """

    def fake_snapshot_download(
        repo_id: str, **kwargs: Any
    ) -> str:
        raise error

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        hf_download, "is_repo_cached", lambda repo_id: cached
    )
    monkeypatch.setattr(
        hf_download, "repo_total_bytes", lambda repo_id: 0
    )
    monkeypatch.setattr(
        hf_download,
        "_repo_blobs_dir",
        lambda repo_id: Path("/nonexistent-cache"),
    )


def _sink() -> Callable[[Dict[str, Any]], None]:
    seen: List[Dict[str, Any]] = []
    return seen.append


def test_offline_and_uncached_names_the_situation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fetch(
        monkeypatch, cached=False, error=_offline_chain()
    )

    with pytest.raises(WeightsUnavailableError) as caught:
        download_with_progress(REPO, sink=_sink())

    message = str(caught.value)
    assert REPO in message
    assert "not downloaded" in message
    assert "could not be reached" in message
    # The remedy, which is the part a retry dump never carried.
    assert "download this model once while online" in message


def test_the_original_failure_is_kept_as_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translated for the user, not thrown away for the log."""
    _install_fetch(
        monkeypatch, cached=False, error=_offline_chain()
    )

    with pytest.raises(WeightsUnavailableError) as caught:
        download_with_progress(REPO, sink=_sink())

    assert caught.value.__cause__ is not None


def test_an_unrelated_failure_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair to the test above. A permissions problem must arrive
    as a permissions problem."""
    _install_fetch(
        monkeypatch,
        cached=False,
        error=PermissionError("403 Forbidden"),
    )

    with pytest.raises(PermissionError):
        download_with_progress(REPO, sink=_sink())


def test_a_cached_repo_never_reaches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path, which is the whole basis for the workers
    loading with local_files_only: a successful return means every
    file is already on disk."""
    calls: List[Dict[str, Any]] = []

    def fake_snapshot_download(
        repo_id: str, **kwargs: Any
    ) -> str:
        calls.append(dict(kwargs))
        return "/cache/snapshot"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        hf_download, "is_repo_cached", lambda repo_id: True
    )

    path = download_with_progress(REPO, sink=_sink())

    assert path == "/cache/snapshot"
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
