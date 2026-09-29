"""The download child fetches a model's borrowed files after it.

Strategy: run `download_main.main` with a chosen argv and the Hub
calls replaced, recording what it asks for. Passing proves the child
fetches the companion after the checkpoint, at the pinned commit, and
refuses a half-described companion before touching the network,
rather than fetching the checkpoint and failing on the rest.
"""

from __future__ import annotations

import sys
from typing import Any, List, Tuple

import pytest

from src.inference import download_main, hf_download

REPO, REVISION = "org/model", "a" * 40
DONOR, DONOR_REVISION = "org/donor", "b" * 40


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[Any, ...]]:
    seen: List[Tuple[Any, ...]] = []

    def snapshot(repo_id: str, **kwargs: Any) -> str:
        seen.append(("snapshot", repo_id, kwargs.get("revision")))
        return "/cache/snapshot"

    def borrow(repo_id: str, files: Any, *, revision: str) -> Any:
        seen.append(("companion", repo_id, tuple(files), revision))
        return {}

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(
        hf_download, "repo_total_bytes", lambda repo, **kw: 0
    )
    monkeypatch.setattr(hf_download, "fetch_companion_files", borrow)
    return seen


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["download_main", *argv])
    return download_main.main()


def test_the_companion_follows_the_checkpoint(
    calls: List[Tuple[Any, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    code = _run(
        monkeypatch,
        "--repo", REPO,
        "--revision", REVISION,
        "--companion-repo", DONOR,
        "--companion-revision", DONOR_REVISION,
        "--companion-file", "tokenizer.json",
    )

    assert code == download_main.DOWNLOAD_EXIT_OK
    assert calls == [
        ("snapshot", REPO, REVISION),
        ("companion", DONOR, ("tokenizer.json",), DONOR_REVISION),
    ]


def test_no_companion_fetches_only_the_checkpoint(
    calls: List[Tuple[Any, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    code = _run(monkeypatch, "--repo", REPO, "--revision", REVISION)

    assert code == download_main.DOWNLOAD_EXIT_OK
    assert calls == [("snapshot", REPO, REVISION)]


@pytest.mark.parametrize(
    "extra",
    [
        ("--companion-repo", DONOR, "--companion-file", "x"),
        ("--companion-repo", DONOR, "--companion-revision", "b" * 40),
        ("--companion-file", "x"),
    ],
    ids=["no-revision", "no-files", "no-repo"],
)
def test_a_half_described_companion_is_refused_first(
    extra: Tuple[str, ...],
    calls: List[Tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, "--repo", REPO, *extra)

    assert calls == []
