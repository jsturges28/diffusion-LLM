"""Files borrowed from another repository never poison its cache.

Strategy: point Hugging Face's cache roots at a temporary directory
and stand in for the one network call, `hf_hub_download`, with a fake
that writes exactly the layout the real one does. Everything else is
the real library reading real directories. Passing proves the hazard
this exists for cannot happen: a borrowed tokenizer fetched on a
machine that never ran the donor model leaves the donor looking
exactly as un-downloaded as before, while the file itself is found
where it was put, and found in the donor's own cache when the donor
is already there, so nothing is fetched twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.inference import hf_download
from src.inference.hf_download import (
    WeightsUnavailableError,
    are_companion_files_cached,
    companion_cache_dir,
    companion_file,
    fetch_companion_files,
)

DONOR = "org/donor"
REVISION = "c" * 40
FILE = "tokenizer.json"


def _place(root: Path, name: str) -> Path:
    """A file where the Hub cache layout puts one, under `root`."""
    folder = root / "models--org--donor" / "snapshots" / REVISION
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private HF_HOME with its hub cache inside, as the default."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        constants, "HF_HUB_CACHE", str(tmp_path / "hub")
    )
    return tmp_path


def _fake_download(
    calls: List[Dict[str, Any]],
) -> Any:
    def download(
        repo_id: str,
        filename: str,
        *,
        revision: str,
        cache_dir: Optional[Any] = None,
    ) -> str:
        assert cache_dir is not None, "fetched into the main cache"
        calls.append({"repo": repo_id, "cache_dir": Path(cache_dir)})
        return str(_place(Path(cache_dir), filename))

    return download


def test_the_companion_cache_is_beside_the_hub_cache(
    home: Path,
) -> None:
    assert companion_cache_dir() == home / "companions"
    assert companion_cache_dir() != home / "hub"


def test_a_fetch_never_makes_the_donor_look_downloaded(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hazard. A snapshot folder in the main cache is all
    `is_repo_cached` needs to answer yes, so the donor's folder must
    not appear there, and the donor must stay un-downloaded."""
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", _fake_download(calls)
    )

    paths = fetch_companion_files(DONOR, [FILE], revision=REVISION)

    assert calls == [
        {"repo": DONOR, "cache_dir": companion_cache_dir()}
    ]
    assert not (home / "hub" / "models--org--donor").exists()
    assert not hf_download.is_repo_cached(DONOR, revision=REVISION)
    assert paths[FILE].is_relative_to(companion_cache_dir())
    found = companion_file(DONOR, FILE, revision=REVISION)
    assert found == paths[FILE]


def test_the_donors_own_file_is_used_when_it_is_there(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On any machine that has run the donor, nothing is fetched."""
    donor_file = _place(home / "hub", FILE)
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", _fake_download(calls)
    )

    paths = fetch_companion_files(DONOR, [FILE], revision=REVISION)

    assert paths[FILE] == donor_file
    assert calls == []


def test_every_named_file_has_to_be_present(
    home: Path,
) -> None:
    _place(companion_cache_dir(), FILE)

    assert are_companion_files_cached(
        DONOR, [FILE], revision=REVISION
    )
    assert not are_companion_files_cached(
        DONOR, [FILE, "tokenizer_config.json"], revision=REVISION
    )


def test_another_commit_does_not_count(home: Path) -> None:
    """Pinned means pinned: the same file at another commit is not
    the file the model was checked against."""
    _place(companion_cache_dir(), FILE)

    assert companion_file(DONOR, FILE, revision="d" * 40) is None


def test_offline_with_the_file_missing_says_so(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentence a missing checkpoint gets, not a retry dump."""

    def offline(repo_id: str, filename: str, **kwargs: Any) -> str:
        raise ConnectionError("no route to host")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)

    with pytest.raises(WeightsUnavailableError) as caught:
        fetch_companion_files(DONOR, [FILE], revision=REVISION)

    assert DONOR in str(caught.value)
