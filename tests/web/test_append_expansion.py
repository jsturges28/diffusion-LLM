"""A run sent as positions lands on disk as a run sent as frames.

Strategy: save the same generation twice through the real endpoint,
once in each wire shape, into two temporary results directories, and
compare the files byte for byte. Nothing is inspected or mocked in
between; the comparison is of what a reader would later open.

That equality is the entire promise of this stage. The browser stops
holding N(N+1)/2 token records and the wire stops carrying them, but
`tokens.json`, `frames.jsonl` and `history.txt` keep the shape every
existing reader was written against, so the run-store schema version
does not move, Analytics is untouched, and the 238 runs already on
disk stay readable by exactly the code that read them yesterday.
Stage two is where the stored form changes; this stage is where it
provably does not.

Passing proves the expansion reproduces the per-frame arrays, that it
does so for the pre-edit baseline as well as the run, that a request
carrying both shapes is refused rather than silently resolved, and
that a request carrying neither is refused too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.web import server

WORDS = ["The", " cat", " sat", " on", " the", " mat"]


def _positions(words: List[str]) -> List[Dict[str, Any]]:
    """One record per token, as an append client sends them."""
    return [
        {
            "t": word,
            "m": False,
            "id": 1000 + at,
            "c": round(0.5 + at / 100, 4),
            "e": round(1.5 - at / 100, 4),
        }
        for at, word in enumerate(words)
    ]


def _as_frames(
    positions: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """The same run as a snapshot client sends it: every prefix."""
    return [
        positions[: count + 1] for count in range(len(positions))
    ]


def _as_texts(positions: List[Dict[str, Any]]) -> List[str]:
    return [
        "".join(p["t"] for p in positions[: count + 1])
        for count in range(len(positions))
    ]


def _client(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(server, "RESULTS_DIR", root)
    return TestClient(server.app)


def _base(**overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": "smollm3",
        "prompt": "explain REST",
        "final_text": "The cat sat on the mat",
    }
    body.update(overrides)
    return body


def _save(client: TestClient, body: Dict[str, Any]) -> Path:
    response = client.post("/api/save", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["success"] is True, result
    return Path(result["path"])


def _files(root: Path) -> Dict[str, bytes]:
    """Every file of the one run under ``root``, by name."""
    runs = [d for d in root.iterdir() if d.is_dir()
            and not d.name.startswith(".")]
    assert len(runs) == 1, f"expected one run, found {runs}"
    return {
        f.name: f.read_bytes()
        for f in sorted(runs[0].iterdir())
        if f.is_file()
    }


@pytest.fixture()
def both_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Dict[str, bytes]]:
    """The same run saved each way, as two sets of files."""
    positions = _positions(WORDS)

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    client = _client(snapshot_root, monkeypatch)
    _save(client, _base(
        frames=_as_texts(positions),
        frame_tokens=_as_frames(positions),
    ))

    append_root = tmp_path / "append"
    append_root.mkdir()
    client = _client(append_root, monkeypatch)
    _save(client, _base(frame_positions=positions))

    return {
        "snapshot": _files(snapshot_root),
        "append": _files(append_root),
    }


# -- the same bytes --


def test_both_shapes_write_the_same_files(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    assert sorted(both_shapes["snapshot"]) == sorted(
        both_shapes["append"]
    )


@pytest.mark.parametrize(
    "name", ["tokens.json", "frames.jsonl", "history.txt"]
)
def test_a_file_is_byte_identical(
    both_shapes: Dict[str, Dict[str, bytes]], name: str
) -> None:
    """The three the expansion is responsible for. Compared as bytes
    rather than as parsed JSON, because a reader that mmaps or
    line-counts a file cares about the bytes and an equality on
    parsed objects would hide a formatting change."""
    assert both_shapes["append"][name] == (
        both_shapes["snapshot"][name]
    )


def test_the_expansion_is_the_run_and_not_a_prefix_of_it(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    """A guard against the files matching because both are empty."""
    stored = json.loads(both_shapes["append"]["tokens.json"])

    assert len(stored) == len(WORDS)
    assert [len(frame) for frame in stored] == [1, 2, 3, 4, 5, 6]


# -- the pre-edit baseline, which pays the same cost twice --


def test_an_edited_run_expands_its_baseline_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`original_tokens.json` came out nearly as large as
    `tokens.json` on the long runs, because the baseline is a frozen
    copy of the same frames. It is sent flat and expanded the same
    way."""
    positions = _positions(WORDS)
    edits = [{"frame_index": 2, "token_positions": [1]}]

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    client = _client(snapshot_root, monkeypatch)
    _save(client, _base(
        frames=_as_texts(positions),
        frame_tokens=_as_frames(positions),
        original_frame_tokens=_as_frames(positions),
        remask_edits=edits,
    ))

    append_root = tmp_path / "append"
    append_root.mkdir()
    client = _client(append_root, monkeypatch)
    _save(client, _base(
        frame_positions=positions,
        original_frame_positions=positions,
        remask_edits=edits,
    ))

    assert _files(append_root)["original_tokens.json"] == (
        _files(snapshot_root)["original_tokens.json"]
    )


# -- a request that cannot be resolved --


def test_a_run_carrying_both_shapes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two descriptions of one run with nothing deciding which is
    true. Preferring either quietly would make a client bug into a
    saved run that disagrees with what was on screen."""
    positions = _positions(WORDS)
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/save", json=_base(
        frames=_as_texts(positions),
        frame_positions=positions,
    ))

    assert response.status_code == 500
    assert "not both" in response.json()["message"]


def test_a_run_carrying_neither_shape_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`frames` stopped being required when it stopped being the only
    way to describe a run, so the check that a run has any frames at
    all moved here rather than disappearing."""
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/save", json=_base())

    assert response.status_code == 500
    assert "frames" in response.json()["message"]


def test_an_empty_position_list_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary: a request that names the append shape and then
    carries nothing is as empty as one that names neither."""
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/save", json=_base(frame_positions=[])
    )

    assert response.status_code == 500
