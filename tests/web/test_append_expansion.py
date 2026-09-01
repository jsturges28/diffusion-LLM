"""A run sent as positions is the same run as one sent as frames.

Strategy: save the same generation twice through the real endpoint,
once in each wire shape, into two temporary results directories, then
compare what a reader gets back. Nothing is inspected or mocked in
between.

**What is compared changed with stage two, and the change is the
point.** Stage one expanded positions on the way in, so the two saves
produced byte-identical files and this suite compared bytes. Stage
two stops expanding: an append run's `tokens.json` is now one flat
list, which is 131 MiB of a long run's 282 becoming under a megabyte.
The bytes are deliberately different now.

What survives is the run. `load_run_frames` returns the same frames
either way, `history.txt` and `frames.jsonl` are still identical
because the text is still expanded, and every reader downstream sees
what it always saw. Byte equality was a means; reading back the same
run is the end, and only one of those was ever worth asserting.

Passing proves both wire shapes reach the same run, that the flat one
is dramatically smaller on disk, that the pre-edit baseline gets the
same treatment, and that a request carrying both shapes or neither is
refused rather than silently resolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from src.analytics import metrics
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


def _run_dir(root: Path) -> Path:
    """The one run under ``root``."""
    runs = [d for d in root.iterdir() if d.is_dir()
            and not d.name.startswith(".")]
    assert len(runs) == 1, f"expected one run, found {runs}"
    return runs[0]


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


# -- the same run --


def test_both_shapes_write_the_same_files(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    assert sorted(both_shapes["snapshot"]) == sorted(
        both_shapes["append"]
    )


@pytest.mark.parametrize("name", ["frames.jsonl", "history.txt"])
def test_the_text_files_are_byte_identical(
    both_shapes: Dict[str, Dict[str, bytes]], name: str
) -> None:
    """The text is still expanded on the way in, so these two are
    unchanged by the append shape. Compared as bytes rather than as
    parsed content, because a reader that line-counts a file cares
    about the bytes."""
    assert both_shapes["append"][name] == (
        both_shapes["snapshot"][name]
    )


def test_the_flat_run_is_the_smaller_file(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    """The whole of stage two in one assertion. Six tokens is a small
    saving; the ratio is what matters, and it grows with the run.
    """
    flat = len(both_shapes["append"]["tokens.json"])
    framed = len(both_shapes["snapshot"]["tokens.json"])

    assert flat < framed
    assert flat * 3 < framed, (
        f"flat {flat} should be far under framed {framed}"
    )


def test_the_flat_file_is_one_record_per_token(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    """Not per frame, which is what it used to be, and not empty,
    which is how a size comparison could pass for the wrong reason."""
    stored = json.loads(both_shapes["append"]["tokens.json"])

    assert len(stored) == len(WORDS)
    assert [record["t"] for record in stored] == WORDS


def test_the_framed_file_still_holds_every_prefix(
    both_shapes: Dict[str, Dict[str, bytes]],
) -> None:
    """The shape a diffusion run keeps, pinned here so the two can be
    told apart by more than their size."""
    stored = json.loads(both_shapes["snapshot"]["tokens.json"])

    assert [len(frame) for frame in stored] == [1, 2, 3, 4, 5, 6]


def test_a_reader_gets_the_same_frames_either_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What byte equality was standing in for, asserted directly.
    Every server-side consumer goes through this one function, so
    agreement here is agreement everywhere downstream."""
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

    framed = metrics.load_run_frames(_run_dir(snapshot_root))
    flat = metrics.load_run_frames(_run_dir(append_root))

    assert flat["frames"] == framed["frames"]
    assert flat["records_available"] == framed["records_available"]
    # Both read as append, and that is the recovery working rather
    # than a mix-up: the run only grows, so it *is* append-shaped
    # however it happened to be sent. The file written per-frame
    # still has prefixes, so its last frame is the flat run and the
    # reader takes it without anything being rewritten.
    assert flat["shape"] == "append"
    assert framed["shape"] == "append"
    assert flat["positions"] == framed["positions"]


def test_a_run_whose_positions_change_stays_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard on that recovery. A diffusion run revises positions
    behind the newest one, so its frames are not prefixes and reading
    the last one as the whole run would throw away everything the
    earlier frames said."""
    positions = _positions(WORDS)
    revised = _as_frames(positions)
    # One position changes after it was written, which is the whole
    # difference between the two shapes.
    revised[-1] = [dict(r) for r in revised[-1]]
    revised[-1][0] = {**revised[-1][0], "t": " A", "id": 77}

    client = _client(tmp_path, monkeypatch)
    _save(client, _base(
        frames=_as_texts(positions), frame_tokens=revised,
    ))

    loaded = metrics.load_run_frames(_run_dir(tmp_path))

    assert loaded["shape"] == "snapshot"
    assert loaded["positions"] is None
    assert loaded["frames"] == revised


# -- the pre-edit baseline, which pays the same cost twice --


def test_an_edited_run_expands_its_baseline_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`original_tokens.json` came out nearly as large as
    `tokens.json` on the long runs, because the baseline is a frozen
    copy of the same frames. It is sent flat, stored flat, and read
    back the same either way."""
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

    flat = _files(append_root)["original_tokens.json"]
    framed = _files(snapshot_root)["original_tokens.json"]
    assert len(flat) * 3 < len(framed), (
        f"baseline flat {len(flat)} against framed {len(framed)}"
    )

    # Smaller, and the same run: the comparison views read frames,
    # and they must not be able to tell which way it was written.
    assert (
        metrics.load_run_frames(_run_dir(append_root))[
            "original_frames"
        ]
        == metrics.load_run_frames(_run_dir(snapshot_root))[
            "original_frames"
        ]
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
