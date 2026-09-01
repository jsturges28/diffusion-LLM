"""One run, three eras, one answer.

Strategy: build the same generation as a v0, a v1 and a v2 run on
disk, read each through `load_run_frames`, and require that they
agree. The point is not that the reader parses three files; it is
that a run written years apart in three arrangements is still the
same run to everything downstream.

Two of those arrangements are per-frame and one is flat. That
difference is the whole of stage two, and it is invisible above this
function by design: only `load_run_frames` knows, and every server
consumer goes through it.

The flat shape is not merely smaller. An autoregressive run's frames
were always prefixes of one another, which is why a v0 or v1 run can
be read flat with nothing rewritten: its last frame *is* the run.
That recovery is checked rather than assumed, because a run whose
positions genuinely change has no such last frame, and reading one
that way would silently discard everything the earlier frames said.

Passing proves the three eras agree, that a declared shape is
believed and an undeclared one is inferred safely, that a run with
changing positions stays per-frame, and that a forward version is
refused rather than guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analytics import metrics

WORDS = ["The", " cat", " sat", " on", " the", " mat"]


def _positions(words: List[str]) -> List[Dict[str, Any]]:
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


def _framed(positions: List[Dict[str, Any]]) -> List[Any]:
    return [positions[: n + 1] for n in range(len(positions))]


def _write_run(
    root: Path,
    name: str,
    tokens: Any,
    metadata: Dict[str, Any],
) -> Path:
    run = root / name
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run / "tokens.json").write_text(
        json.dumps(tokens), encoding="utf-8"
    )
    return run


def _v0(root: Path, positions: List[Dict[str, Any]]) -> Path:
    """Before versioning: no schema key, no capture manifest."""
    return _write_run(
        root, "v0", _framed(positions), {"backend": "smollm3"}
    )


def _v1(root: Path, positions: List[Dict[str, Any]]) -> Path:
    """Per-frame, with a capture manifest."""
    return _write_run(
        root,
        "v1",
        _framed(positions),
        {
            "backend": "smollm3",
            "schema_version": 1,
            "capture": {"frame_tokens": True},
        },
    )


def _v2(root: Path, positions: List[Dict[str, Any]]) -> Path:
    """Flat, and says so."""
    return _write_run(
        root,
        "v2",
        positions,
        {
            "backend": "smollm3",
            "schema_version": 2,
            "frame_shape": "append",
            "capture": {"frame_tokens": True},
        },
    )


@pytest.fixture()
def eras(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    """The same run in all three arrangements, already read."""
    positions = _positions(WORDS)
    return {
        "v0": metrics.load_run_frames(_v0(tmp_path, positions)),
        "v1": metrics.load_run_frames(_v1(tmp_path, positions)),
        "v2": metrics.load_run_frames(_v2(tmp_path, positions)),
    }


# -- the three eras agree --


def test_every_era_reads_the_same_frames(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    assert eras["v0"]["frames"] == eras["v1"]["frames"]
    assert eras["v1"]["frames"] == eras["v2"]["frames"]


def test_every_era_reads_the_same_positions(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    """Including the two that never stored them. Their frames were
    prefixes all along, so the last one is the run."""
    assert eras["v0"]["positions"] == eras["v2"]["positions"]
    assert eras["v1"]["positions"] == eras["v2"]["positions"]
    assert len(eras["v2"]["positions"]) == len(WORDS)


def test_every_era_reads_as_append(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    for era, loaded in eras.items():
        assert loaded["shape"] == "append", era


def test_every_era_reports_records_available(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    assert eras["v1"]["records_available"] is True
    assert eras["v0"]["records_available"] is True
    assert eras["v2"]["records_available"] is True


def test_a_v1_run_still_reads_its_frame_stream(
    tmp_path: Path,
) -> None:
    """The other gate the version bump could have moved.

    `frames.jsonl` arrived in v1 and replaced `history.txt` because a
    frame boundary in the transcript can be forged by the text a
    model writes. Comparing against the latest version rather than
    against the one that introduced the file would send every v1 run
    back to the transcript, and a run whose output contained the
    separator would come back with the wrong number of frames.
    """
    run = tmp_path / "v1-frames"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    (run / "frames.jsonl").write_text(
        '{"i": 0, "text": "one"}\n'
        '{"i": 1, "text": "one two"}\n',
        encoding="utf-8",
    )
    # Present and deliberately different: reading this instead would
    # pass a laxer assertion, so the two files disagree on purpose.
    (run / "history.txt").write_text("wrong", encoding="utf-8")

    assert metrics.read_frame_texts(run) == ["one", "one two"]


def test_a_v1_manifest_outranks_sniffing_the_frames(
    tmp_path: Path,
) -> None:
    """The manifest gate is on the version that introduced it, not
    on the latest one. Those were the same number until version 2,
    so raising the gate with the version would quietly drop every v1
    run back to inferring what it captured.

    Only a run where the two answers differ can tell. A run whose
    frames are all empty sniffs as having no records, while its
    manifest says the tokens were captured, which is the case the
    manifest exists for: an empty frame is a fact about the frame,
    not about the run.
    """
    run = _write_run(
        tmp_path,
        "empty-frames",
        [[], [], []],
        {
            "schema_version": 1,
            "capture": {"frame_tokens": True},
        },
    )

    loaded = metrics.load_run_frames(run)

    assert loaded["records_available"] is True


def test_every_era_gives_the_same_convergence(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    series = [
        metrics.convergence_from_records(loaded["frames"])
        for loaded in eras.values()
    ]

    assert series[0] == series[1] == series[2]


# -- the closed form this stage rests on --


def test_the_closed_form_equals_walking_the_frames(
    eras: Dict[str, Dict[str, Any]],
) -> None:
    """The equality that lets the server skip building N frames to
    count masks in them. Asserted against the general function
    rather than against a literal, so a change to either has to move
    both."""
    loaded = eras["v2"]

    closed = metrics.convergence_from_positions(
        len(loaded["positions"])
    )

    assert closed == metrics.convergence_from_records(
        loaded["frames"]
    )


@pytest.mark.parametrize("count", [1, 2, 7, 128])
def test_the_closed_form_holds_at_every_length(
    count: int, tmp_path: Path
) -> None:
    positions = _positions(["x"] * count)

    closed = metrics.convergence_from_positions(count)

    assert closed == metrics.convergence_from_records(
        _framed(positions)
    )


def test_a_run_with_no_positions_is_refused() -> None:
    """The boundary. A run has at least one frame, and a caller that
    reached here with zero has a bug worth stopping on."""
    with pytest.raises(AssertionError):
        metrics.convergence_from_positions(0)


# -- when the recovery must not happen --


def test_a_run_whose_positions_change_stays_per_frame(
    tmp_path: Path,
) -> None:
    """A diffusion run revises positions behind the newest one, so
    its last frame is not the whole run and flattening would throw
    the rest away."""
    positions = _positions(WORDS)
    frames = [list(f) for f in _framed(positions)]
    frames[-1] = [dict(r) for r in frames[-1]]
    frames[-1][0] = {**frames[-1][0], "id": 77, "t": " A"}
    run = _write_run(
        tmp_path, "revised", frames, {"schema_version": 1}
    )

    loaded = metrics.load_run_frames(run)

    assert loaded["shape"] == "snapshot"
    assert loaded["positions"] is None
    assert loaded["frames"] == frames


def test_a_repeated_final_frame_stays_per_frame(
    tmp_path: Path,
) -> None:
    """The case the length check catches and the prefix check does
    not. Frames [a], [a b], [a b] are each an extension of the last
    in the loose sense, so comparing prefixes alone would accept
    them and hand back two positions for a three-frame run. Every
    frame index would then be off by one past the repeat.
    """
    flat = _positions(WORDS[:2])
    repeated = [flat[:1], flat[:2], flat[:2]]
    run = _write_run(
        tmp_path, "repeat", repeated, {"schema_version": 1}
    )

    loaded = metrics.load_run_frames(run)

    assert loaded["shape"] == "snapshot"
    assert loaded["positions"] is None
    assert len(loaded["frames"]) == 3


def test_frames_of_uneven_length_stay_per_frame(
    tmp_path: Path,
) -> None:
    """A diffusion canvas is a fixed width from the first frame, so
    its lengths never count 1, 2, 3. That alone rules the recovery
    out before any record is compared."""
    wide = [_positions(WORDS) for _ in range(3)]
    run = _write_run(
        tmp_path, "wide", wide, {"schema_version": 1}
    )

    loaded = metrics.load_run_frames(run)

    assert loaded["shape"] == "snapshot"
    assert loaded["positions"] is None


def test_a_declared_flat_run_holding_frames_is_refused(
    tmp_path: Path,
) -> None:
    """Metadata and file disagreeing is not something to resolve by
    preference. One of them is wrong and neither says which."""
    run = _write_run(
        tmp_path,
        "lying",
        _framed(_positions(WORDS)),
        {"schema_version": 2, "frame_shape": "append"},
    )

    with pytest.raises(ValueError, match="declares the flat shape"):
        metrics.load_run_frames(run)


def test_an_unknown_shape_is_refused(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "future",
        _positions(WORDS),
        {"schema_version": 2, "frame_shape": "sideways"},
    )

    with pytest.raises(ValueError, match="unknown frame shape"):
        metrics.load_run_frames(run)


def test_a_forward_version_is_refused(tmp_path: Path) -> None:
    """The run is fine; this build is the old one. Saying so is the
    difference between "update the app" and "this run is broken"."""
    run = _write_run(
        tmp_path,
        "newer",
        _positions(WORDS),
        {"schema_version": 99, "frame_shape": "append"},
    )

    with pytest.raises(metrics.UnsupportedRunVersionError):
        metrics.load_run_frames(run)


# -- the baseline, which pays the same cost twice --


def test_the_baseline_is_read_the_same_way(
    tmp_path: Path,
) -> None:
    """`original_tokens.json` came out within a megabyte of
    `tokens.json` on the long runs, because the baseline is a frozen
    copy of the same frames."""
    positions = _positions(WORDS)
    run = _write_run(
        tmp_path,
        "edited",
        positions,
        {
            "schema_version": 2,
            "frame_shape": "append",
            "capture": {"frame_tokens": True},
        },
    )
    (run / "original_tokens.json").write_text(
        json.dumps(positions), encoding="utf-8"
    )

    loaded = metrics.load_run_frames(run)

    assert loaded["original_positions"] == positions
    assert loaded["original_frames"] == _framed(positions)
