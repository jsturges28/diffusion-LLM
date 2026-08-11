"""Both eras of saved run stay readable through one shape.

Strategy: build a golden fixture per era, byte-for-byte the way that
era's writer produced it, and read both through the same public
functions. The v0 fixture is a transcription of a real saved run's
layout: no version field, no manifest, frame text only in the
transcript. The v1 fixture is produced by the current writer, so the
test moves with it rather than freezing a copy that drifts.

Passing proves three things. A run saved before versioning existed
still loads, which matters because 182 of them exist and none is
being rewritten. A run saved today loads through the JSON-lines
stream, not the transcript. And callers downstream get the same keys
and types either way, so nothing outside this module branches on
version.

Also pins the writer's on-disk vocabulary against the reader's copy of
it, since the two modules name those strings independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analytics.metrics import (
    CAPTURE_KEY,
    SCHEMA_VERSION_LATEST,
    SCHEMA_VERSION_LEGACY,
    UnsupportedRunVersionError,
    compute_convergence,
    load_run_frames,
    load_run_metadata,
    parse_frames_jsonl,
    read_frame_texts,
    run_schema_version,
)
from src.web import run_store
from src.web.run_store import RunBundle

FRAME_TEXTS = [
    "░░░░ ░░░░ ░░░░",
    "The ░░░░ ░░░░",
    "The cat sat",
]

TOKEN_FRAMES = [
    [
        {"t": "░", "m": True, "id": 126336},
        {"t": "░", "m": True, "id": 126336},
    ],
    [
        {"t": "The", "m": False, "id": 791, "c": 0.91},
        {"t": "░", "m": True, "id": 126336},
    ],
    [
        {"t": "The", "m": False, "id": 791, "c": 0.91},
        {"t": " cat", "m": False, "id": 8415, "c": 0.77},
    ],
]

ALTERNATIVES = [
    [{"t": "The", "p": 0.91}, {"t": "A", "p": 0.06}],
    [{"t": " cat", "p": 0.77}, {"t": " dog", "p": 0.12}],
]


def _write_v0_run(root: Path, run_id: str) -> Path:
    """A run exactly as the app wrote them before versioning.

    Deliberately hand-built rather than generated: the point is to
    hold the old bytes still, so a future change to the writer cannot
    quietly redefine what "legacy" means and make this pass by moving
    the target.
    """
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    metadata: Dict[str, Any] = {
        "prompt": "Complete this",
        "model": "LLaDA-8B-Instruct",
        "created_at": "2025-11-02T14:31:07",
        "steps": 64,
        "gen_length": 128,
        "elapsed_seconds": 12.5,
        "per_frame_elapsed": [4.1, 8.3, 12.5],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (run_dir / "final.txt").write_text(
        FRAME_TEXTS[-1], encoding="utf-8"
    )

    lines: List[str] = []
    for index, text in enumerate(FRAME_TEXTS):
        lines.append(f"\n===== FRAME {index} =====\n{text}\n")
    (run_dir / "history.txt").write_text(
        "".join(lines), encoding="utf-8"
    )

    (run_dir / "tokens.json").write_text(
        json.dumps(TOKEN_FRAMES), encoding="utf-8"
    )
    return run_dir


def _write_v1_run(root: Path) -> Path:
    """A run as the current writer produces it."""
    bundle = RunBundle(
        metadata={
            "prompt": "Complete this",
            "model": "LLaDA-8B-Instruct",
            "created_at": "2026-08-11T03:00:00",
            "steps": 64,
            "gen_length": 128,
            "elapsed_seconds": 12.5,
            "per_frame_elapsed": [4.1, 8.3, 12.5],
        },
        final_text=FRAME_TEXTS[-1],
        frames=list(FRAME_TEXTS),
        frame_tokens=TOKEN_FRAMES,
        original_frame_tokens=None,
        alternatives=ALTERNATIVES,
        original_alternatives=None,
    )
    run_id, _revision = run_store.save(
        root, bundle, model_id="LLaDA-8B-Instruct"
    )
    return root / run_id


def _save_frames(root: Path, frames: List[str]) -> Path:
    """Publish a run carrying just these frame texts."""
    bundle = RunBundle(
        metadata={"prompt": "p", "model": "m"},
        final_text=frames[-1] if frames else "",
        frames=list(frames),
        frame_tokens=None,
        original_frame_tokens=None,
        alternatives=None,
        original_alternatives=None,
    )
    run_id, _revision = run_store.save(root, bundle, model_id="m")
    return root / run_id


# -- the two eras agree on what a reader gets --


def test_a_legacy_run_reports_version_zero(
    tmp_path: Path,
) -> None:
    run_dir = _write_v0_run(tmp_path, "20251102_143107_legacy")

    meta = load_run_metadata(run_dir)

    assert run_schema_version(meta) == SCHEMA_VERSION_LEGACY


def test_a_new_run_states_its_version(tmp_path: Path) -> None:
    run_dir = _write_v1_run(tmp_path)

    meta = load_run_metadata(run_dir)

    assert run_schema_version(meta) == SCHEMA_VERSION_LATEST


def test_both_eras_yield_the_same_frames_in_order(
    tmp_path: Path,
) -> None:
    """Same count, same order, same content once the transcript's
    own artifact is accounted for."""
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")
    current = _write_v1_run(tmp_path)

    from_legacy = read_frame_texts(legacy)
    from_current = read_frame_texts(current)

    assert from_current == FRAME_TEXTS
    assert len(from_legacy) == len(from_current)
    assert [text.rstrip("\n") for text in from_legacy] == FRAME_TEXTS


def test_the_transcript_adds_a_trailing_newline_v1_does_not(
    tmp_path: Path,
) -> None:
    """A quirk of the delimiter format, pinned rather than fixed.

    The writer puts a newline after each frame body and the next
    header begins with another, so every frame but the last comes
    back from the transcript with a trailing newline that was never
    in the frame. Runs are not being rewritten, so v0 keeps it; the
    next test is what makes it harmless.
    """
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")

    from_legacy = read_frame_texts(legacy)

    assert from_legacy[0] == FRAME_TEXTS[0] + "\n"
    assert from_legacy[-1] == FRAME_TEXTS[-1]


def test_convergence_is_identical_across_eras(
    tmp_path: Path,
) -> None:
    """Downstream analytics must not be able to tell the eras apart,
    which is what makes the version dispatch safe to add.

    This is also what makes the transcript's trailing newline a
    non-issue: convergence strips each frame before counting, so the
    two eras produce identical numbers from slightly different text.
    """
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")
    current = _write_v1_run(tmp_path)

    assert compute_convergence(
        read_frame_texts(legacy)
    ) == compute_convergence(read_frame_texts(current))


def test_token_stream_keys_match_across_eras(
    tmp_path: Path,
) -> None:
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")
    current = _write_v1_run(tmp_path)

    from_legacy = load_run_frames(legacy)
    from_current = load_run_frames(current)

    assert set(from_legacy) == set(from_current)
    assert from_legacy["frames"] == from_current["frames"]
    assert from_legacy["records_available"] is True
    assert from_current["records_available"] is True


def test_a_new_run_reads_frames_from_the_json_stream(
    tmp_path: Path,
) -> None:
    """Removing the transcript must not affect a v1 run. If it does,
    something still parses the forgeable format."""
    run_dir = _write_v1_run(tmp_path)
    (run_dir / "history.txt").unlink()

    assert read_frame_texts(run_dir) == FRAME_TEXTS


def test_a_legacy_run_still_needs_its_transcript(
    tmp_path: Path,
) -> None:
    """The other half of the pair: v0 has nowhere else to read from,
    so the dispatch must not have silently switched it.

    FileNotFoundError specifically, because that is what the metrics
    route turns into a 404 rather than a 500.
    """
    run_dir = _write_v0_run(tmp_path, "20251102_143107_legacy")
    (run_dir / "history.txt").unlink()

    with pytest.raises(FileNotFoundError, match="history.txt"):
        read_frame_texts(run_dir)


def test_a_new_run_missing_its_frame_stream_is_not_found(
    tmp_path: Path,
) -> None:
    run_dir = _write_v1_run(tmp_path)
    (run_dir / "frames.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="frames.jsonl"):
        read_frame_texts(run_dir)


# -- what the transcript could never do --


def test_model_output_cannot_forge_a_frame_boundary(
    tmp_path: Path,
) -> None:
    """The reason v1 exists. A model that emits the transcript's
    delimiter splits a v0 frame in two; in v1 it is just text."""
    forged = "before\n===== FRAME 99 =====\nafter"
    run_dir = _save_frames(tmp_path, ["clean", forged])

    from src.analytics.metrics import parse_history

    assert read_frame_texts(run_dir) == ["clean", forged]
    # The transcript, still written for humans, is fooled. That is
    # the behavior being left behind, asserted so the contrast is a
    # fact rather than a claim in a comment.
    assert len(parse_history(run_dir / "history.txt")) == 3


def test_frame_text_survives_a_round_trip_unchanged(
    tmp_path: Path,
) -> None:
    """Newlines, unicode masks and quotes all have to come back
    byte-identical, since frame text is compared across frames to
    compute convergence."""
    awkward = 'line\n\ttab "quoted" ░ \\ backslash\r\nend'

    run_dir = _save_frames(tmp_path, [awkward])

    assert read_frame_texts(run_dir) == [awkward]


def test_empty_frames_keep_their_positions(
    tmp_path: Path,
) -> None:
    """An empty frame is a real frame. The transcript drops trailing
    empties, so this is a behavior difference worth pinning."""
    run_dir = _save_frames(tmp_path, ["", "a", ""])

    assert read_frame_texts(run_dir) == ["", "a", ""]


# -- the manifest states what used to be guessed --


def test_the_manifest_records_which_signals_were_captured(
    tmp_path: Path,
) -> None:
    run_dir = _write_v1_run(tmp_path)

    manifest = load_run_metadata(run_dir)[CAPTURE_KEY]

    assert manifest["frames"] is True
    assert manifest["frame_tokens"] is True
    assert manifest["alternatives"] is True
    assert manifest["original_frame_tokens"] is False
    assert manifest["original_alternatives"] is False


def test_the_manifest_covers_every_optional_sidecar(
    tmp_path: Path,
) -> None:
    """A sidecar added later without a manifest entry would be
    invisible to readers that trust the manifest."""
    run_dir = _write_v1_run(tmp_path)

    manifest = load_run_metadata(run_dir)[CAPTURE_KEY]

    declared = set(manifest) - {"frames"}
    written = {name for name, _file in run_store.SIDECAR_NAMES}
    assert declared == written


# -- malformed frame streams are refused, not misread --


def test_a_frame_stream_with_a_gap_is_refused(
    tmp_path: Path,
) -> None:
    """A missing index would renumber every later frame in the UI,
    which reads as a wrong analysis rather than a broken file."""
    path = tmp_path / "frames.jsonl"
    path.write_text(
        '{"i": 0, "text": "a"}\n{"i": 2, "text": "c"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        parse_frames_jsonl(path)


def test_a_frame_stream_out_of_order_is_still_read(
    tmp_path: Path,
) -> None:
    """Order on disk is not the contract; the index is."""
    path = tmp_path / "frames.jsonl"
    path.write_text(
        '{"i": 1, "text": "b"}\n{"i": 0, "text": "a"}\n',
        encoding="utf-8",
    )

    assert parse_frames_jsonl(path) == ["a", "b"]


def test_a_frame_record_without_text_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text('{"i": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="frame text"):
        parse_frames_jsonl(path)


def test_a_frame_record_with_a_boolean_index_is_refused(
    tmp_path: Path,
) -> None:
    """True is an int in Python and would land at index 1."""
    path = tmp_path / "frames.jsonl"
    path.write_text('{"i": true, "text": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="integer frame index"):
        parse_frames_jsonl(path)


def test_blank_lines_in_a_frame_stream_are_ignored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text(
        '{"i": 0, "text": "a"}\n\n{"i": 1, "text": "b"}\n',
        encoding="utf-8",
    )

    assert parse_frames_jsonl(path) == ["a", "b"]


# -- a version this build cannot read --


def test_a_future_version_is_refused_by_name(
    tmp_path: Path,
) -> None:
    version = SCHEMA_VERSION_LATEST + 1

    with pytest.raises(UnsupportedRunVersionError) as caught:
        run_schema_version(
            {"schema_version": version, "run_id": "future_run"}
        )

    assert caught.value.version == version
    assert "future_run" in str(caught.value)


def test_a_boolean_version_is_not_read_as_version_one() -> None:
    """True == 1 in Python, so a membership test alone would accept
    this and read a nonsense run as current."""
    with pytest.raises(UnsupportedRunVersionError):
        run_schema_version({"schema_version": True})


def test_a_string_version_is_refused() -> None:
    with pytest.raises(UnsupportedRunVersionError):
        run_schema_version({"schema_version": "1"})


def test_a_negative_version_is_refused() -> None:
    with pytest.raises(UnsupportedRunVersionError):
        run_schema_version({"schema_version": -1})


# -- writer and reader name the same files --


def test_the_two_modules_agree_on_the_on_disk_names() -> None:
    """The reader keeps its own copy of these strings to avoid
    importing the web layer, so nothing but this test stops the two
    from drifting apart."""
    from src.analytics import metrics

    assert metrics.METADATA_NAME == run_store.METADATA_NAME
    assert metrics.HISTORY_NAME == run_store.HISTORY_NAME
    assert metrics.FRAMES_NAME == run_store.FRAMES_NAME
    assert metrics.CAPTURE_KEY == run_store.CAPTURE_KEY
    assert (
        metrics.SCHEMA_VERSION_KEY == run_store.SCHEMA_VERSION_KEY
    )


def test_the_reader_reads_what_this_build_writes() -> None:
    """A writer bumped past the reader would make every new run
    unreadable by the app that wrote it."""
    assert run_store.SCHEMA_VERSION == SCHEMA_VERSION_LATEST
