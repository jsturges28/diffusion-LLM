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
    convergence_from_records,
    list_runs,
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

    This is the character fallback, the measure a run gets when it
    saved no usable token records. It is still era-parity that
    matters here, not the measure itself: the transcript's trailing
    newline is a non-issue because convergence strips each frame
    before counting, so the two eras produce identical numbers from
    slightly different text.
    """
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")
    current = _write_v1_run(tmp_path)

    assert compute_convergence(
        read_frame_texts(legacy)
    ) == compute_convergence(read_frame_texts(current))


def test_token_convergence_is_identical_across_eras(
    tmp_path: Path,
) -> None:
    """The same parity for the measure runs actually get.

    Both eras store per-token records, so both should be counted in
    positions rather than characters. The eras declare that
    differently, a v1 manifest against a v0 sniff of the first token,
    which is exactly the sort of difference that could leak into the
    numbers if the two paths ever diverged.
    """
    legacy = _write_v0_run(tmp_path, "20251102_143107_legacy")
    current = _write_v1_run(tmp_path)

    from_legacy = load_run_frames(legacy)
    from_current = load_run_frames(current)
    assert from_legacy["records_available"] is True
    assert from_current["records_available"] is True

    assert convergence_from_records(
        from_legacy["frames"]
    ) == convergence_from_records(from_current["frames"])


def test_the_two_measures_disagree_on_this_fixture(
    tmp_path: Path,
) -> None:
    """The reason the basis has to be reported at all.

    The fixture's tokens are different lengths ("The" against
    " cat"), which is the whole defect: counting characters makes a
    long token look like more progress than a short one. If these
    two ever agreed, the token path would not be doing anything and
    the caption would be pointless.
    """
    current = _write_v1_run(tmp_path)

    by_chars = compute_convergence(read_frame_texts(current))
    by_tokens = convergence_from_records(
        load_run_frames(current)["frames"]
    )

    chars_mid = by_chars[1]["resolved_ratio"]
    tokens_mid = by_tokens[1]["resolved_ratio"]
    assert chars_mid != tokens_mid
    # One of two positions resolved, whatever the text says.
    assert tokens_mid == 0.5


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


# -- one bad run does not take the catalog with it --


def _write_raw_metadata(root: Path, run_id: str, raw: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(raw, encoding="utf-8")
    return run_dir


def test_a_healthy_run_still_lists_beside_a_broken_one(
    tmp_path: Path,
) -> None:
    """The finding in one assertion: a single unreadable directory
    used to be able to fail the whole request."""
    _write_v0_run(tmp_path, "20251102_143107_legacy")
    _write_raw_metadata(tmp_path, "20251103_000000_broken", "{oops")

    runs = list_runs(tmp_path)

    assert len(runs) == 2
    healthy = [r for r in runs if not r.get("invalid")]
    assert len(healthy) == 1
    assert healthy[0]["run_id"] == "20251102_143107_legacy"


def test_a_broken_run_is_listed_rather_than_hidden(
    tmp_path: Path,
) -> None:
    """Skipping it silently reads as a deleted run, and the obvious
    response to that is to save it again."""
    _write_raw_metadata(tmp_path, "20251103_000000_broken", "{oops")

    runs = list_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["invalid"] is True
    assert runs[0]["run_id"] == "20251103_000000_broken"
    assert "Unreadable metadata" in runs[0]["error"]


@pytest.mark.parametrize(
    "raw",
    ["[1, 2, 3]", '"just a string"', "42", "null"],
)
def test_metadata_that_is_not_an_object_is_an_invalid_entry(
    tmp_path: Path, raw: str
) -> None:
    """These used to raise TypeError on item assignment, which no
    handler caught, so the catalog returned a 500 for all runs."""
    _write_raw_metadata(tmp_path, "20251103_000000_odd", raw)

    runs = list_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["invalid"] is True


def test_an_invalid_entry_carries_what_the_table_needs(
    tmp_path: Path,
) -> None:
    """The row is rendered next to healthy ones, so a missing key
    here becomes a broken table rather than a broken run."""
    _write_raw_metadata(tmp_path, "20251103_000000_broken", "{oops")

    entry = list_runs(tmp_path)[0]

    for key in ("run_id", "prompt", "model", "created_at",
                "has_diff", "error", "invalid"):
        assert key in entry, key


def test_the_catalog_survives_a_directory_of_broken_runs(
    tmp_path: Path,
) -> None:
    """Bound the blast radius: many bad runs still return, in order,
    without raising."""
    for index in range(12):
        _write_raw_metadata(
            tmp_path, f"20251103_0000{index:02d}_broken", "{oops"
        )
    _write_v0_run(tmp_path, "20251102_143107_legacy")

    runs = list_runs(tmp_path)

    assert len(runs) == 13
    assert sum(1 for r in runs if r.get("invalid")) == 12


# -- a run from a future build --


def _write_future_run(root: Path, run_id: str) -> Path:
    """A run claiming a format this build has never seen."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "prompt": "from tomorrow",
        "model": "LLaDA-8B-Instruct",
        "created_at": "2027-01-01T00:00:00",
        "schema_version": SCHEMA_VERSION_LATEST + 1,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return run_dir


def test_a_future_run_lists_with_a_compatibility_message(
    tmp_path: Path,
) -> None:
    _write_future_run(tmp_path, "20270101_000000_future")

    entry = list_runs(tmp_path)[0]

    assert entry["invalid"] is True
    assert "newer version" in entry["error"]
    assert "Update" in entry["error"]


def test_a_future_run_is_not_reported_as_corrupt(
    tmp_path: Path,
) -> None:
    """The run is probably fine; this build is the old one. Saying
    "unreadable" would invite deleting a good run."""
    _write_future_run(tmp_path, "20270101_000000_future")

    entry = list_runs(tmp_path)[0]

    assert "Unreadable" not in entry["error"]
    assert "corrupt" not in entry["error"].lower()


def test_no_field_of_a_future_run_is_interpreted(
    tmp_path: Path,
) -> None:
    """Its fields were written by a build this one does not know, so
    reading any of them would be a guess shown as a fact."""
    run_dir = _write_future_run(tmp_path, "20270101_000000_future")
    (run_dir / "original_tokens.json").write_text(
        "[]", encoding="utf-8"
    )

    entry = list_runs(tmp_path)[0]

    assert entry["has_diff"] is False
    assert entry["prompt"] == ""
    assert "elapsed_seconds" not in entry


def test_a_future_run_never_reaches_the_frame_readers(
    tmp_path: Path,
) -> None:
    """Refused at the version check, before anything tries to parse
    a file whose format is unknown."""
    run_dir = _write_future_run(tmp_path, "20270101_000000_future")

    with pytest.raises(UnsupportedRunVersionError):
        read_frame_texts(run_dir)
    with pytest.raises(UnsupportedRunVersionError):
        load_run_frames(run_dir)


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
