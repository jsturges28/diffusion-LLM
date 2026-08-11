"""Compute intrinsic diffusion metrics from saved runs.

Reads a run directory and produces convergence and timing statistics
for the Analytics Suite frontend.

This module is the read boundary for saved runs, so it is where the
on-disk schema version is dispatched on. Two eras exist and both must
keep working:

- **v0**, every run saved before versioning: no version field, frame
  text recoverable only from the human transcript, and which signals
  were captured inferable only from which files are present and what
  the first token looks like.
- **v1**: states its version and a capture manifest, and carries a
  JSON-lines frame stream that no model output can forge a delimiter
  in.

An adapter per era normalizes into one shape, so nothing downstream of
here branches on version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

MASK_CHAR = "\u2591"  # ░
FRAME_HEADER_RE = re.compile(
    r"^={5}\s+FRAME\s+\d+\s+={5}$"
)

# Kept in step with `src/web/run_store.py`, which writes them. Named
# here rather than imported because analytics does not otherwise
# depend on the web layer, and one import is not worth inverting that;
# `tests/analytics/test_run_schema.py` fails if the two ever disagree.
METADATA_NAME = "metadata.json"
HISTORY_NAME = "history.txt"
FRAMES_NAME = "frames.jsonl"
SCHEMA_VERSION_KEY = "schema_version"
CAPTURE_KEY = "capture"

# Absent means v0: the corpus that predates versioning is far larger
# than the one that follows it, so the unmarked case has to be a
# supported era rather than an error.
SCHEMA_VERSION_LEGACY = 0
SCHEMA_VERSION_LATEST = 1
SUPPORTED_SCHEMA_VERSIONS = (
    SCHEMA_VERSION_LEGACY,
    SCHEMA_VERSION_LATEST,
)

assert SCHEMA_VERSION_LATEST in SUPPORTED_SCHEMA_VERSIONS
assert SCHEMA_VERSION_LEGACY in SUPPORTED_SCHEMA_VERSIONS


class UnsupportedRunVersionError(ValueError):
    """A run was written by a newer version of this app.

    Raised instead of guessing. A forward version means fields this
    code has never heard of, and a reader that pushes on anyway would
    show a confidently wrong run rather than an honest refusal.
    """

    def __init__(self, version: object, run_id: str) -> None:
        super().__init__(
            f"run {run_id} uses schema version {version!r};"
            f" this build reads up to {SCHEMA_VERSION_LATEST}"
        )
        self.version = version
        self.run_id = run_id


def run_schema_version(metadata: Dict[str, Any]) -> int:
    """The era a run was written in.

    Raises ``UnsupportedRunVersionError`` for anything this build
    cannot read, including a non-integer version, which means the
    field was written by something that did not share this contract.
    """
    raw = metadata.get(SCHEMA_VERSION_KEY)
    if raw is None:
        return SCHEMA_VERSION_LEGACY
    run_id = str(metadata.get("run_id", "<unknown>"))
    # bool is an int in Python and `True == 1` would pass the
    # membership test below, quietly reading a nonsense run as v1.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise UnsupportedRunVersionError(raw, run_id)
    if raw not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedRunVersionError(raw, run_id)
    return raw


def read_frame_texts(
    run_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Per-frame text for a run, whichever era wrote it.

    Callers get frame *i* at index *i* and never learn which file it
    came from. Pass ``metadata`` when it is already loaded to avoid
    reading it twice.

    Raises ``FileNotFoundError`` if the era's frame file is missing.
    That is an explicit raise rather than an assertion because a run
    directory missing a file is a damaged run, not a broken caller,
    and the route above turns it into a 404. An assertion would
    vanish under ``python -O`` and take the 404 with it.
    """
    assert run_dir.is_dir(), f"run dir not found: {run_dir}"

    if metadata is None:
        metadata = load_run_metadata(run_dir)
    version = run_schema_version(metadata)

    if version >= SCHEMA_VERSION_LATEST:
        frames_path = run_dir / FRAMES_NAME
    else:
        frames_path = run_dir / HISTORY_NAME
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"{frames_path.name} missing for run {run_dir.name}"
        )

    if version >= SCHEMA_VERSION_LATEST:
        return parse_frames_jsonl(frames_path)
    return parse_history(frames_path)


def parse_frames_jsonl(
    frames_path: Path,
) -> List[str]:
    """Read the v1 frame stream: one JSON object per line.

    Unlike the transcript this replaces, a frame boundary is a line
    break between JSON documents, so text inside a frame cannot
    produce one no matter what the model emitted.

    Trusts the index each record carries rather than its position, and
    rejects a stream whose indices are not exactly 0..n-1, because a
    gap would silently shift every later frame's number in the UI.
    """
    assert frames_path.is_file(), (
        f"frames.jsonl not found: {frames_path}"
    )

    raw = frames_path.read_text(encoding="utf-8")
    by_index: Dict[int, str] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(
                f"{frames_path.name} line {line_number} is not"
                " an object"
            )
        index = record.get("i")
        text = record.get("text")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                f"{frames_path.name} line {line_number} has no"
                " integer frame index"
            )
        if not isinstance(text, str):
            raise ValueError(
                f"{frames_path.name} line {line_number} has no"
                " frame text"
            )
        by_index[index] = text

    expected = set(range(len(by_index)))
    if set(by_index) != expected:
        raise ValueError(
            f"{frames_path.name} frame indices are not"
            f" contiguous from 0 ({len(by_index)} records)"
        )
    return [by_index[i] for i in range(len(by_index))]


def parse_history(
    history_path: Path,
) -> List[str]:
    """Split history.txt into per-frame text strings.

    The v0 frame reader. Each frame is delimited by
    ``===== FRAME N =====`` headers. Returns a list where index *i*
    holds the text body of frame *i*.

    The delimiter is unescaped, so a model that emits the header line
    splits one frame into two here. That is why v1 does not use this
    file; it stays because the runs that only have it are real.
    """
    assert history_path.is_file(), (
        f"history.txt not found: {history_path}"
    )

    raw = history_path.read_text(encoding="utf-8")
    frames: List[str] = []
    current_lines: List[str] = []
    inside_frame = False

    for line in raw.splitlines():
        if FRAME_HEADER_RE.match(line.strip()):
            if inside_frame:
                frames.append(
                    "\n".join(current_lines)
                )
            current_lines = []
            inside_frame = True
        elif inside_frame:
            current_lines.append(line)

    if inside_frame and current_lines:
        frames.append("\n".join(current_lines))

    assert len(frames) > 0, (
        f"No frames parsed from {history_path}"
    )
    return frames


def compute_convergence(
    frames: List[str],
) -> List[Dict[str, Any]]:
    """Compute mask-count convergence across frames.

    Returns one dict per frame with keys:
      frame:          0-based frame index
      mask_count:     number of mask characters
      total_chars:    total non-whitespace characters
      resolved_ratio: fraction of tokens resolved
    """
    assert len(frames) > 0

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(frames):
        stripped = text.replace("\n", "").replace(
            "\r", ""
        )
        total = len(stripped)
        if total == 0:
            results.append({
                "frame": i,
                "mask_count": 0,
                "total_chars": 0,
                "resolved_ratio": 1.0,
            })
            continue

        mask_count = stripped.count(MASK_CHAR)
        resolved = total - mask_count
        results.append({
            "frame": i,
            "mask_count": mask_count,
            "total_chars": total,
            "resolved_ratio": round(
                resolved / total, 6
            ),
        })

    return results


def canvas_boundaries(
    canvas_index: List[int],
) -> List[int]:
    """Frame indices where a new canvas (block) begins.

    Empty for single-canvas runs (e.g. LLaDA, where every frame
    shares canvas index 0).
    """
    return [
        i
        for i in range(1, len(canvas_index))
        if canvas_index[i] != canvas_index[i - 1]
    ]


def total_elapsed_seconds(
    per_frame_elapsed: Any,
) -> Optional[float]:
    """Wall-clock seconds for a whole run, resumes included.

    The worker restarts its clock for each generate/resume/substitute
    segment, so a run edited before the client began carrying the
    offset forward has an elapsed series that drops back toward zero
    at each branch, and its stored ``elapsed_seconds`` (the last
    sample) covers only the final segment. Each drop marks a boundary,
    so summing the last value of every segment recovers the true
    total.

    Series saved since that fix are already monotonic, so there is no
    drop and this returns the final sample unchanged. Applying it to
    every run is therefore idempotent.

    Returns None for an empty or non-numeric series, which tells the
    caller to leave whatever was stored alone.
    """
    if not isinstance(per_frame_elapsed, list):
        return None
    if len(per_frame_elapsed) == 0:
        return None
    for value in per_frame_elapsed:
        if not isinstance(value, (int, float)):
            return None
        if isinstance(value, bool):
            return None

    total = 0.0
    for i in range(1, len(per_frame_elapsed)):
        if per_frame_elapsed[i] < per_frame_elapsed[i - 1]:
            total += per_frame_elapsed[i - 1]
    total += per_frame_elapsed[-1]

    assert total >= 0.0, "elapsed total went negative"
    assert total >= per_frame_elapsed[-1], (
        "total is shorter than its final segment"
    )
    return round(total, 2)


def _frames_have_records(frames: List[Any]) -> bool:
    """True if a token stream stores rich records, not legacy ids.

    Legacy runs saved only integer ids per token, which cannot drive
    the token overlays (no display text / mask flag). Rich runs store
    ``{t, m, id, c?}`` dicts. Scans until the first populated frame.
    """
    for frame in frames:
        if not frame:
            continue
        return isinstance(frame[0], dict)
    return False


def load_run_frames(
    run_dir: Path,
) -> Dict[str, Any]:
    """Load persisted per-token frame streams for a run.

    Reads ``tokens.json`` (primary / possibly edited run), the
    optional ``original_tokens.json`` (pre-edit snapshot), and the
    optional ``alternatives.json`` / ``original_alternatives.json``
    (per-position candidate sets for each run, only written when the
    capture was enabled). Tolerates legacy files that stored only
    integer ids: those cannot drive the token overlays, so
    ``records_available`` is False.

    Returns a dict with ``frames``, ``original_frames`` (or None),
    ``records_available``, ``alternatives`` (or None),
    ``alternatives_available``, and ``original_alternatives`` (or
    None). Raises ``ValueError`` on malformed files.

    The shape is the same for both eras. What differs is where
    ``records_available`` comes from: a v1 run declares it in the
    capture manifest, a v0 run has it inferred from the type of its
    first token.
    """
    assert run_dir.is_dir(), f"run dir not found: {run_dir}"

    manifest = _capture_manifest(run_dir)

    result: Dict[str, Any] = {
        "frames": None,
        "original_frames": None,
        "records_available": False,
        "alternatives": None,
        "alternatives_available": False,
        "original_alternatives": None,
    }

    tokens_path = run_dir / "tokens.json"
    if not tokens_path.is_file():
        return result
    frames = json.loads(tokens_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        raise ValueError(
            f"tokens.json is malformed in {run_dir}"
        )
    result["frames"] = frames
    if manifest is None:
        result["records_available"] = _frames_have_records(frames)
    else:
        # Every v1 run writes rich records, so the manifest saying
        # tokens were captured settles it. Sniffing would agree, but
        # only for as long as no frame is legitimately empty.
        result["records_available"] = bool(
            manifest.get("frame_tokens")
        )

    original_path = run_dir / "original_tokens.json"
    if original_path.is_file():
        original = json.loads(
            original_path.read_text(encoding="utf-8")
        )
        if not isinstance(original, list):
            raise ValueError(
                "original_tokens.json is malformed in"
                f" {run_dir}"
            )
        result["original_frames"] = original

    # Position-indexed, not per-frame: a position's candidate set is
    # fixed when it is sampled, so one entry per position covers the
    # whole run (see ar_sampler._build_frame).
    alternatives = _load_alternatives(
        run_dir / "alternatives.json", run_dir
    )
    if alternatives is not None:
        result["alternatives"] = alternatives
        result["alternatives_available"] = any(
            isinstance(entry, list) and len(entry) > 0
            for entry in alternatives
        )

    # The pre-edit run's candidate sets. Unlike its tokens these
    # cannot be reconstructed after the fact, because a substitution
    # discards the candidates from the edit position onward.
    result["original_alternatives"] = _load_alternatives(
        run_dir / "original_alternatives.json", run_dir
    )

    return result


def _load_alternatives(
    path: Path,
    run_dir: Path,
) -> Optional[List[Any]]:
    """Read one position-indexed candidate file, or None if absent."""
    if not path.is_file():
        return None
    alternatives = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(alternatives, list):
        raise ValueError(
            f"{path.name} is malformed in {run_dir}"
        )
    return alternatives


def _capture_manifest(
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """A run's declared capture list, or None if it has none.

    None means the caller has to infer what was captured, which is
    what every reader did before versioning and what v0 runs still
    need. Tolerant of a missing metadata file on purpose: this is
    reached from the token loader, whose subject is the sidecars, and
    a directory holding those without metadata is not a run the
    catalog would list anyway.
    """
    metadata = _load_metadata_if_present(run_dir)
    if run_schema_version(metadata) < SCHEMA_VERSION_LATEST:
        return None
    manifest = metadata.get(CAPTURE_KEY)
    if not isinstance(manifest, dict):
        return None
    return manifest


def _load_metadata_if_present(
    run_dir: Path,
) -> Dict[str, Any]:
    """Metadata for a directory that may not have any.

    An empty dict reads as v0 downstream, which is the right era for
    a run old enough to be missing pieces.
    """
    meta_path = run_dir / METADATA_NAME
    if not meta_path.is_file():
        return {}
    return load_run_metadata(run_dir)


def load_run_metadata(
    run_dir: Path,
) -> Dict[str, Any]:
    """Read and return metadata.json from a run dir.

    Adds the directory name as ``run_id`` for API use.

    Raises ``ValueError`` if the file does not hold a JSON object.
    That check exists because the next line assigns into the result:
    a metadata file holding a list or a string used to fail there
    with a bare ``TypeError`` about item assignment, which said
    nothing about which run was broken.
    """
    meta_path = run_dir / METADATA_NAME
    assert meta_path.is_file(), (
        f"metadata.json not found in {run_dir}"
    )

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"metadata.json in {run_dir.name} holds a"
            f" {type(data).__name__}, not an object"
        )
    data["run_id"] = run_dir.name
    return data


def _invalid_run_entry(
    run_id: str, reason: str
) -> Dict[str, Any]:
    """A catalog row for a run that could not be read.

    An entry rather than an omission. A run that silently vanishes
    from the list looks like a deleted run, and the natural next step
    is to save it again; a row that says what is wrong can be read,
    and deleted on purpose.

    Carries the keys the table sorts and groups on so a damaged run
    cannot break the rendering of the healthy ones around it.
    """
    return {
        "run_id": run_id,
        "invalid": True,
        "error": reason,
        "prompt": "",
        "model": "",
        "created_at": "",
        "has_diff": False,
    }


def list_runs(
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Scan results/ for all saved runs.

    Returns a list of metadata dicts sorted by
    created_at (newest first), falling back to
    directory name sort when created_at is absent.

    One damaged run cannot empty the catalog or fail the request. It
    becomes an entry carrying ``invalid`` and a reason, so the other
    runs still list and the broken one is visible enough to delete.
    """
    if not results_dir.is_dir():
        return []

    runs: List[Dict[str, Any]] = []
    for child in sorted(
        results_dir.iterdir(), reverse=True
    ):
        if not child.is_dir():
            continue
        # The run store's working directories (.staging, .trash) live
        # here too. They contain no metadata of their own, so they
        # would be skipped below anyway; skipping them by name says
        # so on purpose rather than by luck.
        if child.name.startswith("."):
            continue
        if not (child / METADATA_NAME).is_file():
            continue
        runs.append(_read_catalog_entry(child))

    runs.sort(
        key=lambda r: r.get(
            "created_at", r.get("run_id", "")
        ),
        reverse=True,
    )
    return runs


def _read_catalog_entry(
    run_dir: Path,
) -> Dict[str, Any]:
    """One row of the catalog, valid or explicitly not.

    Every failure mode of reading a run directory converges here, so
    that the loop above has no error handling of its own and cannot
    grow a path that lets one run take down the request.
    """
    try:
        meta = load_run_metadata(run_dir)
        version = run_schema_version(meta)
    except UnsupportedRunVersionError as exc:
        # Deliberately the end of the road for this run: nothing else
        # is read from it. Its fields were written by a build this
        # one does not know, so interpreting any of them, including
        # the timings repaired below, would be a guess presented as a
        # fact.
        return _invalid_run_entry(
            run_dir.name,
            "Saved by a newer version of this app"
            f" (format {exc.version}). Update to open it.",
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return _invalid_run_entry(
            run_dir.name, f"Unreadable metadata: {exc}"
        )
    except (OSError, AssertionError) as exc:
        return _invalid_run_entry(
            run_dir.name, f"Could not be read: {exc}"
        )

    assert version in SUPPORTED_SCHEMA_VERSIONS, version

    # A durable counterfactual diff is available only when the
    # pre-edit snapshot was saved alongside the run.
    meta["has_diff"] = (run_dir / "original_tokens.json").is_file()
    # Repairs edited runs saved before the elapsed series was made
    # cumulative, whose stored value covers only the final segment. A
    # no-op for every other run.
    repaired = total_elapsed_seconds(meta.get("per_frame_elapsed"))
    if repaired is not None:
        meta["elapsed_seconds"] = repaired
    return meta
