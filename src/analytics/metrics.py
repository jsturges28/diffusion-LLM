"""Compute intrinsic diffusion metrics from saved runs.

Reads history.txt and metadata.json from results/ directories
and produces convergence and timing statistics for the
Analytics Suite frontend.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

MASK_CHAR = "\u2591"  # ░
FRAME_HEADER_RE = re.compile(
    r"^={5}\s+FRAME\s+\d+\s+={5}$"
)


def parse_history(
    history_path: Path,
) -> List[str]:
    """Split history.txt into per-frame text strings.

    Each frame is delimited by ``===== FRAME N =====``
    headers. Returns a list where index *i* holds the
    text body of frame *i*.
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
      frame        — 0-based frame index
      mask_count   — number of mask characters
      total_chars  — total non-whitespace characters
      resolved_ratio — fraction of tokens resolved
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
    optional ``alternatives.json`` (per-position candidate sets, only
    written when the capture was enabled). Tolerates legacy files
    that stored only integer ids: those cannot drive the token
    overlays, so ``records_available`` is False.

    Returns a dict with ``frames``, ``original_frames`` (or None),
    ``records_available``, ``alternatives`` (or None), and
    ``alternatives_available``. Raises ``ValueError`` on malformed
    files.
    """
    assert run_dir.is_dir(), f"run dir not found: {run_dir}"

    result: Dict[str, Any] = {
        "frames": None,
        "original_frames": None,
        "records_available": False,
        "alternatives": None,
        "alternatives_available": False,
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
    result["records_available"] = _frames_have_records(frames)

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
    alternatives_path = run_dir / "alternatives.json"
    if alternatives_path.is_file():
        alternatives = json.loads(
            alternatives_path.read_text(encoding="utf-8")
        )
        if not isinstance(alternatives, list):
            raise ValueError(
                f"alternatives.json is malformed in {run_dir}"
            )
        result["alternatives"] = alternatives
        result["alternatives_available"] = any(
            isinstance(entry, list) and len(entry) > 0
            for entry in alternatives
        )

    return result


def load_run_metadata(
    run_dir: Path,
) -> Dict[str, Any]:
    """Read and return metadata.json from a run dir.

    Adds the directory name as ``run_id`` for API use.
    """
    meta_path = run_dir / "metadata.json"
    assert meta_path.is_file(), (
        f"metadata.json not found in {run_dir}"
    )

    data: Dict[str, Any] = json.loads(
        meta_path.read_text(encoding="utf-8")
    )
    data["run_id"] = run_dir.name
    return data


def list_runs(
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Scan results/ for all saved runs.

    Returns a list of metadata dicts sorted by
    created_at (newest first), falling back to
    directory name sort when created_at is absent.
    """
    if not results_dir.is_dir():
        return []

    runs: List[Dict[str, Any]] = []
    for child in sorted(
        results_dir.iterdir(), reverse=True
    ):
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = load_run_metadata(child)
            # A durable counterfactual diff is available only when the
            # pre-edit snapshot was saved alongside the run.
            meta["has_diff"] = (
                child / "original_tokens.json"
            ).is_file()
            runs.append(meta)
        except (json.JSONDecodeError, AssertionError):
            continue

    runs.sort(
        key=lambda r: r.get(
            "created_at", r.get("run_id", "")
        ),
        reverse=True,
    )
    return runs
