"""Durable, origin-independent UI state for the visualizer frontend.

The desktop app (pywebview/QtWebEngine) keys ``localStorage`` by window
origin (``scheme://host:port``). Because the launcher's port can vary
between runs, localStorage-backed UI state (Settings, the analytics
"new run" cue, prompt history, the generate teaser) appeared to reset
across restarts. This module persists those values server-side in a
single JSON file under ``results/``, so they survive regardless of the
window origin and are shared between the browser and desktop entry
points.

Values are stored verbatim as the strings the frontend keeps in
localStorage: the server is a durable key/value mirror, not a schema,
so the client's existing (synchronous) localStorage reads keep working
unchanged after a one-time hydrate on boot. Writes are atomic (temp
file + ``os.replace``) and serialized with a process-wide lock.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict

# Allowed keys mapped to the maximum accepted value length (characters).
# Bounding the size stops a runaway client from growing the file without
# limit (TigerStyle: put a limit on everything). Prompt history and the
# new-run cue are the largest, so they get more room than the small
# settings/flag values.
UI_STATE_KEYS: Dict[str, int] = {
    "diffusion_settings": 8_192,
    "diffusion_new_runs": 262_144,
    "diffusion_prompt_history": 262_144,
    "diffusion_generate_teased": 64,
    "diffusion_download_toast_corner": 32,
    # Overlay drawer vertical offset, one key per page: the two
    # drawers sit in containers of different heights, so a shared
    # offset would land sensibly on at most one of them. A plain
    # number, so 32 characters is generous.
    "diffusion_overlay_drawer_top_generator": 32,
    "diffusion_overlay_drawer_top_analytics": 32,
}

# Serialize read-modify-write so concurrent PUTs cannot clobber each
# other or observe a half-written file.
_LOCK = threading.Lock()


def _state_path(results_dir: Path) -> Path:
    return results_dir / "ui_state.json"


def load_ui_state(results_dir: Path) -> Dict[str, str]:
    """Return the stored UI-state mapping, or ``{}`` when absent/corrupt.

    Never raises for a missing or malformed file: UI state is a
    convenience cache, so a bad file degrades to defaults rather than
    breaking page loads. Only known, string-valued keys are returned.
    """
    assert isinstance(results_dir, Path), "results_dir must be a Path"
    path = _state_path(results_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in data.items():
        if key in UI_STATE_KEYS and isinstance(value, str):
            result[key] = value
    return result


def set_ui_state_key(
    results_dir: Path, key: str, value: str
) -> Dict[str, str]:
    """Set one UI-state key and return the full updated mapping.

    Raises ``KeyError`` for an unknown key and ``ValueError`` for a
    non-string or oversized value: these are client-contract violations
    (operating errors) that the caller surfaces as 4xx responses.
    """
    assert isinstance(results_dir, Path), "results_dir must be a Path"
    if key not in UI_STATE_KEYS:
        raise KeyError(f"unknown ui-state key: {key}")
    if not isinstance(value, str):
        raise ValueError("ui-state value must be a string")
    limit = UI_STATE_KEYS[key]
    if len(value) > limit:
        raise ValueError(
            f"ui-state value for {key} exceeds {limit} characters"
        )
    with _LOCK:
        state = load_ui_state(results_dir)
        state[key] = value
        _write_atomic(results_dir, state)
    return state


def _write_atomic(results_dir: Path, state: Dict[str, str]) -> None:
    """Write the state file atomically (temp file + ``os.replace``).

    A crash mid-write leaves the previous file intact rather than a
    truncated one, since ``os.replace`` swaps the fully written temp
    file into place in a single step.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(results_dir)
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    handle_fd, tmp_name = tempfile.mkstemp(
        dir=str(results_dir), prefix=".ui_state.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass  # Temp file already gone; nothing to clean up.
        raise
