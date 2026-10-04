"""Durable, origin-independent UI state for the visualizer frontend.

The desktop app (pywebview/QtWebEngine) keys ``localStorage`` by
window origin (``scheme://host:port``). Because the launcher's port
can vary between runs, localStorage-backed UI state (Settings, the
analytics "new run" cue, prompt history, the generate teaser) appeared
to reset across restarts. This module persists those values
server-side in a single JSON file under ``results/``, so they survive
regardless of the window origin and are shared between the browser
and desktop entry points.

Values are stored verbatim as the strings the frontend keeps in
localStorage: the server is a durable key/value mirror, not a schema,
so the client's existing (synchronous) localStorage reads keep working
unchanged after a one-time hydrate on boot. Writes are atomic (temp
file + ``os.replace``) and serialized by a lock every supervisor
shares.

Most of these keys are caches, where losing one costs a preference.
``diffusion_collections`` is not: it holds which runs the user filed
into which collection, which is intent they expressed and cannot be
recomputed from anything on disk. The mechanism is deliberately the
same anyway, since it is already atomic and origin-independent, but
that key is the reason this file is worth not corrupting.

Atomicity is not the same as not losing writes, and this module used
to provide only the first. Two things were missing, both from
``DATA-02``:

- The lock was a ``threading.Lock``, which is process-local. The
  browser supervisor and the desktop supervisor are separate
  processes writing one file, so each could read, modify, and write
  over the other. An ``flock`` on a sidecar file now covers that,
  through ``data_root_lock``.
- A caller that read the file, computed a new value, and then called
  ``set_ui_state_key`` was doing a read-modify-write with the lock
  held for only the write half, so a value computed from an older
  snapshot could land on top of a newer one. ``mutate_ui_state_key``
  exists so that whole sequence happens under one hold.

What is deliberately *not* here is conflict semantics: two clients
each replacing the whole value still means the later one wins, and
deciding between server-authoritative collection operations and a
revision scheme is the open half of ``DATA-02``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

from src.web.data_root_lock import DataRootLock

# Allowed keys mapped to the maximum accepted value length, in
# characters. Bounding the size stops a runaway client from growing
# the file without limit (TigerStyle: put a limit on everything).
# Prompt history and the new-run cue are the largest, so they get more
# room than the small settings/flag values.
UI_STATE_KEYS: Dict[str, int] = {
    "diffusion_settings": 8_192,
    "diffusion_new_runs": 262_144,
    "diffusion_prompt_history": 262_144,
    "diffusion_generate_teased": 64,
    "diffusion_download_toast_corner": 32,
    # The durable conversation the generator restores, as a compact
    # JSON object carrying its id and last observed revision.
    "diffusion_active_conversation": 512,
    # Overlay drawer vertical offset, one key per page: the two
    # drawers sit in containers of different heights, so a shared
    # offset would land sensibly on at most one of them. A plain
    # number, so 32 characters is generous.
    "diffusion_overlay_drawer_top_generator": 32,
    "diffusion_overlay_drawer_top_analytics": 32,
    # Analytics collections: [{id, name, runs: [run_id]}]. Sized to
    # match diffusion_new_runs, which holds the same kind of thing (a
    # list of run ids) and so runs out at roughly the same point.
    "diffusion_collections": 262_144,
}

# Holds every read-modify-write of the state file, in any process.
# The sidecar keeps the name it has always had, so a supervisor still
# running older code contends on the same file as this one.
_STATE_LOCK = DataRootLock("ui_state.lock")


def _state_path(results_dir: Path) -> Path:
    return results_dir / "ui_state.json"


def load_ui_state(results_dir: Path) -> Dict[str, str]:
    """The stored UI-state mapping, or ``{}`` when absent or corrupt.

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
    non-string or oversized value: these are client-contract
    violations (operating errors) that the caller surfaces as 4xx
    responses.
    """
    assert isinstance(results_dir, Path), "results_dir must be a Path"
    _validate_key_value(key, value)
    with _STATE_LOCK.held(results_dir):
        state = load_ui_state(results_dir)
        state[key] = value
        _write_atomic(results_dir, state)
    return state


def mutate_ui_state_key(
    results_dir: Path,
    key: str,
    mutate: Callable[[Optional[str]], Optional[str]],
) -> Dict[str, str]:
    """Read a key, transform it, and write it back under one lock.

    For a caller that derives a new value from the current one. Doing
    that as a read followed by ``set_ui_state_key`` leaves a gap where
    another writer's value lands and is then overwritten by one
    computed before it existed, which is how a GET-time reconcile
    could undo a concurrent PUT.

    *mutate* receives the stored raw value, or ``None`` when the key
    is absent, and returns the replacement, or ``None`` to leave the
    file untouched. It runs with the lock held, so it must not block
    on anything slow and must not call back into this module.

    Returns the mapping as it stands afterwards, written or not.
    """
    assert isinstance(results_dir, Path), "results_dir must be a Path"
    if key not in UI_STATE_KEYS:
        raise KeyError(f"unknown ui-state key: {key}")
    with _STATE_LOCK.held(results_dir):
        state = load_ui_state(results_dir)
        value = mutate(state.get(key))
        if value is None:
            return state
        _validate_key_value(key, value)
        state[key] = value
        _write_atomic(results_dir, state)
    return state


def _validate_key_value(key: str, value: str) -> None:
    """Reject what the client contract does not allow.

    Bounding the size stops a runaway client growing the file without
    limit; the key check stops it inventing storage.
    """
    if key not in UI_STATE_KEYS:
        raise KeyError(f"unknown ui-state key: {key}")
    if not isinstance(value, str):
        raise ValueError("ui-state value must be a string")
    limit = UI_STATE_KEYS[key]
    if len(value) > limit:
        raise ValueError(
            f"ui-state value for {key} exceeds {limit} characters"
        )


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
        Path(tmp_name).replace(path)
    except OSError:
        # Best-effort cleanup; the temp file may already be gone.
        # The write's own failure is what matters and is reraised.
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise
