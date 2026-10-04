"""Independent clients filing at once, and none of them losing.

Strategy: skip HTTP and race `_collections_apply`, which is the whole
of what an endpoint does once the body is parsed. Threads cover two
tabs against one supervisor; separate processes cover the browser
entry point and the desktop app pointed at one results directory,
which is the case a `threading.Lock` cannot reach.

This is `DATA-02`'s stated verification, and it is the reason the
operations exist at all. Under the write path they replaced, every
client read the array, computed a successor from what it had, and
wrote the whole thing. Racing that loses filings by construction, and
no amount of locking the write alone repairs it, because the loss
happens between the read and the write rather than during either.

An operation carries only the gesture, so the list it lands on is
whichever one is current when the lock is taken. A test that passes
here would still pass with the lock removed if the transform were
applied to a stale snapshot, which is why the assertion counts every
filing rather than checking the file is valid JSON.

Passing proves N clients each filing a different run all survive, that
the same holds across processes, and that a run filed and unfiled by
two racing clients settles one way rather than corrupting the list.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

import src.web.server as server
from src.web import collections as ops
from src.web.ui_state import load_ui_state

from process_race import race_context

_KEY = "diffusion_collections"

# Enough contention to fail reliably when the lock is wrong. The
# threaded case failed on the first attempt with a read-then-write
# transform at this width.
THREADS = 16
PROCESSES = 8

# Long enough that processes the lock fails to exclude all read the
# list before any of them writes it.
PAUSE_SECONDS = 0.05


def _run_id(index: int) -> str:
    return f"2026-01-01_00-00-{index:02d}_llada"


def _seed_runs(root: Path, count: int) -> None:
    """Real run folders, since filing refuses a run with none."""
    for index in range(count):
        run_dir = root / _run_id(index)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(
            json.dumps({"backend": "llada", "prompt": "p"}),
            encoding="utf-8",
        )


def _seed_collection(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ui_state.json").write_text(
        json.dumps(
            {
                _KEY: ops.encode(
                    [{"id": "papers", "name": "Papers", "runs": []}]
                )
            }
        ),
        encoding="utf-8",
    )


def _file_one(root: Path, index: int) -> None:
    """One client's whole job: file one run, through the real path."""
    server.RESULTS_DIR = root
    existing = {_run_id(i) for i in range(THREADS + PROCESSES)}
    server._collections_apply(
        lambda current: ops.add_run(
            current, "papers", _run_id(index), existing
        )
    )


def _file_one_paused(root: Path, index: int) -> None:
    """``_file_one`` with the filing paused between its read and its
    write, installed here because a process from ``race_context``
    inherits no patch from the test.

    Threads start together, so their race contends as it is. A process
    acts only once it has imported the server, which spreads their
    starts far wider than one filing takes, so without the pause they
    mostly file one after another and a missing lock goes unseen.
    """
    real_add_run = ops.add_run

    def add_run_paused(*args: Any, **kwargs: Any) -> Any:
        filed = real_add_run(*args, **kwargs)
        time.sleep(PAUSE_SECONDS)
        return filed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ops, "add_run", add_run_paused)
        _file_one(root, index)


def _filed(root: Path) -> List[str]:
    stored = load_ui_state(root).get(_KEY)
    collections: List[Dict[str, Any]] = ops.decode(stored)
    assert len(collections) == 1, collections
    return collections[0]["runs"]


@pytest.fixture()
def results_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    _seed_runs(tmp_path, THREADS + PROCESSES)
    _seed_collection(tmp_path)
    return tmp_path


# -- two tabs --


def test_threads_filing_at_once_lose_nothing(
    results_dir: Path,
) -> None:
    """Sixteen clients, sixteen different runs, one collection.

    The old write path could not pass this: each client would have
    computed its successor from the list it read, so the last writer
    would have stored a list containing its own run and whichever
    others happened to be visible when it read.
    """
    threads = [
        threading.Thread(target=_file_one, args=(results_dir, i))
        for i in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    filed = _filed(results_dir)
    assert sorted(filed) == sorted(
        _run_id(i) for i in range(THREADS)
    )


def test_no_filing_is_duplicated(results_dir: Path) -> None:
    """The other way a race shows: the same run appearing twice
    because two clients both read a list without it."""
    threads = [
        threading.Thread(target=_file_one, args=(results_dir, 0))
        for _ in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert _filed(results_dir) == [_run_id(0)]


# -- two supervisors --


def test_processes_filing_at_once_lose_nothing(
    results_dir: Path,
) -> None:
    """The half a threading lock cannot give. Two supervisors is not
    hypothetical: the browser entry point and the desktop app are
    separate processes pointed at one results directory."""
    context = race_context()
    procs = [
        context.Process(
            target=_file_one_paused, args=(results_dir, index)
        )
        for index in range(PROCESSES)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)

    assert all(proc.exitcode == 0 for proc in procs)
    assert sorted(_filed(results_dir)) == sorted(
        _run_id(i) for i in range(PROCESSES)
    )


# -- gestures that disagree --


def test_a_star_racing_a_filing_leaves_a_coherent_list(
    results_dir: Path,
) -> None:
    """Two clients acting on one run in opposite directions.

    Either order is a correct outcome, so this asserts coherence
    rather than a winner: the run is either filed or not, the list is
    readable, and no collection was lost on the way.
    """
    existing = {_run_id(i) for i in range(THREADS)}

    def star() -> None:
        server.RESULTS_DIR = results_dir
        server._collections_apply(
            lambda current: ops.toggle_favorite(
                current, _run_id(0), existing
            )
        )

    threads = [
        threading.Thread(target=_file_one, args=(results_dir, 0)),
        threading.Thread(target=star),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    collections = ops.decode(load_ui_state(results_dir).get(_KEY))
    ids = [c["id"] for c in collections]
    assert "papers" in ids
    for entry in collections:
        assert entry["runs"].count(_run_id(0)) <= 1
