"""Conversation compare-and-swap across independent supervisors.

Strategy: forkserver children all use the same expected revision and
pause immediately before manifest publication. Without the shared
filesystem lock they all read the old manifest before any writes;
with it, exactly one commits and every later child observes a stale
revision.

Passing proves both assistant versioning and user append reserve one
winner across processes, with no hybrid versions or lost turn pairs.
"""

from __future__ import annotations

import time
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Callable

from src.web import conversation_store as store

from process_race import race_context


PROCESS_COUNT = 6
PAUSE_SECONDS = 0.05


def _pause_manifest_publication() -> None:
    real_write = store._write_manifest

    def paused(
        conversation_dir: Path,
        manifest: store.ConversationManifest,
    ) -> None:
        time.sleep(PAUSE_SECONDS)
        real_write(conversation_dir, manifest)

    store._write_manifest = paused


def _record(
    root: Path,
    operation: str,
    index: int,
    mutate: Callable[[], None],
) -> None:
    _pause_manifest_publication()
    try:
        mutate()
    except store.ConversationRevisionConflictError:
        outcome = "conflict"
    else:
        outcome = "committed"
    (root / f"{operation}-{index}.txt").write_text(
        outcome, encoding="utf-8"
    )


def _race_update(
    root: Path,
    conversation_id: str,
    assistant_turn_id: str,
    index: int,
) -> None:
    def mutate() -> None:
        store.update_assistant(
            root,
            conversation_id,
            assistant_turn_id,
            expected_revision=2,
            text=f"Answer {index}",
            partial=False,
        )

    _record(root, "update", index, mutate)


def _race_append(
    root: Path,
    conversation_id: str,
    index: int,
) -> None:
    def mutate() -> None:
        store.append_user(
            root,
            conversation_id,
            expected_revision=3,
            text=f"Question {index}",
            model_id="llada",
            input_mode="chat",
        )

    _record(root, "append", index, mutate)


def _outcomes(root: Path, operation: str) -> list[str]:
    return [
        (root / f"{operation}-{index}.txt").read_text(
            encoding="utf-8"
        )
        for index in range(PROCESS_COUNT)
    ]


def _run_processes(processes: list[BaseProcess]) -> None:
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    assert all(process.exitcode == 0 for process in processes)


def test_assistant_race_has_one_winner(tmp_path: Path) -> None:
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    context = race_context()
    processes = [
        context.Process(
            target=_race_update,
            args=(
                tmp_path,
                created.id,
                appended.assistant_turn.turn_id,
                index,
            ),
        )
        for index in range(PROCESS_COUNT)
    ]

    _run_processes(processes)

    outcomes = _outcomes(tmp_path, "update")
    page = store.get_turns(tmp_path, created.id)
    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == PROCESS_COUNT - 1
    assert store.get_manifest(tmp_path, created.id).revision == 3
    assert page.turns[-1].text.startswith("Answer ")
    assert page.turns[-1].version == 2


def test_user_append_race_reserves_one_pair(tmp_path: Path) -> None:
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    context = race_context()
    processes = [
        context.Process(
            target=_race_append,
            args=(tmp_path, created.id, index),
        )
        for index in range(PROCESS_COUNT)
    ]

    _run_processes(processes)

    outcomes = _outcomes(tmp_path, "append")
    manifest = store.get_manifest(tmp_path, created.id)
    page = store.get_turns(tmp_path, created.id)
    assert completed.manifest.revision == 3
    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == PROCESS_COUNT - 1
    assert manifest.revision == 4
    assert manifest.turn_count == 4
    assert manifest.pending_assistant_id == "00000004"
    assert page.turns[-2].text.startswith("Question ")
    assert page.turns[-1].turn_id == "00000004"
