"""Conversation compare-and-swap across independent supervisors.

Strategy: forkserver children race one branch revision, independent
branch revisions, or one catalog revision while publication pauses.

Passing proves same-branch writes reserve one winner, unrelated
branches both advance, catalog CAS reserves one fork, and no process
publishes hybrid versions or loses a turn pair.
"""

from __future__ import annotations

import time
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Callable

from src.web import _conversation_branch_store as branch_store
from src.web import _conversation_store_core as core_store
from src.web import conversation_generation
from src.web import conversation_store as store

from process_race import race_context


PROCESS_COUNT = 6
PAUSE_SECONDS = 0.05


def _generation_configuration() -> dict[str, object]:
    """Build the exact deferred snapshot shared by race processes."""
    return {
        "codec_version": (
            store.GENERATION_CONFIGURATION_CODEC_VERSION
        ),
        "model_id": "llada",
        "input_mode": "chat",
        "device": "cuda",
        "schema_id": (
            conversation_generation.registry_generation_schema_id(
                "llada", "cuda"
            )
        ),
        "experimental": False,
        "parameters": {
            "steps": 128,
            "gen_length": 160,
            "block_length": 160,
            "temperature": 0.7,
            "cfg_scale": 0.0,
            "seed": -1,
            "remasking": "low_confidence",
            "alternatives": True,
        },
    }


def _pause_manifest_publication() -> None:
    real_write = core_store.write_legacy_manifest
    real_branch_write = branch_store.write_branch_manifest

    def paused(
        conversation_dir: Path,
        manifest: store.ConversationManifest,
    ) -> None:
        time.sleep(PAUSE_SECONDS)
        real_write(conversation_dir, manifest)

    def paused_branch(
        branch_dir: Path,
        branch: branch_store.StoredBranch,
    ) -> None:
        time.sleep(PAUSE_SECONDS)
        real_branch_write(branch_dir, branch)

    core_store.write_legacy_manifest = paused
    branch_store.write_branch_manifest = paused_branch


def _record(
    root: Path,
    operation: str,
    index: int,
    mutate: Callable[[], None],
) -> None:
    _pause_manifest_publication()
    try:
        mutate()
    except (
        store.ConversationRevisionConflictError,
        store.ConversationCatalogRevisionConflictError,
    ):
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
    branch_id: str,
    index: int,
) -> None:
    def mutate() -> None:
        store.update_assistant(
            root,
            conversation_id,
            assistant_turn_id,
            branch_id=branch_id,
            expected_revision=2,
            text=f"Answer {index}",
            partial=False,
        )

    _record(root, "update", index, mutate)


def _race_append(
    root: Path,
    conversation_id: str,
    branch_id: str,
    index: int,
) -> None:
    def mutate() -> None:
        store.append_user(
            root,
            conversation_id,
            branch_id=branch_id,
            expected_revision=3,
            text=f"Question {index}",
            model_id="llada",
            input_mode="chat",
        )

    _record(root, "append", index, mutate)


def _race_explicit_update(
    root: Path,
    conversation_id: str,
    assistant_turn_id: str,
    branch_id: str,
    expected_revision: int,
    index: int,
) -> None:
    def mutate() -> None:
        store.update_assistant(
            root,
            conversation_id,
            assistant_turn_id,
            branch_id=branch_id,
            expected_revision=expected_revision,
            text=f"Branch answer {index}",
            partial=False,
        )

    _record(root, "independent", index, mutate)


def _race_retry_fork(
    root: Path,
    conversation_id: str,
    assistant_turn_id: str,
    branch_id: str,
    expected_revision: int,
    expected_catalog_revision: int,
    index: int,
) -> None:
    def mutate() -> None:
        store.fork_retry_assistant(
            root,
            conversation_id,
            assistant_turn_id,
            operation_id=f"{index:032x}",
            branch_id=branch_id,
            expected_revision=expected_revision,
            expected_catalog_revision=expected_catalog_revision,
            model_id="llada",
            input_mode="chat",
            generation_configuration=_generation_configuration(),
        )

    _record(root, "fork", index, mutate)


def _race_duplicate_retry(
    root: Path,
    conversation_id: str,
    assistant_turn_id: str,
    branch_id: str,
    expected_revision: int,
    expected_catalog_revision: int,
    index: int,
) -> None:
    _pause_manifest_publication()
    result = store.fork_retry_assistant(
        root,
        conversation_id,
        assistant_turn_id,
        operation_id="d" * 32,
        branch_id=branch_id,
        expected_revision=expected_revision,
        expected_catalog_revision=expected_catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(),
    )
    (root / f"duplicate-{index}.txt").write_text(
        result.branch.branch_id,
        encoding="utf-8",
    )


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


def _legacy_completed(
    root: Path,
) -> tuple[store.AppendResult, store.ConversationMutation]:
    conversation_id = "a" * 32
    conversation_dir = (
        root / store.CONVERSATIONS_DIR_NAME / conversation_id
    )
    (conversation_dir / store.TURNS_DIR_NAME).mkdir(parents=True)
    timestamp = "2026-01-01T00:00:00.000Z"
    manifest = store.ConversationManifest(
        id=conversation_id,
        title="Legacy race",
        revision=1,
        created_at=timestamp,
        updated_at=timestamp,
        turn_count=0,
        tail_role=None,
        tail_turn_id=None,
        tail_version=None,
        pending_assistant_id=None,
    )
    core_store.write_legacy_manifest(conversation_dir, manifest)
    appended = store.append_user(
        root,
        conversation_id,
        expected_revision=1,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        root,
        conversation_id,
        appended.assistant_turn.turn_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    return appended, completed


def test_assistant_race_has_one_winner(tmp_path: Path) -> None:
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    context = race_context()
    assert appended.manifest.branch_id is not None
    processes = [
        context.Process(
            target=_race_update,
            args=(
                tmp_path,
                created.id,
                appended.assistant_turn.turn_id,
                appended.manifest.branch_id,
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
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    context = race_context()
    assert completed.manifest.branch_id is not None
    processes = [
        context.Process(
            target=_race_append,
            args=(
                tmp_path,
                created.id,
                completed.manifest.branch_id,
                index,
            ),
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
    assert manifest.pending_assistant_id == page.turns[-1].turn_id
    assert page.turns[-2].text.startswith("Question ")
    assert page.turns[-1].index == 4


def test_unrelated_branch_updates_both_commit(
    tmp_path: Path,
) -> None:
    """Two processes prove branch-local CAS is independent."""
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="Original",
        partial=False,
    )
    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="a" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    assert completed.manifest.branch_id is not None
    context = race_context()
    processes = [
        context.Process(
            target=_race_explicit_update,
            args=(
                tmp_path,
                created.id,
                appended.assistant_turn.turn_id,
                completed.manifest.branch_id,
                completed.manifest.revision,
                0,
            ),
        ),
        context.Process(
            target=_race_explicit_update,
            args=(
                tmp_path,
                created.id,
                retried.assistant_turn.turn_id,
                retried.branch.branch_id,
                retried.manifest.revision,
                1,
            ),
        ),
    ]

    _run_processes(processes)

    outcomes = [
        (tmp_path / f"independent-{index}.txt").read_text()
        for index in range(2)
    ]
    original = store.get_turns(
        tmp_path,
        created.id,
        branch_id=completed.manifest.branch_id,
    )
    alternate = store.get_turns(
        tmp_path,
        created.id,
        branch_id=retried.branch.branch_id,
    )
    assert outcomes == ["committed", "committed"]
    assert original.turns[-1].text == "Branch answer 0"
    assert alternate.turns[-1].text == "Branch answer 1"


def test_catalog_cas_gives_one_concurrent_fork_winner(
    tmp_path: Path,
) -> None:
    """Concurrent forks share catalog CAS."""
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    catalog = store.get_catalog(tmp_path, created.id)
    assert completed.manifest.branch_id is not None
    context = race_context()
    processes = [
        context.Process(
            target=_race_retry_fork,
            args=(
                tmp_path,
                created.id,
                appended.assistant_turn.turn_id,
                completed.manifest.branch_id,
                completed.manifest.revision,
                catalog.revision,
                index,
            ),
        )
        for index in range(PROCESS_COUNT)
    ]

    _run_processes(processes)

    outcomes = _outcomes(tmp_path, "fork")
    branches = store.list_branches(tmp_path, created.id)
    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == PROCESS_COUNT - 1
    assert len(branches.branches) == 2
    assert branches.catalog.revision == catalog.revision + 1


def test_concurrent_duplicate_fork_replays_one_branch(
    tmp_path: Path,
) -> None:
    """One operation id serializes to one commit and six successes."""
    created = store.create(tmp_path)
    appended = store.append_user(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    completed = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    catalog = store.get_catalog(tmp_path, created.id)
    assert completed.manifest.branch_id is not None
    context = race_context()
    processes = [
        context.Process(
            target=_race_duplicate_retry,
            args=(
                tmp_path,
                created.id,
                appended.assistant_turn.turn_id,
                completed.manifest.branch_id,
                completed.manifest.revision,
                catalog.revision,
                index,
            ),
        )
        for index in range(PROCESS_COUNT)
    ]

    _run_processes(processes)

    branch_ids = [
        (tmp_path / f"duplicate-{index}.txt").read_text(
            encoding="utf-8"
        )
        for index in range(PROCESS_COUNT)
    ]
    listed = store.list_branches(tmp_path, created.id)
    operations_root = (
        tmp_path
        / store.CONVERSATIONS_DIR_NAME
        / created.id
        / store.OPERATIONS_DIR_NAME
    )

    assert len(set(branch_ids)) == 1
    assert len(listed.branches) == 2
    assert listed.catalog.revision == catalog.revision + 1
    assert {path.name for path in operations_root.iterdir()} == {
        f"{'d' * 32}.json"
    }
    pending = store.get_turns(
        tmp_path,
        created.id,
        branch_id=listed.catalog.default_branch_id,
    ).turns[-1]
    assert pending.metadata == {
        store.PENDING_GENERATION_KEY: _generation_configuration()
    }


def test_lazy_upgrade_race_has_one_catalog_winner(
    tmp_path: Path,
) -> None:
    """Concurrent v1 forks publish one upgrade and one child."""
    appended, completed = _legacy_completed(tmp_path)
    branch_id = f"b_{completed.manifest.id}"
    context = race_context()
    processes = [
        context.Process(
            target=_race_retry_fork,
            args=(
                tmp_path,
                completed.manifest.id,
                appended.assistant_turn.turn_id,
                branch_id,
                completed.manifest.revision,
                0,
                index,
            ),
        )
        for index in range(PROCESS_COUNT)
    ]

    _run_processes(processes)

    outcomes = _outcomes(tmp_path, "fork")
    branches = store.list_branches(tmp_path, completed.manifest.id)
    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == PROCESS_COUNT - 1
    assert len(branches.branches) == 2
    assert branches.catalog.revision == 1
