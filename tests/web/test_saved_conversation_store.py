"""Immutable selected-path conversation snapshot storage.

Strategy: create real durable conversations and saved runs under a
temporary data root, snapshot exact heads, then mutate or delete every
source. Passing proves snapshot text and pinned XAI stay independent,
publication is idempotent, paging is bounded, and mutable titles alone
use revision CAS.
"""

from __future__ import annotations

import errno
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from src.web import conversation_store
from src.web import run_store
from src.web import saved_conversation_store as snapshots


IMPORT_PROBE = """
import sys
import src.web.saved_conversation_store
forbidden = {"fastapi", "pydantic", "torch", "transformers"}
print(",".join(sorted(set(sys.modules) & forbidden)))
"""


def test_store_imports_neither_framework_nor_models() -> None:
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ""


def _run(
    root: Path, *, text: str = "saved answer"
) -> tuple[str, int]:
    bundle = run_store.RunBundle(
        metadata={"backend": "llada", "prompt": "question"},
        final_text=text,
        frames=[text],
    )
    return run_store.save(root, bundle, model_id="llada")


def _append_pair(
    root: Path,
    manifest: conversation_store.ConversationManifest,
    *,
    question: str = "Question",
    answer: str = "Answer",
) -> tuple[
    conversation_store.AppendResult,
    conversation_store.ConversationMutation,
]:
    appended = conversation_store.append_user(
        root,
        manifest.id,
        branch_id=manifest.branch_id,
        expected_revision=manifest.revision,
        text=question,
        model_id="llada",
        input_mode="chat",
    )
    completed = conversation_store.update_assistant(
        root,
        manifest.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text=answer,
        partial=False,
        context_pack={"prompt_token_count": 3},
        metadata={"status": "completed"},
    )
    return appended, completed


def _conversation(
    root: Path,
    *,
    linked: bool = False,
) -> tuple[
    conversation_store.ConversationManifest,
    conversation_store.AppendResult,
    conversation_store.ConversationMutation,
    str | None,
]:
    created = conversation_store.create(root)
    appended, completed = _append_pair(root, created)
    if not linked:
        return created, appended, completed, None
    run_id, revision = _run(root)
    mutation = conversation_store.set_run_link(
        root,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=conversation_store.RunLink(run_id, revision),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )
    return created, appended, mutation, run_id


def _head(
    mutation: conversation_store.ConversationMutation,
) -> dict[str, object]:
    manifest = mutation.manifest
    assert manifest.branch_id is not None
    assert manifest.tail_turn_id is not None
    assert manifest.tail_version is not None
    return {
        "conversation_id": manifest.id,
        "branch_id": manifest.branch_id,
        "branch_revision": manifest.revision,
        "turn_count": manifest.turn_count,
        "tail_turn_id": manifest.tail_turn_id,
        "tail_version": manifest.tail_version,
    }


def _create(
    root: Path,
    mutation: conversation_store.ConversationMutation,
    *,
    operation_id: str | None = None,
    title: str = "Research thread",
) -> snapshots.SnapshotResult:
    return snapshots.create_snapshot(
        root,
        operation_id=operation_id or uuid4().hex,
        title=title,
        **_head(mutation),
    )


def test_preview_reads_the_full_exact_path(tmp_path: Path) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)

    preview = snapshots.preview_snapshot(tmp_path, **_head(completed))

    assert preview.default_title == "Question"
    assert preview.turn_count == 2
    assert preview.exchange_count == 1
    assert preview.xai_count == 0
    assert preview.text_only_count == 1
    assert preview.unavailable_count == 0
    assert preview.tail_xai_status == snapshots.STATUS_TEXT_ONLY


def test_snapshot_survives_source_deletion(tmp_path: Path) -> None:
    created, appended, linked, run_id = _conversation(
        tmp_path, linked=True
    )
    assert run_id is not None
    result = _create(tmp_path, linked)

    conversation_store.delete(tmp_path, created.id)
    run_store.delete(tmp_path, run_id)

    page = snapshots.page_turns(tmp_path, result.snapshot_id)
    assert [turn["text"] for turn in page["turns"]] == [
        "Question",
        "Answer",
    ]
    assistant = page["turns"][1]
    assert assistant["xai"]["status"] == snapshots.STATUS_PINNED
    pinned = snapshots.resolve_pinned_run_dir(
        tmp_path, result.snapshot_id, appended.assistant_turn.turn_id
    )
    assert (pinned / run_store.FINAL_TEXT_NAME).read_text(
        encoding="utf-8"
    ) == "saved answer"


def test_schema_one_link_map_remains_readable(tmp_path: Path) -> None:
    _created, appended, linked, _run_id = _conversation(
        tmp_path, linked=True
    )
    result = _create(tmp_path, linked)
    snapshot_dir = (
        tmp_path / snapshots.ROOT_NAME / result.snapshot_id
    )
    metadata_path = snapshot_dir / snapshots.METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    link_path = (
        snapshot_dir
        / snapshots.LINKS_DIR_NAME
        / f"{appended.assistant_turn.turn_id}.json"
    )
    link = json.loads(link_path.read_text(encoding="utf-8"))
    (snapshot_dir / snapshots.LEGACY_LINKS_NAME).write_text(
        json.dumps({appended.assistant_turn.turn_id: link}),
        encoding="utf-8",
    )
    shutil.rmtree(snapshot_dir / snapshots.LINKS_DIR_NAME)

    page = snapshots.page_turns(tmp_path, result.snapshot_id)
    pinned = snapshots.resolve_pinned_run_dir(
        tmp_path, result.snapshot_id, appended.assistant_turn.turn_id
    )
    replay = _create(
        tmp_path,
        linked,
        operation_id=result.snapshot_id,
    )

    assert page["turns"][1]["xai"]["status"] == "pinned"
    assert pinned.is_dir()
    assert replay.replayed is True


def test_pinned_revision_survives_run_replacement(
    tmp_path: Path,
) -> None:
    _created, appended, linked, run_id = _conversation(
        tmp_path, linked=True
    )
    assert run_id is not None
    result = _create(tmp_path, linked)
    first = snapshots.resolve_pinned_run_dir(
        tmp_path, result.snapshot_id, appended.assistant_turn.turn_id
    )

    bundle = run_store.RunBundle(
        metadata={"backend": "llada", "prompt": "changed"},
        final_text="replacement",
        frames=["replacement"],
    )
    run_store.save(
        tmp_path,
        bundle,
        model_id="llada",
        run_id=run_id,
        expected_revision=1,
    )

    assert (first / run_store.FINAL_TEXT_NAME).read_text(
        encoding="utf-8"
    ) == "saved answer"


def test_missing_link_is_preserved_as_unavailable(
    tmp_path: Path,
) -> None:
    created, appended, completed, _run_id = _conversation(tmp_path)
    missing = conversation_store.set_run_link(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=conversation_store.RunLink("missing-run", 1),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )

    preview = snapshots.preview_snapshot(tmp_path, **_head(missing))
    result = _create(tmp_path, missing)
    page = snapshots.page_turns(tmp_path, result.snapshot_id)

    assert preview.unavailable_count == 1
    assert page["turns"][1]["xai"]["status"] == (
        snapshots.STATUS_UNAVAILABLE
    )

    with pytest.raises(
        snapshots.SnapshotOperationConflictError,
        match="active tail XAI changed",
    ):
        snapshots.create_snapshot(
            tmp_path,
            operation_id=uuid4().hex,
            title="Requires XAI",
            require_tail_xai=True,
            **_head(missing),
        )


def test_operation_replay_is_idempotent(tmp_path: Path) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    operation_id = uuid4().hex

    first = _create(
        tmp_path, completed, operation_id=operation_id
    )
    replay = _create(
        tmp_path, completed, operation_id=operation_id
    )

    assert first.snapshot_id == replay.snapshot_id
    assert first.replayed is False
    assert replay.replayed is True
    assert len(snapshots.list_snapshots(tmp_path)) == 1


def test_operation_reuse_with_other_semantics_is_refused(
    tmp_path: Path,
) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    operation_id = uuid4().hex
    _create(tmp_path, completed, operation_id=operation_id)

    with pytest.raises(snapshots.SnapshotOperationConflictError):
        _create(
            tmp_path,
            completed,
            operation_id=operation_id,
            title="Different",
        )


def test_title_rename_uses_revision_cas(tmp_path: Path) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    result = _create(tmp_path, completed)

    renamed = snapshots.rename_snapshot(
        tmp_path,
        result.snapshot_id,
        title="Renamed",
        expected_revision=1,
    )

    assert renamed["title"] == "Renamed"
    assert renamed["title_revision"] == 2
    with pytest.raises(snapshots.SnapshotRevisionConflictError):
        snapshots.rename_snapshot(
            tmp_path,
            result.snapshot_id,
            title="Stale",
            expected_revision=1,
        )


def test_delete_removes_only_the_snapshot(tmp_path: Path) -> None:
    created, _appended, linked, run_id = _conversation(
        tmp_path, linked=True
    )
    assert run_id is not None
    result = _create(tmp_path, linked)

    snapshots.delete_snapshot(tmp_path, result.snapshot_id)

    assert snapshots.list_snapshots(tmp_path) == []
    assert conversation_store.get_manifest(tmp_path, created.id)
    assert run_store.resolve_run_dir(tmp_path, run_id).is_dir()


def test_deleted_operation_does_not_resurrect_snapshot(
    tmp_path: Path,
) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    operation_id = uuid4().hex
    result = _create(
        tmp_path, completed, operation_id=operation_id
    )
    snapshots.delete_snapshot(tmp_path, result.snapshot_id)

    with pytest.raises(
        snapshots.SnapshotOperationConflictError,
        match="deleted",
    ):
        _create(tmp_path, completed, operation_id=operation_id)


def test_receipt_failure_cannot_resurrect_after_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    operation_id = uuid4().hex
    real_receipt = snapshots._write_receipt  # noqa: SLF001

    def fail_receipt(*_args: object, **_kwargs: object) -> None:
        raise OSError("receipt unavailable")

    monkeypatch.setattr(snapshots, "_write_receipt", fail_receipt)
    result = _create(
        tmp_path, completed, operation_id=operation_id
    )
    monkeypatch.setattr(snapshots, "_write_receipt", real_receipt)
    snapshots.delete_snapshot(tmp_path, result.snapshot_id)

    with pytest.raises(snapshots.SnapshotOperationConflictError):
        _create(tmp_path, completed, operation_id=operation_id)


def test_copy_fallback_preserves_turns_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _created, _appended, linked, _run_id = _conversation(
        tmp_path, linked=True
    )

    def refuse_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "forced copy fallback")

    monkeypatch.setattr(snapshots.os, "link", refuse_link)
    monkeypatch.setattr(conversation_store.os, "link", refuse_link)
    result = _create(tmp_path, linked)
    metadata = snapshots.read_metadata(tmp_path, result.snapshot_id)
    page = snapshots.page_turns(tmp_path, result.snapshot_id)

    assert metadata["xai_count"] == 1
    assert page["turns"][0]["text"] == "Question"
    assert page["turns"][1]["xai"]["storage"] == "copy"


def test_stale_or_pending_heads_are_refused(tmp_path: Path) -> None:
    created = conversation_store.create(tmp_path)
    appended = conversation_store.append_user(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
    )
    pending_head = {
        "conversation_id": created.id,
        "branch_id": appended.manifest.branch_id,
        "branch_revision": appended.manifest.revision,
        "turn_count": appended.manifest.turn_count,
        "tail_turn_id": appended.assistant_turn.turn_id,
        "tail_version": appended.assistant_turn.version,
    }
    with pytest.raises(
        conversation_store.ConversationStateError, match="pending"
    ):
        snapshots.preview_snapshot(tmp_path, **pending_head)

    completed = conversation_store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=appended.manifest.branch_id,
        expected_revision=appended.manifest.revision,
        text="Answer",
        partial=False,
    )
    stale = _head(completed)
    stale["branch_revision"] = completed.manifest.revision - 1
    with pytest.raises(
        conversation_store.ConversationRevisionConflictError
    ):
        snapshots.preview_snapshot(tmp_path, **stale)


def test_snapshot_pages_stay_chronological(tmp_path: Path) -> None:
    created = conversation_store.create(tmp_path)
    _first, completed = _append_pair(tmp_path, created)
    _second, completed = _append_pair(
        tmp_path,
        completed.manifest,
        question="Second",
        answer="Second answer",
    )
    result = _create(tmp_path, completed)

    newest = snapshots.page_turns(
        tmp_path, result.snapshot_id, limit=2
    )
    older = snapshots.page_turns(
        tmp_path,
        result.snapshot_id,
        before=newest["next_before"],
        limit=2,
    )

    assert [turn["text"] for turn in newest["turns"]] == [
        "Second",
        "Second answer",
    ]
    assert [turn["text"] for turn in older["turns"]] == [
        "Question",
        "Answer",
    ]


def test_metadata_is_the_visibility_marker(tmp_path: Path) -> None:
    root = tmp_path / snapshots.ROOT_NAME
    hidden = root / ("a" * 32)
    hidden.mkdir(parents=True)
    (hidden / snapshots.PATH_NAME).write_text(
        json.dumps({"partial": True}), encoding="utf-8"
    )

    assert snapshots.list_snapshots(tmp_path) == []


def test_racing_replays_publish_one_snapshot(tmp_path: Path) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    operation_id = uuid4().hex
    start = threading.Barrier(2)

    def worker(_index: int) -> snapshots.SnapshotResult:
        start.wait()
        return _create(
            tmp_path,
            completed,
            operation_id=operation_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, range(2)))

    assert {result.snapshot_id for result in results} == {
        operation_id
    }
    replayed = sorted(result.replayed for result in results)
    assert replayed == [False, True]
    assert len(snapshots.list_snapshots(tmp_path)) == 1


def test_unrelated_catalog_fork_does_not_stale_selected_head(
    tmp_path: Path,
) -> None:
    created, appended, completed, _run_id = _conversation(tmp_path)
    conversation_store.fork_delete_from_path(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=uuid4().hex,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=(
            completed.manifest.catalog_revision
        ),
    )

    result = _create(tmp_path, completed)

    source = snapshots.read_metadata(
        tmp_path, result.snapshot_id
    )["source"]
    assert source["catalog_revision"] == 2
    assert source["branch_revision"] == completed.manifest.revision


def test_failure_before_metadata_keeps_snapshot_invisible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _created, _appended, completed, _run_id = _conversation(tmp_path)
    real_replace = snapshots.conversation_core.replace_durable

    def fail_metadata(source: Path, target: Path) -> None:
        if target.name == snapshots.METADATA_NAME:
            raise OSError("metadata publish failed")
        real_replace(source, target)

    monkeypatch.setattr(
        snapshots.conversation_core,
        "replace_durable",
        fail_metadata,
    )

    with pytest.raises(OSError, match="metadata publish failed"):
        _create(tmp_path, completed)
    assert snapshots.list_snapshots(tmp_path) == []
