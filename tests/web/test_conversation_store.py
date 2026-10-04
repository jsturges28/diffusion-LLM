"""The durable conversation store, independent of HTTP and models.

Strategy: drive the standard-library store directly against temporary
data roots. Inspect both its typed results and the files it publishes,
then corrupt or orphan individual files to prove the manifest remains
the only commit point.

Passing proves conversations append in bounded work, user turns stay
immutable, only the tail assistant can version, stale clients cannot
write, old assistants freeze, pages are deterministic, and malformed
paths or committed data fail without touching legacy run folders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from src.web import conversation_store as store


REPO_ROOT = Path(__file__).resolve().parents[2]

IMPORT_PROBE = """
import sys

import src.web.conversation_store

loaded = set(sys.modules)
forbidden = {"fastapi", "pydantic", "torch", "transformers"}
print(",".join(sorted(loaded & forbidden)))
"""


def _append(
    root: Path,
    manifest: store.ConversationManifest,
    *,
    text: str = "Question",
    model_id: str = "llada",
    input_mode: store.InputMode = "chat",
) -> store.AppendResult:
    return store.append_user(
        root,
        manifest.id,
        expected_revision=manifest.revision,
        text=text,
        model_id=model_id,
        input_mode=input_mode,
    )


def _complete(
    root: Path,
    appended: store.AppendResult,
    *,
    text: str = "Answer",
    partial: bool = False,
) -> store.ConversationMutation:
    return store.update_assistant(
        root,
        appended.manifest.id,
        appended.assistant_turn.turn_id,
        expected_revision=appended.manifest.revision,
        text=text,
        partial=partial,
        context_pack={
            "included_turn_ids": [appended.user_turn.turn_id]
        },
        metadata={"source": "worker"},
    )


def _ready_pair(
    root: Path,
) -> Tuple[
    store.ConversationManifest,
    store.AppendResult,
    store.ConversationMutation,
]:
    created = store.create(root, title="Example")
    appended = _append(root, created)
    completed = _complete(root, appended)
    return created, appended, completed


def _conversation_dir(
    root: Path,
    conversation_id: str,
) -> Path:
    return root / store.CONVERSATIONS_DIR_NAME / conversation_id


def _manifest_path(root: Path, conversation_id: str) -> Path:
    conversation_dir = _conversation_dir(root, conversation_id)
    return conversation_dir / store.MANIFEST_NAME


def _turn_dir(
    root: Path,
    conversation_id: str,
    turn_id: str,
) -> Path:
    return (
        _conversation_dir(root, conversation_id)
        / store.TURNS_DIR_NAME
        / turn_id
    )


# -- import and type boundaries --


def test_store_imports_without_frameworks_or_models() -> None:
    """A fresh interpreter exposes every transitive import."""
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_public_records_are_frozen(tmp_path: Path) -> None:
    manifest = store.create(tmp_path)

    with pytest.raises(FrozenInstanceError):
        manifest.title = "changed"  # type: ignore[misc]


# -- creation and small manifests --


def test_create_publishes_only_a_small_manifest(
    tmp_path: Path,
) -> None:
    manifest = store.create(tmp_path, title="  Research  ")
    conversation_dir = _conversation_dir(tmp_path, manifest.id)
    manifest_path = _manifest_path(tmp_path, manifest.id)
    raw = json.loads(manifest_path.read_text())

    assert manifest.title == "Research"
    assert manifest.revision == 1
    assert manifest.turn_count == 0
    assert set(raw) == set(store.ManifestPayload.__required_keys__)
    assert {path.name for path in conversation_dir.iterdir()} == {
        store.TURNS_DIR_NAME,
        store.MANIFEST_NAME,
    }
    assert raw["tail_turn_id"] is None
    assert "turns" not in raw


def test_create_ids_are_direct_lowercase_uuid_names(
    tmp_path: Path,
) -> None:
    first = store.create(tmp_path)
    second = store.create(tmp_path)

    assert first.id != second.id
    assert len(first.id) == 32
    assert first.id == first.id.lower()
    assert Path(first.id).name == first.id


def test_titles_accept_the_limit_and_refuse_one_more(
    tmp_path: Path,
) -> None:
    accepted = store.create(
        tmp_path, title="t" * store.TITLE_CHARS_MAX
    )

    assert len(accepted.title) == store.TITLE_CHARS_MAX
    with pytest.raises(ValueError, match="title exceeds"):
        store.create(
            tmp_path,
            title="t" * (store.TITLE_CHARS_MAX + 1),
        )


@pytest.mark.parametrize("title", ["", " ", "\n\t"])
def test_blank_titles_are_refused(
    tmp_path: Path,
    title: str,
) -> None:
    with pytest.raises(ValueError, match="blank"):
        store.create(tmp_path, title=title)


def test_list_is_lightweight_and_newest_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps: Iterator[str] = iter(
        [
            "2026-10-04T10:00:00.000Z",
            "2026-10-04T11:00:00.000Z",
        ]
    )
    monkeypatch.setattr(store, "_timestamp", lambda: next(timestamps))
    older = store.create(tmp_path, title="Older")
    newer = store.create(tmp_path, title="Newer")

    listed = store.list_conversations(tmp_path)

    assert [item.id for item in listed] == [newer.id, older.id]
    assert all(item.turn_count == 0 for item in listed)


def test_list_ignores_orphan_and_corrupt_directories(
    tmp_path: Path,
) -> None:
    valid = store.create(tmp_path)
    root = tmp_path / store.CONVERSATIONS_DIR_NAME
    orphan = root / ("a" * 32)
    corrupt = root / ("b" * 32)
    orphan.mkdir()
    corrupt.mkdir()
    (corrupt / store.MANIFEST_NAME).write_text(
        "{not json", encoding="utf-8"
    )

    listed = store.list_conversations(tmp_path)
    assert [item.id for item in listed] == [valid.id]


# -- append and assistant state --


def test_append_writes_numeric_turns_before_manifest(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    turns_root = (
        _conversation_dir(tmp_path, created.id) / store.TURNS_DIR_NAME
    )

    assert sorted(path.name for path in turns_root.iterdir()) == [
        "00000001",
        "00000002",
    ]
    assert appended.user_turn.role == "user"
    assert appended.user_turn.version == 1
    assert appended.assistant_turn.role == "assistant"
    assert appended.assistant_turn.version == 1
    assert appended.assistant_turn.partial is True
    assert appended.assistant_turn.text == ""
    assert appended.manifest.pending_assistant_id == "00000002"


def test_append_reserves_model_and_input_mode(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(
        tmp_path,
        created,
        model_id="mamba3",
        input_mode="completion",
    )

    assert appended.assistant_turn.model_id == "mamba3"
    assert appended.assistant_turn.input_mode == "completion"


def test_user_text_accepts_the_limit_and_refuses_one_more(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    accepted = _append(
        tmp_path,
        created,
        text="u" * store.TEXT_CHARS_MAX,
    )

    assert len(accepted.user_turn.text) == store.TEXT_CHARS_MAX
    with pytest.raises(ValueError, match="user text exceeds"):
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=accepted.manifest.revision,
            text="u" * (store.TEXT_CHARS_MAX + 1),
            model_id="llada",
            input_mode="chat",
        )


@pytest.mark.parametrize("text", ["", " ", "\n"])
def test_blank_user_text_is_refused(
    tmp_path: Path,
    text: str,
) -> None:
    created = store.create(tmp_path)

    with pytest.raises(ValueError, match="must not be blank"):
        _append(tmp_path, created, text=text)


def test_metadata_accepts_its_json_boundary(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    overhead = len('{"x":""}')
    remaining = store.METADATA_JSON_CHARS_MAX - overhead
    accepted = store.append_user(
        tmp_path,
        created.id,
        expected_revision=created.revision,
        text="Question",
        model_id="llada",
        input_mode="chat",
        metadata={"x": "m" * remaining},
    )

    assert len(accepted.user_turn.metadata["x"]) == remaining


def test_metadata_refuses_one_json_character_over(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    overhead = len('{"x":""}')
    remaining = store.METADATA_JSON_CHARS_MAX - overhead + 1

    with pytest.raises(ValueError, match="metadata exceeds"):
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=created.revision,
            text="Question",
            model_id="llada",
            input_mode="chat",
            metadata={"x": "m" * remaining},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"value": float("nan")},
        {"value": object()},
        {1: "not a string key"},
    ],
)
def test_metadata_refuses_non_json_values(
    tmp_path: Path,
    metadata: dict[object, object],
) -> None:
    created = store.create(tmp_path)

    with pytest.raises(ValueError):
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=created.revision,
            text="Question",
            model_id="llada",
            input_mode="chat",
            metadata=metadata,  # type: ignore[arg-type]
        )


def test_a_second_user_waits_for_the_reserved_assistant(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)

    with pytest.raises(store.ConversationStateError, match="pending"):
        _append(tmp_path, appended.manifest, text="Too soon")


def test_complete_persists_full_assistant_fields(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    completed = _complete(tmp_path, appended)

    assert completed.turn.text == "Answer"
    assert completed.turn.partial is False
    assert completed.turn.version == 2
    assert completed.turn.model_id == "llada"
    assert completed.turn.context_pack == {
        "included_turn_ids": ["00000001"]
    }
    assert completed.turn.metadata == {"source": "worker"}
    assert completed.manifest.pending_assistant_id is None


def test_partial_assistant_is_terminal_and_appendable(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    partial = _complete(
        tmp_path,
        appended,
        text="Interrupted",
        partial=True,
    )
    next_pair = _append(tmp_path, partial.manifest, text="Continue")

    assert partial.turn.partial is True
    assert next_pair.manifest.turn_count == 4


def test_empty_assistant_text_is_valid(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    completed = _complete(tmp_path, appended, text="")

    assert completed.turn.text == ""
    assert completed.turn.partial is False


# -- compare-and-swap and tail-only mutation --


def test_stale_append_is_rejected_without_files(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)

    with pytest.raises(
        store.ConversationRevisionConflictError
    ) as captured:
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=created.revision,
            text="Stale",
            model_id="llada",
            input_mode="chat",
        )

    assert captured.value.actual == appended.manifest.revision
    assert (
        store.get_manifest(tmp_path, created.id) == appended.manifest
    )


def test_stale_assistant_update_is_rejected(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    completed = _complete(tmp_path, appended)

    with pytest.raises(store.ConversationRevisionConflictError):
        store.update_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            expected_revision=appended.manifest.revision,
            text="Stale revision",
            partial=False,
        )

    assert (
        store.get_manifest(tmp_path, created.id) == completed.manifest
    )


def test_bool_is_not_an_expected_revision(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)

    with pytest.raises(ValueError, match="integer"):
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=True,
            text="Question",
            model_id="llada",
            input_mode="chat",
        )


def test_only_the_tail_assistant_can_update(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)

    with pytest.raises(store.ConversationStateError, match="tail"):
        store.update_assistant(
            tmp_path,
            completed.manifest.id,
            appended.user_turn.turn_id,
            expected_revision=completed.manifest.revision,
            text="Wrong role",
            partial=False,
        )


# -- immutable versions and freezing --


def test_assistant_revisions_keep_every_prior_version(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    assistant_dir = _turn_dir(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
    )
    version_one = (assistant_dir / "00000001.json").read_bytes()
    revised = store.update_assistant(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        expected_revision=completed.manifest.revision,
        text="Revised answer",
        partial=False,
        metadata={"revision": 2},
    )

    names = sorted(path.name for path in assistant_dir.glob("*.json"))
    assert names == [
        "00000001.json",
        "00000002.json",
        "00000003.json",
    ]
    assert (
        assistant_dir / "00000001.json"
    ).read_bytes() == version_one
    assert revised.turn.version == 3
    assert revised.turn.text == "Revised answer"


def test_appending_freezes_the_previous_assistant(
    tmp_path: Path,
) -> None:
    _created, first, completed = _ready_pair(tmp_path)
    second = _append(
        tmp_path,
        completed.manifest,
        text="Second question",
    )
    frozen_path = (
        _turn_dir(
            tmp_path,
            completed.manifest.id,
            first.assistant_turn.turn_id,
        )
        / store.FROZEN_NAME
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    assert frozen == {
        "schema_version": 1,
        "turn_id": "00000002",
        "version": 2,
    }
    with pytest.raises(store.ConversationStateError, match="tail"):
        store.update_assistant(
            tmp_path,
            completed.manifest.id,
            first.assistant_turn.turn_id,
            expected_revision=second.manifest.revision,
            text="Too late",
            partial=False,
        )


def test_page_reads_the_frozen_version_not_an_orphan(
    tmp_path: Path,
) -> None:
    _created, first, completed = _ready_pair(tmp_path)
    second = _append(
        tmp_path,
        completed.manifest,
        text="Second question",
    )
    assistant_dir = _turn_dir(
        tmp_path,
        completed.manifest.id,
        first.assistant_turn.turn_id,
    )
    (assistant_dir / "00000063.json").write_text(
        "{not json", encoding="utf-8"
    )
    page = store.get_turns(tmp_path, second.manifest.id)

    assert [turn.text for turn in page.turns[:3]] == [
        "Question",
        "Answer",
        "Second question",
    ]


def test_tail_version_limit_refuses_another_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(store, "TAIL_VERSIONS_MAX", 2)

    with pytest.raises(store.ConversationLimitError, match="limit"):
        store.update_assistant(
            tmp_path,
            completed.manifest.id,
            appended.assistant_turn.turn_id,
            expected_revision=completed.manifest.revision,
            text="One too many",
            partial=False,
        )


# -- optional run links --


def test_run_link_and_unlink_are_immutable_tail_versions(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    linked = store.set_run_link(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        expected_revision=completed.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 3),
    )
    unlinked = store.set_run_link(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        expected_revision=linked.manifest.revision,
        run_link=None,
    )

    assert linked.turn.run_link == store.RunLink(
        "2026-01-01_llada", 3
    )
    assert linked.turn.version == 3
    assert unlinked.turn.run_link is None
    assert unlinked.turn.version == 4


def test_duplicate_run_link_is_an_idempotent_noop(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    link = store.RunLink("2026-01-01_llada", 1)
    linked = store.set_run_link(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        expected_revision=completed.manifest.revision,
        run_link=link,
    )
    duplicate = store.set_run_link(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        expected_revision=linked.manifest.revision,
        run_link=link,
    )

    assert duplicate == linked


def test_revising_assistant_text_clears_its_old_run_link(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    linked = store.set_run_link(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        expected_revision=completed.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 1),
    )

    revised = store.update_assistant(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        expected_revision=linked.manifest.revision,
        text="Edited answer",
        partial=False,
    )

    assert revised.turn.text == "Edited answer"
    assert revised.turn.run_link is None


def test_pending_assistant_cannot_link_a_run(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)

    with pytest.raises(store.ConversationStateError, match="pending"):
        store.set_run_link(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            expected_revision=appended.manifest.revision,
            run_link=store.RunLink("2026-01-01_llada", 1),
        )


@pytest.mark.parametrize(
    "link",
    [
        store.RunLink("../run", 1),
        store.RunLink("run/name", 1),
        store.RunLink("run", -1),
        store.RunLink("run", True),
    ],
)
def test_invalid_run_links_are_refused(
    tmp_path: Path,
    link: store.RunLink,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)

    with pytest.raises(ValueError):
        store.set_run_link(
            tmp_path,
            completed.manifest.id,
            appended.assistant_turn.turn_id,
            expected_revision=completed.manifest.revision,
            run_link=link,
        )


# -- pagination --


def test_turn_pages_are_bounded_and_chronological(
    tmp_path: Path,
) -> None:
    manifest = store.create(tmp_path)
    for index in range(30):
        appended = _append(
            tmp_path,
            manifest,
            text=f"Question {index}",
        )
        completed = _complete(
            tmp_path,
            appended,
            text=f"Answer {index}",
        )
        manifest = completed.manifest

    newest = store.get_turns(tmp_path, manifest.id)
    older = store.get_turns(
        tmp_path,
        manifest.id,
        before=newest.next_before,
    )

    assert len(newest.turns) == store.PAGE_SIZE_DEFAULT
    assert newest.turns[0].turn_id == "00000011"
    assert newest.turns[-1].turn_id == "00000060"
    assert newest.next_before == "00000011"
    assert newest.has_more is True
    assert [turn.turn_id for turn in older.turns] == [
        f"{index:08d}" for index in range(1, 11)
    ]
    assert older.next_before is None
    assert older.has_more is False


def test_page_limit_boundaries(
    tmp_path: Path,
) -> None:
    _created, _appended, completed = _ready_pair(tmp_path)

    assert (
        len(
            store.get_turns(
                tmp_path,
                completed.manifest.id,
                limit=1,
            ).turns
        )
        == 1
    )
    assert (
        len(
            store.get_turns(
                tmp_path,
                completed.manifest.id,
                limit=store.PAGE_SIZE_MAX,
            ).turns
        )
        == 2
    )
    for invalid in (0, store.PAGE_SIZE_MAX + 1, True):
        with pytest.raises(ValueError, match="page limit"):
            store.get_turns(
                tmp_path,
                completed.manifest.id,
                limit=invalid,
            )


def test_invalid_before_cursors_are_refused(
    tmp_path: Path,
) -> None:
    _created, _appended, completed = _ready_pair(tmp_path)

    for before in ("0", "00000000", "00000004", "../00000001"):
        with pytest.raises(ValueError):
            store.get_turns(
                tmp_path,
                completed.manifest.id,
                before=before,
            )


# -- publication failures, corruption, and orphans --


def test_orphan_tail_version_is_invisible_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    real_write = store._write_manifest

    def fail_manifest(
        conversation_dir: Path,
        manifest: store.ConversationManifest,
    ) -> None:
        if manifest.revision == 3:
            raise OSError("injected manifest failure")
        real_write(conversation_dir, manifest)

    monkeypatch.setattr(store, "_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        _complete(tmp_path, appended)

    old = store.get_manifest(tmp_path, created.id)
    old_page = store.get_turns(tmp_path, created.id)
    assert old.revision == 2
    assert old_page.turns[-1].version == 1
    assert old_page.turns[-1].text == ""

    monkeypatch.setattr(store, "_write_manifest", real_write)
    retried = _complete(tmp_path, appended, text="Recovered")
    assert retried.turn.version == 2
    assert retried.turn.text == "Recovered"


def test_orphan_future_turns_are_ignored_and_replaced(
    tmp_path: Path,
) -> None:
    _created, _appended, completed = _ready_pair(tmp_path)
    turns_root = (
        _conversation_dir(tmp_path, completed.manifest.id)
        / store.TURNS_DIR_NAME
    )
    orphan = turns_root / "00000003"
    orphan.mkdir()
    (orphan / "garbage").write_text("orphan", encoding="utf-8")

    before = store.get_turns(tmp_path, completed.manifest.id)
    appended = _append(
        tmp_path,
        completed.manifest,
        text="After orphan",
    )

    assert len(before.turns) == 2
    assert appended.user_turn.turn_id == "00000003"
    assert not (orphan / "garbage").exists()


def test_corrupt_manifest_fails_direct_reads_but_not_list(
    tmp_path: Path,
) -> None:
    manifest = store.create(tmp_path)
    _manifest_path(tmp_path, manifest.id).write_text(
        "{not json", encoding="utf-8"
    )

    with pytest.raises(store.ConversationCorruptError):
        store.get_manifest(tmp_path, manifest.id)
    assert store.list_conversations(tmp_path) == []


def test_extra_manifest_field_is_corruption(
    tmp_path: Path,
) -> None:
    manifest = store.create(tmp_path)
    path = _manifest_path(tmp_path, manifest.id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["turns"] = []
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError, match="fields differ"
    ):
        store.get_manifest(tmp_path, manifest.id)


def test_corrupt_committed_turn_fails_the_page(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    assistant_dir = _turn_dir(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
    )
    (assistant_dir / "00000002.json").write_text(
        "{not json", encoding="utf-8"
    )

    with pytest.raises(store.ConversationCorruptError):
        store.get_turns(tmp_path, completed.manifest.id)


def test_pending_turn_must_remain_an_empty_placeholder(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    assistant_dir = _turn_dir(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
    )
    path = assistant_dir / "00000001.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["text"] = "uncommitted output"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError, match="placeholder"
    ):
        store.get_turns(tmp_path, created.id)


def test_user_turn_cannot_carry_a_run_link(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    user_dir = _turn_dir(
        tmp_path,
        completed.manifest.id,
        appended.user_turn.turn_id,
    )
    path = user_dir / "00000001.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["run_link"] = {"run_id": "2026-01-01_llada", "revision": 1}
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError, match="user turn"
    ):
        store.get_turns(tmp_path, completed.manifest.id)


# -- path safety and deletion --


@pytest.mark.parametrize(
    "conversation_id",
    [
        "../escape",
        "../../etc",
        "nested/conversation",
        "/absolute",
        "." * 32,
        "g" * 32,
        "A" * 32,
        "",
    ],
)
def test_invalid_conversation_ids_are_refused(
    tmp_path: Path,
    conversation_id: str,
) -> None:
    with pytest.raises(store.InvalidConversationIdError):
        store.resolve_conversation_dir(tmp_path, conversation_id)


def test_symlink_conversation_is_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / store.CONVERSATIONS_DIR_NAME
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    conversation_id = "a" * 32
    (root / conversation_id).symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(store.InvalidConversationIdError):
        store.resolve_conversation_dir(tmp_path, conversation_id)


def test_delete_removes_the_visible_conversation(
    tmp_path: Path,
) -> None:
    manifest = store.create(tmp_path)

    store.delete(tmp_path, manifest.id)

    with pytest.raises(store.ConversationNotFoundError):
        store.get_manifest(tmp_path, manifest.id)
    assert store.list_conversations(tmp_path) == []


def test_delete_missing_conversation_is_not_found(
    tmp_path: Path,
) -> None:
    with pytest.raises(store.ConversationNotFoundError):
        store.delete(tmp_path, "a" * 32)


def test_legacy_run_folder_is_untouched(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "2026-01-01_00-00-00_llada"
    run_dir.mkdir()
    metadata = run_dir / "metadata.json"
    metadata.write_text('{"revision":1}', encoding="utf-8")

    conversation = store.create(tmp_path)
    store.delete(tmp_path, conversation.id)

    assert metadata.read_text(encoding="utf-8") == '{"revision":1}'
