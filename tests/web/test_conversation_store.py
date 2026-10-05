"""The durable conversation store, independent of HTTP and models.

Strategy: drive the standard-library store directly against temporary
data roots. Inspect both its typed results and the files it publishes,
then corrupt or orphan individual files to prove manifests remain the
only commit points.

Passing proves schema-v2 branches share fixed prefixes, lazy v1
upgrade preserves old turns, branch CAS stays independent, forks and
pages are bounded, old links remain readable, and malformed paths or
committed data fail without touching legacy run folders.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Iterator, Optional, Tuple
from uuid import uuid4

import pytest

from src.backends.protocol import ParamGroup, ParamSpec, ParamType
from src.backends.registry import REGISTRY
from src.web import _conversation_branch_store as branch_store
from src.web import _conversation_store_core as core_store
from src.web import conversation_generation
from src.web import conversation_store as store


REPO_ROOT = Path(__file__).resolve().parents[2]

IMPORT_PROBE = """
import sys

import src.web.conversation_store

loaded = set(sys.modules)
forbidden = {"fastapi", "pydantic", "torch", "transformers"}
print(",".join(sorted(loaded & forbidden)))
"""


def _operation_id() -> str:
    return uuid4().hex


def _semantic_digest(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    configuration = canonical.get("generation_configuration")
    if configuration is not None:
        canonical["generation_configuration"] = (
            _canonical_generation_numbers(configuration)
        )
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_generation_numbers(value: object) -> object:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [
            _canonical_generation_numbers(item) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _canonical_generation_numbers(item)
            for key, item in value.items()
        }
    raise TypeError("fixture contains a non-JSON value")


def _generation_configuration(
    *,
    temperature: float = 0.0,
    experimental: bool = False,
) -> dict[str, object]:
    """Build one exact LLaDA panel snapshot for fork tests."""
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
        "experimental": experimental,
        "parameters": {
            "steps": 128,
            "gen_length": 160,
            "block_length": 160,
            "temperature": temperature,
            "cfg_scale": 0.0,
            "seed": -1,
            "remasking": "low_confidence",
            "alternatives": True,
        },
    }


def _smollm_configuration(
    *,
    device: str,
    max_new_tokens: int,
) -> dict[str, object]:
    """Build a non-Experimental SmolLM3 snapshot."""
    return {
        "codec_version": (
            store.GENERATION_CONFIGURATION_CODEC_VERSION
        ),
        "model_id": "smollm3",
        "input_mode": "chat",
        "device": device,
        "schema_id": (
            conversation_generation.registry_generation_schema_id(
                "smollm3", device
            )
        ),
        "experimental": False,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": -1,
            "seed": -1,
            "thinking": False,
            "alternatives": True,
        },
    }


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
        branch_id=manifest.branch_id,
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
        branch_id=appended.manifest.branch_id,
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


def _append_complete(
    root: Path,
    manifest: store.ConversationManifest,
    *,
    question: str,
    answer: str,
) -> Tuple[store.AppendResult, store.ConversationMutation]:
    appended = _append(root, manifest, text=question)
    completed = _complete(root, appended, text=answer)
    return appended, completed


def _legacy_ready_pair(
    root: Path,
) -> Tuple[
    store.ConversationManifest,
    store.AppendResult,
    store.ConversationMutation,
]:
    conversation_id = "a" * 32
    conversation_dir = _conversation_dir(root, conversation_id)
    (conversation_dir / store.TURNS_DIR_NAME).mkdir(parents=True)
    timestamp = "2026-01-01T00:00:00.000Z"
    created = store.ConversationManifest(
        id=conversation_id,
        title="Legacy fixture",
        revision=1,
        created_at=timestamp,
        updated_at=timestamp,
        turn_count=0,
        tail_role=None,
        tail_turn_id=None,
        tail_version=None,
        pending_assistant_id=None,
    )
    core_store.write_legacy_manifest(conversation_dir, created)
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
    if turn_id.startswith("t_"):
        branch_hex = turn_id.split("_", maxsplit=2)[1]
        branch_id = f"b_{branch_hex}"
        return (
            _conversation_dir(root, conversation_id)
            / store.BRANCHES_DIR_NAME
            / branch_id
            / store.TURNS_DIR_NAME
            / turn_id
        )
    return (
        _conversation_dir(root, conversation_id)
        / store.TURNS_DIR_NAME
        / turn_id
    )


def _rewrite_fork_as_preconfiguration_fixture(
    root: Path,
    *,
    conversation_id: str,
    assistant_turn_id: str,
    operation_id: str,
    semantic_payload: dict[str, object],
) -> None:
    """Replace current fork artifacts with exact pre-config shapes."""
    turn_path = (
        _turn_dir(root, conversation_id, assistant_turn_id)
        / "00000001.json"
    )
    turn = json.loads(turn_path.read_text(encoding="utf-8"))
    turn["metadata"] = {}
    turn_path.write_text(json.dumps(turn), encoding="utf-8")
    receipt_path = (
        _conversation_dir(root, conversation_id)
        / store.OPERATIONS_DIR_NAME
        / f"{operation_id}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["request_digest"] = _semantic_digest(semantic_payload)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def _branch_turns_root(
    root: Path,
    manifest: store.ConversationManifest,
) -> Path:
    assert manifest.branch_id is not None
    return (
        _conversation_dir(root, manifest.id)
        / store.BRANCHES_DIR_NAME
        / manifest.branch_id
        / store.TURNS_DIR_NAME
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
    catalog = store.get_catalog(tmp_path, manifest.id)
    assert manifest.branch_id is not None
    branch = store.get_branch(
        tmp_path, manifest.id, manifest.branch_id
    )

    with pytest.raises(FrozenInstanceError):
        manifest.title = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        catalog.title = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        branch.revision = 9  # type: ignore[misc]


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
    assert manifest.schema_version == store.SCHEMA_VERSION
    assert manifest.branch_id == manifest.default_branch_id
    assert set(raw) == set(store.CatalogPayload.__required_keys__)
    assert {path.name for path in conversation_dir.iterdir()} == {
        store.BRANCHES_DIR_NAME,
        store.MANIFEST_NAME,
        store.OPERATIONS_DIR_NAME,
    }
    assert raw["schema_version"] == store.SCHEMA_VERSION
    assert raw["default_branch_id"] == manifest.branch_id
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
    monkeypatch.setattr(
        core_store,
        "timestamp",
        lambda: next(timestamps),
    )
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


def test_append_writes_opaque_local_turns_before_manifest(
    tmp_path: Path,
) -> None:
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    turns_root = _branch_turns_root(tmp_path, appended.manifest)

    assert sorted(path.name for path in turns_root.iterdir()) == [
        appended.user_turn.turn_id,
        appended.assistant_turn.turn_id,
    ]
    assert not appended.user_turn.turn_id.isdigit()
    assert not appended.assistant_turn.turn_id.isdigit()
    assert appended.user_turn.index == 1
    assert appended.assistant_turn.index == 2
    assert appended.user_turn.role == "user"
    assert appended.user_turn.version == 1
    assert appended.assistant_turn.role == "assistant"
    assert appended.assistant_turn.version == 1
    assert appended.assistant_turn.partial is True
    assert appended.assistant_turn.text == ""
    assert (
        appended.manifest.pending_assistant_id
        == appended.assistant_turn.turn_id
    )


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


def test_astral_user_text_at_limit_round_trips_and_refuses_more(
    tmp_path: Path,
) -> None:
    """A maximum astral turn remains readable under the byte bound."""
    created = store.create(tmp_path)
    text = "\U0001f642" * store.TEXT_CHARS_MAX
    accepted = _append(
        tmp_path,
        created,
        text=text,
    )
    restored = store.get_turns(
        tmp_path,
        created.id,
        branch_id=created.branch_id,
    )

    assert len(accepted.user_turn.text) == store.TEXT_CHARS_MAX
    assert restored.turns[0].text == text
    with pytest.raises(ValueError, match="user text exceeds"):
        store.append_user(
            tmp_path,
            created.id,
            branch_id=created.branch_id,
            expected_revision=accepted.manifest.revision,
            text="u" * (store.TEXT_CHARS_MAX + 1),
            model_id="llada",
            input_mode="chat",
        )


def test_oversized_json_refuses_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact byte refusal leaves no version or manifest commit."""
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    conversation_dir = _conversation_dir(tmp_path, created.id)
    committed_bytes = _manifest_path(
        tmp_path, created.id
    ).read_bytes()
    committed_sizes = [
        path.stat().st_size
        for path in conversation_dir.rglob("*.json")
    ]
    byte_limit = max(committed_sizes) + 128
    monkeypatch.setattr(
        core_store, "JSON_FILE_BYTES_MAX", byte_limit
    )

    with pytest.raises(ValueError, match="serialized JSON exceeds"):
        _complete(
            tmp_path,
            appended,
            text="x" * byte_limit,
        )

    assistant_dir = _turn_dir(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
    )
    assert _manifest_path(tmp_path, created.id).read_bytes() == (
        committed_bytes
    )
    version_names = sorted(
        path.name for path in assistant_dir.glob("*.json")
    )
    assert version_names == [
        "00000001.json"
    ]
    assert list(assistant_dir.glob("*.tmp")) == []


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
        branch_id=created.branch_id,
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
            branch_id=created.branch_id,
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
            branch_id=created.branch_id,
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
        "included_turn_ids": [appended.user_turn.turn_id]
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
            branch_id=created.branch_id,
            expected_revision=created.revision,
            text="Stale",
            model_id="llada",
            input_mode="chat",
        )

    assert captured.value.actual == appended.manifest.revision
    assert (
        store.get_manifest(tmp_path, created.id) == appended.manifest
    )


def test_stale_legacy_append_names_its_synthetic_branch(
    tmp_path: Path,
) -> None:
    created, _appended, completed = _legacy_ready_pair(tmp_path)
    branch_id = store.legacy_branch_id(created.id)

    with pytest.raises(
        store.ConversationRevisionConflictError
    ) as captured:
        store.append_user(
            tmp_path,
            created.id,
            branch_id=branch_id,
            expected_revision=created.revision,
            text="Stale",
            model_id="llada",
            input_mode="chat",
        )

    assert captured.value.branch_id == branch_id
    assert captured.value.actual == completed.manifest.revision


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
            branch_id=completed.manifest.branch_id,
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
            branch_id=created.branch_id,
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
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            text="Wrong role",
            partial=False,
        )


def test_v2_mutations_require_an_explicit_branch(
    tmp_path: Path,
) -> None:
    """Every schema-v2 write refuses implicit default selection."""
    created, appended, completed = _ready_pair(tmp_path)
    before = store.get_catalog(tmp_path, created.id)

    with pytest.raises(ValueError, match="branch_id is required"):
        store.append_user(
            tmp_path,
            created.id,
            expected_revision=completed.manifest.revision,
            text="Implicit append",
            model_id="llada",
            input_mode="chat",
        )
    with pytest.raises(ValueError, match="branch_id is required"):
        store.update_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            expected_revision=completed.manifest.revision,
            text="Implicit update",
            partial=False,
        )
    with pytest.raises(ValueError, match="branch_id is required"):
        store.set_run_link(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            expected_revision=completed.manifest.revision,
            run_link=store.RunLink("2026-01-01_llada", 1),
            expected_turn_index=completed.turn.index,
            expected_turn_version=completed.turn.version,
        )
    with pytest.raises(ValueError, match="branch_id is required"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )

    assert store.get_catalog(tmp_path, created.id) == before


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
        branch_id=completed.manifest.branch_id,
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
        "schema_version": store.SCHEMA_VERSION,
        "turn_id": first.assistant_turn.turn_id,
        "version": 2,
    }
    with pytest.raises(store.ConversationStateError, match="tail"):
        store.update_assistant(
            tmp_path,
            completed.manifest.id,
            first.assistant_turn.turn_id,
            branch_id=second.manifest.branch_id,
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
    monkeypatch.setattr(core_store, "TAIL_VERSIONS_MAX", 2)

    with pytest.raises(store.ConversationLimitError, match="limit"):
        store.update_assistant(
            tmp_path,
            completed.manifest.id,
            appended.assistant_turn.turn_id,
            branch_id=completed.manifest.branch_id,
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
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 3),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )
    unlinked = store.set_run_link(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        branch_id=linked.manifest.branch_id,
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
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=link,
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )
    duplicate = store.set_run_link(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        branch_id=linked.manifest.branch_id,
        expected_revision=linked.manifest.revision,
        run_link=link,
        expected_turn_index=linked.turn.index,
        expected_turn_version=linked.turn.version,
    )

    assert duplicate == linked


def test_run_link_requires_the_exact_tail_version(
    tmp_path: Path,
) -> None:
    """A saved stale version cannot attach to a revised assistant."""
    _created, appended, completed = _ready_pair(tmp_path)

    with pytest.raises(
        store.ConversationStateError,
        match="current assistant version",
    ):
        store.set_run_link(
            tmp_path,
            completed.manifest.id,
            appended.assistant_turn.turn_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            run_link=store.RunLink("2026-01-01_llada", 1),
            expected_turn_index=completed.turn.index,
            expected_turn_version=completed.turn.version - 1,
        )


def test_revising_assistant_text_clears_its_old_run_link(
    tmp_path: Path,
) -> None:
    _created, appended, completed = _ready_pair(tmp_path)
    linked = store.set_run_link(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 1),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )

    revised = store.update_assistant(
        tmp_path,
        linked.manifest.id,
        linked.turn.turn_id,
        branch_id=linked.manifest.branch_id,
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
            branch_id=appended.manifest.branch_id,
            expected_revision=appended.manifest.revision,
            run_link=store.RunLink("2026-01-01_llada", 1),
            expected_turn_index=appended.assistant_turn.index,
            expected_turn_version=appended.assistant_turn.version,
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
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            run_link=link,
            expected_turn_index=completed.turn.index,
            expected_turn_version=completed.turn.version,
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
    assert newest.turns[0].index == 11
    assert newest.turns[-1].index == 60
    assert newest.next_before == "00000011"
    assert newest.has_more is True
    assert [turn.index for turn in older.turns] == list(range(1, 11))
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
    real_write = branch_store.write_branch_manifest

    def fail_manifest(
        branch_dir: Path,
        branch: branch_store.StoredBranch,
    ) -> None:
        if branch.record.revision == 3:
            raise OSError("injected manifest failure")
        real_write(branch_dir, branch)

    monkeypatch.setattr(
        branch_store,
        "write_branch_manifest",
        fail_manifest,
    )
    with pytest.raises(OSError, match="injected"):
        _complete(tmp_path, appended)

    old = store.get_manifest(tmp_path, created.id)
    old_page = store.get_turns(tmp_path, created.id)
    assert old.revision == 2
    assert old_page.turns[-1].version == 1
    assert old_page.turns[-1].text == ""

    monkeypatch.setattr(
        branch_store,
        "write_branch_manifest",
        real_write,
    )
    retried = _complete(tmp_path, appended, text="Recovered")
    assert retried.turn.version == 2
    assert retried.turn.text == "Recovered"


def test_orphan_future_turns_are_ignored_and_replaced(
    tmp_path: Path,
) -> None:
    _created, _appended, completed = _ready_pair(tmp_path)
    turns_root = _branch_turns_root(tmp_path, completed.manifest)
    assert completed.manifest.branch_id is not None
    orphan_id = branch_store.opaque_turn_id(
        completed.manifest.branch_id,
        3,
    )
    orphan = turns_root / orphan_id
    orphan.mkdir()
    (orphan / "garbage").write_text("orphan", encoding="utf-8")

    before = store.get_turns(tmp_path, completed.manifest.id)
    appended = _append(
        tmp_path,
        completed.manifest,
        text="After orphan",
    )

    assert len(before.turns) == 2
    assert appended.user_turn.turn_id == orphan_id
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


# -- schema-v2 branches and the lazy v1 adapter --


def test_v1_fixture_reads_and_upgrades_only_on_fork(
    tmp_path: Path,
) -> None:
    """Lazy upgrade leaves legacy turns byte exact."""
    _created, appended, completed = _legacy_ready_pair(tmp_path)
    conversation_dir = _conversation_dir(
        tmp_path, completed.manifest.id
    )
    turns_root = conversation_dir / store.TURNS_DIR_NAME
    before = {
        path.relative_to(turns_root): path.read_bytes()
        for path in turns_root.rglob("*")
        if path.is_file()
    }

    page = store.get_turns(tmp_path, completed.manifest.id)

    assert [turn.turn_id for turn in page.turns] == [
        "00000001",
        "00000002",
    ]
    assert not (conversation_dir / store.BRANCHES_DIR_NAME).exists()

    forked = store.fork_retry_assistant(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=0,
        model_id="llada",
        input_mode="chat",
    )
    after = {
        path.relative_to(turns_root): path.read_bytes()
        for path in turns_root.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert forked.catalog.revision == 1
    assert (
        len(
            store.list_branches(
                tmp_path, completed.manifest.id
            ).branches
        )
        == 2
    )
    assert (
        store.get_manifest(
            tmp_path, completed.manifest.id
        ).schema_version
        == store.SCHEMA_VERSION
    )


def test_v1_user_and_completed_reserved_key_stays_ordinary(
    tmp_path: Path,
) -> None:
    """A later reservation codec cannot reinterpret old metadata."""
    _created, appended, completed = _legacy_ready_pair(tmp_path)
    values = {
        appended.user_turn.turn_id: {"historical": "user"},
        appended.assistant_turn.turn_id: {
            "historical": "completed"
        },
    }
    for turn_id, historical in values.items():
        version = 1 if turn_id == appended.user_turn.turn_id else 2
        path = (
            _turn_dir(tmp_path, completed.manifest.id, turn_id)
            / f"{version:08d}.json"
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["metadata"][store.PENDING_GENERATION_KEY] = historical
        path.write_text(json.dumps(raw), encoding="utf-8")

    page = store.get_turns(tmp_path, completed.manifest.id)

    assert page.turns[0].metadata[store.PENDING_GENERATION_KEY] == {
        "historical": "user"
    }
    assert page.turns[1].metadata[store.PENDING_GENERATION_KEY] == {
        "historical": "completed"
    }


def test_v1_pending_reserved_key_stays_ordinary(
    tmp_path: Path,
) -> None:
    """A schema-v1 reservation predates the reserved-key contract."""
    conversation_id = "d" * 32
    conversation_dir = _conversation_dir(tmp_path, conversation_id)
    (conversation_dir / store.TURNS_DIR_NAME).mkdir(parents=True)
    timestamp = "2026-01-01T00:00:00.000Z"
    created = store.ConversationManifest(
        id=conversation_id,
        title="Legacy pending fixture",
        revision=1,
        created_at=timestamp,
        updated_at=timestamp,
        turn_count=0,
        tail_role=None,
        tail_turn_id=None,
        tail_version=None,
        pending_assistant_id=None,
    )
    core_store.write_legacy_manifest(conversation_dir, created)
    appended = _append(tmp_path, created)
    path = (
        _turn_dir(
            tmp_path,
            conversation_id,
            appended.assistant_turn.turn_id,
        )
        / "00000001.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["metadata"][store.PENDING_GENERATION_KEY] = {
        "historical": "pending"
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    page = store.get_turns(tmp_path, conversation_id)

    assert page.turns[-1].metadata[store.PENDING_GENERATION_KEY] == {
        "historical": "pending"
    }


def test_lost_v1_fork_replays_without_a_new_v2_selector(
    tmp_path: Path,
) -> None:
    """The committed receipt resolves before v2 requires branch_id."""
    _created, appended, completed = _legacy_ready_pair(tmp_path)
    operation_id = "a" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=0,
        model_id="llada",
        input_mode="chat",
    )

    replayed = store.fork_retry_assistant(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        expected_revision=completed.manifest.revision + 99,
        expected_catalog_revision=first.catalog.revision + 99,
        model_id="llada",
        input_mode="chat",
    )

    assert replayed == first
    assert len(
        store.list_branches(
            tmp_path, completed.manifest.id
        ).catalog.branch_ids
    ) == 2


def test_invalid_v1_fork_does_not_upgrade(
    tmp_path: Path,
) -> None:
    """Validation happens before the v1 catalog is rewritten."""
    _created, _appended, completed = _legacy_ready_pair(tmp_path)
    conversation_dir = _conversation_dir(
        tmp_path, completed.manifest.id
    )

    with pytest.raises(store.ConversationStateError):
        store.fork_retry_assistant(
            tmp_path,
            completed.manifest.id,
            "00000001",
            operation_id=_operation_id(),
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )

    raw = json.loads(
        (conversation_dir / store.MANIFEST_NAME).read_text()
    )
    assert raw["schema_version"] == store.LEGACY_SCHEMA_VERSION
    assert not (conversation_dir / store.BRANCHES_DIR_NAME).exists()


def test_upgraded_v1_main_branch_can_extend_with_opaque_turns(
    tmp_path: Path,
) -> None:
    """The adapter mixes numeric and new opaque turns."""
    _created, appended, completed = _legacy_ready_pair(tmp_path)
    store.fork_retry_assistant(
        tmp_path,
        completed.manifest.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    legacy_branch_id = f"b_{completed.manifest.id}"

    extended = store.append_user(
        tmp_path,
        completed.manifest.id,
        branch_id=legacy_branch_id,
        expected_revision=completed.manifest.revision,
        text="New question",
        model_id="llada",
        input_mode="chat",
    )
    page = store.get_turns(
        tmp_path,
        completed.manifest.id,
        branch_id=legacy_branch_id,
    )

    assert [turn.turn_id for turn in page.turns[:2]] == [
        "00000001",
        "00000002",
    ]
    assert extended.user_turn.turn_id.startswith("t_")
    assert page.turns[-2].turn_id == extended.user_turn.turn_id


def test_edit_fork_resolves_a_shared_prefix_without_copying(
    tmp_path: Path,
) -> None:
    """The child stores only its replacement pair after the prefix."""
    created = store.create(tmp_path)
    first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    second, second_done = _append_complete(
        tmp_path,
        first_done.manifest,
        question="Question 2",
        answer="Answer 2",
    )

    forked = store.fork_edit_user(
        tmp_path,
        created.id,
        second.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
        text="Edited question",
        model_id="llada",
        input_mode="chat",
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=forked.branch.branch_id,
        limit=2,
    )
    older = store.get_turns(
        tmp_path,
        created.id,
        branch_id=forked.branch.branch_id,
        before=page.next_before,
        limit=2,
    )
    child_turns = _branch_turns_root(tmp_path, forked.manifest)
    turns = older.turns + page.turns

    assert [turn.turn_id for turn in turns[:2]] == [
        first.user_turn.turn_id,
        first.assistant_turn.turn_id,
    ]
    assert [turn.text for turn in turns] == [
        "Question 1",
        "Answer 1",
        "Edited question",
        "",
    ]
    assert len(list(child_turns.iterdir())) == 2
    assert forked.branch.prefix_turn_count == 2
    assert page.branch_points == (
        store.BranchPoint(
            turn_index=3,
            source_branch_id=second_done.manifest.branch_id,
            selected_branch_id=forked.branch.branch_id,
            branch_ids=(
                second_done.manifest.branch_id,
                forked.branch.branch_id,
            ),
            deleted_branch_ids=(),
        ),
    )
    assert older.branch_points == ()


def test_nested_retry_uses_bounded_parent_segments(
    tmp_path: Path,
) -> None:
    """A second-level fork composes both ancestors chronologically."""
    created = store.create(tmp_path)
    first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question",
        answer="Original",
    )
    edited = store.fork_edit_user(
        tmp_path,
        created.id,
        first.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=first_done.manifest.branch_id,
        expected_revision=first_done.manifest.revision,
        text="Edited",
        model_id="llada",
        input_mode="chat",
    )
    edited_done = store.update_assistant(
        tmp_path,
        created.id,
        edited.assistant_turn.turn_id,
        branch_id=edited.branch.branch_id,
        expected_revision=edited.manifest.revision,
        text="Edited answer",
        partial=False,
    )

    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        edited.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=edited.branch.branch_id,
        expected_revision=edited_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=retried.branch.branch_id,
    )

    assert retried.branch.depth == 2
    assert retried.branch.parent_branch_id == edited.branch.branch_id
    assert [turn.text for turn in page.turns] == ["Edited", ""]
    assert page.turns[0].turn_id == edited.user_turn.turn_id
    assert [
        (
            point.turn_index,
            point.source_branch_id,
            point.selected_branch_id,
        )
        for point in page.branch_points
    ] == [
        (
            1,
            first_done.manifest.branch_id,
            edited.branch.branch_id,
        ),
        (
            2,
            edited.branch.branch_id,
            retried.branch.branch_id,
        ),
    ]


def test_retry_of_retry_coalesces_one_logical_branch_point(
    tmp_path: Path,
) -> None:
    """Nested retries at one slot are one set of alternatives."""
    created, appended, completed = _ready_pair(tmp_path)
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    first_done = store.update_assistant(
        tmp_path,
        created.id,
        first.assistant_turn.turn_id,
        branch_id=first.branch.branch_id,
        expected_revision=first.manifest.revision,
        text="First retry",
        partial=False,
    )
    second = store.fork_retry_assistant(
        tmp_path,
        created.id,
        first.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=first.branch.branch_id,
        expected_revision=first_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=second.branch.branch_id,
    )

    assert len(page.branch_points) == 1
    point = page.branch_points[0]
    assert point.turn_index == 2
    assert point.selected_branch_id == second.branch.branch_id
    assert point.branch_ids == (
        completed.manifest.branch_id,
        first.branch.branch_id,
        second.branch.branch_id,
    )


def test_early_descendant_edit_hides_stale_ancestor_point(
    tmp_path: Path,
) -> None:
    """An earlier edit cuts off a later ancestor continuation."""
    created = store.create(tmp_path)
    first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    second, second_done = _append_complete(
        tmp_path,
        first_done.manifest,
        question="Question 2",
        answer="Answer 2",
    )
    retry = store.fork_retry_assistant(
        tmp_path,
        created.id,
        second.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    retry_done = store.update_assistant(
        tmp_path,
        created.id,
        retry.assistant_turn.turn_id,
        branch_id=retry.branch.branch_id,
        expected_revision=retry.manifest.revision,
        text="Alternate answer 2",
        partial=False,
    )
    edited = store.fork_edit_user(
        tmp_path,
        created.id,
        first.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=retry.branch.branch_id,
        expected_revision=retry_done.manifest.revision,
        text="Edited question 1",
        model_id="llada",
        input_mode="chat",
    )

    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=edited.branch.branch_id,
    )

    assert len(page.branch_points) == 1
    point = page.branch_points[0]
    assert point.turn_index == 1
    assert point.source_branch_id == retry.branch.branch_id
    assert point.selected_branch_id == edited.branch.branch_id
    assert second_done.manifest.branch_id not in point.branch_ids


def test_nested_delete_keeps_only_its_effective_marker(
    tmp_path: Path,
) -> None:
    """A same-slot delete stays marked after alternatives merge."""
    created, appended, completed = _ready_pair(tmp_path)
    edited = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Edited question",
        model_id="llada",
        input_mode="chat",
    )
    edited_done = store.update_assistant(
        tmp_path,
        created.id,
        edited.assistant_turn.turn_id,
        branch_id=edited.branch.branch_id,
        expected_revision=edited.manifest.revision,
        text="Edited answer",
        partial=False,
    )
    deleted = store.fork_delete_from_path(
        tmp_path,
        created.id,
        edited.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=edited.branch.branch_id,
        expected_revision=edited_done.manifest.revision,
    )

    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=deleted.branch.branch_id,
    )

    assert len(page.branch_points) == 1
    point = page.branch_points[0]
    assert point.turn_index == 1
    assert point.selected_branch_id == deleted.branch.branch_id
    assert point.branch_ids == (
        completed.manifest.branch_id,
        edited.branch.branch_id,
        deleted.branch.branch_id,
    )
    assert point.deleted_branch_ids == (
        deleted.branch.branch_id,
    )


def test_nested_forks_resolve_inherited_turns(
    tmp_path: Path,
) -> None:
    """Second-level forks can target both inherited turn roles."""
    created = store.create(tmp_path)
    first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    second, second_done = _append_complete(
        tmp_path,
        first_done.manifest,
        question="Question 2",
        answer="Answer 2",
    )
    child = store.fork_retry_assistant(
        tmp_path,
        created.id,
        second.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    child_done = store.update_assistant(
        tmp_path,
        created.id,
        child.assistant_turn.turn_id,
        branch_id=child.branch.branch_id,
        expected_revision=child.manifest.revision,
        text="Alternate answer 2",
        partial=False,
    )

    edited = store.fork_edit_user(
        tmp_path,
        created.id,
        first.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=child.branch.branch_id,
        expected_revision=child_done.manifest.revision,
        text="Edited question 1",
        model_id="llada",
        input_mode="chat",
    )
    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        first.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=child.branch.branch_id,
        expected_revision=child_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    edited_page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=edited.branch.branch_id,
    )
    retried_page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=retried.branch.branch_id,
    )

    assert edited.branch.parent_branch_id == child.branch.branch_id
    assert retried.branch.parent_branch_id == child.branch.branch_id
    assert edited.branch.depth == 2
    assert retried.branch.depth == 2
    assert [turn.text for turn in edited_page.turns] == [
        "Edited question 1",
        "",
    ]
    assert [turn.text for turn in retried_page.turns] == [
        "Question 1",
        "",
    ]


def test_parent_extension_does_not_change_child_path(
    tmp_path: Path,
) -> None:
    """A child keeps its fixed prefix while its parent grows."""
    created = store.create(tmp_path)
    first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    child = store.fork_retry_assistant(
        tmp_path,
        created.id,
        first.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=first_done.manifest.branch_id,
        expected_revision=first_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    assert first_done.manifest.branch_id is not None
    parent_append = store.append_user(
        tmp_path,
        created.id,
        branch_id=first_done.manifest.branch_id,
        expected_revision=first_done.manifest.revision,
        text="Question 2",
        model_id="llada",
        input_mode="chat",
    )
    parent_done = store.update_assistant(
        tmp_path,
        created.id,
        parent_append.assistant_turn.turn_id,
        branch_id=first_done.manifest.branch_id,
        expected_revision=parent_append.manifest.revision,
        text="Answer 2",
        partial=False,
    )

    child_done = store.update_assistant(
        tmp_path,
        created.id,
        child.assistant_turn.turn_id,
        branch_id=child.branch.branch_id,
        expected_revision=child.manifest.revision,
        text="Alternate",
        partial=False,
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=child.branch.branch_id,
    )

    assert parent_done.manifest.turn_count == 4
    assert child_done.manifest.turn_count == 2
    assert [turn.text for turn in page.turns] == [
        "Question 1",
        "Alternate",
    ]


def test_delete_fork_ends_before_the_selected_user(
    tmp_path: Path,
) -> None:
    """Delete retains the prefix with no local turn directory."""
    created = store.create(tmp_path)
    _first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    second, second_done = _append_complete(
        tmp_path,
        first_done.manifest,
        question="Question 2",
        answer="Answer 2",
    )

    deleted = store.fork_delete_from_path(
        tmp_path,
        created.id,
        second.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=deleted.branch.branch_id,
    )

    assert [turn.text for turn in page.turns] == [
        "Question 1",
        "Answer 1",
    ]
    assert deleted.removed_turn_count == 2
    assert page.branch_points[0].turn_index == 3
    assert page.branch_points[0].selected_branch_id == (
        deleted.branch.branch_id
    )
    assert page.branch_points[0].deleted_branch_ids == (
        deleted.branch.branch_id,
    )
    assert (
        list(_branch_turns_root(tmp_path, deleted.manifest).iterdir())
        == []
    )


def test_delete_first_user_creates_an_empty_path(
    tmp_path: Path,
) -> None:
    """Deleting the first exchange produces a valid empty child."""
    created, appended, completed = _ready_pair(tmp_path)

    deleted = store.fork_delete_from_path(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=deleted.branch.branch_id,
    )

    assert deleted.branch.turn_count == 0
    assert deleted.branch.tail_turn_id is None
    assert page.turns == ()
    assert page.branch_points[0].turn_index == 1
    assert page.branch_points[0].deleted_branch_ids == (
        deleted.branch.branch_id,
    )


def test_append_reactivates_a_deleted_path_marker(
    tmp_path: Path,
) -> None:
    """Appending at the deleted slot removes that branch's marker."""
    created, appended, completed = _ready_pair(tmp_path)
    deleted = store.fork_delete_from_path(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
    )

    resumed = store.append_user(
        tmp_path,
        created.id,
        branch_id=deleted.manifest.branch_id,
        expected_revision=deleted.manifest.revision,
        text="Replacement path",
        model_id="llada",
        input_mode="chat",
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=resumed.manifest.branch_id,
    )

    assert [turn.text for turn in page.turns] == [
        "Replacement path",
        "",
    ]
    assert len(page.branch_points) == 1
    assert page.branch_points[0].deleted_branch_ids == ()


def test_every_retry_allocates_a_new_branch_and_node(
    tmp_path: Path,
) -> None:
    """Repeated tail retries never revise the original assistant."""
    created, appended, completed = _ready_pair(tmp_path)

    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    second = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    assert first.branch.branch_id != second.branch.branch_id
    assert (
        first.assistant_turn.turn_id != second.assistant_turn.turn_id
    )
    assert (
        first.assistant_turn.turn_id
        != appended.assistant_turn.turn_id
    )
    assert (
        second.branch.parent_branch_id == completed.manifest.branch_id
    )


def test_lost_edit_response_replays_exact_result(
    tmp_path: Path,
) -> None:
    """A retry returns the edit despite stale CAS values."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "1" * 32
    first = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        text="Edited question",
        model_id="llada",
        input_mode="chat",
        metadata={"client": "edit"},
    )
    revised = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Revised original",
        partial=False,
    )
    store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="2" * 32,
        branch_id=revised.manifest.branch_id,
        expected_revision=revised.manifest.revision,
        expected_catalog_revision=first.catalog.revision,
        model_id="llada",
        input_mode="chat",
    )
    before = store.list_branches(tmp_path, created.id)

    replayed = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        text="Edited question",
        model_id="llada",
        input_mode="chat",
        metadata={"client": "edit"},
    )

    assert replayed == first
    after = store.list_branches(tmp_path, created.id)
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    receipt_files = list(operations_root.iterdir())
    assert after == before
    assert len(receipt_files) == 2
    assert len(receipt_files) < len(after.catalog.branch_ids)
    receipt_path = operations_root / f"{operation_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == set(
        store.OperationReceiptPayload.__required_keys__
    )
    assert receipt["operation_id"] == operation_id
    assert receipt["result_branch_id"] == first.branch.branch_id
    assert len(receipt["request_digest"]) == 64


def test_lost_delete_response_replays_exact_result(
    tmp_path: Path,
) -> None:
    """A delete receipt wins over a stale branch revision."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "3" * 32
    first = store.fork_delete_from_path(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
    )
    store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Revised original",
        partial=False,
    )
    before = store.list_branches(tmp_path, created.id)

    replayed = store.fork_delete_from_path(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
    )

    assert replayed == first
    assert store.list_branches(tmp_path, created.id) == before


def test_lost_retry_response_replays_exact_result(
    tmp_path: Path,
) -> None:
    """A retry receipt wins over stale source coordinates."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "4" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
    )
    store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Revised original",
        partial=False,
    )
    before = store.list_branches(tmp_path, created.id)

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
    )

    assert replayed == first
    assert store.list_branches(tmp_path, created.id) == before


def test_operation_replay_ignores_changed_cas_preconditions(
    tmp_path: Path,
) -> None:
    """CAS coordinates are not part of operation semantics."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "b" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
    )
    revised = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Changed source revision",
        partial=False,
    )

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=revised.manifest.revision,
        expected_catalog_revision=first.catalog.revision,
        model_id="llada",
        input_mode="chat",
    )

    assert replayed == first


def test_operation_id_collision_refuses_changed_edit_payload(
    tmp_path: Path,
) -> None:
    """A committed operation id cannot acquire new semantics."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "5" * 32
    store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="First edit",
        model_id="llada",
        input_mode="chat",
        metadata={"version": 1},
    )
    before = store.list_branches(tmp_path, created.id)

    with pytest.raises(store.ConversationOperationConflictError):
        store.fork_edit_user(
            tmp_path,
            created.id,
            appended.user_turn.turn_id,
            operation_id=operation_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            text="Changed edit",
            model_id="llada",
            input_mode="chat",
            metadata={"version": 2},
        )

    assert store.list_branches(tmp_path, created.id) == before


def test_edit_configuration_round_trips_and_is_pending_only(
    tmp_path: Path,
) -> None:
    """Edit stores the exact snapshot only on its reservation."""
    created, appended, completed = _ready_pair(tmp_path)
    configuration = _generation_configuration(temperature=0.75)
    forked = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id="6" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        text="Configured edit",
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )

    assert forked.generation_configuration == configuration
    assert forked.assistant_turn.metadata == {
        store.PENDING_GENERATION_KEY: configuration
    }
    receipt_path = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
        / f"{'6' * 32}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["request_digest"] == _semantic_digest(
        {
            "kind": "edit_user",
            "source_branch_id": completed.manifest.branch_id,
            "target_turn_id": appended.user_turn.turn_id,
            "text": "Configured edit",
            "model_id": "llada",
            "input_mode": "chat",
            "generation_configuration": configuration,
            "metadata": {},
        }
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=forked.branch.branch_id,
    )
    assert page.turns[-1].metadata == forked.assistant_turn.metadata

    completed_fork = store.update_assistant(
        tmp_path,
        created.id,
        forked.assistant_turn.turn_id,
        branch_id=forked.branch.branch_id,
        expected_revision=forked.manifest.revision,
        text="Configured answer",
        partial=False,
        metadata={"source": "worker"},
    )

    assert completed_fork.turn.metadata == {"source": "worker"}
    assert (
        store.PENDING_GENERATION_KEY
        not in completed_fork.turn.metadata
    )


def test_public_user_metadata_refuses_reserved_generation_key(
    tmp_path: Path,
) -> None:
    """Only the store may create pending-generation metadata."""
    created = store.create(tmp_path)

    with pytest.raises(ValueError, match="cannot contain"):
        store.append_user(
            tmp_path,
            created.id,
            branch_id=created.branch_id,
            expected_revision=created.revision,
            text="Question",
            model_id="llada",
            input_mode="chat",
            metadata={store.PENDING_GENERATION_KEY: {}},
        )

    assert store.get_manifest(tmp_path, created.id) == created


def test_public_completion_metadata_refuses_reserved_generation_key(
    tmp_path: Path,
) -> None:
    """Completion clears rather than accepting reserved spoofing."""
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)

    with pytest.raises(ValueError, match="cannot contain"):
        store.update_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            branch_id=created.branch_id,
            expected_revision=appended.manifest.revision,
            text="Answer",
            partial=False,
            metadata={store.PENDING_GENERATION_KEY: {}},
        )

    assert (
        store.get_manifest(tmp_path, created.id).pending_assistant_id
        == appended.assistant_turn.turn_id
    )


def test_retry_receipt_replays_the_same_configuration(
    tmp_path: Path,
) -> None:
    """Receipt replay reconstructs version-one pending metadata."""
    created, appended, completed = _ready_pair(tmp_path)
    configuration = _generation_configuration(temperature=0.65)
    operation_id = "7" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision + 99,
        expected_catalog_revision=first.catalog.revision + 99,
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )

    assert replayed == first
    assert replayed.generation_configuration == configuration
    assert replayed.assistant_turn.metadata == {
        store.PENDING_GENERATION_KEY: configuration
    }


def test_receipt_digest_canonicalizes_equivalent_json_numbers(
    tmp_path: Path,
) -> None:
    """A persisted float replays a request that arrived as an int."""
    created, appended, completed = _ready_pair(tmp_path)
    configuration = _generation_configuration()
    parameters = configuration["parameters"]
    assert isinstance(parameters, dict)
    parameters["temperature"] = 0
    operation_id = "e" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )
    stored = first.generation_configuration
    assert stored is not None

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision + 1,
        expected_catalog_revision=first.catalog.revision + 1,
        model_id="llada",
        input_mode="chat",
        generation_configuration=stored,
    )

    assert replayed == first
    assert stored["parameters"]["temperature"] == 0.0


def test_preconfiguration_retry_receipt_replays_without_duplicate(
    tmp_path: Path,
) -> None:
    """An old receipt stays pending without invented settings."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "c" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(
            temperature=0.25
        ),
    )
    _rewrite_fork_as_preconfiguration_fixture(
        tmp_path,
        conversation_id=created.id,
        assistant_turn_id=first.assistant_turn.turn_id,
        operation_id=operation_id,
        semantic_payload={
            "kind": "retry_assistant",
            "source_branch_id": completed.manifest.branch_id,
            "target_turn_id": appended.assistant_turn.turn_id,
            "model_id": "llada",
            "input_mode": "chat",
        },
    )
    before = store.list_branches(tmp_path, created.id)

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision + 99,
        expected_catalog_revision=first.catalog.revision + 99,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(
            temperature=0.9
        ),
    )

    assert replayed.branch.branch_id == first.branch.branch_id
    assert replayed.assistant_turn.metadata == {}
    assert replayed.generation_configuration is None
    assert store.list_branches(tmp_path, created.id) == before


@pytest.mark.parametrize("evolution", ["add", "remove", "change"])
def test_schema_evolution_keeps_reads_and_receipt_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evolution: str,
) -> None:
    """Old data reads, stale writes fail, and receipts replay once."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "7" * 32
    configuration = _generation_configuration(temperature=0.25)
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )
    original = REGISTRY["llada"]
    specs = list(original.param_specs)
    if evolution == "add":
        specs.append(
            ParamSpec(
                name="future_parameter",
                label="Future Parameter",
                type=ParamType.INT,
                default=1,
                group=ParamGroup.OUTPUT,
                recommended=(1, 2),
            )
        )
    elif evolution == "remove":
        specs.pop()
    else:
        specs[0] = specs[0].model_copy(
            update={"default": specs[0].default + 1}
        )
    monkeypatch.setitem(
        REGISTRY,
        "llada",
        original.model_copy(update={"param_specs": specs}),
    )

    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=first.branch.branch_id,
    )
    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision + 99,
        expected_catalog_revision=first.catalog.revision + 99,
        model_id="llada",
        input_mode="chat",
        generation_configuration=configuration,
    )

    assert page.turns[-1].metadata[
        store.PENDING_GENERATION_KEY
    ] == configuration
    assert replayed.branch.branch_id == first.branch.branch_id
    assert len(
        store.list_branches(tmp_path, created.id).branches
    ) == 2
    with pytest.raises(ValueError, match="schema id"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=first.catalog.revision,
            model_id="llada",
            input_mode="chat",
            generation_configuration=configuration,
        )


def test_preconfiguration_edit_receipt_replays_without_duplicate(
    tmp_path: Path,
) -> None:
    """A committed old edit keeps its branch without invented data."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "d" * 32
    metadata = {"source": "legacy-client"}
    first = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        text="Legacy edit",
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(
            temperature=0.25
        ),
        metadata=metadata,
    )
    _rewrite_fork_as_preconfiguration_fixture(
        tmp_path,
        conversation_id=created.id,
        assistant_turn_id=first.assistant_turn.turn_id,
        operation_id=operation_id,
        semantic_payload={
            "kind": "edit_user",
            "source_branch_id": completed.manifest.branch_id,
            "target_turn_id": appended.user_turn.turn_id,
            "text": "Legacy edit",
            "model_id": "llada",
            "input_mode": "chat",
            "metadata": metadata,
        },
    )
    before = store.list_branches(tmp_path, created.id)

    replayed = store.fork_edit_user(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision + 99,
        expected_catalog_revision=first.catalog.revision + 99,
        text="Legacy edit",
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(
            temperature=0.9
        ),
        metadata=metadata,
    )

    assert replayed.branch.branch_id == first.branch.branch_id
    assert replayed.assistant_turn.metadata == {}
    assert replayed.generation_configuration is None
    assert store.list_branches(tmp_path, created.id) == before


def test_operation_id_collision_refuses_changed_configuration(
    tmp_path: Path,
) -> None:
    """Hyperparameters are part of fork operation semantics."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "8" * 32
    store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(
            temperature=0.25
        ),
    )

    with pytest.raises(store.ConversationOperationConflictError):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=operation_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=completed.manifest.catalog_revision,
            model_id="llada",
            input_mode="chat",
            generation_configuration=_generation_configuration(
                temperature=0.5
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "bool_as_int",
        "nonfinite",
        "wrong_select",
        "out_of_bounds",
        "experimental_bool",
        "unsupported_device",
        "huge_integer",
        "huge_float",
    ],
)
def test_retry_configuration_rejects_invalid_schema_values(
    tmp_path: Path,
    case: str,
) -> None:
    """Malformed snapshots fail before allocating a branch."""
    created, appended, completed = _ready_pair(tmp_path)
    configuration = _generation_configuration()
    parameters = configuration["parameters"]
    assert isinstance(parameters, dict)
    if case == "unknown":
        parameters["unknown"] = 1
    elif case == "bool_as_int":
        parameters["steps"] = True
    elif case == "nonfinite":
        parameters["temperature"] = float("inf")
    elif case == "wrong_select":
        parameters["remasking"] = 1
    elif case == "out_of_bounds":
        parameters["steps"] = 151
    elif case == "experimental_bool":
        configuration["experimental"] = 1
    elif case == "huge_integer":
        parameters["steps"] = 10**1000
    elif case == "huge_float":
        parameters["temperature"] = 1e101
    else:
        assert case == "unsupported_device"
        configuration["device"] = "cpu"
    before = store.get_catalog(tmp_path, created.id)

    with pytest.raises(ValueError):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=(
                completed.manifest.catalog_revision
            ),
            model_id="llada",
            input_mode="chat",
            generation_configuration=configuration,
        )

    assert store.get_catalog(tmp_path, created.id) == before


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("gen_length", 159, "divisible by block_length"),
        ("block_length", 32, "steps .* divisible by"),
    ],
)
def test_retry_configuration_rejects_llada_relational_rules(
    tmp_path: Path,
    name: str,
    value: int,
    message: str,
) -> None:
    """LLaDA arithmetic is checked before branch publication."""
    created, appended, completed = _ready_pair(tmp_path)
    configuration = _generation_configuration()
    parameters = configuration["parameters"]
    assert isinstance(parameters, dict)
    parameters[name] = value
    before = store.get_catalog(tmp_path, created.id)

    with pytest.raises(ValueError, match=message):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=(
                completed.manifest.catalog_revision
            ),
            model_id="llada",
            input_mode="chat",
            generation_configuration=configuration,
        )

    assert store.get_catalog(tmp_path, created.id) == before


def test_retry_configuration_uses_device_specific_bounds(
    tmp_path: Path,
) -> None:
    """The same token budget is valid on GPU but not ordinary CPU."""
    created, appended, completed = _ready_pair(tmp_path)
    with pytest.raises(ValueError, match="active cpu bounds"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id="a" * 32,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=(
                completed.manifest.catalog_revision
            ),
            model_id="smollm3",
            input_mode="chat",
            generation_configuration=_smollm_configuration(
                device="cpu",
                max_new_tokens=200,
            ),
        )

    forked = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="b" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="smollm3",
        input_mode="chat",
        generation_configuration=_smollm_configuration(
            device="cuda",
            max_new_tokens=200,
        ),
    )

    assert forked.generation_configuration["device"] == "cuda"
    assert (
        forked.generation_configuration["parameters"][
            "max_new_tokens"
        ]
        == 200
    )


@pytest.mark.parametrize("registry_change", ["added", "removed"])
def test_pending_configuration_read_ignores_registry_drift(
    tmp_path: Path,
    registry_change: str,
) -> None:
    """Durable reads preserve snapshots from older registries."""
    created, appended, completed = _ready_pair(tmp_path)
    forked = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="9" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(),
    )
    assistant_dir = _turn_dir(
        tmp_path,
        created.id,
        forked.assistant_turn.turn_id,
    )
    path = assistant_dir / "00000001.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    pending = raw["metadata"][store.PENDING_GENERATION_KEY]
    if registry_change == "added":
        del pending["parameters"]["alternatives"]
    else:
        assert registry_change == "removed"
        pending["parameters"]["retired_parameter"] = 17
    path.write_text(json.dumps(raw), encoding="utf-8")

    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=forked.branch.branch_id,
    )

    stored = page.turns[-1].metadata[
        store.PENDING_GENERATION_KEY
    ]
    assert stored == pending


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("codec", "unsupported generation configuration codec"),
        ("device", "device must be cpu or cuda"),
        ("identity", "model_id does not match"),
        ("type", "temperature must be a boolean, number, or string"),
        ("magnitude", "temperature exceeds the numeric limit"),
        ("field", "unknown fields"),
        ("serialized_size", "exceeds 16384 bytes"),
    ],
)
def test_corrupt_pending_configuration_structure_is_refused(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """The registry-independent reader still enforces codec bounds."""
    created, appended, completed = _ready_pair(tmp_path)
    forked = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="9" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
        generation_configuration=_generation_configuration(),
    )
    path = (
        _turn_dir(
            tmp_path,
            created.id,
            forked.assistant_turn.turn_id,
        )
        / "00000001.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    pending = raw["metadata"][store.PENDING_GENERATION_KEY]
    parameters = pending["parameters"]
    if case == "codec":
        pending["codec_version"] = 2
    elif case == "device":
        pending["device"] = "cuda:0"
    elif case == "identity":
        pending["model_id"] = "mamba3"
    elif case == "type":
        parameters["temperature"] = []
    elif case == "magnitude":
        parameters["temperature"] = 10**101
    elif case == "field":
        pending["unexpected"] = True
    else:
        assert case == "serialized_size"
        pending["parameters"] = {
            f"retired_{index}": "\U0001f4a5" * 1024
            for index in range(5)
        }
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError,
        match=message,
    ):
        store.get_turns(
            tmp_path,
            created.id,
            branch_id=forked.branch.branch_id,
        )


def test_authoritative_replay_does_not_clean_other_debris(
    tmp_path: Path,
) -> None:
    """A receipt replay performs no unrelated storage mutation."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "9" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    conversation_dir = _conversation_dir(tmp_path, created.id)
    orphan_branch_id = "b_" + "f" * 32
    orphan_branch = (
        conversation_dir / store.BRANCHES_DIR_NAME / orphan_branch_id
    )
    orphan_branch.mkdir()
    orphan_receipt = (
        conversation_dir
        / store.OPERATIONS_DIR_NAME
        / f"{'e' * 32}.json"
    )
    orphan_receipt.write_text(
        json.dumps(
            {
                "schema_version": store.SCHEMA_VERSION,
                "operation_id": "e" * 32,
                "request_digest": "0" * 64,
                "kind": "retry_assistant",
                "source_branch_id": completed.manifest.branch_id,
                "target_turn_id": appended.assistant_turn.turn_id,
                "result_branch_id": orphan_branch_id,
                "catalog_revision": first.catalog.revision + 1,
                "removed_turn_count": None,
            }
        ),
        encoding="utf-8",
    )

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    assert replayed == first
    assert orphan_branch.is_dir()
    assert orphan_receipt.is_file()


def test_tail_revision_stays_on_the_retry_node(
    tmp_path: Path,
) -> None:
    """Edit Frames-style updates version one retry node in place."""
    created, appended, completed = _ready_pair(tmp_path)
    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    first = store.update_assistant(
        tmp_path,
        created.id,
        retried.assistant_turn.turn_id,
        branch_id=retried.branch.branch_id,
        expected_revision=retried.manifest.revision,
        text="Retry answer",
        partial=False,
    )
    revised = store.update_assistant(
        tmp_path,
        created.id,
        retried.assistant_turn.turn_id,
        branch_id=retried.branch.branch_id,
        expected_revision=first.manifest.revision,
        text="Revised retry",
        partial=False,
    )

    assert revised.turn.turn_id == retried.assistant_turn.turn_id
    assert revised.turn.version == 3
    assert revised.turn.text == "Revised retry"


def test_old_run_links_survive_retry_branches(
    tmp_path: Path,
) -> None:
    """A retry leaves the source node and its saved link readable."""
    created, appended, completed = _ready_pair(tmp_path)
    linked = store.set_run_link(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 7),
        expected_turn_index=completed.turn.index,
        expected_turn_version=completed.turn.version,
    )

    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=linked.manifest.branch_id,
        expected_revision=linked.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    source_page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=linked.manifest.branch_id,
    )

    assert source_page.turns[-1].run_link == store.RunLink(
        "2026-01-01_llada", 7
    )
    assert retried.assistant_turn.run_link is None


def test_inherited_prefix_keeps_its_run_link(
    tmp_path: Path,
) -> None:
    """A saved link remains attached inside a shared prefix."""
    created, first, first_done = _ready_pair(tmp_path)
    linked = store.set_run_link(
        tmp_path,
        created.id,
        first.assistant_turn.turn_id,
        branch_id=first_done.manifest.branch_id,
        expected_revision=first_done.manifest.revision,
        run_link=store.RunLink("2026-01-01_llada", 4),
        expected_turn_index=first_done.turn.index,
        expected_turn_version=first_done.turn.version,
    )
    second, second_done = _append_complete(
        tmp_path,
        linked.manifest,
        question="Question 2",
        answer="Answer 2",
    )

    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        second.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    page = store.get_turns(
        tmp_path,
        created.id,
        branch_id=retried.branch.branch_id,
    )

    assert page.turns[1].run_link == store.RunLink(
        "2026-01-01_llada", 4
    )


def test_unrelated_branch_cas_does_not_conflict(
    tmp_path: Path,
) -> None:
    """Each branch can advance from its own unchanged revision."""
    created, appended, completed = _ready_pair(tmp_path)
    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    original = store.update_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        text="Revised original",
        partial=False,
    )
    alternate = store.update_assistant(
        tmp_path,
        created.id,
        retried.assistant_turn.turn_id,
        branch_id=retried.branch.branch_id,
        expected_revision=retried.manifest.revision,
        text="Alternate",
        partial=False,
    )
    catalog = store.get_catalog(tmp_path, created.id)

    assert (
        original.manifest.revision == completed.manifest.revision + 1
    )
    assert (
        alternate.manifest.revision == retried.manifest.revision + 1
    )
    assert catalog.default_branch_id == retried.branch.branch_id
    assert catalog.revision == 2


def test_branch_browsing_is_catalog_read_only(
    tmp_path: Path,
) -> None:
    """Explicit reads never change the durable default or revision."""
    created, appended, completed = _ready_pair(tmp_path)
    retried = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    before = store.get_catalog(tmp_path, created.id)

    store.get_manifest(
        tmp_path,
        created.id,
        branch_id=completed.manifest.branch_id,
    )
    store.get_turns(
        tmp_path,
        created.id,
        branch_id=completed.manifest.branch_id,
    )
    store.list_branches(tmp_path, created.id)
    after = store.get_catalog(tmp_path, created.id)

    assert before == after
    assert after.default_branch_id == retried.branch.branch_id


def test_branch_page_crosses_the_shared_prefix_boundary(
    tmp_path: Path,
) -> None:
    """Paging composes an inherited prefix with one local retry."""
    manifest = store.create(tmp_path)
    last_append: Optional[store.AppendResult] = None
    for index in range(30):
        last_append, completed = _append_complete(
            tmp_path,
            manifest,
            question=f"Question {index}",
            answer=f"Answer {index}",
        )
        manifest = completed.manifest
    assert last_append is not None
    retried = store.fork_retry_assistant(
        tmp_path,
        manifest.id,
        last_append.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=manifest.branch_id,
        expected_revision=manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    newest = store.get_turns(
        tmp_path,
        manifest.id,
        branch_id=retried.branch.branch_id,
    )
    older = store.get_turns(
        tmp_path,
        manifest.id,
        branch_id=retried.branch.branch_id,
        before=newest.next_before,
    )

    assert [turn.index for turn in newest.turns] == list(
        range(11, 61)
    )
    assert newest.turns[-1].turn_id == retried.assistant_turn.turn_id
    assert newest.next_before == "00000011"
    assert [turn.index for turn in older.turns] == list(range(1, 11))


def test_branch_count_limit_refuses_another_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog honors the fixed branch bound."""
    created, appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(core_store, "BRANCH_COUNT_MAX", 2)
    store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    with pytest.raises(
        store.ConversationLimitError, match="branches"
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_sibling_limit_is_per_fork_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sibling cap applies at one parent prefix."""
    created, appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(
        core_store,
        "BRANCH_SIBLINGS_MAX",
        1,
    )
    store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    with pytest.raises(store.ConversationLimitError, match="sibling"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_depth_limit_refuses_a_nested_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child at the depth bound cannot allocate another child."""
    created, appended, completed = _ready_pair(tmp_path)
    child = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    child_done = store.update_assistant(
        tmp_path,
        created.id,
        child.assistant_turn.turn_id,
        branch_id=child.branch.branch_id,
        expected_revision=child.manifest.revision,
        text="Alternate",
        partial=False,
    )
    monkeypatch.setattr(core_store, "BRANCH_DEPTH_MAX", 1)

    with pytest.raises(store.ConversationLimitError, match="depth"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            child.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=child.branch.branch_id,
            expected_revision=child_done.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_turn_count_limit_applies_to_each_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pair that would cross the path turn bound is refused."""
    _created, _appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(core_store, "TURN_COUNT_MAX", 3)

    with pytest.raises(store.ConversationLimitError, match="turns"):
        _append(tmp_path, completed.manifest, text="Too many")


def test_local_turn_limit_applies_to_one_branch_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch cannot grow beyond its bounded local suffix."""
    _created, _appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(
        core_store,
        "BRANCH_LOCAL_TURNS_MAX",
        3,
    )

    with pytest.raises(
        store.ConversationLimitError, match="local turn"
    ):
        _append(tmp_path, completed.manifest, text="Too local")


def test_catalog_revision_limit_refuses_branch_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork cannot wrap or exceed the bounded catalog revision."""
    created, appended, completed = _ready_pair(tmp_path)
    monkeypatch.setattr(
        core_store,
        "CATALOG_REVISION_MAX",
        1,
    )

    with pytest.raises(
        store.ConversationLimitError, match="catalog revision"
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_stale_catalog_cas_refuses_fork_without_files(
    tmp_path: Path,
) -> None:
    """Catalog CAS fails before branch allocation or publication."""
    created, appended, completed = _ready_pair(tmp_path)
    before = store.list_branches(tmp_path, created.id)

    with pytest.raises(
        store.ConversationCatalogRevisionConflictError
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=0,
            model_id="llada",
            input_mode="chat",
        )

    after = store.list_branches(tmp_path, created.id)
    assert after == before


def test_catalog_recovers_orphan_branch_and_receipt(
    tmp_path: Path,
) -> None:
    """The next fork removes an unpublished operation."""
    created, appended, completed = _ready_pair(tmp_path)
    assert completed.manifest.branch_id is not None
    conversation_dir = _conversation_dir(tmp_path, created.id)
    branches_root = (
        conversation_dir / store.BRANCHES_DIR_NAME
    )
    operations_root = conversation_dir / store.OPERATIONS_DIR_NAME
    unpublished_id = "b_" + "f" * 32
    assert unpublished_id != completed.manifest.branch_id
    unpublished = branches_root / unpublished_id
    unpublished.mkdir()
    (unpublished / "partial").write_text(
        "uncommitted",
        encoding="utf-8",
    )
    orphan_operation_id = "f" * 32
    orphan_receipt = operations_root / f"{orphan_operation_id}.json"
    orphan_receipt.write_text(
        json.dumps(
            {
                "schema_version": store.SCHEMA_VERSION,
                "operation_id": orphan_operation_id,
                "request_digest": "0" * 64,
                "kind": "retry_assistant",
                "source_branch_id": completed.manifest.branch_id,
                "target_turn_id": appended.assistant_turn.turn_id,
                "result_branch_id": unpublished_id,
                "catalog_revision": 2,
                "removed_turn_count": None,
            }
        ),
        encoding="utf-8",
    )

    before = store.list_branches(tmp_path, created.id)
    assert [item.branch_id for item in before.branches] == [
        completed.manifest.branch_id
    ]

    operation_id = "e" * 32
    forked = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    after = store.list_branches(tmp_path, created.id)

    assert not unpublished.exists()
    assert not orphan_receipt.exists()
    assert set(path.name for path in branches_root.iterdir()) == set(
        after.catalog.branch_ids
    )
    assert {path.name for path in operations_root.iterdir()} == {
        f"{operation_id}.json"
    }
    assert forked.branch.branch_id in after.catalog.branch_ids


def test_operation_recovery_removes_atomic_writer_temporary(
    tmp_path: Path,
) -> None:
    """A crashed receipt replace leaves bounded removable debris."""
    created, appended, completed = _ready_pair(tmp_path)
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    temporary = (
        operations_root
        / f".{'c' * 32}.json.abcdefgh.tmp"
    )
    temporary.write_text("partial", encoding="utf-8")

    store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="d" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    assert not temporary.exists()


def test_receipt_replay_recovers_an_atomic_writer_temporary(
    tmp_path: Path,
) -> None:
    """Committed replay is not permanently blocked by temp debris."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "3" * 32
    first = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    temporary = (
        operations_root
        / f".{'4' * 32}.json.abcdefgh.tmp"
    )
    temporary.write_text("partial", encoding="utf-8")

    replayed = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    assert replayed == first
    assert not temporary.exists()


@pytest.mark.parametrize(
    "name",
    [
        ".arbitrary.tmp",
        f".{'e' * 32}.json.too-short.tmp",
        "receipt.partial",
    ],
)
def test_operation_recovery_rejects_arbitrary_files(
    tmp_path: Path,
    name: str,
) -> None:
    """Only the store's exact atomic temporary shape is removable."""
    created, appended, completed = _ready_pair(tmp_path)
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    (operations_root / name).write_text(
        "unsafe", encoding="utf-8"
    )

    with pytest.raises(store.ConversationCorruptError):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id="f" * 32,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_operation_recovery_rejects_atomic_temporary_symlink(
    tmp_path: Path,
) -> None:
    """A matching name never makes a symlink writer-owned."""
    created, appended, completed = _ready_pair(tmp_path)
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    temporary = (
        operations_root
        / f".{'c' * 32}.json.abcdefgh.tmp"
    )
    temporary.symlink_to(tmp_path / "outside")

    with pytest.raises(
        store.ConversationCorruptError, match="temporary.*unsafe"
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id="d" * 32,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_fork_barriers_precede_the_catalog_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branch data and receipt parents sync before the catalog."""
    created, appended, completed = _ready_pair(tmp_path)
    conversation_dir = _conversation_dir(tmp_path, created.id)
    synced: list[Path] = []
    monkeypatch.setattr(
        core_store,
        "fsync_directory",
        lambda path: synced.append(path),
    )

    forked = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id="1" * 32,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )

    branch_dir = (
        conversation_dir
        / store.BRANCHES_DIR_NAME
        / forked.branch.branch_id
    )
    operations_root = conversation_dir / store.OPERATIONS_DIR_NAME
    assert branch_dir in synced
    assert operations_root in synced
    assert synced.index(branch_dir) < len(synced) - 1
    assert synced.index(operations_root) < len(synced) - 1
    assert synced[-1] == conversation_dir


def test_catalog_directory_barrier_failure_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported commit-boundary directory fsync is not hidden."""
    created, appended, completed = _ready_pair(tmp_path)
    conversation_dir = _conversation_dir(tmp_path, created.id)
    real_fsync = core_store.fsync_directory

    def fail_catalog_parent(path: Path) -> None:
        if path == conversation_dir:
            raise OSError("directory fsync unsupported")
        real_fsync(path)

    monkeypatch.setattr(
        core_store, "fsync_directory", fail_catalog_parent
    )

    with pytest.raises(OSError, match="fsync unsupported"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id="2" * 32,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_failed_catalog_publication_stays_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed catalog write stays invisible and retryable."""
    created, appended, completed = _ready_pair(tmp_path)
    real_write = branch_store.write_catalog
    branches_root = (
        _conversation_dir(tmp_path, created.id)
        / store.BRANCHES_DIR_NAME
    )

    def fail_catalog(
        conversation_dir: Path,
        catalog: store.ConversationCatalog,
    ) -> None:
        if catalog.revision == 2:
            raise OSError("injected catalog failure")
        real_write(conversation_dir, catalog)

    monkeypatch.setattr(
        branch_store,
        "write_catalog",
        fail_catalog,
    )
    with pytest.raises(OSError, match="injected"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=_operation_id(),
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )

    branches = store.list_branches(tmp_path, created.id)
    assert branches.catalog.revision == 1
    assert len(branches.branches) == 1
    unpublished = {
        path.name for path in branches_root.iterdir()
    } - set(branches.catalog.branch_ids)
    assert len(unpublished) == 1

    monkeypatch.setattr(
        branch_store,
        "write_catalog",
        real_write,
    )
    recovered = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    after = store.list_branches(tmp_path, created.id)

    assert unpublished.isdisjoint(
        path.name for path in branches_root.iterdir()
    )
    assert set(after.catalog.branch_ids) == {
        path.name for path in branches_root.iterdir()
    }
    assert recovered.branch.branch_id in after.catalog.branch_ids


def test_retry_after_catalog_crash_replaces_orphan_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branch and receipt debris stay uncommitted and are replaced."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "6" * 32
    real_write = branch_store.write_catalog
    conversation_dir = _conversation_dir(tmp_path, created.id)
    branches_root = conversation_dir / store.BRANCHES_DIR_NAME
    receipt_path = (
        conversation_dir
        / store.OPERATIONS_DIR_NAME
        / f"{operation_id}.json"
    )

    def fail_catalog(
        _conversation_dir: Path,
        _catalog: store.ConversationCatalog,
    ) -> None:
        raise OSError("injected catalog failure")

    monkeypatch.setattr(
        branch_store,
        "write_catalog",
        fail_catalog,
    )
    with pytest.raises(OSError, match="injected"):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=operation_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            expected_catalog_revision=completed.manifest.catalog_revision,
            model_id="llada",
            input_mode="chat",
        )
    orphan_receipt = json.loads(
        receipt_path.read_text(encoding="utf-8")
    )
    orphan_branch_id = orphan_receipt["result_branch_id"]

    assert orphan_branch_id not in store.get_catalog(
        tmp_path, created.id
    ).branch_ids
    assert (branches_root / orphan_branch_id).is_dir()

    monkeypatch.setattr(
        branch_store,
        "write_catalog",
        real_write,
    )
    recovered = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=operation_id,
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        expected_catalog_revision=completed.manifest.catalog_revision,
        model_id="llada",
        input_mode="chat",
    )
    committed_receipt = json.loads(
        receipt_path.read_text(encoding="utf-8")
    )
    listed = store.list_branches(tmp_path, created.id)

    assert recovered.branch.branch_id != orphan_branch_id
    assert not (branches_root / orphan_branch_id).exists()
    assert (
        committed_receipt["result_branch_id"]
        == recovered.branch.branch_id
    )
    assert len(listed.catalog.branch_ids) == 2
    assert len(list(receipt_path.parent.iterdir())) == 1


def test_extra_branch_manifest_field_is_corruption(
    tmp_path: Path,
) -> None:
    """Strict branch parsing rejects unknown committed fields."""
    manifest = store.create(tmp_path)
    assert manifest.branch_id is not None
    path = (
        _conversation_dir(tmp_path, manifest.id)
        / store.BRANCHES_DIR_NAME
        / manifest.branch_id
        / store.MANIFEST_NAME
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["history"] = []
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError, match="fields differ"
    ):
        store.get_manifest(tmp_path, manifest.id)


def test_resolved_inherited_tail_mismatch_is_corruption(
    tmp_path: Path,
) -> None:
    """A child manifest cannot misdescribe its inherited tail."""
    created = store.create(tmp_path)
    _first, first_done = _append_complete(
        tmp_path,
        created,
        question="Question 1",
        answer="Answer 1",
    )
    second, second_done = _append_complete(
        tmp_path,
        first_done.manifest,
        question="Question 2",
        answer="Answer 2",
    )
    deleted = store.fork_delete_from_path(
        tmp_path,
        created.id,
        second.user_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=second_done.manifest.branch_id,
        expected_revision=second_done.manifest.revision,
    )
    path = (
        _conversation_dir(tmp_path, created.id)
        / store.BRANCHES_DIR_NAME
        / deleted.branch.branch_id
        / store.MANIFEST_NAME
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["tail_turn_id"] = second.assistant_turn.turn_id
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError,
        match="resolved tail id",
    ):
        store.get_turns(
            tmp_path,
            created.id,
            branch_id=deleted.branch.branch_id,
        )


def test_branch_parent_cycle_is_corruption(
    tmp_path: Path,
) -> None:
    """A two-branch ancestry cycle is rejected before traversal."""
    created, appended, completed = _ready_pair(tmp_path)
    child = store.fork_retry_assistant(
        tmp_path,
        created.id,
        appended.assistant_turn.turn_id,
        operation_id=_operation_id(),
        branch_id=completed.manifest.branch_id,
        expected_revision=completed.manifest.revision,
        model_id="llada",
        input_mode="chat",
    )
    assert completed.manifest.branch_id is not None
    path = (
        _conversation_dir(tmp_path, created.id)
        / store.BRANCHES_DIR_NAME
        / completed.manifest.branch_id
        / store.MANIFEST_NAME
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["parent_branch_id"] = child.branch.branch_id
    raw["depth"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        store.ConversationCorruptError, match="ancestry.*cycle"
    ):
        store.get_turns(
            tmp_path,
            created.id,
            branch_id=completed.manifest.branch_id,
        )


def test_symlink_branch_turn_directory_is_corruption(
    tmp_path: Path,
) -> None:
    """Committed turn directories cannot escape through symlinks."""
    created = store.create(tmp_path)
    appended = _append(tmp_path, created)
    turn_dir = _turn_dir(
        tmp_path,
        created.id,
        appended.user_turn.turn_id,
    )
    outside = tmp_path / "outside-turn"
    turn_dir.rename(outside)
    turn_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        store.ConversationCorruptError, match="unsafe"
    ):
        store.get_turns(tmp_path, created.id)


def test_duplicate_catalog_key_is_corruption(
    tmp_path: Path,
) -> None:
    """The strict JSON reader refuses ambiguous duplicate keys."""
    manifest = store.create(tmp_path)
    path = _manifest_path(tmp_path, manifest.id)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{")
    path.write_text(
        '{"schema_version":2,' + text[1:],
        encoding="utf-8",
    )

    with pytest.raises(
        store.ConversationCorruptError, match="unreadable"
    ):
        store.get_catalog(tmp_path, manifest.id)


@pytest.mark.parametrize(
    "branch_id",
    ["../branch", "branch/name", "/absolute", "b_" + "g" * 32],
)
def test_invalid_branch_ids_are_refused(
    tmp_path: Path,
    branch_id: str,
) -> None:
    """Branch lookup accepts only one bounded direct-child name."""
    manifest = store.create(tmp_path)

    with pytest.raises(store.InvalidBranchIdError):
        store.get_branch(tmp_path, manifest.id, branch_id)


def test_unknown_valid_branch_is_not_found(tmp_path: Path) -> None:
    manifest = store.create(tmp_path)

    with pytest.raises(store.BranchNotFoundError):
        store.get_branch(
            tmp_path,
            manifest.id,
            "b_" + "f" * 32,
        )


@pytest.mark.parametrize(
    "operation_id",
    [
        "../operation",
        "A" * 32,
        "a" * 31,
        "g" * 32,
        "",
    ],
)
def test_invalid_fork_operation_ids_are_refused(
    tmp_path: Path,
    operation_id: str,
) -> None:
    """Store callers cannot traverse or use non-canonical ids."""
    created, appended, completed = _ready_pair(tmp_path)

    with pytest.raises(store.InvalidOperationIdError):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=operation_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_symlink_operation_receipt_is_corruption(
    tmp_path: Path,
) -> None:
    """Receipt lookup never follows a link outside its fixed root."""
    created, appended, completed = _ready_pair(tmp_path)
    operation_id = "7" * 32
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    (operations_root / f"{operation_id}.json").symlink_to(
        tmp_path / "missing-receipt"
    )

    with pytest.raises(
        store.ConversationCorruptError, match="safe regular file"
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id=operation_id,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )


def test_operation_receipt_scan_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fork cleanup refuses an oversized receipt set."""
    created, appended, completed = _ready_pair(tmp_path)
    operations_root = (
        _conversation_dir(tmp_path, created.id)
        / store.OPERATIONS_DIR_NAME
    )
    monkeypatch.setattr(
        core_store,
        "OPERATION_RECEIPT_SCAN_MAX",
        1,
    )
    for index in range(2):
        (operations_root / f"{index:032x}.json").write_text(
            "{}",
            encoding="utf-8",
        )

    with pytest.raises(
        store.ConversationCorruptError, match="receipt count"
    ):
        store.fork_retry_assistant(
            tmp_path,
            created.id,
            appended.assistant_turn.turn_id,
            operation_id="8" * 32,
            branch_id=completed.manifest.branch_id,
            expected_revision=completed.manifest.revision,
            model_id="llada",
            input_mode="chat",
        )
