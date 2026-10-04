"""Exact bounded conversation packing without a model.

Strategy: use deterministic counters over typed messages, including a
deliberately non-monotonic one. This isolates suffix selection from
tokenizer behavior while proving every candidate is still counted by
the supplied exact path.

Passing proves the pending user is never dropped, only complete oldest
exchanges are omitted, output tokens are reserved, policy and loaded
window caps both apply, malformed transcripts fail by stable protocol
code, and work is bounded by the hard candidate limit.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Dict, List, Tuple

import pytest

from src.backends.context_pack import (
    MESSAGE_CANDIDATES_MAX,
    MESSAGE_CHARS_MAX,
    ContextRequestError,
    MessageRecord,
    pack_context,
    parse_messages,
)
from src.backends.protocol import (
    ERROR_CONTEXT_BOUNDS,
    ERROR_INVALID_MESSAGE_ORDER,
    ERROR_INVALID_MESSAGE_ROLE,
    ERROR_MALFORMED_MESSAGES,
)


def _messages(
    exchanges: int,
) -> Tuple[MessageRecord, ...]:
    records: List[MessageRecord] = []
    for index in range(exchanges):
        records.append(
            MessageRecord(
                role="user",
                content=f"user {index}",
                turn_id=f"u{index}",
            )
        )
        records.append(
            MessageRecord(
                role="assistant",
                content=f"assistant {index}",
                turn_id=f"a{index}",
            )
        )
    records.append(
        MessageRecord(
            role="user",
            content="pending",
            turn_id="pending",
        )
    )
    return tuple(records)


def _ten_each(messages: Tuple[MessageRecord, ...]) -> int:
    return len(messages) * 10


def _pack(
    messages: Tuple[MessageRecord, ...],
    **overrides: object,
):
    options: Dict[str, object] = {
        "count_tokens": _ten_each,
        "output_reserve": 10,
        "policy_default_tokens": 100,
        "policy_max_tokens": 200,
    }
    options.update(overrides)
    return pack_context(messages, **options)


def test_every_message_is_kept_when_the_full_context_fits() -> None:
    packed = _pack(_messages(2))

    assert packed.messages == _messages(2)
    assert packed.manifest.first_included_index == 0
    assert packed.manifest.omitted_turn_count == 0


def test_oldest_complete_exchanges_are_dropped() -> None:
    packed = _pack(
        _messages(2),
        requested_total_budget=40,
    )

    assert [message.turn_id for message in packed.messages] == [
        "u1",
        "a1",
        "pending",
    ]
    assert packed.manifest.first_included_index == 2
    assert packed.manifest.omitted_turn_count == 2


def test_a_bounded_candidate_tail_reports_absolute_indices() -> None:
    packed = _pack(
        _messages(2),
        requested_total_budget=40,
        candidate_turn_offset=150,
    )

    assert packed.manifest.first_included_index == 152
    assert packed.manifest.omitted_turn_count == 152


def test_parse_accepts_an_absolute_candidate_offset() -> None:
    messages, conversation = parse_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "question",
                    "turn_id": "00000199",
                }
            ],
            "candidate_turn_offset": 198,
            "conversation_id": "a" * 32,
            "conversation_revision": 101,
            "assistant_turn_id": "00000200",
        }
    )

    assert messages[-1].turn_id == "00000199"
    assert conversation is not None


@pytest.mark.parametrize("offset", [-2, 1, True, "2"])
def test_invalid_candidate_offsets_are_refused(
    offset: object,
) -> None:
    with pytest.raises(ContextRequestError):
        parse_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question",
                        "turn_id": "00000003",
                    }
                ],
                "candidate_turn_offset": offset,
            }
        )


def test_candidate_location_ends_before_reserved_assistant() -> None:
    with pytest.raises(ContextRequestError) as raised:
        parse_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question",
                        "turn_id": "00000003",
                    }
                ],
                "candidate_turn_offset": 0,
                "conversation_id": "a" * 32,
                "conversation_revision": 3,
                "assistant_turn_id": "00000004",
            }
        )

    assert raised.value.code == ERROR_MALFORMED_MESSAGES


def test_the_pending_user_is_kept_on_the_smallest_suffix() -> None:
    packed = _pack(
        _messages(3),
        requested_total_budget=20,
    )

    assert packed.messages == (_messages(3)[-1],)
    assert packed.manifest.included_turn_ids == ("pending",)


def test_output_tokens_are_reserved_before_history() -> None:
    packed = _pack(
        _messages(1),
        output_reserve=20,
        requested_total_budget=40,
    )

    assert packed.manifest.prompt_token_count == 10
    assert packed.manifest.output_reserve == 20
    assert packed.manifest.omitted_turn_count == 2


def test_a_pending_turn_that_cannot_fit_is_refused() -> None:
    with pytest.raises(
        ContextRequestError, match="pending user"
    ) as raised:
        _pack(
            _messages(0),
            output_reserve=11,
            requested_total_budget=20,
        )

    assert raised.value.code == ERROR_CONTEXT_BOUNDS


def test_a_loaded_checkpoint_lowers_the_effective_budget() -> None:
    packed = _pack(
        _messages(2),
        requested_total_budget=100,
        checkpoint_window=40,
    )

    manifest = packed.manifest
    assert manifest.requested_total_budget == 100
    assert manifest.effective_total_budget == 40
    assert manifest.omitted_turn_count == 2


def test_an_absent_request_uses_the_policy_default() -> None:
    packed = _pack(_messages(0))

    assert packed.manifest.requested_total_budget == 100
    assert packed.manifest.effective_total_budget == 100


def test_a_request_past_policy_is_refused_not_clamped() -> None:
    with pytest.raises(ContextRequestError) as raised:
        _pack(
            _messages(0),
            requested_total_budget=201,
        )

    assert raised.value.code == ERROR_CONTEXT_BOUNDS


def test_non_monotonic_counts_are_still_searched_exactly() -> None:
    messages = _messages(2)

    def non_monotonic(
        candidate: Tuple[MessageRecord, ...],
    ) -> int:
        first = candidate[0].turn_id
        return {"pending": 5, "u1": 100, "u0": 20}[first]

    packed = _pack(
        messages,
        count_tokens=non_monotonic,
        requested_total_budget=30,
    )

    assert packed.messages == messages
    assert packed.manifest.prompt_token_count == 20


def test_search_work_is_bounded_by_candidate_count() -> None:
    calls = 0
    messages = _messages((MESSAGE_CANDIDATES_MAX - 1) // 2)

    def counted(candidate: Tuple[MessageRecord, ...]) -> int:
        nonlocal calls
        calls += 1
        if len(candidate) == 1:
            return 1
        return 1024

    packed = _pack(
        messages,
        count_tokens=counted,
        output_reserve=1,
        policy_default_tokens=2,
        policy_max_tokens=2,
    )

    assert packed.manifest.first_included_index == (
        MESSAGE_CANDIDATES_MAX - 1
    )
    assert calls == (MESSAGE_CANDIDATES_MAX + 1) // 2


def test_the_manifest_is_immutable() -> None:
    manifest = _pack(_messages(0)).manifest

    with pytest.raises(FrozenInstanceError):
        manifest.prompt_token_count = 1  # type: ignore[misc]


def test_parse_accepts_exact_turn_and_conversation_metadata() -> None:
    messages, conversation = parse_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "question",
                    "turn_id": "00000001",
                }
            ],
            "conversation_id": "a" * 32,
            "conversation_revision": 2,
            "assistant_turn_id": "00000002",
        }
    )

    assert messages[0].content == "question"
    assert conversation is not None
    assert conversation.assistant_turn_id == "00000002"


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        (
            [
                {
                    "role": "system",
                    "content": "no",
                    "turn_id": "1",
                }
            ],
            ERROR_INVALID_MESSAGE_ROLE,
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "no",
                    "turn_id": "1",
                }
            ],
            ERROR_INVALID_MESSAGE_ORDER,
        ),
        (
            [
                {
                    "role": "user",
                    "content": "one",
                    "turn_id": "1",
                },
                {
                    "role": "assistant",
                    "content": "two",
                    "turn_id": "2",
                },
            ],
            ERROR_INVALID_MESSAGE_ORDER,
        ),
        (
            [
                {
                    "role": "user",
                    "content": "one",
                    "turn_id": "1",
                    "extra": True,
                }
            ],
            ERROR_MALFORMED_MESSAGES,
        ),
    ],
)
def test_malformed_roles_order_and_shape_have_stable_codes(
    messages: List[Dict[str, object]],
    code: str,
) -> None:
    with pytest.raises(ContextRequestError) as raised:
        parse_messages({"messages": messages})

    assert raised.value.code == code


def test_duplicate_turn_ids_are_refused() -> None:
    with pytest.raises(ContextRequestError) as raised:
        parse_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "one",
                        "turn_id": "same",
                    },
                    {
                        "role": "assistant",
                        "content": "two",
                        "turn_id": "same",
                    },
                    {
                        "role": "user",
                        "content": "three",
                        "turn_id": "last",
                    },
                ]
            }
        )

    assert raised.value.code == ERROR_MALFORMED_MESSAGES


def test_candidate_message_bound_is_enforced_before_iteration(
) -> None:
    raw = [
        {
            "role": "user",
            "content": "x",
            "turn_id": str(index),
        }
        for index in range(MESSAGE_CANDIDATES_MAX + 1)
    ]

    with pytest.raises(ContextRequestError) as raised:
        parse_messages({"messages": raw})

    assert raised.value.code == ERROR_CONTEXT_BOUNDS


def test_candidate_character_bound_is_enforced() -> None:
    with pytest.raises(ContextRequestError) as raised:
        parse_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "x" * (MESSAGE_CHARS_MAX + 1),
                        "turn_id": "one",
                    }
                ]
            }
        )

    assert raised.value.code == ERROR_CONTEXT_BOUNDS
