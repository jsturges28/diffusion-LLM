"""Exact bounded packing for structured conversation context.

This module is dependency-light because every worker environment
imports it. It owns only protocol validation and deterministic suffix
selection. Tokenization remains the text adapter's job and arrives as
an exact counter callback.

A candidate transcript is zero or more complete user/assistant
exchanges followed by the pending user turn. Packing always keeps that
pending turn and may remove only whole exchanges from the oldest end.
There are no summaries and no partial messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

from src.backends.protocol import (
    ERROR_CONTEXT_BOUNDS,
    ERROR_INVALID_MESSAGE_ORDER,
    ERROR_INVALID_MESSAGE_ROLE,
    ERROR_MALFORMED_MESSAGES,
    PROMPT_CHARS_MAX,
)

MESSAGE_CANDIDATES_MAX = 257
MESSAGE_CHARS_MAX = PROMPT_CHARS_MAX
IDENTIFIER_CHARS_MAX = 128

assert MESSAGE_CANDIDATES_MAX % 2 == 1
assert MESSAGE_CANDIDATES_MAX > 1
assert MESSAGE_CHARS_MAX >= PROMPT_CHARS_MAX

MessageRole: TypeAlias = Literal["user", "assistant"]


class ContextRequestError(ValueError):
    """A structured-context request the protocol must refuse."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """One exact conversation turn supplied to a text adapter."""

    role: MessageRole
    content: str
    turn_id: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ContextRequestError(
                f"invalid message role: {self.role!r}",
                code=ERROR_INVALID_MESSAGE_ROLE,
            )
        if not isinstance(self.content, str):
            raise ContextRequestError(
                "message content must be a string",
                code=ERROR_MALFORMED_MESSAGES,
            )
        if not isinstance(self.turn_id, str) or not self.turn_id:
            raise ContextRequestError(
                "message turn_id must be a non-empty string",
                code=ERROR_MALFORMED_MESSAGES,
            )
        if len(self.turn_id) > IDENTIFIER_CHARS_MAX:
            raise ContextRequestError(
                "message turn_id exceeds 128 characters",
                code=ERROR_CONTEXT_BOUNDS,
            )


PromptInput: TypeAlias = Union[str, Tuple[MessageRecord, ...]]
TokenCounter: TypeAlias = Callable[[Tuple[MessageRecord, ...]], int]


@dataclass(frozen=True, slots=True)
class ConversationMetadata:
    """The durable conversation location this request belongs to."""

    conversation_id: str
    conversation_revision: int
    assistant_turn_id: str

    def __post_init__(self) -> None:
        _identifier(self.conversation_id, "conversation_id")
        _identifier(self.assistant_turn_id, "assistant_turn_id")
        _positive_int(
            self.conversation_revision, "conversation_revision"
        )

    def to_payload(self) -> Dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "conversation_revision": self.conversation_revision,
            "assistant_turn_id": self.assistant_turn_id,
        }


@dataclass(frozen=True, slots=True)
class ContextPackManifest:
    """The immutable account of one exact suffix decision."""

    included_turn_ids: Tuple[str, ...]
    first_included_index: int
    omitted_turn_count: int
    prompt_token_count: int
    output_reserve: int
    requested_total_budget: int
    effective_total_budget: int

    def __post_init__(self) -> None:
        assert self.included_turn_ids, (
            "a pack includes the pending turn"
        )
        assert self.first_included_index == self.omitted_turn_count
        assert self.omitted_turn_count % 2 == 0
        assert self.prompt_token_count > 0
        assert self.output_reserve > 0
        assert (
            self.prompt_token_count + self.output_reserve
            <= self.effective_total_budget
        )
        assert (
            self.effective_total_budget
            <= self.requested_total_budget
        )

    def to_payload(self) -> Dict[str, object]:
        return {
            "included_turn_ids": list(self.included_turn_ids),
            "first_included_index": self.first_included_index,
            "omitted_turn_count": self.omitted_turn_count,
            "prompt_token_count": self.prompt_token_count,
            "output_reserve": self.output_reserve,
            "requested_total_budget": (
                self.requested_total_budget
            ),
            "effective_total_budget": self.effective_total_budget,
        }


@dataclass(frozen=True, slots=True)
class PackedContext:
    """The selected messages, manifest, and optional durable owner."""

    messages: Tuple[MessageRecord, ...]
    manifest: ContextPackManifest
    conversation: Optional[ConversationMetadata]

    def __post_init__(self) -> None:
        assert self.messages, "a pack includes the pending turn"
        ids = tuple(message.turn_id for message in self.messages)
        assert ids == self.manifest.included_turn_ids
        assert self.messages[-1].role == "user"

    def attestation(self) -> Dict[str, object]:
        payload = self.manifest.to_payload()
        if self.conversation is not None:
            payload["conversation"] = self.conversation.to_payload()
        return payload


def parse_messages(
    data: Mapping[str, object],
) -> Tuple[
    Tuple[MessageRecord, ...],
    Optional[ConversationMetadata],
]:
    """Parse and strictly validate a wire transcript.

    Message objects accept exactly ``role``, ``content`` and
    ``turn_id``. Conversation metadata is all-or-none at the request
    top level so an attestation can never carry a partial owner.
    """
    raw = data.get("messages")
    if not isinstance(raw, list):
        raise ContextRequestError(
            "messages must be a list",
            code=ERROR_MALFORMED_MESSAGES,
        )
    if not raw:
        raise ContextRequestError(
            "messages must contain a pending user turn",
            code=ERROR_INVALID_MESSAGE_ORDER,
        )
    if len(raw) > MESSAGE_CANDIDATES_MAX:
        raise ContextRequestError(
            f"messages holds {len(raw):,} turns; the limit is"
            f" {MESSAGE_CANDIDATES_MAX:,}",
            code=ERROR_CONTEXT_BOUNDS,
        )
    messages = _parse_message_records(raw)
    _validate_message_sequence(messages)
    conversation = _parse_conversation(data, messages)
    return messages, conversation


def pack_context(
    messages: Tuple[MessageRecord, ...],
    *,
    count_tokens: TokenCounter,
    output_reserve: int,
    policy_default_tokens: int,
    policy_max_tokens: int,
    requested_total_budget: Optional[int] = None,
    checkpoint_window: Optional[int] = None,
    conversation: Optional[ConversationMetadata] = None,
) -> PackedContext:
    """Select the longest exact suffix that fits the total budget.

    Every possible whole-exchange suffix is counted at most once.
    The candidate list has a hard maximum, so this exhaustive search
    is bounded while avoiding an unsafe monotonic-token-count
    assumption at raw completion boundaries.
    """
    _validate_message_sequence(messages)
    _validate_message_chars(messages)
    reserve = _positive_int(output_reserve, "output_reserve")
    default = _positive_int(
        policy_default_tokens, "policy default"
    )
    maximum = _positive_int(policy_max_tokens, "policy maximum")
    if default > maximum:
        raise ContextRequestError(
            "context policy default exceeds its maximum",
            code=ERROR_CONTEXT_BOUNDS,
        )
    requested = _requested_budget(
        requested_total_budget,
        default=default,
        maximum=maximum,
    )
    effective = _effective_budget(requested, checkpoint_window)

    pending_index = len(messages) - 1
    pending = messages[pending_index:]
    pending_count = _count_candidate(count_tokens, pending)
    if pending_count + reserve > effective:
        raise ContextRequestError(
            "The pending user turn needs"
            f" {pending_count + reserve:,} tokens including the"
            f" {reserve:,}-token output reserve, but the effective"
            f" context budget is {effective:,}. Shorten it or lower"
            " the output reserve.",
            code=ERROR_CONTEXT_BOUNDS,
        )

    first, prompt_count = _select_suffix(
        messages,
        count_tokens=count_tokens,
        reserve=reserve,
        effective=effective,
        pending_count=pending_count,
    )

    included = messages[first:]
    manifest = ContextPackManifest(
        included_turn_ids=tuple(
            message.turn_id for message in included
        ),
        first_included_index=first,
        omitted_turn_count=first,
        prompt_token_count=prompt_count,
        output_reserve=reserve,
        requested_total_budget=requested,
        effective_total_budget=effective,
    )
    assert manifest.omitted_turn_count % 2 == 0
    assert manifest.included_turn_ids[-1] == messages[-1].turn_id
    return PackedContext(
        messages=included,
        manifest=manifest,
        conversation=conversation,
    )


def _select_suffix(
    messages: Tuple[MessageRecord, ...],
    *,
    count_tokens: TokenCounter,
    reserve: int,
    effective: int,
    pending_count: int,
) -> Tuple[int, int]:
    """The earliest fitting whole-exchange suffix and its count."""
    pending_index = len(messages) - 1
    if pending_index == 0:
        return pending_index, pending_count
    full_count = _count_candidate(count_tokens, messages)
    if full_count + reserve <= effective:
        return 0, full_count

    first = pending_index
    prompt_count = pending_count
    # Examine every bounded middle suffix because tokenization across
    # a raw completion boundary is exact but not formally monotonic.
    for start in range(pending_index - 2, 0, -2):
        candidate_count = _count_candidate(
            count_tokens, messages[start:]
        )
        if candidate_count + reserve <= effective:
            first = start
            prompt_count = candidate_count
    return first, prompt_count


def _parse_message_records(
    raw: List[object],
) -> Tuple[MessageRecord, ...]:
    records: List[MessageRecord] = []
    chars = 0
    expected = {"role", "content", "turn_id"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ContextRequestError(
                f"messages[{index}] must be an object",
                code=ERROR_MALFORMED_MESSAGES,
            )
        if set(item) != expected:
            raise ContextRequestError(
                f"messages[{index}] must contain exactly role,"
                " content and turn_id",
                code=ERROR_MALFORMED_MESSAGES,
            )
        role = item["role"]
        if role not in ("user", "assistant"):
            raise ContextRequestError(
                f"invalid message role at index {index}: {role!r}",
                code=ERROR_INVALID_MESSAGE_ROLE,
            )
        content = item["content"]
        turn_id = item["turn_id"]
        record = MessageRecord(
            role=role,
            content=content,
            turn_id=turn_id,
        )
        chars += len(record.content)
        if chars > MESSAGE_CHARS_MAX:
            raise ContextRequestError(
                f"messages exceed {MESSAGE_CHARS_MAX:,} characters",
                code=ERROR_CONTEXT_BOUNDS,
            )
        records.append(record)
    return tuple(records)


def _validate_message_sequence(
    messages: Tuple[MessageRecord, ...],
) -> None:
    if not messages:
        raise ContextRequestError(
            "messages must contain a pending user turn",
            code=ERROR_INVALID_MESSAGE_ORDER,
        )
    if len(messages) > MESSAGE_CANDIDATES_MAX:
        raise ContextRequestError(
            f"messages exceeds {MESSAGE_CANDIDATES_MAX:,} turns",
            code=ERROR_CONTEXT_BOUNDS,
        )
    if len(messages) % 2 == 0:
        raise ContextRequestError(
            "messages must end in a pending user turn",
            code=ERROR_INVALID_MESSAGE_ORDER,
        )
    seen: set[str] = set()
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise ContextRequestError(
                f"messages[{index}] must have role {expected},"
                f" not {message.role}",
                code=ERROR_INVALID_MESSAGE_ORDER,
            )
        if message.turn_id in seen:
            raise ContextRequestError(
                f"duplicate turn_id: {message.turn_id}",
                code=ERROR_MALFORMED_MESSAGES,
            )
        seen.add(message.turn_id)
    if messages[-1].content.strip() == "":
        raise ContextRequestError(
            "the pending user turn must not be blank",
            code=ERROR_MALFORMED_MESSAGES,
        )


def _validate_message_chars(
    messages: Tuple[MessageRecord, ...],
) -> None:
    chars = 0
    for message in messages:
        chars += len(message.content)
        if chars > MESSAGE_CHARS_MAX:
            raise ContextRequestError(
                f"messages exceed {MESSAGE_CHARS_MAX:,} characters",
                code=ERROR_CONTEXT_BOUNDS,
            )


def _parse_conversation(
    data: Mapping[str, object],
    messages: Tuple[MessageRecord, ...],
) -> Optional[ConversationMetadata]:
    keys = (
        "conversation_id",
        "conversation_revision",
        "assistant_turn_id",
    )
    present = tuple(key in data for key in keys)
    if not any(present):
        return None
    if not all(present):
        raise ContextRequestError(
            "conversation metadata requires conversation_id,"
            " conversation_revision and assistant_turn_id",
            code=ERROR_MALFORMED_MESSAGES,
        )
    conversation_id = _identifier(
        data["conversation_id"], "conversation_id"
    )
    assistant_turn_id = _identifier(
        data["assistant_turn_id"], "assistant_turn_id"
    )
    revision = data["conversation_revision"]
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise ContextRequestError(
            "conversation_revision must be a positive integer",
            code=ERROR_MALFORMED_MESSAGES,
        )
    if assistant_turn_id in {
        message.turn_id for message in messages
    }:
        raise ContextRequestError(
            "assistant_turn_id must name the reserved next turn",
            code=ERROR_MALFORMED_MESSAGES,
        )
    return ConversationMetadata(
        conversation_id=conversation_id,
        conversation_revision=revision,
        assistant_turn_id=assistant_turn_id,
    )


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContextRequestError(
            f"{name} must be a non-empty string",
            code=ERROR_MALFORMED_MESSAGES,
        )
    if len(value) > IDENTIFIER_CHARS_MAX:
        raise ContextRequestError(
            f"{name} exceeds {IDENTIFIER_CHARS_MAX} characters",
            code=ERROR_CONTEXT_BOUNDS,
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextRequestError(
            f"{name} must be a positive integer",
            code=ERROR_CONTEXT_BOUNDS,
        )
    if value < 1:
        raise ContextRequestError(
            f"{name} must be at least 1",
            code=ERROR_CONTEXT_BOUNDS,
        )
    return value


def _requested_budget(
    requested: Optional[int],
    *,
    default: int,
    maximum: int,
) -> int:
    if requested is None:
        return default
    value = _positive_int(requested, "context_budget")
    if value > maximum:
        raise ContextRequestError(
            f"context_budget is {value:,}; this model and device"
            f" allow at most {maximum:,}",
            code=ERROR_CONTEXT_BOUNDS,
        )
    return value


def _effective_budget(
    requested: int,
    checkpoint_window: Optional[int],
) -> int:
    if checkpoint_window is None:
        return requested
    window = _positive_int(
        checkpoint_window, "checkpoint context window"
    )
    return min(requested, window)


def _count_candidate(
    count_tokens: TokenCounter,
    candidate: Tuple[MessageRecord, ...],
) -> int:
    assert candidate, "a suffix always keeps the pending user"
    count = count_tokens(candidate)
    assert isinstance(count, int), "a token count is an integer"
    assert not isinstance(count, bool), "a token count is not a flag"
    assert count > 0, "a non-empty prompt must produce tokens"
    return count
