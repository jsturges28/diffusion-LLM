"""Dependency-light durable conversation identity primitives.

Every worker environment imports this module. It therefore uses only
the standard library and raises ``ValueError`` so each protocol or
storage boundary can translate failures into its own error shape.
"""

from __future__ import annotations

import hashlib
import re


CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"
BRANCH_ID_PATTERN = r"^b_[0-9a-f]{32}$"
OPERATION_ID_PATTERN = r"^[0-9a-f]{32}$"
LEGACY_TURN_ID_PATTERN = r"^[0-9]{8}$"
OPAQUE_TURN_ID_PATTERN = (
    r"^t_([0-9a-f]{32})_([0-9]{8})_([0-9a-f]{16})$"
)
TURN_ID_PATTERN = (
    r"^(?:[0-9]{8}|t_[0-9a-f]{32}_[0-9]{8}_"
    r"[0-9a-f]{16})$"
)

CONVERSATION_ID_RE = re.compile(CONVERSATION_ID_PATTERN)
BRANCH_ID_RE = re.compile(BRANCH_ID_PATTERN)
OPERATION_ID_RE = re.compile(OPERATION_ID_PATTERN)
LEGACY_TURN_ID_RE = re.compile(LEGACY_TURN_ID_PATTERN)
OPAQUE_TURN_ID_RE = re.compile(OPAQUE_TURN_ID_PATTERN)

TURN_ID_WIDTH = 8
TURN_INDEX_MAX = 1_000_000

assert TURN_INDEX_MAX < 10**TURN_ID_WIDTH


def validate_conversation_id(value: object) -> str:
    """Return one canonical conversation id."""
    if not isinstance(value, str):
        raise ValueError("conversation id must be a string")
    if CONVERSATION_ID_RE.fullmatch(value) is None:
        raise ValueError(f"invalid conversation id: {value}")
    return value


def validate_branch_id(value: object) -> str:
    """Return one canonical branch id."""
    if not isinstance(value, str):
        raise ValueError("branch id must be a string")
    if BRANCH_ID_RE.fullmatch(value) is None:
        raise ValueError(f"invalid branch id: {value}")
    return value


def validate_operation_id(value: object) -> str:
    """Return one canonical fork operation id."""
    if not isinstance(value, str):
        raise ValueError("operation id must be a string")
    if OPERATION_ID_RE.fullmatch(value) is None:
        raise ValueError(f"invalid operation id: {value}")
    return value


def legacy_branch_id(conversation_id: object) -> str:
    """Derive schema-v1's stable synthetic branch identity."""
    validated = validate_conversation_id(conversation_id)
    return f"b_{validated}"


def validate_turn_index(
    value: object,
    *,
    name: str = "turn index",
) -> int:
    """Return one bounded integer turn index without coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= TURN_INDEX_MAX:
        raise ValueError(
            f"{name} must be between 1 and {TURN_INDEX_MAX}"
        )
    return value


def legacy_turn_id(index: object) -> str:
    """Encode one schema-v1 absolute turn index."""
    validated = validate_turn_index(index)
    return f"{validated:0{TURN_ID_WIDTH}d}"


def legacy_turn_index(turn_id: object) -> int:
    """Decode one canonical schema-v1 numeric turn id."""
    if not isinstance(turn_id, str):
        raise ValueError("turn id must be a string")
    if LEGACY_TURN_ID_RE.fullmatch(turn_id) is None:
        raise ValueError(f"invalid turn id: {turn_id}")
    index = validate_turn_index(int(turn_id))
    if legacy_turn_id(index) != turn_id:
        raise ValueError(f"invalid turn id: {turn_id}")
    return index


def is_legacy_turn_id(turn_id: object) -> bool:
    """Whether a value has schema-v1's canonical numeric shape."""
    if not isinstance(turn_id, str):
        return False
    return LEGACY_TURN_ID_RE.fullmatch(turn_id) is not None


def opaque_turn_id(branch_id: object, local_slot: object) -> str:
    """Create one checksummed schema-v2 branch-local turn id."""
    branch = validate_branch_id(branch_id)
    slot = validate_turn_index(local_slot, name="local turn slot")
    source = f"{branch}:{slot}".encode("ascii")
    digest = hashlib.blake2s(source, digest_size=8).hexdigest()
    return f"t_{branch[2:]}_{slot:0{TURN_ID_WIDTH}d}_{digest}"


def opaque_turn_parts(turn_id: object) -> tuple[str, int]:
    """Validate and decode one checksummed schema-v2 turn id."""
    if not isinstance(turn_id, str):
        raise ValueError("turn id must be a string")
    matched = OPAQUE_TURN_ID_RE.fullmatch(turn_id)
    if matched is None:
        raise ValueError(f"invalid turn id: {turn_id}")
    branch_id = validate_branch_id(f"b_{matched.group(1)}")
    local_slot = validate_turn_index(
        int(matched.group(2)),
        name="local turn slot",
    )
    if opaque_turn_id(branch_id, local_slot) != turn_id:
        raise ValueError(f"invalid turn id: {turn_id}")
    return branch_id, local_slot


def validate_any_turn_id(turn_id: object) -> str:
    """Return one canonical schema-v1 or schema-v2 turn id."""
    if is_legacy_turn_id(turn_id):
        legacy_turn_index(turn_id)
    else:
        opaque_turn_parts(turn_id)
    assert isinstance(turn_id, str)
    return turn_id


def validate_assistant_turn_identity(
    *,
    conversation_id: object,
    branch_id: object,
    assistant_turn_id: object,
    assistant_turn_index: object,
) -> None:
    """Validate one reserved assistant's complete durable identity."""
    validate_conversation_id(conversation_id)
    branch = validate_branch_id(branch_id)
    index = validate_turn_index(
        assistant_turn_index,
        name="assistant_turn_index",
    )
    if is_legacy_turn_id(assistant_turn_id):
        numeric_index = legacy_turn_index(assistant_turn_id)
        if numeric_index != index:
            raise ValueError(
                "numeric assistant_turn_id must equal"
                " assistant_turn_index"
            )
        return
    owner_branch, _local_slot = opaque_turn_parts(assistant_turn_id)
    if owner_branch != branch:
        raise ValueError(
            "opaque assistant_turn_id belongs to another branch"
        )
