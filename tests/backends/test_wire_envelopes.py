"""Tests for the worker's error envelopes.

Strategy: build frames directly and read the fields. These are plain
dicts by design, so there is nothing to mock and the whole contract is
observable from the return value.

What passing proves is the property `PROTOCOL-01` names: how far a
failure reaches is a fact about the operation that failed, not about
the site that noticed it. Before this every error left as
``{"type": "error", "message": <prose>}``, so the browser had one
handler for all of them and a probe refused because a generation was
running tore down the whole What If session. The pairing test at the
bottom is the one that matters: the same refusal, for two different
requests, must not carry the same scope.

The codes exist so the client can branch without matching on prose
written for a human. They are asserted to be distinct, because a
collision would silently merge two branches.
"""

from __future__ import annotations

import pytest

from src.backends import protocol
from src.backends.protocol import (
    ERROR_BUSY,
    ERROR_GENERATION_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_SCOPE_FATAL,
    ERROR_SCOPE_REQUEST,
    ERROR_SCOPE_RUN,
    ERROR_SCOPES,
    MSG_COUNT_PROMPT,
    MSG_ERROR,
    MSG_GENERATE,
    MSG_PROBE,
    MSG_RESUME,
    MSG_SUBSTITUTE,
    MSG_TOKENIZE,
    REQUEST_SCOPES,
    request_error,
    request_id_of,
    wire_error,
)

# Everything the worker's message loop dispatches on. Kept here rather
# than imported so that adding a request type to the loop without a
# scope fails this file rather than defaulting quietly.
DISPATCHED = (
    MSG_GENERATE,
    MSG_RESUME,
    MSG_SUBSTITUTE,
    MSG_TOKENIZE,
    MSG_COUNT_PROMPT,
    MSG_PROBE,
)


# -- shape --


def test_an_error_frame_carries_all_four_fields() -> None:
    frame = wire_error(
        message="it broke",
        code=ERROR_GENERATION_FAILED,
        scope=ERROR_SCOPE_RUN,
    )

    assert frame["type"] == MSG_ERROR
    assert frame["message"] == "it broke"
    assert frame["code"] == ERROR_GENERATION_FAILED
    assert frame["scope"] == ERROR_SCOPE_RUN


def test_an_unowned_error_omits_the_request_fields() -> None:
    """Omitted rather than null, so the client's "is this mine" test
    is a presence check and cannot read a null as an id of zero."""
    frame = wire_error(
        message="gone",
        code=ERROR_GENERATION_FAILED,
        scope=ERROR_SCOPE_FATAL,
    )

    assert "request_type" not in frame
    assert "request_id" not in frame


def test_an_owned_error_carries_both() -> None:
    frame = request_error(
        message="no",
        code=ERROR_INVALID_REQUEST,
        request_type=MSG_PROBE,
        request_id=7,
    )

    assert frame["request_type"] == MSG_PROBE
    assert frame["request_id"] == 7


def test_an_unknown_scope_is_refused() -> None:
    with pytest.raises(AssertionError):
        wire_error(
            message="x", code=ERROR_BUSY, scope="somewhere_else"
        )


def test_a_frame_must_say_something() -> None:
    with pytest.raises(AssertionError):
        wire_error(
            message="", code=ERROR_BUSY, scope=ERROR_SCOPE_RUN
        )


def test_a_frame_must_carry_a_code() -> None:
    with pytest.raises(AssertionError):
        wire_error(message="x", code="", scope=ERROR_SCOPE_RUN)


# -- who owns the failure --


@pytest.mark.parametrize(
    "request_type", [MSG_GENERATE, MSG_RESUME, MSG_SUBSTITUTE]
)
def test_generation_requests_own_the_run(request_type: str) -> None:
    """These truncate the run on the client before the worker
    answers, so a failure has to roll that back."""
    frame = request_error(
        message="x",
        code=ERROR_GENERATION_FAILED,
        request_type=request_type,
    )

    assert frame["scope"] == ERROR_SCOPE_RUN


@pytest.mark.parametrize(
    "request_type", [MSG_TOKENIZE, MSG_COUNT_PROMPT, MSG_PROBE]
)
def test_auxiliary_requests_own_only_themselves(
    request_type: str,
) -> None:
    frame = request_error(
        message="x",
        code=ERROR_INVALID_REQUEST,
        request_type=request_type,
    )

    assert frame["scope"] == ERROR_SCOPE_REQUEST


def test_the_same_refusal_scopes_by_what_it_refused() -> None:
    """The finding, in one assertion. A busy worker turning away a
    second generation ends a run; turning away a probe must leave
    What If exactly as it was. One error type served both."""
    generation = request_error(
        message="busy", code=ERROR_BUSY, request_type=MSG_RESUME
    )
    probe = request_error(
        message="busy", code=ERROR_BUSY, request_type=MSG_PROBE
    )

    assert generation["scope"] == ERROR_SCOPE_RUN
    assert probe["scope"] == ERROR_SCOPE_REQUEST
    assert generation["message"] == probe["message"]


def test_an_unrecognised_request_falls_back_to_the_run() -> None:
    """The cautious reading: too much cleanup is recoverable, a
    half-applied edit left on screen is not."""
    frame = request_error(
        message="x",
        code=ERROR_INVALID_REQUEST,
        request_type="something_new",
    )

    assert frame["scope"] == ERROR_SCOPE_RUN


def test_every_dispatched_request_declares_a_scope() -> None:
    """So a new request type cannot reach the fallback by omission."""
    missing = [t for t in DISPATCHED if t not in REQUEST_SCOPES]

    assert missing == []


def test_every_declared_scope_is_a_real_one() -> None:
    for request_type, scope in REQUEST_SCOPES.items():
        assert scope in ERROR_SCOPES, request_type


# -- request ids --


def test_a_missing_request_id_reads_as_none() -> None:
    assert request_id_of({}) is None


def test_a_non_integer_request_id_reads_as_none() -> None:
    """A client that sends nonsense gets no correlation rather than a
    frame claiming an id it did not ask for."""
    assert request_id_of({"request_id": "3"}) is None


def test_a_zero_request_id_survives() -> None:
    """Zero is a legitimate id and must not be swallowed as falsey,
    which is the bug a truthiness check here would introduce."""
    assert request_id_of({"request_id": 0}) == 0


# -- the codes themselves --


def test_the_codes_are_distinct() -> None:
    """A collision would merge two client branches silently."""
    codes = [
        value
        for name, value in vars(protocol).items()
        if name.startswith("ERROR_")
        and not name.startswith("ERROR_SCOPE")
        and isinstance(value, str)
    ]

    assert len(codes) == len(set(codes))
    assert all(code for code in codes)
