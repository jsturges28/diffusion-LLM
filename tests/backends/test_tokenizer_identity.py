"""Tests for the tokenizer readout and the typed-token preview.

Strategy: both pieces are pure functions over a tokenizer-shaped
object, so these drive them with stubs, no checkpoint and no GPU.
They cover ``describe_tokenizer`` (the /health payload the supervisor
caches and every saved run keeps), ``tokenize_pieces`` (the split the
What If preview shows), and ``Backend.handle_tokenize`` end to end
against a stub WebSocket.

Passing proves three things. The identity is read off the live object,
so it cannot drift from the checkpoint that actually loaded. It
degrades field by field rather than failing, so an unfamiliar
tokenizer costs a row in the UI and not a health check. And the
preview echoes the request id and the text it answered, which is what
lets the client throw away a reply that a later keystroke outran.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from src.backends.worker_base import (
    TOKENIZE_TEXT_MAX_CHARS,
    Backend,
    describe_tokenizer,
    tokenize_pieces,
)

_VOCAB: Dict[str, int] = {"she": 7, " ran": 11}
_REVERSE: Dict[int, str] = {
    value: key for key, value in _VOCAB.items()
}


class _StubTokenizer:
    """Words in the vocabulary are one entry; anything else splits."""

    is_fast = True
    name_or_path = "stub/checkpoint"
    vocab_size = 128

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> List[int]:
        assert add_special_tokens is False, (
            "a spliced fragment must not gain a BOS"
        )
        if text in _VOCAB:
            return [_VOCAB[text]]
        return [ord(char) % 128 for char in text]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        return "".join(
            _REVERSE.get(int(i), chr(int(i))) for i in ids
        )


class _BareTokenizer:
    """Nothing but a class name, the worst case worth surviving."""


class _StubConfig:
    """A padded embedding: wider than the tokenizer that feeds it."""

    vocab_size = 256


class _StubModel:
    config = _StubConfig()


class _StubWebSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


class _StubBackend(Backend):
    """Only the tokenizer matters; the rest is contract filler."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def load(self, *, device: str = "cuda") -> None:
        raise NotImplementedError

    async def handle_generate(
        self, ws: Any, data: Any, cancel_event: Any, stream: Any
    ) -> None:
        raise NotImplementedError


# -- describe_tokenizer --


def test_describe_reports_every_field_it_can_read() -> None:
    described = describe_tokenizer(_StubTokenizer())
    assert described["class"] == "_StubTokenizer"
    assert described["name_or_path"] == "stub/checkpoint"
    assert described["vocab_size"] == 128
    assert described["is_fast"] is True


def test_describe_degrades_to_the_class_alone() -> None:
    """An unfamiliar tokenizer costs a UI row, not a health check."""
    described = describe_tokenizer(_BareTokenizer())
    assert described == {"class": "_BareTokenizer"}


def test_describe_returns_nothing_before_a_load() -> None:
    assert describe_tokenizer(None) == {}


def test_describe_reports_the_models_wider_output() -> None:
    """The pair is the point: a rank is measured against the wider
    figure, and they differ because the embedding is padded."""
    described = describe_tokenizer(_StubTokenizer(), _StubModel())
    assert described["vocab_size"] == 128
    assert described["model_vocab_size"] == 256


def test_describe_omits_the_width_without_a_model() -> None:
    """The health payload is built before the weights are, so the
    field has to be absent rather than zero or guessed."""
    described = describe_tokenizer(_StubTokenizer())
    assert "model_vocab_size" not in described


def test_describe_omits_the_width_from_a_configless_model() -> None:
    described = describe_tokenizer(_StubTokenizer(), object())
    assert "model_vocab_size" not in described


# -- tokenize_pieces --


def test_a_vocabulary_word_is_one_piece() -> None:
    pieces = tokenize_pieces(_StubTokenizer(), "she")
    assert len(pieces) == 1
    assert pieces[0]["id"] == 7
    assert pieces[0]["t"] == "she"


def test_an_unknown_word_splits_into_several() -> None:
    pieces = tokenize_pieces(_StubTokenizer(), "hex")
    assert len(pieces) == 3
    assert [p["t"] for p in pieces] == ["h", "e", "x"]


def test_empty_text_resolves_to_nothing() -> None:
    """The boundary: an empty field is not zero-length text."""
    assert tokenize_pieces(_StubTokenizer(), "") == []


# -- handle_tokenize --


def _tokenize(
    backend: Backend, payload: Dict[str, Any]
) -> _StubWebSocket:
    ws = _StubWebSocket()
    asyncio.run(
        backend.handle_tokenize(ws, payload)  # type: ignore[arg-type]
    )
    return ws


def test_preview_echoes_the_request_id_and_the_text() -> None:
    """Both are what let the client discard a stale reply."""
    backend = _StubBackend(_StubTokenizer())

    ws = _tokenize(
        backend, {"text": " ran", "request_id": 4}
    )

    assert len(ws.sent) == 1
    reply = ws.sent[0]
    assert reply["type"] == "tokenize_result"
    assert reply["request_id"] == 4
    assert reply["text"] == " ran"
    assert reply["count"] == 1
    assert reply["pieces"][0]["id"] == 11


def test_preview_truncates_an_oversized_request() -> None:
    """Bounded work, and the echoed text says where it was cut."""
    backend = _StubBackend(_StubTokenizer())
    oversized = "a" * (TOKENIZE_TEXT_MAX_CHARS + 50)

    ws = _tokenize(
        backend, {"text": oversized, "request_id": 1}
    )

    reply = ws.sent[0]
    assert len(reply["text"]) == TOKENIZE_TEXT_MAX_CHARS
    assert reply["count"] == TOKENIZE_TEXT_MAX_CHARS


def test_preview_without_a_tokenizer_reports_an_error() -> None:
    backend = _StubBackend(None)

    ws = _tokenize(backend, {"text": "she", "request_id": 1})

    assert ws.sent[0]["type"] == "error"
    assert "No tokenizer" in ws.sent[0]["message"]
