"""Tests for the context-window readout: ceiling and prompt count.

Strategy: both halves are functions over stub model, config, and
tokenizer objects, so these run with no checkpoint and no GPU. They
cover ``describe_context_length`` (the /health figure the supervisor
caches and the UI divides by), ``Backend.prompt_token_count`` (the
templated count), ``handle_count_prompt`` end to end against a stub
WebSocket, and ``build_llada_inputs`` (the one encode LLaDA
generation, resume, and counting all share).

Passing proves the readout can only ever quote a measured number.
The ceiling comes off the loaded config and rejects the ``int(1e30)``
sentinel a tokenizer supplies when a checkpoint declares no length,
returning None rather than a guess. The count is of the templated
sequence rather than the user's characters, so it cannot understate
what reaches the model. And LLaDA counts through the same function
its generator encodes with, so the number is the run's, not a
lookalike derived beside it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.backends.worker_base import (
    CONTEXT_LENGTH_SANE_MAX,
    COUNT_PROMPT_MAX_CHARS,
    Backend,
    describe_context_length,
)
from src.inference.streaming_sampler import build_llada_inputs

# What transformers hands back for a checkpoint that declares no
# maximum length. The reason the fallback needs a bound at all.
_UNSPECIFIED_LENGTH = int(1e30)


class _StubWebSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


class _StubConfig:
    def __init__(self, window: Any) -> None:
        self.max_position_embeddings = window


class _StubModel:
    def __init__(self, window: Any) -> None:
        self.config = _StubConfig(window)


class _LengthTokenizer:
    """Only carries the tokenizer-side length convention."""

    def __init__(self, length: Any) -> None:
        self.model_max_length = length


class _TemplateTokenizer:
    """Wraps a prompt in role markers, the way a chat template does.

    The wrapper is deliberately several tokens wide, because the fact
    under test is that a count of the raw prompt would understate the
    templated sequence. Thinking mode adds one more marker, mirroring
    how ``enable_thinking`` changes the template it selects.
    """

    # Tokens the template contributes around the user's words.
    MARKERS_PLAIN = 4
    MARKERS_THINKING = 5

    def __init__(self) -> None:
        self.thinking_seen: List[bool] = []

    def apply_chat_template(
        self,
        chat: List[Dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_tensors: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> Any:
        assert add_generation_prompt is True, (
            "counting must template for a reply, as a run does"
        )
        self.thinking_seen.append(enable_thinking)
        words = chat[0]["content"].split()
        markers = (
            self.MARKERS_THINKING
            if enable_thinking
            else self.MARKERS_PLAIN
        )
        ids = list(range(len(words) + markers))
        if not tokenize:
            return " ".join(words)
        return {"input_ids": _FakeTensor(ids)}


class _FakeTensor:
    """Just the ``.shape[-1]`` the count reads."""

    def __init__(self, ids: List[int]) -> None:
        self.shape = (1, len(ids))


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


# -- describe_context_length --


def test_the_ceiling_comes_from_the_model_config() -> None:
    window = describe_context_length(_StubModel(65_536))
    assert window == 65_536


def test_the_config_wins_over_the_tokenizer() -> None:
    """The config is the architectural fact; the other is a
    convention, and they disagree often enough to matter."""
    window = describe_context_length(
        _StubModel(65_536), _LengthTokenizer(2_048)
    )
    assert window == 65_536


def test_the_tokenizer_answers_when_the_config_cannot() -> None:
    window = describe_context_length(
        _StubModel(None), _LengthTokenizer(4_096)
    )
    assert window == 4_096


def test_the_unspecified_sentinel_is_not_a_ceiling() -> None:
    """The case the bound exists for: transformers reports int(1e30)
    for a checkpoint that declares nothing, and dividing a prompt by
    it would claim any prompt fits."""
    window = describe_context_length(
        _StubModel(None),
        _LengthTokenizer(_UNSPECIFIED_LENGTH),
    )
    assert window is None


def test_a_length_at_the_bound_is_still_accepted() -> None:
    """The boundary itself: the bound separates sentinel from
    measurement, so the largest sane value has to pass."""
    window = describe_context_length(
        _StubModel(None),
        _LengthTokenizer(CONTEXT_LENGTH_SANE_MAX),
    )
    assert window == CONTEXT_LENGTH_SANE_MAX


def test_one_past_the_bound_is_rejected() -> None:
    window = describe_context_length(
        _StubModel(None),
        _LengthTokenizer(CONTEXT_LENGTH_SANE_MAX + 1),
    )
    assert window is None


def test_a_nonpositive_length_is_rejected() -> None:
    assert describe_context_length(_StubModel(0)) is None
    assert describe_context_length(_StubModel(-1)) is None


def test_a_boolean_is_not_a_length() -> None:
    """bool is an int in Python, and True would otherwise read as a
    one-token context window."""
    assert describe_context_length(_StubModel(True)) is None


def test_nothing_is_reported_before_a_load() -> None:
    assert describe_context_length(None) is None


def test_a_configless_model_falls_through() -> None:
    assert describe_context_length(object()) is None


# -- prompt_token_count --


def test_the_count_includes_the_templates_markers() -> None:
    """The reason this runs on the worker: the user typed three
    words, and the model will see the template around them."""
    tokenizer = _TemplateTokenizer()
    backend = _StubBackend(tokenizer)

    count = backend.prompt_token_count("she ran home")

    assert count == 3 + _TemplateTokenizer.MARKERS_PLAIN


def test_thinking_mode_changes_the_count() -> None:
    """``enable_thinking`` selects a different template, so the
    count has to be taken under the flag the run will use."""
    backend = _StubBackend(_TemplateTokenizer())

    plain = backend.prompt_token_count("she ran", thinking=False)
    thinking = backend.prompt_token_count(
        "she ran", thinking=True
    )

    assert thinking > plain


def test_an_empty_prompt_counts_zero() -> None:
    """The boundary. Nothing typed means nothing to report, not the
    template's own overhead, which no run would build."""
    tokenizer = _TemplateTokenizer()
    backend = _StubBackend(tokenizer)

    assert backend.prompt_token_count("") == 0
    assert tokenizer.thinking_seen == []


# -- handle_count_prompt --


def _count(
    backend: Backend, payload: Dict[str, Any]
) -> _StubWebSocket:
    ws = _StubWebSocket()
    asyncio.run(
        backend.handle_count_prompt(  # type: ignore[arg-type]
            ws, payload
        )
    )
    return ws


def test_the_reply_echoes_the_request_id() -> None:
    """What lets the client drop an answer a keystroke outran."""
    backend = _StubBackend(_TemplateTokenizer())

    ws = _count(backend, {"text": "she ran", "request_id": 9})

    assert len(ws.sent) == 1
    reply = ws.sent[0]
    assert reply["type"] == "count_prompt_result"
    assert reply["request_id"] == 9
    assert reply["count"] == 2 + _TemplateTokenizer.MARKERS_PLAIN
    assert reply["truncated"] is False


def test_the_reply_carries_no_per_token_pieces() -> None:
    """The whole reason this is not a flag on tokenize: an imported
    file must cost one integer, not one object per token."""
    backend = _StubBackend(_TemplateTokenizer())

    ws = _count(backend, {"text": "she ran", "request_id": 1})

    assert "pieces" not in ws.sent[0]


def test_an_oversized_prompt_is_truncated_and_says_so() -> None:
    """Bounded work, and the flag lets the readout present the
    count as a floor rather than as the answer."""
    backend = _StubBackend(_TemplateTokenizer())
    oversized = "a " * (COUNT_PROMPT_MAX_CHARS)

    ws = _count(backend, {"text": oversized, "request_id": 1})

    reply = ws.sent[0]
    assert reply["chars"] == COUNT_PROMPT_MAX_CHARS
    assert reply["truncated"] is True


def test_counting_without_a_tokenizer_reports_an_error() -> None:
    backend = _StubBackend(None)

    ws = _count(backend, {"text": "she ran", "request_id": 1})

    assert ws.sent[0]["type"] == "error"
    assert "No tokenizer" in ws.sent[0]["message"]


# -- build_llada_inputs --


class _LladaTokenizer:
    """LLaDA's two-step shape: template to text, then encode it."""

    def __init__(self) -> None:
        self.special_tokens_seen: List[bool] = []

    def apply_chat_template(
        self,
        chat: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> str:
        assert tokenize is False, "LLaDA encodes separately"
        assert add_generation_prompt is True
        return "<|start|> " + chat[0]["content"] + " <|end|>"

    def __call__(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        import torch

        self.special_tokens_seen.append(add_special_tokens)
        count = len(texts[0].split())
        ids = torch.arange(count).unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def test_the_llada_encode_returns_ids_and_a_mask() -> None:
    """The canvas is built from both, so both must come back."""
    encoded = build_llada_inputs(_LladaTokenizer(), "she ran")

    assert encoded["input_ids"].shape == (1, 4)
    assert encoded["attention_mask"].shape == (1, 4)


def test_the_llada_encode_adds_no_second_bos() -> None:
    """The template already placed every special token; another
    would shift every position in the canvas by one."""
    tokenizer = _LladaTokenizer()

    build_llada_inputs(tokenizer, "she ran")

    assert tokenizer.special_tokens_seen == [False]
