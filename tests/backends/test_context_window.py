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
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.backends.context_pack import MessageRecord
from src.backends.protocol import (
    ERROR_INVALID_MESSAGE_ORDER,
    ContextPolicy,
    PROMPT_CHARS_MAX,
)
from src.backends.text_adapter import ChatTextAdapter
from src.backends.worker_base import (
    CONTEXT_LENGTH_SANE_MAX,
    COUNT_PROMPT_MAX_CHARS,
    Backend,
    describe_context_length,
)
from src.inference.streaming_sampler import (
    LLADA_TEXT,
    build_llada_inputs,
    build_llada_message_inputs,
)

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
        words = [
            word
            for message in chat
            for word in message["content"].split()
        ]
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
    """Only the tokenizer, adapter and model matter."""

    def __init__(
        self,
        tokenizer: Any,
        adapter: Any = None,
        model: Any = None,
    ) -> None:
        self.tokenizer = tokenizer
        # A bare chat adapter unless a test names one: the counting
        # under test is the template's, and no model's control tokens
        # or channel convention reaches it.
        self.text_adapter = adapter or ChatTextAdapter()
        # None until a test gives one, which is also what an unloaded
        # backend looks like: the refusal has nothing to read then and
        # must not invent a ceiling.
        self.model = model
        self.effective_device = "cuda"
        self.model_info = SimpleNamespace(
            capabilities=SimpleNamespace(
                context_policy=ContextPolicy(
                    status="provisional",
                    default_tokens=4096,
                    max_tokens=8192,
                )
            )
        )

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


def _message_payload() -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": "first question",
                "turn_id": "00000001",
            },
            {
                "role": "assistant",
                "content": "first answer",
                "turn_id": "00000002",
            },
            {
                "role": "user",
                "content": "next question",
                "turn_id": "00000003",
            },
        ],
        "conversation_id": "a" * 32,
        "branch_id": "b_" + "b" * 32,
        "branch_revision": 3,
        "assistant_turn_id": "00000004",
        "assistant_turn_index": 4,
        "context_budget": 100,
        "output_reserve": 20,
        "request_id": 4,
    }


def test_count_and_generation_use_the_same_context_pack() -> None:
    """The count reply is the generation decision, not an estimate."""
    backend = _StubBackend(
        _TemplateTokenizer(), model=_StubModel(80)
    )
    payload = _message_payload()

    counted = _count(backend, payload).sent[0]
    prepared = backend.prepare_generation_prompt(
        payload,
        output_reserve=20,
        thinking=False,
    )

    assert counted["count"] == (
        prepared.context_pack["prompt_token_count"]
    )
    assert counted["context_pack"] == prepared.context_pack
    assert counted["context_pack"]["effective_total_budget"] == 80


def test_a_malformed_message_count_is_request_scoped() -> None:
    backend = _StubBackend(_TemplateTokenizer())
    payload = _message_payload()
    payload["messages"] = payload["messages"][:-1]

    reply = _count(backend, payload).sent[0]

    assert reply["type"] == "error"
    assert reply["code"] == ERROR_INVALID_MESSAGE_ORDER
    assert reply["scope"] == "request"


# -- build_llada_inputs --


class _LladaTokenizer:
    """LLaDA's two-step shape: template to text, then encode it."""

    def __init__(self) -> None:
        self.special_tokens_seen: List[bool] = []
        self.chats_seen: List[List[Dict[str, str]]] = []

    def apply_chat_template(
        self,
        chat: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> str:
        assert tokenize is False, "LLaDA encodes separately"
        assert add_generation_prompt is True
        self.chats_seen.append([dict(message) for message in chat])
        content = " ".join(
            message["content"] for message in chat
        )
        return "<|start|> " + content + " <|end|>"

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


def test_llada_messages_keep_the_two_step_encode() -> None:
    """Multiple roles template to text, then tokenize once."""
    tokenizer = _LladaTokenizer()
    messages = (
        MessageRecord("user", "first question", "1"),
        MessageRecord("assistant", "first answer", "2"),
        MessageRecord("user", "next question", "3"),
    )

    encoded = build_llada_message_inputs(tokenizer, messages)
    count = LLADA_TEXT.count_message_tokens(
        tokenizer, messages, thinking=False
    )

    assert count == encoded["input_ids"].shape[-1]
    assert tokenizer.chats_seen[0] == [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    assert tokenizer.special_tokens_seen == [False, False]


# -- refusing a prompt that cannot run --
#
# The readout above tells the user; this turns it into something the
# worker enforces. Only the case that cannot run at all: a prompt
# already past the window. One that fits but leaves no room for the
# whole output budget still runs and gets truncated, which the browser
# says beside the counter, and which somebody may have asked for.


def _sized_backend(window: Any) -> _StubBackend:
    """A backend whose config declares ``window`` tokens."""
    return _StubBackend(
        _TemplateTokenizer(), model=_StubModel(window)
    )


def test_a_prompt_inside_the_window_is_allowed() -> None:
    """Four words plus four markers is eight, so a window of eight is
    the largest prompt that fits, not the first that does not."""
    _sized_backend(8).check_prompt_fits("she ran home today")


def test_a_prompt_exactly_at_the_window_is_allowed() -> None:
    """The boundary, stated from the permissive side. Off by one here
    refuses a prompt the model would have accepted."""
    _sized_backend(8).check_prompt_fits("she ran home today")


def test_a_prompt_one_past_the_window_is_refused() -> None:
    """And from the other side, so a comparison flipped either way
    fails one of the two."""
    with pytest.raises(ValueError, match="window"):
        _sized_backend(7).check_prompt_fits("she ran home today")


def test_the_refusal_names_both_numbers() -> None:
    """The user can only act on this by shortening the prompt, which
    needs to know by how much."""
    with pytest.raises(ValueError, match="8.*7|7.*8"):
        _sized_backend(7).check_prompt_fits("she ran home today")


def test_the_refusal_fits_the_status_row() -> None:
    """The status row is one nowrap line that truncates with an
    ellipsis, and it already carries Step, Elapsed and T/s. A refusal
    the user has to act on is the worst thing in the app to clip, and
    the first version of this message was clipped mid-sentence.

    The row now puts any clipped message on a tooltip, which is the
    general answer, because a CUDA out-of-memory report comes from
    torch and is not ours to shorten. This still holds our own
    messages short: a tooltip is a fallback for text we do not
    control, not a licence to write past the row.

    Bounded on the real numbers rather than the stub's, since six
    figures of tokens is what an overflowing prompt actually reports.
    """
    try:
        _sized_backend(7).check_prompt_fits("she ran home today")
    except ValueError as exc:
        message = str(exc).replace("8", "107,304").replace(
            "7", "65,536"
        )
    assert len(f"Error: {message}") < 80, message


def test_an_unreadable_ceiling_refuses_nothing() -> None:
    """``describe_context_length`` answers None when the checkpoint
    declares nothing usable, and inventing a bound there would turn
    away prompts that fit. The readout is blank in this case too."""
    backend = _StubBackend(
        _TemplateTokenizer(),
        model=_StubModel(_UNSPECIFIED_LENGTH),
    )

    backend.check_prompt_fits("she ran home today")


def test_an_unloaded_backend_refuses_nothing() -> None:
    """Not a generation path, since nothing can run before a load, but
    reading a ceiling off a model that is not there would raise where
    a refusal was meant to be reported."""
    _StubBackend(_TemplateTokenizer()).check_prompt_fits("she ran")


def test_the_thinking_flag_reaches_the_refusal() -> None:
    """Thinking selects a wider template, so the same prompt can fit
    one way and not the other. A refusal taken under the wrong flag is
    wrong in whichever direction the flag moved the count."""
    backend = _sized_backend(8)

    backend.check_prompt_fits("she ran home today", thinking=False)
    with pytest.raises(ValueError):
        backend.check_prompt_fits(
            "she ran home today", thinking=True
        )


# -- refusing a prompt past the character cap --
#
# The one bound a model with no window has (`A2-TRUST-02`). Mamba-3
# declares none, so nothing refused a prompt of any length, and the
# save of such a run would then have been refused instead.


def _windowless_backend() -> _StubBackend:
    """A loaded backend whose checkpoint declares no window."""
    return _StubBackend(
        _TemplateTokenizer(),
        model=_StubModel(_UNSPECIFIED_LENGTH),
    )


def test_a_prompt_past_the_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="1,000,000") as raised:
        _windowless_backend().check_prompt_fits(
            "a" * (PROMPT_CHARS_MAX + 1)
        )

    assert len(f"Error: {raised.value}") < 80, str(raised.value)


def test_a_prompt_at_the_cap_goes_on_to_the_window() -> None:
    """The boundary from the permissive side: at the cap the window
    decides, and a model with no window refuses nothing."""
    _windowless_backend().check_prompt_fits("a" * PROMPT_CHARS_MAX)


def test_the_cap_is_checked_before_anything_loads() -> None:
    """It needs no tokenizer, so it is the cheapest check and comes
    first, ahead of the early return for a backend not yet loaded."""
    with pytest.raises(ValueError):
        _StubBackend(_TemplateTokenizer()).check_prompt_fits(
            "a" * (PROMPT_CHARS_MAX + 1)
        )


def test_counting_stops_where_running_does() -> None:
    """A prompt the readout counts in full is one the worker will
    take, so the two bounds are one number."""
    assert COUNT_PROMPT_MAX_CHARS == PROMPT_CHARS_MAX


# -- the count does not stall the socket --


class _SlowTokenizer(_TemplateTokenizer):
    """Templates correctly, and takes real time doing it.

    ``time.sleep`` rather than ``asyncio.sleep`` on purpose: the thing
    under test is whether blocking work reaches the event loop, and a
    cooperative sleep would yield and prove nothing.
    """

    BLOCK_SECONDS = 0.2

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(self.BLOCK_SECONDS)
        return super().apply_chat_template(*args, **kwargs)


def test_the_loop_keeps_ticking_during_a_long_count() -> None:
    """The offload's whole point. This request is bounded at 200,000
    characters, which is real work, and doing it inline held every
    frame and every Cancel behind one keystroke's readout.

    Asserted as the longest gap between ticks rather than as a tick
    count, because a blocked loop still ticks before and after the
    block; the gap is what distinguishes the two.
    """
    stamps: List[float] = []

    async def drive() -> None:
        async def tick() -> None:
            while True:
                stamps.append(time.monotonic())
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.01)
        backend = _StubBackend(_SlowTokenizer())
        await backend.handle_count_prompt(
            _StubWebSocket(), {"text": "she ran home"}
        )
        ticker.cancel()

    asyncio.run(drive())

    assert len(stamps) > 2, "the ticker never ran"
    gaps = [
        later - earlier
        for earlier, later in zip(
            stamps[:-1], stamps[1:], strict=True
        )
    ]
    assert max(gaps) < _SlowTokenizer.BLOCK_SECONDS / 2, (
        f"the loop stalled for {max(gaps):.3f}s while counting"
    )


def test_the_slow_count_still_answers() -> None:
    """Paired with the test above: a count moved off the loop has to
    come back, or the readout would simply never arrive."""
    ws = _StubWebSocket()
    backend = _StubBackend(_SlowTokenizer())

    asyncio.run(backend.handle_count_prompt(ws, {"text": "she ran"}))

    assert len(ws.sent) == 1
    assert ws.sent[0]["count"] > 0
