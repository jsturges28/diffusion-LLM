"""Each model's text conventions, behind one interface.

Strategy: two layers. Most of this runs against stub tokenizers, which
is enough because the adapters are string and dictionary work rather
than inference. The no-chat-template case does not, and is the reason
the finding exists: a base checkpoint may carry no template, and
a stub
that pretends to lack one proves only that the stub was written that
way. So that case loads a real cached tokenizer and clears its
``chat_template``, so the template path behaves as it does on a
base model while the raw encode keeps working.

The central claim is that ``count_prompt_tokens`` agrees with what
``build_inputs`` produces. The count is shown to the user as a
statement about their run, and it used to live in three places (the
base class, LLaDA's override, and each sampler's builder) held in step
by a comment asking them to be.

Passing proves the count is the run's own encode for every adapter and
both thinking settings, that the two channel conventions are separate
algorithms neither of which leaks into the other, that sanitization
removes exactly what a model declares, and that a completion adapter
never reaches for a template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.backends.text_adapter import (
    DGEMMA_TEXT,
    INPUT_MODE_CHAT,
    INPUT_MODE_COMPLETION,
    SMOLLM3_TEXT,
    ChatTextAdapter,
    CompletionTextAdapter,
    DgemmaTextAdapter,
    Smollm3TextAdapter,
)


class _Encoded(dict):
    """What a tokenizer returns: a mapping that can move devices."""

    def __init__(self, ids: List[int]) -> None:
        super().__init__({"input_ids": _Ids(ids)})
        self.moved_to: Optional[str] = None

    def to(self, device: str) -> "_Encoded":
        self.moved_to = device
        return self


class _Ids:
    """Stands in for a tensor: only its trailing dimension is read."""

    def __init__(self, ids: List[int]) -> None:
        self.ids = list(ids)
        self.shape = (1, len(ids))

    def __len__(self) -> int:
        return len(self.ids)


class _Model:
    def __init__(self, eos: Any = 2) -> None:
        self.device = "cuda:0"
        self.generation_config = type(
            "Config", (), {"eos_token_id": eos}
        )()


class _ChatTokenizer:
    """Templates by adding markers, and records what it was asked.

    Several markers wide on purpose: the count under test is of the
    templated sequence, so a stub that added none could not tell a
    correct implementation from one counting the raw prompt.
    """

    MARKERS_PLAIN = 4
    MARKERS_THINKING = 5

    def __init__(self, vocabulary: Optional[Dict[str, int]] = None):
        self.thinking_seen: List[bool] = []
        self.template_calls = 0
        self.raw_calls = 0
        self.eos_token_id = 2
        self.unk_token_id = 0
        # ``is None``, not ``or``: an empty vocabulary is the case a
        # test wants and is falsy, so the shorter form silently handed
        # back the default and the unknown-token test proved nothing.
        self._vocabulary = (
            {"<|im_end|>": 7} if vocabulary is None else vocabulary
        )

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
            "an encode for a reply, as a run does"
        )
        assert tokenize is True, "the adapters ask for ids"
        self.template_calls += 1
        self.thinking_seen.append(enable_thinking)
        words = len(chat[0]["content"].split())
        markers = (
            self.MARKERS_THINKING
            if enable_thinking
            else self.MARKERS_PLAIN
        )
        return _Encoded(list(range(words + markers)))

    def __call__(self, text: str, **kwargs: Any) -> Any:
        self.raw_calls += 1
        return _Encoded(list(range(len(text.split()))))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocabulary.get(token, self.unk_token_id)


# -- the count is the run's own encode --


@pytest.mark.parametrize(
    "adapter",
    [SMOLLM3_TEXT, DGEMMA_TEXT, ChatTextAdapter()],
    ids=["smollm3", "diffusiongemma", "bare"],
)
@pytest.mark.parametrize("thinking", [False, True])
def test_the_count_matches_what_build_inputs_produces(
    adapter: Any, thinking: bool
) -> None:
    """The clause's central claim, for every chat adapter and both
    settings of the flag that changes the template."""
    tokenizer = _ChatTokenizer()
    model = _Model()

    built = adapter.build_inputs(
        tokenizer, model, "she ran home", thinking=thinking
    )
    counted = adapter.count_prompt_tokens(
        tokenizer, "she ran home", thinking=thinking
    )

    assert counted == built["input_ids"].shape[-1]
    # More than the three words, because the template's markers are
    # exactly what a count of the raw prompt would miss.
    assert counted > 3


def test_the_completion_count_matches_its_inputs_too() -> None:
    """The same claim for the adapter with no template at all, which
    is the one nothing exercised before."""
    adapter = CompletionTextAdapter()
    tokenizer = _ChatTokenizer()

    built = adapter.build_inputs(
        tokenizer, _Model(), "she ran home", thinking=False
    )
    counted = adapter.count_prompt_tokens(
        tokenizer, "she ran home", thinking=False
    )

    assert counted == built["input_ids"].shape[-1]
    assert tokenizer.template_calls == 0


@pytest.mark.parametrize(
    "adapter",
    [SMOLLM3_TEXT, DGEMMA_TEXT, CompletionTextAdapter()],
    ids=["smollm3", "diffusiongemma", "completion"],
)
def test_an_empty_prompt_counts_zero(adapter: Any) -> None:
    """Zero rather than the template's own markers: the readout is
    blank before anything is typed, and a number there would be the
    scaffolding rather than the user's prompt."""
    assert (
        adapter.count_prompt_tokens(
            _ChatTokenizer(), "", thinking=False
        )
        == 0
    )


def test_the_flag_reaches_the_template() -> None:
    """``enable_thinking`` selects a different template, so passing it
    is what makes the count the one the run will build."""
    tokenizer = _ChatTokenizer()

    SMOLLM3_TEXT.count_prompt_tokens(
        tokenizer, "she ran", thinking=True
    )

    assert tokenizer.thinking_seen == [True]


def test_inputs_land_on_the_models_device() -> None:
    """A tensor left on the CPU fails deep inside the first forward
    pass rather than here."""
    built = SMOLLM3_TEXT.build_inputs(
        _ChatTokenizer(), _Model(), "she ran", thinking=False
    )

    assert built.moved_to == "cuda:0"


# -- two channel conventions, kept apart --


def test_smollm3_splits_on_its_think_tags() -> None:
    thinking, answer = SMOLLM3_TEXT.split_channels(
        "<think>weighing it up</think>the answer"
    )

    assert thinking == "weighing it up"
    assert answer == "the answer"


def test_smollm3_takes_the_first_close_tag() -> None:
    """The trace comes before the answer, so a later literal inside
    the answer must not move the boundary."""
    thinking, answer = SMOLLM3_TEXT.split_channels(
        "<think>trace</think>mentions </think> in passing"
    )

    assert thinking == "trace"
    assert "in passing" in answer


def test_diffusiongemma_splits_on_its_channel_markers() -> None:
    thinking, answer = DGEMMA_TEXT.split_channels(
        "<|channel>thought weighing it up<channel|>the answer"
    )

    assert thinking == "weighing it up"
    assert answer == "the answer"


def test_diffusiongemma_takes_the_last_close_marker() -> None:
    """``rpartition``, which is the difference from SmolLM3's rule and
    the reason these are two methods rather than one parameterized
    function. Swapping them would pass one of these and fail this."""
    thinking, answer = DGEMMA_TEXT.split_channels(
        "<|channel>thought first<channel|>middle<channel|>last"
    )

    assert answer == "last"
    assert "middle" in thinking


def test_an_unclosed_channel_is_all_reasoning() -> None:
    """What a real run does. DiffusionGemma is a reasoning model and
    256 tokens is not enough to finish reasoning, so the channel opens
    and the budget runs out before it closes. Everything produced is
    then reasoning and there is no answer yet.

    Reporting it as the answer instead put a bare ``thought`` at the
    head of a reply, because sanitizing strips the markers around the
    label but not the label. Found on the first run anyone ever made
    with this model's thinking enabled.
    """
    thinking, answer = DGEMMA_TEXT.split_channels(
        "<|channel>thought\nweighing it up and running out of"
    )

    assert thinking == "weighing it up and running out of"
    assert answer == ""


def test_an_unclosed_channel_is_found_by_its_label_too() -> None:
    """The same case with the opener missing. Which spelling arrives
    depends on where the prompt ended: the template emits
    ``<|channel>thought`` from the model on an ordinary turn and
    pre-fills it on a tool-response turn, leaving only the label in
    the generated slice. A saved run cannot tell the two apart,
    because sanitizing removes the opener either way, so both are
    handled."""
    thinking, answer = DGEMMA_TEXT.split_channels(
        "thought\nweighing it up and running out of"
    )

    assert thinking == "weighing it up and running out of"
    assert answer == ""


def test_an_answer_beginning_with_a_word_is_not_reasoning() -> None:
    """The negative space of the label check, and the reason it is
    anchored at the start: prose mentioning a thought is an answer."""
    thinking, answer = DGEMMA_TEXT.split_channels(
        "A thought experiment is a useful device."
    )

    assert thinking == ""
    assert answer == "A thought experiment is a useful device."


def test_an_unmarked_output_is_still_all_answer() -> None:
    """The negative space of the test above, and the common case with
    thinking off: no opener either, so nothing was reasoning."""
    thinking, answer = DGEMMA_TEXT.split_channels("just the answer")

    assert thinking == ""
    assert answer == "just the answer"


@pytest.mark.parametrize(
    "adapter",
    [SMOLLM3_TEXT, DGEMMA_TEXT, ChatTextAdapter()],
    ids=["smollm3", "diffusiongemma", "bare"],
)
def test_no_channel_means_the_whole_output_is_the_answer(
    adapter: Any,
) -> None:
    """The negative space, and the common case: thinking disabled
    emits no markers at all."""
    thinking, answer = adapter.split_channels("just the answer")

    assert thinking == ""
    assert answer == "just the answer"


def test_neither_convention_answers_the_others_markers() -> None:
    """Each model's split is blind to the other's, which is what makes
    them separate rather than two configurations of one."""
    smollm3_on_gemma = SMOLLM3_TEXT.split_channels(
        "<|channel>thought t<channel|>a"
    )
    gemma_on_smollm3 = DGEMMA_TEXT.split_channels(
        "<think>t</think>a"
    )

    assert smollm3_on_gemma[0] == ""
    assert gemma_on_smollm3[0] == ""


# -- sanitization removes what a model declares, and no more --


# Written out rather than read off the adapters. Iterating
# ``control_tokens`` to check that each is stripped proves only that
# the loop agrees with itself: deleting a token from the tuple removes
# it from the check too, which is the one mistake this should catch.
_EXPECTED_CONTROL_TOKENS = {
    "smollm3": (
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
    ),
    "diffusiongemma": (
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|think|>",
    ),
}


@pytest.mark.parametrize(
    ("name", "adapter"),
    [("smollm3", SMOLLM3_TEXT), ("diffusiongemma", DGEMMA_TEXT)],
)
def test_every_control_token_we_expect_is_stripped(
    name: str, adapter: Any
) -> None:
    """One at a time, so a token that survived could not hide behind
    the others being removed."""
    expected = _EXPECTED_CONTROL_TOKENS[name]
    for token in expected:
        assert adapter.sanitize(f"a{token}b") == "ab", token


@pytest.mark.parametrize(
    ("name", "adapter"),
    [("smollm3", SMOLLM3_TEXT), ("diffusiongemma", DGEMMA_TEXT)],
)
def test_the_declared_set_is_the_set_we_expect(
    name: str, adapter: Any
) -> None:
    """Paired with the test above, which walks the list this pins. A
    token quietly added would be stripped from a user's output without
    anyone deciding to, and one removed would start leaking."""
    assert set(adapter.control_tokens) == set(
        _EXPECTED_CONTROL_TOKENS[name]
    )


def test_sanitizing_leaves_ordinary_text_alone() -> None:
    """The negative space. A sanitizer that stripped more than it was
    told would quietly eat a user's own angle brackets."""
    text = "if a < b and c > d then <notatoken>"

    assert SMOLLM3_TEXT.sanitize(text) == text


def test_a_bare_adapter_strips_nothing() -> None:
    """Declaring no control tokens means declaring none, which is what
    a completion model does."""
    assert (
        ChatTextAdapter().sanitize("<|im_end|>") == "<|im_end|>"
    )


# -- stop ids --


def test_a_chat_adapter_stops_on_eos_and_the_turn_end() -> None:
    """A turn terminator is a distinct token from EOS, and a chat run
    that stops only on EOS walks into the next role marker."""
    ids = SMOLLM3_TEXT.stop_ids(_ChatTokenizer(), _Model())

    assert 2 in ids
    assert 7 in ids


def test_a_list_of_eos_ids_all_count() -> None:
    """Some checkpoints end on more than one token, and missing the
    others stops a run late rather than not at all."""
    ids = SMOLLM3_TEXT.stop_ids(
        _ChatTokenizer(), _Model(eos=[11, 12])
    )

    assert {11, 12} <= ids


def test_an_unknown_turn_token_is_not_a_stop() -> None:
    """A vocabulary without the marker converts it to the unknown id,
    and adding that would end every run at its first odd piece."""
    tokenizer = _ChatTokenizer(vocabulary={})

    ids = SMOLLM3_TEXT.stop_ids(tokenizer, _Model())

    assert tokenizer.unk_token_id not in ids


def test_a_completion_adapter_has_no_turn_to_end() -> None:
    """EOS only, because a base model has no turns."""
    ids = CompletionTextAdapter().stop_ids(
        _ChatTokenizer(), _Model()
    )

    assert ids == {2}


# -- what each adapter declares about itself --


@pytest.mark.parametrize(
    "adapter",
    [SMOLLM3_TEXT, DGEMMA_TEXT],
    ids=["smollm3", "diffusiongemma"],
)
def test_the_chat_adapters_say_so(adapter: Any) -> None:
    assert adapter.input_mode == INPUT_MODE_CHAT


def test_the_completion_adapter_says_so() -> None:
    assert (
        CompletionTextAdapter().input_mode
        == INPUT_MODE_COMPLETION
    )


def test_only_the_chat_model_with_turns_declares_one() -> None:
    """DiffusionGemma ends a canvas through ``generate`` rather
    than on a token, so declaring a terminator it never reads would
    be a claim nothing checks."""
    assert Smollm3TextAdapter().turn_end_token == "<|im_end|>"
    assert DgemmaTextAdapter().turn_end_token is None


# -- the case a stub cannot prove --


def test_the_chat_path_misreads_a_base_model() -> None:
    """Against a real tokenizer with its template cleared, which is
    what a base checkpoint looks like. A stub asserting this would
    only prove the stub was written to.

    Deliberately not asserting an exception, because what transformers
    does here depends on its version and the milder outcome is the
    worse one. On the pinned 4.38.2 a cleared template does not raise:
    it falls back to a default ChatML wrapping and warns, so a base
    model would be fed role markers it never saw in training and the
    run would look fine. Newer versions raise instead.

    Either way the answer disagrees with the text the user wrote,
    which is the claim that holds across versions and the reason a
    completion adapter exists rather than a try/except here.
    """
    tokenizer = _real_tokenizer_without_a_template()
    raw_length = len(tokenizer("she ran home")["input_ids"])

    try:
        templated = ChatTextAdapter().count_prompt_tokens(
            tokenizer, "she ran home", thinking=False
        )
    except Exception:  # noqa: BLE001
        return
    assert templated != raw_length, (
        "the chat path returned the raw length, so this tokenizer"
        " applied no template and the case is untested"
    )


def test_the_completion_adapter_reads_it_correctly() -> None:
    """The same tokenizer, through the adapter built for it. The count
    is the raw encode's own length, so a missing template is not a
    failure to survive but something never consulted."""
    tokenizer = _real_tokenizer_without_a_template()
    raw_length = len(tokenizer("she ran home")["input_ids"])

    count = CompletionTextAdapter().count_prompt_tokens(
        tokenizer, "she ran home", thinking=False
    )

    assert count == raw_length
    assert count > 0


# Tried in order, so this runs wherever the suite runs: LLaDA's
# tokenizer is the one the default venv can parse, and SmolLM3's needs
# the newer ``tokenizers`` its own venv pins.
_CACHED_CHECKPOINTS = (
    "GSAI-ML/LLaDA-8B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
)


def test_no_sampler_kept_its_own_text_helpers() -> None:
    """Source inspection, because the duplication is what the finding
    is about and nothing about behaviour can see it: two samplers each
    carried a builder, a sanitizer and a channel splitter, and the two
    builders were byte-identical.

    A sampler that grew one back would pass every test above while
    quietly putting one model's vocabulary back into the loop the next
    model is supposed to reuse.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "inference"
    for name in ("ar_sampler.py", "dgemma_sampler.py"):
        source = (root / name).read_text(encoding="utf-8")
        for helper in (
            "def _build_inputs(",
            "def _sanitize(",
            "def _split_thinking(",
            "def _stop_ids(",
        ):
            assert helper not in source, f"{name} kept {helper}"


def test_no_sampler_names_a_control_token() -> None:
    """The other half: a sampler could call the adapter and still hold
    a literal to compare against. These strings belong to one model's
    vocabulary and to no shared loop."""
    root = Path(__file__).resolve().parents[2] / "src" / "inference"
    for name in ("ar_sampler.py", "dgemma_sampler.py"):
        source = (root / name).read_text(encoding="utf-8")
        for token in ("<|im_end|>", "<think>", "<|channel>"):
            assert token not in source, f"{name} names {token}"


def _real_tokenizer_without_a_template() -> Any:
    """A cached tokenizer with ``chat_template`` removed.

    Skipped rather than failed when nothing is cached: the point is to
    test against a real encode, and a host without any checkpoint has
    nothing real to test against.
    """
    transformers = pytest.importorskip("transformers")
    reasons = []
    for checkpoint in _CACHED_CHECKPOINTS:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                checkpoint,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{checkpoint}: {exc}")
            continue
        tokenizer.chat_template = None
        return tokenizer
    pytest.skip(
        "no cached tokenizer this venv can load: "
        + "; ".join(reasons)
    )
