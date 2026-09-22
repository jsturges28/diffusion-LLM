"""Per-model text conventions, separated from the sampling math.

The autoregressive sampler is meant to be reused by the next
append-only model, and it could not be: it carried SmolLM3's chat
template, SmolLM3's ChatML turn terminator, SmolLM3's control tokens
and SmolLM3's ``<think>`` channel. DiffusionGemma's sampler carried
its own near-copies, with a byte-identical input builder, an identical
sanitizer over a different token list, and a channel convention that
is a genuinely different algorithm. Reusing the numeric loop for a
base completion model would have changed prompt semantics, stopped on
the wrong token, and split a reasoning channel that does not exist.

This module owns the shared half. What a model declares is its input
mode, its control tokens and its turn terminator; what a model
implements is any channel convention of its own, because those are
not variations on one algorithm and pretending otherwise would invent
a commonality that is not there.

The rule this enforces is that no *sampler* knows a model's text
conventions, which is why every one of them takes an adapter as an
argument. The concrete adapters live here together rather than beside
their workers, for a practical reason: the DiffusionGemma worker
imports bitsandbytes and so can only be loaded in its own venv, and an
adapter nothing else can import is an adapter nothing can test.
LLaDA's is the exception, and sits beside the encode it wraps.

Dependency-light for the same reason ``params.py`` and ``protocol.py``
are: three venvs with incompatible ``transformers`` versions import
it. Nothing here imports torch or transformers; it only calls methods
on the tokenizer and model objects it is handed.
"""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Protocol,
    Set,
    Tuple,
)

# How a prompt reaches the model. ``chat`` wraps it in a template's
# role markers; ``completion`` hands it over as-is for the model to
# continue. The distinction is user-visible, because a base model
# presented as a chat partner is easy to run and hard to interpret.
INPUT_MODE_CHAT = "chat"
INPUT_MODE_COMPLETION = "completion"
INPUT_MODES: Tuple[str, ...] = (
    INPUT_MODE_CHAT,
    INPUT_MODE_COMPLETION,
)


class TextAdapter(Protocol):
    """Everything a sampler needs to know about a model's text.

    Six questions, each of which had a SmolLM3-shaped answer baked
    into a sampler that three model families are meant to share.
    """

    input_mode: str

    def build_inputs(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        *,
        thinking: bool,
    ) -> Any:
        """The prompt as model-ready inputs, on the model's device."""

    def count_prompt_tokens(
        self, tokenizer: Any, prompt: str, *, thinking: bool
    ) -> int:
        """How many tokens ``build_inputs`` would produce."""

    def stop_ids(self, tokenizer: Any, model: Any) -> Set[int]:
        """Token ids that end generation."""

    def sanitize(self, text: str) -> str:
        """``text`` without the scaffolding a user should not see."""

    def split_channels(self, raw: str) -> Tuple[str, str]:
        """``raw`` as (reasoning, answer), both sanitized."""


class ChatTextAdapter:
    """A model whose prompt is wrapped by a chat template.

    Subclass and declare ``control_tokens`` and, where one exists,
    ``turn_end_token``. Override ``split_channels`` only if the model
    emits a reasoning channel; the default reports none, which is the
    right answer for most checkpoints and was previously expressed by
    simply not having the code.
    """

    input_mode: str = INPUT_MODE_CHAT
    # Scaffolding hidden from the per-token display, so the chat
    # markers do not clutter the streamed output.
    control_tokens: Tuple[str, ...] = ()
    # The token that ends one turn, where the template has one. It is
    # distinct from EOS, and a model without it says None rather than
    # having the sampler guess at a string.
    turn_end_token: Optional[str] = None

    def build_inputs(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        *,
        thinking: bool,
    ) -> Any:
        """Template and tokenize in one call, then move to the device.

        ``thinking`` selects the template's reasoning variant where it
        declares one. An adapter whose template ignores the flag
        overrides this rather than passing it, because handing an
        unknown keyword to a template is a silent no-op on some
        versions of transformers and an error on others.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert prompt != "", "cannot build inputs for no prompt"
        encoded = tokenizer.apply_chat_template(
            self._chat(prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        return encoded.to(model.device)

    def count_prompt_tokens(
        self, tokenizer: Any, prompt: str, *, thinking: bool
    ) -> int:
        """The same encode as ``build_inputs``, minus the device move.

        The same encode matters more than it sounds: this number is
        shown to the user as a statement about their run, so a
        lookalike that could drift from what generation builds would
        be worse than no readout.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        if prompt == "":
            return 0
        encoded = tokenizer.apply_chat_template(
            self._chat(prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        count = int(encoded["input_ids"].shape[-1])
        assert count > 0, "a templated prompt has tokens"
        return count

    def stop_ids(self, tokenizer: Any, model: Any) -> Set[int]:
        """EOS, whatever the generation config adds, and the turn end.

        A turn terminator is a distinct token from EOS, and a chat
        model that stops only on EOS runs past the end of its answer
        into the next role marker.
        """
        ids: Set[int] = set()
        _collect_ids(ids, tokenizer.eos_token_id)
        generation_config = getattr(
            model, "generation_config", None
        )
        if generation_config is not None:
            _collect_ids(
                ids, getattr(generation_config, "eos_token_id", None)
            )
        if self.turn_end_token is not None:
            turn_end = _known_token_id(
                tokenizer, self.turn_end_token
            )
            if turn_end is not None:
                ids.add(turn_end)
        return ids

    def sanitize(self, text: str) -> str:
        """Strip every declared control token, and nothing else."""
        assert isinstance(text, str), "sanitize takes a string"
        for token in self.control_tokens:
            text = text.replace(token, "")
        return text

    def split_channels(self, raw: str) -> Tuple[str, str]:
        """No reasoning channel: the whole output is the answer."""
        assert isinstance(raw, str), "split takes a string"
        return "", self.sanitize(raw).strip()

    def _chat(self, prompt: str) -> list:
        """The prompt as the one-turn chat a template expects."""
        return [{"role": "user", "content": prompt}]


class CompletionTextAdapter:
    """A base model that continues text rather than answering it.

    The branch that did not exist and that the next model class needs.
    A base checkpoint may carry no chat template at all, in which case
    ``apply_chat_template`` raises rather than degrading, so nothing
    here calls it. There is no turn terminator either, because there
    are no turns.
    """

    input_mode: str = INPUT_MODE_COMPLETION
    control_tokens: Tuple[str, ...] = ()

    def build_inputs(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        *,
        thinking: bool,
    ) -> Any:
        """Encode the prompt as written.

        ``thinking`` is accepted and ignored to keep one signature
        across adapters; a base model has no reasoning channel for the
        flag to select.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert prompt != "", "cannot build inputs for no prompt"
        encoded = tokenizer(prompt, return_tensors="pt")
        return encoded.to(model.device)

    def count_prompt_tokens(
        self, tokenizer: Any, prompt: str, *, thinking: bool
    ) -> int:
        """``build_inputs``' encode, minus the device move."""
        assert isinstance(prompt, str), "prompt must be a string"
        if prompt == "":
            return 0
        encoded = tokenizer(prompt, return_tensors="pt")
        count = int(encoded["input_ids"].shape[-1])
        assert count > 0, "an encoded prompt has tokens"
        return count

    def stop_ids(self, tokenizer: Any, model: Any) -> Set[int]:
        """EOS only. A completion has no turn to terminate."""
        ids: Set[int] = set()
        _collect_ids(ids, tokenizer.eos_token_id)
        generation_config = getattr(
            model, "generation_config", None
        )
        if generation_config is not None:
            _collect_ids(
                ids, getattr(generation_config, "eos_token_id", None)
            )
        return ids

    def sanitize(self, text: str) -> str:
        """Strip every declared control token, and nothing else."""
        assert isinstance(text, str), "sanitize takes a string"
        for token in self.control_tokens:
            text = text.replace(token, "")
        return text

    def split_channels(self, raw: str) -> Tuple[str, str]:
        """No reasoning channel: the whole output is the answer."""
        assert isinstance(raw, str), "split takes a string"
        return "", self.sanitize(raw).strip()


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class Smollm3TextAdapter(ChatTextAdapter):
    """SmolLM3's ChatML scaffolding and its ``<think>`` channel."""

    control_tokens = (
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        _THINK_OPEN,
        _THINK_CLOSE,
    )
    # A distinct token from EOS, so a run that stops only on EOS walks
    # on into the next role marker.
    turn_end_token = "<|im_end|>"

    def split_channels(self, raw: str) -> Tuple[str, str]:
        """Reasoning in ``<think> ... </think>``, then the answer.

        Partitions on the *first* close tag, because the trace comes
        before the answer and a later literal inside the answer must
        not move the boundary. With thinking disabled there is no tag
        at all and the whole output is the answer.
        """
        assert isinstance(raw, str), "split takes a string"
        if _THINK_CLOSE not in raw:
            return "", self.sanitize(raw).strip()
        head, _, answer = raw.partition(_THINK_CLOSE)
        head = head.replace(_THINK_OPEN, "", 1)
        return (
            self.sanitize(head).strip(),
            self.sanitize(answer).strip(),
        )


_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"


class DgemmaTextAdapter(ChatTextAdapter):
    """DiffusionGemma's Gemma scaffolding and channel convention.

    Its reasoning channel is not a variation on SmolLM3's think tags,
    which is why ``split_channels`` is a method each model implements
    rather than one function taking a pair of delimiters. This one
    partitions on the *last* close marker and carries a literal
    ``thought`` label inside the opener.

    No turn terminator: this model runs through ``model.generate``,
    which ends a canvas itself, so nothing reads its stop set.
    """

    control_tokens = (
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<|turn>",
        "<turn|>",
        _CHANNEL_OPEN,
        _CHANNEL_CLOSE,
        "<|think|>",
    )

    def split_channels(self, raw: str) -> Tuple[str, str]:
        """``<|channel>thought ... <channel|>``, then the answer."""
        assert isinstance(raw, str), "split takes a string"
        if _CHANNEL_CLOSE not in raw:
            return "", self.sanitize(raw).strip()
        head, _, answer = raw.rpartition(_CHANNEL_CLOSE)
        if _CHANNEL_OPEN in head:
            head = head.split(_CHANNEL_OPEN, 1)[1]
        head = head.replace("thought", "", 1)
        return (
            self.sanitize(head).strip(),
            self.sanitize(answer).strip(),
        )


SMOLLM3_TEXT = Smollm3TextAdapter()
DGEMMA_TEXT = DgemmaTextAdapter()


def _collect_ids(ids: Set[int], value: Any) -> None:
    """Add ``value`` to ``ids``, which may be one id or several.

    ``eos_token_id`` is an int on most checkpoints and a list on the
    ones that end on more than one token, and both shapes have to
    count or a run stops late.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        ids.add(int(value))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool):
                ids.add(int(item))


def _known_token_id(
    tokenizer: Any, token: str
) -> Optional[int]:
    """The id for ``token``, or None if the vocabulary lacks it.

    An absent token converts to the unknown id rather than failing, so
    the unknown id has to be ruled out explicitly. Adding it to the
    stop set would end every run at its first unrecognised piece.
    """
    assert token, "cannot look up an empty token"
    resolved = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(resolved, int):
        return None
    if resolved < 0:
        return None
    if resolved == tokenizer.unk_token_id:
        return None
    return resolved
