"""Mamba-3 in the calling shape the autoregressive sampler drives.

`src/inference/ar_sampler.py` was written for Hugging Face causal
language models: it calls `model(input_ids=..., attention_mask=...,
past_key_values=..., use_cache=True)`, reads `.logits` and
`.past_key_values`, and asks `model.device` where tensors go. This
adapter gives `Mamba3LM` exactly that shape, with the recurrent
states standing where the key-value cache would. The sampler needs no
branch for it, and What If needs none either: a list of states cannot
be sliced like a cache, so the sampler's cache helpers decline it and
every substitution replays the prompt and the kept prefix, which is
exact in float32.

It also reports, for every token it reads, what reading that token
erased from the state (`mamba3_memory.forgetting`), under the token
record's key `f`. `emits_token_signals` tells the sampler to send
each token only once it has been read, so that value belongs to the
token it is shown on.

There is no `max_position_embeddings`. A state-space model has no
positional ceiling, only the length it was trained on, so the prompt
check has nothing it could honestly refuse.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from src.inference import mamba3_memory as memory
from src.inference.mamba3 import LayerState, Mamba3LM

FORGETTING_KEY = "f"


class Mamba3Output(NamedTuple):
    """One forward's results, named as Hugging Face names them."""

    logits: Tensor  # (batch, length, vocab), float32
    past_key_values: List[LayerState]
    token_signals: Dict[str, Tensor]  # key to (batch, length)


class Mamba3Config(NamedTuple):
    """The one config field the worker shell reads off a model."""

    vocab_size: int


class Mamba3CausalLM:
    """`Mamba3LM` as the sampler calls a causal language model."""

    emits_token_signals = True
    # Stop tokens come from the tokenizer; there is no generation
    # config to consult.
    generation_config = None

    def __init__(self, model: Mamba3LM) -> None:
        self.model = model
        self.config = Mamba3Config(
            vocab_size=model.config.vocab_size
        )

    @property
    def device(self) -> torch.device:
        return self.model.lm_head.weight.device

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[LayerState]] = None,
        use_cache: bool = True,
    ) -> Mamba3Output:
        """Read `input_ids` on from `past_key_values`, or from the
        empty state when there are none.

        `attention_mask` is accepted and ignored: a recurrence reads
        every token it is given, in order, and has nothing a mask
        could hide. `use_cache` is true in effect whatever it says,
        since the new state is the result.
        """
        assert input_ids.dim() == 2, "one row of ids per batch entry"
        states = past_key_values
        if states is None:
            states = self.model.empty_states(int(input_ids.shape[0]))
        decays: List[Tensor] = []
        with torch.no_grad():
            logits, after = self.model(
                input_ids, states, core=memory.recording_core(decays)
            )
        forgetting = memory.forgetting(decays)
        assert forgetting.shape == input_ids.shape, (
            "one forgetting value per token read"
        )
        return Mamba3Output(
            logits=logits,
            past_key_values=after,
            token_signals={FORGETTING_KEY: forgetting},
        )
