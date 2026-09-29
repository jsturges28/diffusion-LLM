"""Mamba-3 answers the sampler the way a Hugging Face model does.

Strategy: a tiny random model, the real checkpoint's shape at a
fraction of its size, is called through the adapter and directly, and
the two are compared. Passing proves the adapter adds nothing to the
arithmetic; that the state it returns is the recurrence's own, so
reading on from it is the same as reading everything at once, which
is what makes the sampler's token-at-a-time loop and What If's replay
the same run; and that the forgetting reported for a token does not
depend on how that token was read.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from src.backends.worker_base import (
    describe_context_length,
    describe_output_width,
)
from src.inference.mamba3 import LayerState
from src.inference.mamba3_causal import FORGETTING_KEY, Mamba3CausalLM
from src.inference.mamba3_memory import forgetting, recording_core
from tests.inference.test_mamba3_memory import _tiny_model

IDS = torch.tensor([[3, 17, 5, 9, 30, 2, 41, 8]])
SPLIT = 5


def _read_one_at_a_time(
    adapter: Mamba3CausalLM, ids: torch.Tensor
) -> List[torch.Tensor]:
    """Each token read on its own call, the way the sampler reads
    what it generates. Returns each call's forgetting values."""
    past: Optional[List[LayerState]] = None
    values: List[torch.Tensor] = []
    for index in range(int(ids.shape[1])):
        out = adapter(
            input_ids=ids[:, index:index + 1], past_key_values=past
        )
        past = out.past_key_values
        values.append(out.token_signals[FORGETTING_KEY])
    return values


def test_the_logits_are_the_models_own() -> None:
    """The recording core runs the same recurrence, so what the
    sampler reads is exactly what the model computes."""
    model = _tiny_model()

    out = Mamba3CausalLM(model)(input_ids=IDS)
    with torch.no_grad():
        logits, _ = model(IDS, model.empty_states(1))

    assert torch.equal(out.logits, logits)
    assert out.logits.dtype == torch.float32


def test_no_past_is_the_empty_state() -> None:
    """The sampler's first call passes None for the cache, which here
    has to mean a state nothing has been read into."""
    model = _tiny_model()
    adapter = Mamba3CausalLM(model)

    fresh = adapter(input_ids=IDS, past_key_values=None)
    empty = adapter(
        input_ids=IDS, past_key_values=model.empty_states(1)
    )

    assert torch.equal(fresh.logits, empty.logits)


def test_reading_on_from_the_state_is_reading_it_all() -> None:
    """The state round trip. The sampler reads the prompt in one pass
    and then one token per call, and What If replays the kept prefix
    in one pass; both describe the same run only if a split read
    gives what the whole read gives."""
    adapter = Mamba3CausalLM(_tiny_model())
    whole = adapter(input_ids=IDS)

    head = adapter(input_ids=IDS[:, :SPLIT])
    past = head.past_key_values
    logits = [head.logits]
    for index in range(SPLIT, int(IDS.shape[1])):
        step = adapter(
            input_ids=IDS[:, index:index + 1], past_key_values=past
        )
        past = step.past_key_values
        logits.append(step.logits)

    torch.testing.assert_close(torch.cat(logits, dim=1), whole.logits)


def test_each_token_reports_what_reading_it_erased() -> None:
    """The value is `forgetting` over the decays this very pass used,
    one per token read, and a share of the state."""
    model = _tiny_model()
    decays: List[torch.Tensor] = []
    with torch.no_grad():
        model(IDS, model.empty_states(1), core=recording_core(decays))

    values = Mamba3CausalLM(model)(input_ids=IDS).token_signals[
        FORGETTING_KEY
    ]

    assert values.shape == IDS.shape
    assert torch.equal(values, forgetting(decays))
    assert bool((values >= 0.0).all())
    assert bool((values <= 1.0).all())


def test_a_tokens_value_does_not_depend_on_how_it_was_read() -> None:
    """The sampler reads generated tokens one call at a time and a
    What If replay reads the same prefix in one call. A token's value
    has to come out the same either way, or a branch would disagree
    with the run it branched from about the tokens they share."""
    adapter = Mamba3CausalLM(_tiny_model())
    whole = adapter(input_ids=IDS).token_signals[FORGETTING_KEY]

    stepped = _read_one_at_a_time(adapter, IDS)

    torch.testing.assert_close(torch.cat(stepped, dim=1), whole)


def test_the_value_depends_on_what_came_before() -> None:
    """The negative space of the test above: the same token read after
    a different history erases a different amount, so the agreement
    there is the state being carried and not a per-id constant."""
    adapter = Mamba3CausalLM(_tiny_model())
    first = adapter(input_ids=IDS).token_signals[FORGETTING_KEY]
    other = torch.tensor([[29, 4, 44, 12, 6, 2, 41, 8]])

    second = adapter(input_ids=other).token_signals[FORGETTING_KEY]

    # Positions 5 to 7 hold the same ids after different histories.
    assert not torch.allclose(first[:, 5:], second[:, 5:])


def test_the_mask_changes_nothing() -> None:
    """Accepted because the sampler always sends one, and ignored
    because a recurrence has nothing it could hide."""
    adapter = Mamba3CausalLM(_tiny_model())

    plain = adapter(input_ids=IDS)
    masked = adapter(
        input_ids=IDS, attention_mask=torch.zeros_like(IDS)
    )

    assert torch.equal(plain.logits, masked.logits)


def test_a_flat_row_of_ids_is_refused() -> None:
    """One row per batch entry, as the sampler always sends; a flat
    tensor is a caller's mistake, not a batch of one."""
    adapter = Mamba3CausalLM(_tiny_model())

    with pytest.raises(AssertionError):
        adapter(input_ids=IDS[0])


def test_the_shell_reads_a_width_and_no_window() -> None:
    """What the worker shell asks of any model: an output width, for
    range checks, and a context window, for the prompt check. A
    state-space model has the first and honestly lacks the second,
    so no prompt is refused for a ceiling it does not have."""
    model = _tiny_model()
    adapter = Mamba3CausalLM(model)

    assert describe_output_width(adapter) == model.config.vocab_size
    assert describe_context_length(adapter, SimpleNamespace()) is None
    assert adapter.device == torch.device("cpu")
    assert adapter.emits_token_signals is True
