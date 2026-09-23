"""The kernel still computes what LLaDA's own sampler computes.

Strategy: drive `src/inference/llada_kernel.py` and the quarantined
`reference/llada/llada_reference.py` over the same stub model, from
the same seed, and require the same canvas and the same reveal order
out of each. The stub returns deterministic logits from a seeded
generator, so there is no model, no weights and no GPU.

What passing proves is that consolidating the algorithm did not change
it. Production streaming used to import two helpers from the reference
and then re-implement its CFG, Gumbel, remasking, transfer and block
schedule; those are now one implementation, and this is the evidence
that the surviving one is the same algorithm rather than merely a
plausible one. It is also the reason the reference was quarantined
instead of deleted: with it gone there would be nothing to compare to,
and any future drift in the kernel would be invisible.

The comparison is per step, not only at the end. Two samplers can
agree on a final canvas while revealing positions in a different
order, and the order is exactly what this app draws: commit order,
per-frame confidence, and the diff overlay all read it.

What this does not cover: the async wrapper. `streaming_generate` and
`streaming_resume` own their loops, and the resume one deliberately
treats the whole canvas as a single block, because a resume cannot
know which block a saved frame came from. Those are covered by
`test_llada_resume_conf.py` and `test_llada_mask_candidate.py`. This
file is about the step and the schedule they share.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference.llada import llada_reference  # noqa: E402
from src.inference.llada_kernel import (  # noqa: E402
    MASK_ID,
    block_schedule,
    diffusion_step,
)

VOCAB = 32
PROMPT_LEN = 3
# Small enough to read a failure, large enough that several steps
# reveal more than one position and the block boundary matters.
GEN_LENGTH = 8


class _StubModel:
    """Deterministic logits, and a record of what it was asked.

    Seeded per instance rather than globally so the logits depend only
    on the call index, which is what lets two independent runs see the
    same model without sharing state. The batch dimension doubles
    under guidance, so the shape is taken from the input rather than
    assumed.
    """

    device = torch.device("cpu")

    def __init__(self, seed: int = 1234) -> None:
        self.seed = seed
        self.calls = 0
        self.batch_shapes: List[Tuple[int, ...]] = []

    def __call__(
        self, x: torch.Tensor, attention_mask: Any = None
    ) -> Any:
        del attention_mask
        self.batch_shapes.append(tuple(x.shape))
        generator = torch.Generator().manual_seed(
            self.seed + self.calls
        )
        self.calls += 1
        logits = torch.randn(
            x.shape[0],
            x.shape[1],
            VOCAB,
            generator=generator,
            dtype=torch.float32,
        )
        return _Output(logits)


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


def _prompt() -> torch.Tensor:
    """A prompt of ordinary token ids, none of them the mask."""
    ids = torch.arange(1, PROMPT_LEN + 1, dtype=torch.long)
    return ids.unsqueeze(0)


def _prompt_mask() -> torch.Tensor:
    """An all-ones mask over the prompt, as the tokenizer returns.

    Supplied rather than passing None, because production always has
    one: `build_llada_inputs` returns it and `streaming_generate`
    extends it over the generated region before the first step. It
    also avoids an upstream defect. With guidance and no mask, the
    reference builds `attention_mask_` inside `if attention_mask is
    not None` and then reads it unconditionally, so it raises
    UnboundLocalError. The kernel handles that branch correctly, which
    would show up here as a crash on one side rather than as a
    disagreement, and comparing against a path production never takes
    would prove nothing either way.
    """
    return torch.ones((1, PROMPT_LEN), dtype=torch.long)


def _extended_mask() -> torch.Tensor:
    """The prompt mask extended over the canvas.

    What `streaming_generate` hands the step function. The reference
    does this concatenation itself, from the prompt-length mask, so
    each side is given the form it expects.
    """
    return torch.ones(
        (1, PROMPT_LEN + GEN_LENGTH), dtype=torch.long
    )


def _run_reference(
    *,
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    seed: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Upstream's loop, with history so reveal order is comparable."""
    torch.manual_seed(seed)
    out, history = llada_reference.generate(
        _StubModel(),
        _prompt(),
        attention_mask=_prompt_mask(),
        steps=steps,
        gen_length=GEN_LENGTH,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_ID,
        record_history=True,
    )
    return out, history


def _run_kernel(
    *,
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    seed: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """The kernel, driven the way `streaming_generate` drives it.

    Deliberately mirrors `streaming_sampler.streaming_generate`: the
    same schedule call, the same `block_end`, the same argument order
    into `diffusion_step`. If that wrapper's loop changes shape, this
    has to change with it, which is the price of comparing a
    synchronous kernel against a synchronous reference.
    """
    torch.manual_seed(seed)
    model = _StubModel()
    prompt = _prompt()
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + GEN_LENGTH),
        MASK_ID,
        dtype=torch.long,
    )
    x[:, :prompt_len] = prompt.clone()
    prompt_index = x != MASK_ID

    attention_mask = _extended_mask()
    history = [x.clone()]
    schedule = block_schedule(
        gen_length=GEN_LENGTH,
        block_length=block_length,
        steps=steps,
    )
    for num_block in range(schedule.num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_ID
        num_transfer_tokens = (
            llada_reference.get_num_transfer_tokens(
                block_mask_index, schedule.steps_per_block
            )
        )
        for step in range(schedule.steps_per_block):
            x, _conf, _transfer, _guess = diffusion_step(
                x,
                model,
                attention_mask,
                prompt_index,
                cfg_scale,
                temperature,
                remasking,
                block_end,
                num_transfer_tokens,
                step,
            )
            history.append(x.clone())
    return x, history


# Every combination that changes a branch in either implementation.
# temperature is 0.0 and 0.7 because 0.0 returns the logits untouched
# and consumes no random state, so the two paths through
# add_gumbel_noise are genuinely different code.
CASES = [
    pytest.param(8, 8, 0.0, 0.0, "low_confidence", id="plain"),
    pytest.param(8, 8, 0.7, 0.0, "low_confidence", id="temperature"),
    pytest.param(8, 8, 0.0, 1.5, "low_confidence", id="cfg"),
    pytest.param(8, 8, 0.7, 1.5, "low_confidence", id="cfg-temp"),
    pytest.param(8, 8, 0.0, 0.0, "random", id="random-remask"),
    pytest.param(8, 4, 0.0, 0.0, "low_confidence", id="two-blocks"),
    pytest.param(8, 2, 0.0, 0.0, "low_confidence", id="four-blocks"),
    pytest.param(8, 2, 0.7, 1.5, "random", id="everything-at-once"),
    pytest.param(4, 4, 0.0, 0.0, "low_confidence", id="fewer-steps"),
    pytest.param(16, 8, 0.0, 0.0, "low_confidence", id="more-steps"),
]


@pytest.mark.parametrize(
    "steps,block_length,temperature,cfg_scale,remasking", CASES
)
def test_the_final_canvas_matches(
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
) -> None:
    kwargs = {
        "steps": steps,
        "block_length": block_length,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
        "remasking": remasking,
        "seed": 7,
    }
    reference_out, _ = _run_reference(**kwargs)
    kernel_out, _ = _run_kernel(**kwargs)

    assert torch.equal(kernel_out, reference_out)


@pytest.mark.parametrize(
    "steps,block_length,temperature,cfg_scale,remasking", CASES
)
def test_every_step_reveals_the_same_positions(
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
) -> None:
    """The stronger claim, and the one this app actually depends on.

    Agreeing on a final canvas while committing positions in a
    different order would still break commit-order colouring, the
    per-frame confidence readout and the diff overlay, all of which
    read the order rather than the result.
    """
    kwargs = {
        "steps": steps,
        "block_length": block_length,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
        "remasking": remasking,
        "seed": 7,
    }
    _, reference_history = _run_reference(**kwargs)
    _, kernel_history = _run_kernel(**kwargs)

    assert len(kernel_history) == len(reference_history)
    for index, (mine, theirs) in enumerate(
        zip(kernel_history, reference_history, strict=True)
    ):
        assert torch.equal(mine, theirs), f"frame {index} differs"


def test_the_run_actually_reveals_something() -> None:
    """Guards the comparison itself.

    Two implementations that both did nothing would match perfectly.
    This pins that the fixture resolves the whole generated region, so
    the agreement above is about a real run.
    """
    out, history = _run_kernel(
        steps=8,
        block_length=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        seed=7,
    )

    assert len(history) == 9
    assert not bool((out == MASK_ID).any())
    assert bool((history[0] == MASK_ID).any())


def test_guidance_doubles_the_batch() -> None:
    """Pins that the cfg case exercises the guided branch.

    Without this, a cfg_scale the kernel silently ignored would still
    match a reference that ignored it the same way.
    """
    model = _StubModel()
    prompt = _prompt()
    x = torch.full(
        (1, PROMPT_LEN + GEN_LENGTH), MASK_ID, dtype=torch.long
    )
    x[:, :PROMPT_LEN] = prompt
    prompt_index = x != MASK_ID
    num_transfer_tokens = torch.ones((1, 1), dtype=torch.int64)

    diffusion_step(
        x,
        model,
        _extended_mask(),
        prompt_index,
        1.5,
        0.0,
        "low_confidence",
        PROMPT_LEN + GEN_LENGTH,
        num_transfer_tokens,
        0,
    )

    assert model.batch_shapes[0][0] == 2


def test_nothing_past_the_block_end_is_revealed() -> None:
    """The semi-autoregressive property, stated directly.

    Both implementations mask confidence past the block end, and they
    arrive at that index differently: the reference slices at
    `prompt.shape[1] + (num_block + 1) * block_length` and the wrapper
    passes a precomputed `block_end`. Pinning the behaviour means a
    future change to either arithmetic has to keep them equal.
    """
    out, history = _run_kernel(
        steps=8,
        block_length=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        seed=7,
    )
    del out
    first_block_end = PROMPT_LEN + 4

    # Halfway through, only the first block may have resolved.
    midpoint = history[4]
    tail = midpoint[0, first_block_end:]
    assert bool((tail == MASK_ID).all()), (
        "a position past the block end committed early"
    )


# -- the schedule, which the reference derives inline --


def test_the_schedule_matches_the_reference_arithmetic() -> None:
    """`block_schedule` replaced three copies of this division, one of
    which was a pair of bare asserts inside the generate loop."""
    schedule = block_schedule(
        gen_length=160, block_length=32, steps=160
    )

    assert schedule.num_blocks == 5
    assert schedule.steps_per_block == 32
    assert schedule.total_steps == 160


def test_one_block_is_the_default_shape() -> None:
    """The registry ships gen_length 160 and block_length 160, so the
    common case is a single block over the whole canvas."""
    schedule = block_schedule(
        gen_length=160, block_length=160, steps=128
    )

    assert schedule.num_blocks == 1
    assert schedule.steps_per_block == 128


def test_an_indivisible_canvas_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="divisible by block_length"):
        block_schedule(
            gen_length=160, block_length=48, steps=128
        )


def test_indivisible_steps_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match="divisible by"):
        block_schedule(
            gen_length=160, block_length=32, steps=129
        )


@pytest.mark.parametrize("block_length", [0, -1])
def test_a_nonpositive_block_is_refused_before_dividing(
    block_length: int,
) -> None:
    """Negative space. A zero would be a ZeroDivisionError, which
    escapes the worker's ValueError handler and reads as a dead run
    rather than a rejected request."""
    with pytest.raises(ValueError, match="must be positive"):
        block_schedule(
            gen_length=160,
            block_length=block_length,
            steps=128,
        )


def test_zero_steps_are_refused() -> None:
    with pytest.raises(ValueError, match="steps"):
        block_schedule(
            gen_length=160, block_length=160, steps=0
        )


def test_a_schedule_cannot_be_edited_after_validation() -> None:
    """Frozen so a caller cannot disagree with the validation that
    produced it."""
    schedule = block_schedule(
        gen_length=160, block_length=160, steps=128
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        schedule.num_blocks = 4  # type: ignore[misc]
