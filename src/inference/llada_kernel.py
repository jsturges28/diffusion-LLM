"""The LLaDA diffusion algorithm: one step, and the schedule of steps.

Everything here is synchronous and takes plain arguments, so it can be
read, tested and driven from more than one place. What drives it lives
elsewhere: `streaming_sampler.py` wraps these in an async generator
that yields frames, and the worker validates a request against the
schedule before starting a run.

The split matters because there used to be two implementations of this
algorithm. The live one was inlined in the streaming wrapper, and a
second, dormant copy sat in `llada_sampler.py` with its own CFG,
Gumbel, remasking, transfer and block-schedule logic. A LLaDA change
had two plausible places to land, and the diffusion entropy and top-k
work coming next would have had to choose. The dormant copy is now a
quarantined reference under `reference/llada/`, kept only so a test
can prove this file still agrees with it. See finding `ORG-03`.

`streaming_resume` deliberately drives these differently from
`streaming_generate`: it treats the whole canvas as a single block,
because a resume from an arbitrary saved frame cannot reconstruct
which block that frame was in. That is why the loops stay with their
callers and only the step and the schedule are shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# LLaDA's [MASK] token. Every masked position on the canvas holds this
# id until a step reveals it, so it is both the sentinel the sampler
# tests against and the value the frontend renders as a mask glyph.
MASK_ID: int = 126336

assert MASK_ID > 0, "the mask id is a real vocabulary entry"


@dataclass(frozen=True)
class BlockSchedule:
    """How a run is divided into blocks, and steps within a block.

    Frozen because it is derived once per run and read many times; a
    caller that could edit it would be able to disagree with the
    validation that produced it.
    """

    num_blocks: int
    steps_per_block: int

    @property
    def total_steps(self) -> int:
        return self.num_blocks * self.steps_per_block


def block_schedule(
    *, gen_length: int, block_length: int, steps: int
) -> BlockSchedule:
    """Divide ``gen_length`` into blocks and ``steps`` between them.

    Raises ``ValueError`` naming the offending pair when the division
    does not come out even. That is an operating error, not a broken
    invariant: the three values arrive from a request, so the worker
    turns this into an invalid-request envelope rather than crashing.

    One owner for arithmetic that used to be written three times: here
    by way of the worker's validation, as bare asserts inside the
    generate loop, and in the browser's `validateDivisibility`. The
    browser's copy stays, because disabling the button beats refusing
    a request, but it is now the only duplicate and it is on the other
    side of a network boundary.
    """
    if block_length <= 0:
        raise ValueError(
            f"block_length ({block_length}) must be positive"
        )
    if steps <= 0:
        raise ValueError(f"steps ({steps}) must be positive")
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be"
            f" divisible by block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by"
            f" num_blocks ({num_blocks})"
        )
    schedule = BlockSchedule(
        num_blocks=num_blocks,
        steps_per_block=steps // num_blocks,
    )
    assert schedule.total_steps == steps, "steps must be preserved"
    return schedule


def add_gumbel_noise(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Gumbel-max sampling over the vocabulary, in float64.

    float64 rather than the model's bf16 on purpose. arXiv:2409.02908
    reports that low-precision Gumbel-max improves perplexity for
    masked diffusion models while making the generations worse, which
    is the wrong trade for a tool whose output people read.

    ``temperature == 0`` returns the logits untouched, which makes the
    caller's argmax deterministic. Worth knowing for tests: this path
    consumes no random state, so a seeded comparison only lines up
    across implementations when both take the same branch.
    """
    assert temperature >= 0.0, "temperature cannot be negative"
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """How many positions each step is allowed to reveal.

    LLaDA uses a linear noise schedule, so the reverse process should
    move the same number of tokens per step. This precomputes that:
    the masked count divided evenly across ``steps``, with the
    remainder spread one each over the earliest steps rather than
    dropped.

    Returns a tensor of shape (batch, steps). The caller reads column
    ``i`` on step ``i`` and reveals that many highest-confidence
    positions, which is what makes the strategy rank-based rather than
    threshold-based: a low-confidence token still commits if it is the
    best of what is left.
    """
    assert steps > 0, "a schedule needs at least one step"
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    assert num_transfer_tokens.shape[1] == steps, "one column a step"
    return num_transfer_tokens


def forward_with_cfg(
    model: Any,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """One model forward pass, with classifier-free guidance if asked.

    With guidance the batch is doubled: the canvas as it stands, and a
    copy with the prompt itself masked out. The difference between the
    two is what the prompt is contributing, and scaling it pushes the
    generation further toward the prompt than the model would go on
    its own.
    """
    assert cfg_scale >= 0.0, "guidance scale cannot be negative"
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = MASK_ID
        x_cat = torch.cat([x, un_x], dim=0)
        if attention_mask is not None:
            attention_mask_cat = torch.cat(
                [attention_mask, attention_mask], dim=0
            )
        else:
            attention_mask_cat = None
        logits = model(
            x_cat, attention_mask=attention_mask_cat
        ).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        return un_logits + (cfg_scale + 1) * (logits - un_logits)
    return model(x, attention_mask=attention_mask).logits


@torch.no_grad()
def diffusion_step(
    x: torch.Tensor,
    model: Any,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    cfg_scale: float,
    temperature: float,
    remasking: str,
    block_end: int,
    num_transfer_tokens: torch.Tensor,
    step_in_block: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Execute one synchronous diffusion step, mutating x.

    Returns (x, true_conf, transfer_index, x0): the mutated
    sequence, the per-position softmax confidence of the argmax
    prediction, the boolean mask of positions revealed this step,
    and the argmax prediction itself for every position.
    """
    mask_index = x == MASK_ID

    logits = forward_with_cfg(
        model, x, attention_mask, prompt_index, cfg_scale
    )

    logits_with_noise = add_gumbel_noise(
        logits, temperature=temperature
    )
    x0 = torch.argmax(logits_with_noise, dim=-1)

    # True per-token confidence: softmax prob of the argmax
    # prediction. Computed for every strategy, not just
    # low_confidence, because the heatmap shows it either way. The
    # extra softmax under random remasking is deliberate: it costs one
    # pass and is the only thing that makes those frames readable.
    p = F.softmax(logits, dim=-1)
    true_conf = torch.squeeze(
        torch.gather(
            p,
            dim=-1,
            index=torch.unsqueeze(x0, -1),
        ),
        -1,
    ).float()
    if remasking == "low_confidence":
        x0_p = true_conf.clone()
    elif remasking == "random":
        x0_p = torch.rand(
            (x0.shape[0], x0.shape[1]),
            device=x0.device,
        )
    else:
        raise NotImplementedError(remasking)

    # Nothing past this block's end may commit yet, which is what
    # makes the process semi-autoregressive rather than free over the
    # whole canvas.
    x0_p[:, block_end:] = -np.inf

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(
        x0, dtype=torch.bool, device=x0.device
    )
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j, step_in_block].item())
        if k <= 0:
            continue
        k = min(k, confidence[j].numel())
        _, select_index = torch.topk(confidence[j], k=k)
        transfer_index[j, select_index] = True

    x[transfer_index] = x0[transfer_index]
    # x0 goes back out as well as into x. It is the model's pick for
    # every position, settled or not, and the line above keeps only
    # the few that were revealed this step. The rest are what a
    # masked position is currently holding out for, and the display
    # had no way to name them because they stopped here.
    return x, true_conf, transfer_index, x0
