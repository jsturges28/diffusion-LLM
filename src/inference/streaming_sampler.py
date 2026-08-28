"""Async generator wrapper around the LLaDA diffusion loop.

Yields decoded text frames one at a time so a WebSocket handler
can stream them to the browser without waiting for the full run.
Reuses helper functions from llada_sampler.py; the core sampling
file stays untouched.
"""

from __future__ import annotations

import asyncio
import threading
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
)

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.checkpoint import (
    CheckpointBudget,
    FrameCheckpoint,
    LladaFrame,
    RngState,
    rng_restore,
)
from src.inference.llada_sampler import (
    add_gumbel_noise,
    get_num_transfer_tokens,
)
from src.inference.reveal import newly_revealed

MASK_ID: int = 126336


def checkpoint_append(
    history: List[FrameCheckpoint] | None,
    budget: CheckpointBudget,
    x: torch.Tensor,
    prompt_len: int,
    reveal_conf: torch.Tensor,
) -> None:
    """Record what a resume would need to re-enter this frame.

    The canvas and its reveal confidence are copied to CPU because
    the checkpoint outlives the run, and holding device tensors for
    every frame of a 128-step run would keep that memory from the
    next model to load.

    The random state is taken here, after the step that produced the
    frame, so restoring it puts the sampler exactly where the *next*
    step would have drawn from.
    """
    if history is None:
        return
    assert prompt_len >= 0, "a prompt cannot have negative length"
    assert x.dim() == 2, "the canvas carries a batch dimension"
    history.append(
        FrameCheckpoint(
            ids=x[:, prompt_len:].clone().cpu(),
            canvas_index=0,
            rng=budget.capture_rng(),
            extra=LladaFrame(
                reveal_conf=reveal_conf.clone().cpu()
            ),
        )
    )


def resume_reveal_conf(
    base_conf: torch.Tensor,
    remask_positions: List[int],
    gen_length: int,
) -> torch.Tensor:
    """The confidence a resumed canvas starts from.

    Every surviving position keeps the probability it was actually
    revealed at. Before this, they were all set to 1.0, which made an
    edited run's heatmap and mean confidence report a certainty the
    model never expressed.

    Remasked positions go to zero rather than keeping their old
    value: the user threw that token away, so the canvas genuinely
    does not know what belongs there until a step reveals it again.
    """
    assert base_conf.dim() == 1, "confidence is one per position"
    assert base_conf.numel() == gen_length, (
        "the checkpoint covers the whole generation region"
    )
    conf = base_conf.clone()
    for pos in remask_positions:
        assert 0 <= pos < gen_length, "remask is in range"
        conf[pos] = 0.0
    return conf


def build_llada_inputs(
    tokenizer: Any, prompt: str
) -> Dict[str, torch.Tensor]:
    """Encode a chat prompt the way a LLaDA run does.

    The single definition of "what tokens does this prompt become",
    shared by generation, the resume state, and the prompt token
    count. It was previously written out three times, and the count is
    what made one copy insufficient: a readout telling the user their
    prompt is N tokens has to be N tokens of the same encode the run
    performs, not of a lookalike that could drift from it.

    Templated to a string and then encoded separately, rather than
    ``tokenize=True`` in one call, because the canvas is built around
    ``attention_mask`` and this shape returns both. ``padding=True``
    is kept for the same reason the batch dimension is: the loop
    indexes ``[1, seq]`` throughout.

    ``add_special_tokens=False`` because the template has already
    placed every special token this sequence should carry, and letting
    the tokenizer add another BOS would shift every position by one.

    No ``enable_thinking``: LLaDA exposes no reasoning channel, and
    passing the flag to a template that does not declare it is a
    silent no-op on some versions and an error on others.
    """
    assert isinstance(prompt, str), "prompt must be a string"
    message = {"role": "user", "content": prompt}
    chat_text = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(
        [chat_text],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    assert encoded["input_ids"].dim() == 2, (
        "expected (batch, seq) ids"
    )
    return encoded


def sanitize_frame(text: str) -> str:
    """Replace raw mask / special tokens with display chars."""
    text = text.replace("<|mdm_mask|>", "░")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|endoftext|>", "")
    return text


def _build_token_list(
    x: torch.Tensor,
    prompt_len: int,
    tokenizer: Any,
    reveal_conf: torch.Tensor | None = None,
    mask_conf: torch.Tensor | None = None,
) -> List[Dict[str, Any]]:
    """Build per-token metadata for the generation region.

    Each dict has keys: t (display text), m (is mask), id (token
    id), and c (confidence 0..1). For a resolved token c is the
    frozen reveal-time confidence; for a still-masked token c is the
    model's current predicted confidence (top-prob) at this step,
    which the UI maps to mask opacity. c is omitted on masked tokens
    when no prediction exists yet (the initial all-masked frame),
    which the UI treats as fully solid.
    """
    gen_ids = x[0, prompt_len:].tolist()
    conf = (
        reveal_conf.tolist() if reveal_conf is not None else None
    )
    pred = (
        mask_conf.tolist() if mask_conf is not None else None
    )
    tokens: List[Dict[str, Any]] = []
    for i, token_id in enumerate(gen_ids):
        is_mask = token_id == MASK_ID
        if is_mask:
            display = "░"
        else:
            raw = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
            display = sanitize_frame(raw)
        token: Dict[str, Any] = {
            "t": display, "m": is_mask, "id": token_id
        }
        if is_mask:
            if pred is not None:
                token["c"] = round(float(pred[i]), 4)
        elif conf is not None:
            token["c"] = round(float(conf[i]), 4)
        tokens.append(token)
    return tokens


def _resolved_flags(
    x: torch.Tensor,
    prompt_len: int,
) -> List[bool]:
    """Per-position "no longer a mask" over the generated region."""
    return (x[0, prompt_len:] != MASK_ID).tolist()


def _mean_conf(
    x: torch.Tensor,
    prompt_len: int,
    reveal_conf: torch.Tensor | None,
) -> float:
    """Mean reveal-time confidence over resolved positions."""
    if reveal_conf is None:
        return 0.0
    resolved = x[0, prompt_len:] != MASK_ID
    if not bool(resolved.any()):
        return 0.0
    return round(float(reveal_conf[resolved].mean().item()), 4)


def _forward_pass(
    model: Any,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """Run a single model forward pass (blocking)."""
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
        logits, un_logits = torch.chunk(
            logits, 2, dim=0
        )
        logits = un_logits + (cfg_scale + 1) * (
            logits - un_logits
        )
    else:
        logits = model(
            x, attention_mask=attention_mask
        ).logits
    return logits


@torch.no_grad()
def _diffusion_step(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one synchronous diffusion step, mutating x.

    Returns (x, true_conf, transfer_index): the mutated sequence,
    the per-position softmax confidence of the argmax prediction,
    and the boolean mask of positions revealed this step.
    """
    mask_index = x == MASK_ID

    logits = _forward_pass(
        model, x, attention_mask, prompt_index, cfg_scale
    )

    logits_with_noise = add_gumbel_noise(
        logits, temperature=temperature
    )
    x0 = torch.argmax(logits_with_noise, dim=-1)

    # True per-token confidence: softmax prob of the argmax
    # prediction. Used for the heatmap regardless of the
    # remasking strategy.
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

    x0_p[:, block_end:] = -np.inf

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(
        mask_index, x0_p, -np.inf
    )

    transfer_index = torch.zeros_like(
        x0, dtype=torch.bool, device=x0.device
    )
    for j in range(confidence.shape[0]):
        k = int(
            num_transfer_tokens[j, step_in_block].item()
        )
        if k <= 0:
            continue
        k = min(k, confidence[j].numel())
        _, select_index = torch.topk(
            confidence[j], k=k
        )
        transfer_index[j, select_index] = True

    x[transfer_index] = x0[transfer_index]
    return x, true_conf, transfer_index


async def streaming_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    cancel_event: threading.Event | None = None,
    frame_checkpoints: List[FrameCheckpoint] | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator that yields one dict per diffusion step.

    Yields
    ------
    {"type": "frame", "index": int, "total_steps": int,
     "text": str, "tokens": list, "revealed": list[int]}
        After each diffusion step (including initial masked state).
        ``revealed`` holds the positions unmasked by this step and
        never seen before; see src/inference/reveal.py.
    {"type": "done", "final_text": str}
        After the last step with skip_special_tokens decoding.

    Parameters
    ----------
    frame_checkpoints :
        If provided, each frame's checkpoint (canvas ids, reveal
        confidence, random state) is appended here so the server can
        support resume-from-frame.
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_steps = steps_per_block * num_blocks

    encoded = build_llada_inputs(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(
        model.device
    )
    prompt_len: int = input_ids.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length),
        MASK_ID,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_len] = input_ids.clone()

    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (1, gen_length),
                dtype=attention_mask.dtype,
                device=model.device,
            ),
        ],
        dim=-1,
    )

    prompt_index = x != MASK_ID

    reveal_conf = torch.zeros(gen_length, device=x.device)
    # Positions already counted as born, so a token is only ever
    # revealed once. Empty here because the canvas starts all masked.
    seen_revealed: set[int] = set()

    budget = CheckpointBudget()
    checkpoint_append(
        frame_checkpoints, budget, x, prompt_len, reveal_conf
    )

    initial_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=False
    )[0]
    yield {
        "type": "frame",
        "index": 0,
        "total_steps": total_steps,
        "canvas_index": 0,
        "mean_conf": 0.0,
        "text": sanitize_frame(initial_text),
        "tokens": _build_token_list(
            x, prompt_len, tokenizer, reveal_conf
        ),
        "revealed": [],
    }

    frame_index = 1

    for num_block in range(num_blocks):
        block_start = (
            prompt_len + num_block * block_length
        )
        block_end = (
            prompt_len + (num_block + 1) * block_length
        )
        block_mask_index = (
            x[:, block_start:block_end] == MASK_ID
        )
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )

        for i in range(steps_per_block):
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                return

            x, step_conf, step_transfer = (
                await asyncio.to_thread(
                    _diffusion_step,
                    x,
                    model,
                    attention_mask,
                    prompt_index,
                    cfg_scale,
                    temperature,
                    remasking,
                    block_end,
                    num_transfer_tokens,
                    i,
                )
            )
            gen_transfer = step_transfer[0, prompt_len:]
            gen_step_conf = step_conf[0, prompt_len:]
            reveal_conf[gen_transfer] = (
                gen_step_conf[gen_transfer]
            )

            checkpoint_append(
                frame_checkpoints,
                budget,
                x,
                prompt_len,
                reveal_conf,
            )

            step_text = tokenizer.batch_decode(
                x[:, prompt_len:],
                skip_special_tokens=False,
            )[0]
            born = newly_revealed(
                _resolved_flags(x, prompt_len), seen_revealed
            )
            seen_revealed.update(born)
            yield {
                "type": "frame",
                "index": frame_index,
                "total_steps": total_steps,
                "canvas_index": 0,
                "mean_conf": _mean_conf(
                    x, prompt_len, reveal_conf
                ),
                "text": sanitize_frame(step_text),
                "tokens": _build_token_list(
                    x, prompt_len, tokenizer, reveal_conf,
                    gen_step_conf,
                ),
                "revealed": born,
            }
            frame_index += 1

    final_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=True
    )[0]
    yield {
        "type": "done",
        "final_text": final_text,
        # The templated length this run actually built, so the saved
        # run records a measurement rather than the client's estimate.
        "prompt_len": prompt_len,
    }


async def streaming_resume(
    model: Any,
    tokenizer: Any,
    *,
    base_tokens: torch.Tensor,
    base_conf: torch.Tensor,
    base_rng: RngState | None,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    remask_positions: List[int],
    remaining_steps: int,
    gen_length: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    cancel_event: threading.Event | None = None,
    frame_checkpoints: List[FrameCheckpoint] | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Resume diffusion from a saved frame with user remasks.

    Reconstructs the full sequence tensor from prompt_ids and
    base_tokens, applies remasks, then runs remaining_steps of
    diffusion treating the whole generation region as one block.

    Parameters
    ----------
    base_tokens :
        Generation-region tensor at the resume frame,
        shape (1, gen_length), on CPU.
    base_conf :
        Reveal confidence at that frame, shape (gen_length,), so
        surviving tokens keep the certainty they were written with.
    base_rng :
        The generator state captured at that frame, restored before
        the first step so two attempts at one edit agree. None when
        the run outran its checkpoint budget, in which case the
        resume draws from wherever the process left off.
    prompt_ids :
        Original prompt token IDs, shape (1, prompt_len).
    attention_mask :
        Full attention mask, shape (1, prompt_len + gen_length).
    remask_positions :
        Indices within the generation region (0-based) to
        set back to MASK_ID before resuming.
    remaining_steps :
        How many diffusion steps to run from this point.
    """
    assert remaining_steps > 0
    assert len(remask_positions) > 0

    # Before the canvas is touched, so the first step draws from the
    # state the chosen frame left behind rather than from whatever
    # random work has happened since the run that produced it.
    if base_rng is not None:
        rng_restore(base_rng)

    prompt_len: int = prompt_ids.shape[1]
    device = model.device

    x = torch.cat(
        [prompt_ids.to(device), base_tokens.to(device)],
        dim=1,
    )
    attention_mask = attention_mask.to(device)

    for pos in remask_positions:
        assert 0 <= pos < gen_length
        x[0, prompt_len + pos] = MASK_ID

    prompt_index = torch.zeros_like(
        x, dtype=torch.bool
    )
    prompt_index[:, :prompt_len] = True

    block_end = prompt_len + gen_length

    block_mask_index = (
        x[:, prompt_len:block_end] == MASK_ID
    )
    num_transfer_tokens = get_num_transfer_tokens(
        block_mask_index, remaining_steps
    )

    # Surviving positions keep the confidence they were revealed at
    # in the run this branched from; remasked ones drop to zero.
    reveal_conf = resume_reveal_conf(
        base_conf, remask_positions, gen_length
    ).to(x.device)

    budget = CheckpointBudget()
    checkpoint_append(
        frame_checkpoints, budget, x, prompt_len, reveal_conf
    )

    # A resume starts with most of the canvas already written by the
    # run it branched from. Seeding from that state (rather than
    # starting empty, as generate does) is what stops the whole
    # surviving prefix from being reported as born on frame 0.
    seen_revealed: set[int] = set(
        newly_revealed(_resolved_flags(x, prompt_len), set())
    )

    initial_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=False
    )[0]
    yield {
        "type": "frame",
        "index": 0,
        "total_steps": remaining_steps,
        "canvas_index": 0,
        "mean_conf": _mean_conf(x, prompt_len, reveal_conf),
        "text": sanitize_frame(initial_text),
        "tokens": _build_token_list(
            x, prompt_len, tokenizer, reveal_conf
        ),
        "revealed": [],
    }

    for i in range(remaining_steps):
        if (
            cancel_event is not None
            and cancel_event.is_set()
        ):
            return

        x, step_conf, step_transfer = (
            await asyncio.to_thread(
                _diffusion_step,
                x,
                model,
                attention_mask,
                prompt_index,
                cfg_scale,
                temperature,
                remasking,
                block_end,
                num_transfer_tokens,
                i,
            )
        )
        gen_transfer = step_transfer[0, prompt_len:]
        gen_step_conf = step_conf[0, prompt_len:]
        reveal_conf[gen_transfer] = (
            gen_step_conf[gen_transfer]
        )

        checkpoint_append(
            frame_checkpoints, budget, x, prompt_len, reveal_conf
        )

        step_text = tokenizer.batch_decode(
            x[:, prompt_len:],
            skip_special_tokens=False,
        )[0]
        born = newly_revealed(
            _resolved_flags(x, prompt_len), seen_revealed
        )
        seen_revealed.update(born)
        yield {
            "type": "frame",
            "index": i + 1,
            "total_steps": remaining_steps,
            "canvas_index": 0,
            "mean_conf": _mean_conf(
                x, prompt_len, reveal_conf
            ),
            "text": sanitize_frame(step_text),
            "tokens": _build_token_list(
                x, prompt_len, tokenizer, reveal_conf,
                gen_step_conf,
            ),
            "revealed": born,
        }

    final_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=True
    )[0]
    yield {
        "type": "done",
        "final_text": final_text,
        "prompt_len": prompt_len,
    }
