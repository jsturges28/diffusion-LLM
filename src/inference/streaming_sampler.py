"""Async generator wrapper around the LLaDA diffusion loop.

Yields decoded text frames one at a time so a WebSocket handler
can stream them to the browser without waiting for the full run.
Reuses helper functions from llada_sampler.py — the core sampling
file stays untouched.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
)

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.llada_sampler import (
    add_gumbel_noise,
    get_num_transfer_tokens,
)

MASK_ID: int = 126336


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
) -> List[Dict[str, Any]]:
    """Build per-token metadata for the generation region.

    Returns a list of dicts, one per token in x[0, prompt_len:].
    Each dict has keys: t (display text), m (is mask), id (token id).
    """
    gen_ids = x[0, prompt_len:].tolist()
    tokens: List[Dict[str, Any]] = []
    for token_id in gen_ids:
        is_mask = token_id == MASK_ID
        if is_mask:
            display = "░"
        else:
            raw = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
            display = sanitize_frame(raw)
        tokens.append(
            {"t": display, "m": is_mask, "id": token_id}
        )
    return tokens


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
) -> torch.Tensor:
    """Execute one synchronous diffusion step, mutating x.

    Separated so both generate and resume can share logic
    without duplicating the core loop body.
    """
    mask_index = x == MASK_ID

    logits = _forward_pass(
        model, x, attention_mask, prompt_index, cfg_scale
    )

    logits_with_noise = add_gumbel_noise(
        logits, temperature=temperature
    )
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(
                p,
                dim=-1,
                index=torch.unsqueeze(x0, -1),
            ),
            -1,
        )
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
    return x


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
    cancel_event: asyncio.Event | None = None,
    tensor_history: List[torch.Tensor] | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator that yields one dict per diffusion step.

    Yields
    ------
    {"type": "frame", "index": int, "total_steps": int,
     "text": str, "tokens": list}
        After each diffusion step (including initial masked state).
    {"type": "done", "final_text": str}
        After the last step with skip_special_tokens decoding.

    Parameters
    ----------
    tensor_history :
        If provided, each frame's generation-region tensor
        (shape (1, gen_length), on CPU) is appended here so
        the server can support resume-from-frame.
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_steps = steps_per_block * num_blocks

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

    if tensor_history is not None:
        tensor_history.append(
            x[:, prompt_len:].clone().cpu()
        )

    initial_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=False
    )[0]
    yield {
        "type": "frame",
        "index": 0,
        "total_steps": total_steps,
        "text": sanitize_frame(initial_text),
        "tokens": _build_token_list(
            x, prompt_len, tokenizer
        ),
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

            x = await asyncio.to_thread(
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

            if tensor_history is not None:
                tensor_history.append(
                    x[:, prompt_len:].clone().cpu()
                )

            step_text = tokenizer.batch_decode(
                x[:, prompt_len:],
                skip_special_tokens=False,
            )[0]
            yield {
                "type": "frame",
                "index": frame_index,
                "total_steps": total_steps,
                "text": sanitize_frame(step_text),
                "tokens": _build_token_list(
                    x, prompt_len, tokenizer
                ),
            }
            frame_index += 1

    final_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=True
    )[0]
    yield {"type": "done", "final_text": final_text}


async def streaming_resume(
    model: Any,
    tokenizer: Any,
    *,
    base_tokens: torch.Tensor,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    remask_positions: List[int],
    remaining_steps: int,
    gen_length: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    cancel_event: asyncio.Event | None = None,
    tensor_history: List[torch.Tensor] | None = None,
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

    if tensor_history is not None:
        tensor_history.append(
            x[:, prompt_len:].clone().cpu()
        )

    initial_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=False
    )[0]
    yield {
        "type": "frame",
        "index": 0,
        "total_steps": remaining_steps,
        "text": sanitize_frame(initial_text),
        "tokens": _build_token_list(
            x, prompt_len, tokenizer
        ),
    }

    for i in range(remaining_steps):
        if (
            cancel_event is not None
            and cancel_event.is_set()
        ):
            return

        x = await asyncio.to_thread(
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

        if tensor_history is not None:
            tensor_history.append(
                x[:, prompt_len:].clone().cpu()
            )

        step_text = tokenizer.batch_decode(
            x[:, prompt_len:],
            skip_special_tokens=False,
        )[0]
        yield {
            "type": "frame",
            "index": i + 1,
            "total_steps": remaining_steps,
            "text": sanitize_frame(step_text),
            "tokens": _build_token_list(
                x, prompt_len, tokenizer
            ),
        }

    final_text = tokenizer.batch_decode(
        x[:, prompt_len:], skip_special_tokens=True
    )[0]
    yield {"type": "done", "final_text": final_text}
