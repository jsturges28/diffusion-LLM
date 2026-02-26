from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MDMSamplerConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 128
    steps: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: Optional[float] = None
    seed: int = 42

    # Controls how many masked tokens are filled per step.
    # If None, uses a simple linear schedule: fill more early, less later.
    fill_schedule: str = "linear"  # "linear" | "sqrt"

    # Rendering: what token is used for masks?
    mask_token: str = "[MASK]"


def _top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    # Top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)

    # Top-p (nucleus)
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits.float(), dim=-1)
        cum = torch.cumsum(probs, dim=-1)

        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask = mask.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    return logits


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      sampled_ids: [...]
      confidence:  [...] probability of sampled id
    """
    if temperature > 0:
        logits = logits / temperature
    logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        sampled = torch.distributions.Categorical(probs=probs).sample()
    else:
        sampled = probs.argmax(dim=-1)

    conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, conf


def _num_to_fill(total_masked: int, step: int, steps: int, schedule: str) -> int:
    # fraction filled by end of this step
    t = (step + 1) / float(steps)
    if schedule == "sqrt":
        frac = t**0.5
    else:
        frac = t  # linear
    target_filled = int(round(frac * total_masked))
    return target_filled


@torch.no_grad()
def mdm_generate_with_history(model, tokenizer, prompt: str, cfg: MDMSamplerConfig):
    """
    Returns:
      final_text: str
      history_texts: List[str]  # decoded intermediate sequences
    """
    torch.manual_seed(cfg.seed)

    device = next(model.parameters()).device
    model.eval()

    # Ensure [MASK] exists like Open-dLLM does (they add it if missing). :contentReference[oaicite:7]{index=7}
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": cfg.mask_token})
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token_id is None:
        # decoder-only models often use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # [1, P]
    B, P = prompt_ids.shape

    mask_id = tokenizer.mask_token_id
    # Create masked "canvas"
    canvas = torch.full((B, cfg.max_new_tokens), mask_id, device=device, dtype=torch.long)
    x = torch.cat([prompt_ids, canvas], dim=1)  # [1, P+N]
    total_len = x.size(1)

    # Fix prompt tokens (never change them)
    fix_mask = torch.zeros_like(x, dtype=torch.bool)
    fix_mask[:, :P] = True

    history: List[str] = []

    # Track how many of the completion positions we've filled so far
    completion_positions = torch.arange(P, total_len, device=device)
    total_masked = completion_positions.numel()
    filled_so_far = 0

    for s in range(cfg.steps):
        mask_positions = (x == mask_id) & (~fix_mask)
        if not mask_positions.any():
            break

        outputs = model(input_ids=x, attention_mask=(x != tokenizer.pad_token_id).long())
        logits = outputs.logits  # [B, L, V]

        # Align logits with token positions for decoder-only LMs:
        # Open-dLLM shifts logits right by 1 (so position i predicts token at i). :contentReference[oaicite:8]{index=8}
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # [B, L, V]

        # Only consider masked positions
        masked_logits = logits[mask_positions]  # [M, V]
        sampled_ids, conf = _sample_from_logits(
            masked_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )  # [M], [M]

        # Decide how many new tokens to fill this step
        target_filled = _num_to_fill(total_masked, s, cfg.steps, cfg.fill_schedule)
        to_fill_now = max(0, target_filled - filled_so_far)
        if to_fill_now == 0:
            # still record state for animation smoothness
            decoded = tokenizer.decode(x[0], skip_special_tokens=False)
            history.append(decoded)
            continue

        # Choose highest-confidence masked positions to fill
        # conf aligns with masked_logits order; map it back to positions.
        mask_idx_flat = mask_positions.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [M] indices in flattened x
        # Pick top confidence indices
        k = min(to_fill_now, conf.numel())
        topk = torch.topk(conf, k=k, largest=True).indices  # indices into [M]

        chosen_flat = mask_idx_flat[topk]
        chosen_token_ids = sampled_ids[topk]

        # Write them into x
        x_flat = x.view(-1)
        x_flat[chosen_flat] = chosen_token_ids
        x = x_flat.view_as(x)

        filled_so_far += k

        decoded = tokenizer.decode(x[0], skip_special_tokens=False)
        history.append(decoded)

    # Final decode: only return newly generated part in user-facing output
    final_text = tokenizer.decode(x[0, P:], skip_special_tokens=True)
    return final_text, history