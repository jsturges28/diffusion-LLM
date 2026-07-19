"""NF4 quantization for DiffusionGemma's fused MoE experts.

The experts (128 per layer x 30 layers) hold roughly 22.7B of the
model's 25.2B parameters. Quantizing only the fused expert stacks
to 4-bit (NF4) while leaving attention, dense MLP, norms,
embeddings, and the vision tower in bf16 brings the resident
footprint to ~16 GB, which fits a 24 GB RTX 4090.

Encoder and decoder experts are tied (share the same parameter
tensors), so each unique expert stack is quantized once and the
resulting module is reused wherever the tie points.

bitsandbytes needs the torch-bundled CUDA libraries on
``LD_LIBRARY_PATH``; the worker/quantizer processes set that in
their environment before importing this module.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.functional as bnbF
from bitsandbytes.functional import QuantState

logger = logging.getLogger("dgemma_nf4")

NF4_BLOCKSIZE = 64
EXPERTS_CLASS_NAME = "DiffusionGemmaTextExperts"
STATE_DICT_NAME = "model_nf4.pt"


def _tie_key(name: str) -> str:
    """Normalize an expert module path to its tied identity.

    Encoder and decoder experts are tied; mapping the encoder path
    onto the decoder path lets us dedupe them without relying on
    tensor storage identity (which is unavailable on meta tensors).
    """
    return name.replace(
        "encoder.language_model.layers", "decoder.layers"
    )


class Experts4bit(nn.Module):
    """NF4-quantized drop-in for ``DiffusionGemmaTextExperts``.

    Stores per-expert packed 4-bit weights and dequantizes only the
    experts hit by the current tokens, mirroring the reference
    module's top-k routing forward.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        blocksize: int = NF4_BLOCKSIZE,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.blocksize = blocksize
        self.compute_dtype = compute_dtype

        # gate_up per expert: (2 * intermediate, hidden)
        self.gate_up_shape = (
            2 * intermediate_dim,
            hidden_dim,
        )
        # down per expert: (hidden, intermediate)
        self.down_shape = (hidden_dim, intermediate_dim)

        gu_numel = self.gate_up_shape[0] * self.gate_up_shape[1]
        dp_numel = self.down_shape[0] * self.down_shape[1]
        assert gu_numel % blocksize == 0
        assert dp_numel % blocksize == 0
        gu_bytes = gu_numel // 2
        dp_bytes = dp_numel // 2
        gu_blocks = gu_numel // blocksize
        dp_blocks = dp_numel // blocksize

        # Registered so they serialize via state_dict.
        self.register_buffer(
            "gate_up_packed",
            torch.zeros(num_experts, gu_bytes, dtype=torch.uint8),
        )
        self.register_buffer(
            "gate_up_absmax",
            torch.zeros(num_experts, gu_blocks, dtype=torch.float32),
        )
        self.register_buffer(
            "down_packed",
            torch.zeros(num_experts, dp_bytes, dtype=torch.uint8),
        )
        self.register_buffer(
            "down_absmax",
            torch.zeros(num_experts, dp_blocks, dtype=torch.float32),
        )
        # NF4 codebook (16 levels), shared across all experts.
        self.register_buffer(
            "code", torch.zeros(16, dtype=torch.float32)
        )

    @classmethod
    @torch.no_grad()
    def from_experts(
        cls,
        module: nn.Module,
        *,
        blocksize: int = NF4_BLOCKSIZE,
        quant_device: str = "cuda",
    ) -> "Experts4bit":
        """Quantize a bf16 ``DiffusionGemmaTextExperts`` module."""
        num_experts = int(module.num_experts)
        hidden_dim = int(module.hidden_dim)
        intermediate_dim = int(module.intermediate_dim)
        obj = cls(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=module.act_fn,
            blocksize=blocksize,
        )
        gate_up = module.gate_up_proj  # (E, 2I, H)
        down = module.down_proj  # (E, H, I)
        code: torch.Tensor | None = None
        for e in range(num_experts):
            gu = (
                gate_up[e]
                .to(quant_device, torch.bfloat16)
                .contiguous()
            )
            packed_gu, qs_gu = bnbF.quantize_4bit(
                gu, quant_type="nf4", blocksize=blocksize
            )
            obj.gate_up_packed[e] = (
                packed_gu.reshape(-1).to("cpu")
            )
            obj.gate_up_absmax[e] = qs_gu.absmax.to("cpu")

            dp = (
                down[e]
                .to(quant_device, torch.bfloat16)
                .contiguous()
            )
            packed_dp, qs_dp = bnbF.quantize_4bit(
                dp, quant_type="nf4", blocksize=blocksize
            )
            obj.down_packed[e] = packed_dp.reshape(-1).to("cpu")
            obj.down_absmax[e] = qs_dp.absmax.to("cpu")
            if code is None:
                code = qs_gu.code.to("cpu").float()
        assert code is not None
        obj.code = code
        return obj

    def _dequant(
        self,
        packed_row: torch.Tensor,
        absmax_row: torch.Tensor,
        shape: Tuple[int, int],
    ) -> torch.Tensor:
        quant_state = QuantState(
            absmax=absmax_row,
            shape=torch.Size(shape),
            code=self.code,
            blocksize=self.blocksize,
            quant_type="nf4",
            dtype=self.compute_dtype,
        )
        weight = bnbF.dequantize_4bit(
            packed_row.reshape(-1, 1),
            quant_state,
            quant_type="nf4",
        )
        return weight.to(self.compute_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for entry in expert_hit:
            expert_idx = entry[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(
                expert_mask[expert_idx]
            )
            current_state = hidden_states[token_idx]
            gate_up_weight = self._dequant(
                self.gate_up_packed[expert_idx],
                self.gate_up_absmax[expert_idx],
                self.gate_up_shape,
            )
            gate, up = F.linear(
                current_state, gate_up_weight
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_weight = self._dequant(
                self.down_packed[expert_idx],
                self.down_absmax[expert_idx],
                self.down_shape,
            )
            current_hidden_states = F.linear(
                current_hidden_states, down_weight
            )
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(
                    final_hidden_states.dtype
                ),
            )
        return final_hidden_states


def _iter_expert_modules(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == EXPERTS_CLASS_NAME:
            out.append((name, module))
    return out


def _set_submodule(
    model: nn.Module, path: str, new_module: nn.Module
) -> None:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _parent_and_attr(
    model: nn.Module, name: str
) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@torch.no_grad()
def _move_preserving_sharing(
    model: nn.Module, device: str
) -> None:
    """Move params/buffers to ``device``, allocating each unique
    storage only once.

    ``nn.Module.to`` copies tied tensors (e.g. the 1.38 GB
    embedding shared by the decoder, encoder, and lm_head) into
    separate device tensors, which wastes several GB. Mapping by
    storage identity keeps ties shared on the target device.
    """
    seen: Dict[int, torch.Tensor] = {}

    def relocate(name: str, tensor: torch.Tensor, is_param: bool):
        if tensor.is_meta:
            return
        ptr = tensor.untyped_storage().data_ptr()
        moved = seen.get(ptr)
        if moved is None:
            moved = tensor.to(device)
            seen[ptr] = moved
        parent, attr = _parent_and_attr(model, name)
        if is_param:
            parent._parameters[attr] = nn.Parameter(
                moved, requires_grad=False
            )
        else:
            parent._buffers[attr] = moved

    for name, tensor in list(
        model.named_parameters(remove_duplicate=False)
    ):
        relocate(name, tensor, True)
    for name, tensor in list(
        model.named_buffers(remove_duplicate=False)
    ):
        relocate(name, tensor, False)


@torch.no_grad()
def quantize_experts_inplace(
    model: nn.Module, *, blocksize: int = NF4_BLOCKSIZE
) -> int:
    """Replace every expert stack with an ``Experts4bit``.

    Tied expert stacks (same underlying storage) are quantized
    once and shared. Returns the number of unique stacks
    quantized.
    """
    targets = _iter_expert_modules(model)
    by_storage: Dict[str, Experts4bit] = {}
    unique = 0
    for name, module in targets:
        key = _tie_key(name)
        quantized = by_storage.get(key)
        if quantized is None:
            quantized = Experts4bit.from_experts(
                module, blocksize=blocksize
            )
            by_storage[key] = quantized
            unique += 1
            logger.info("quantized experts at %s", name)
        _set_submodule(model, name, quantized)
    return unique


def swap_experts_for_load(
    model: nn.Module, *, blocksize: int = NF4_BLOCKSIZE
) -> None:
    """Replace expert stacks with empty ``Experts4bit`` skeletons.

    Used before ``load_state_dict`` when reconstructing a saved
    NF4 model. Ties are preserved by sharing one skeleton across
    modules that referenced the same original storage.
    """
    targets = _iter_expert_modules(model)
    by_storage: Dict[str, Experts4bit] = {}
    for name, module in targets:
        key = _tie_key(name)
        skeleton = by_storage.get(key)
        if skeleton is None:
            skeleton = Experts4bit(
                num_experts=int(module.num_experts),
                hidden_dim=int(module.hidden_dim),
                intermediate_dim=int(module.intermediate_dim),
                act_fn=module.act_fn,
                blocksize=blocksize,
            )
            by_storage[key] = skeleton
        _set_submodule(model, name, skeleton)


@torch.no_grad()
def load_quantized(
    nf4_dir: str,
    *,
    device: str = "cuda",
) -> Any:
    """Reconstruct an NF4 DiffusionGemma from a saved checkpoint.

    Builds the model with meta parameters but real (correctly
    initialized) buffers, swaps in ``Experts4bit`` skeletons, then
    assigns the saved weights. No bf16 base is required.
    """
    from pathlib import Path

    from accelerate import init_empty_weights
    from transformers import (  # type: ignore[attr-defined]
        AutoConfig,
        DiffusionGemmaForBlockDiffusion,
    )

    nf4_path = Path(nf4_dir)
    config = AutoConfig.from_pretrained(str(nf4_path))

    # Params on meta (no allocation), buffers real so RoPE and
    # other non-persistent buffers keep correct values.
    with init_empty_weights(include_buffers=False):
        model = DiffusionGemmaForBlockDiffusion(config)

    swap_experts_for_load(model)

    state_dict = torch.load(
        str(nf4_path / STATE_DICT_NAME), map_location="cpu"
    )
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False, assign=True
    )
    leftover_meta = [
        name
        for name, param in model.named_parameters()
        if param.is_meta
    ]
    assert not leftover_meta, (
        f"parameters still on meta after load: {leftover_meta[:5]}"
    )

    # Note: tie_weights() is intentionally skipped. The expert tie
    # is handled by sharing Experts4bit skeletons, and lm_head /
    # embedding ties are preserved as shared storage through the
    # saved state dict. Calling tie_weights() here fails because
    # the quantized experts no longer expose gate_up_proj params.
    _move_preserving_sharing(model, device)
    model.eval()
    logger.info(
        "loaded NF4 model (missing=%d, unexpected=%d)",
        len(missing),
        len(unexpected),
    )
    return model
