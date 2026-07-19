"""Quantize DiffusionGemma's MoE experts to NF4 and save.

Loads the bf16 base on CPU, replaces every fused expert stack with
an NF4 ``Experts4bit`` (leaving attention/dense/norms/embeddings/
vision in bf16), and writes a ~16 GB checkpoint that the worker
reloads with ``src.inference.dgemma_nf4``.

Run in the DiffusionGemma venv, with the bundled CUDA libs on the
linker path (bitsandbytes needs them):

    LD_LIBRARY_PATH="$PWD/.venv-dgemma/lib/python3.12/\
site-packages/nvidia/cu13/lib:..." \
    .venv-dgemma/bin/python scripts/quantize_diffusiongemma_nf4.py \
        --base ~/models/diffusiongemma-26B-A4B-it-bf16 \
        --out  ~/models/diffusiongemma-26B-A4B-it-nf4
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from transformers import (  # type: ignore[attr-defined]
    DiffusionGemmaForBlockDiffusion,
)

from src.inference.dgemma_nf4 import quantize_experts_inplace

# Files copied verbatim from the base checkpoint so the NF4 dir is
# self-contained for the tokenizer / config / chat template.
COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
]

STATE_DICT_NAME = "model_nf4.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF4-quantize DiffusionGemma experts."
    )
    parser.add_argument(
        "--base",
        type=str,
        default="~/models/diffusiongemma-26B-A4B-it-bf16",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="~/models/diffusiongemma-26B-A4B-it-nf4",
    )
    return parser.parse_args()


def _dir_size_gib(path: Path) -> float:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / 1024**3


def main() -> None:
    args = parse_args()
    base = Path(args.base).expanduser()
    out = Path(args.out).expanduser()
    assert base.is_dir(), f"base not found: {base}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for NF4 quantization.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading bf16 base from {base} (CPU) ...", flush=True)
    load_start = time.monotonic()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        str(base),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    print(
        f"Loaded in {time.monotonic() - load_start:.1f}s",
        flush=True,
    )

    print("Quantizing experts to NF4 ...", flush=True)
    quant_start = time.monotonic()
    unique = quantize_experts_inplace(model)
    print(
        f"Quantized {unique} unique expert stacks in"
        f" {time.monotonic() - quant_start:.1f}s",
        flush=True,
    )

    print("Saving NF4 state dict ...", flush=True)
    save_start = time.monotonic()
    torch.save(
        model.state_dict(), str(out / STATE_DICT_NAME)
    )
    print(
        f"Saved in {time.monotonic() - save_start:.1f}s",
        flush=True,
    )

    for name in COPY_FILES:
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)

    print(
        f"Done. Output: {out} ({_dir_size_gib(out):.1f} GiB)"
    )


if __name__ == "__main__":
    main()
