"""Spike: load 4-bit DiffusionGemma and verify per-frame diffusion capture.

This is a throwaway feasibility probe, not production code. It answers three
questions for the RTX 4090 (24 GB) target:

  1. Does the 4-bit (compressed-tensors W4A16) checkpoint load under
     Transformers v5 on the GPU, and what is the real VRAM footprint?
  2. Does a custom streamer subclass capture intermediate denoising frames
     (via ``put_draft``) and committed canvases (via ``put``)?
  3. Roughly how fast is one canvas, and how many frames does it emit?

Run (from repo root, using the isolated DiffusionGemma venv, unsandboxed for
GPU access):

    .venv-dgemma/bin/python scripts/spike_diffusiongemma.py \
        --model-path ~/models/diffusiongemma-26B-A4B-it-AWQ-W4A16 \
        --prompt "Why is the sky blue?" \
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List

import torch
from transformers import (  # type: ignore[attr-defined]
    AutoTokenizer,
    DiffusionGemmaForBlockDiffusion,
)
from transformers.generation.streamers import BaseStreamer


class CapturingStreamer(BaseStreamer):
    """Captures diffusion frames instead of printing them to the terminal.

    The DiffusionGemma generate loop calls:
      - ``put(input_ids)`` once at the start (the prompt),
      - ``put_draft(canvas)`` on every denoising step (the frames we want),
      - ``put(canvas)`` when each 256-token canvas is committed.

    We store CPU copies so VRAM is not held by the frame history.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.draft_frames: List[torch.Tensor] = []
        self.committed_blocks: List[torch.Tensor] = []
        # Keep logits out of put_draft; on a 262K vocab they are huge.
        self._takes_logits = False
        self._prompt_captured = False

    @staticmethod
    def _to_cpu_1d(value: torch.Tensor) -> torch.Tensor:
        tensor = value
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().to("cpu")
        if tensor.dim() > 1:
            assert tensor.shape[0] == 1, (
                f"spike assumes batch size 1, got shape {tuple(tensor.shape)}"
            )
            tensor = tensor[0]
        return tensor

    def put(self, value: torch.Tensor) -> None:
        tensor = self._to_cpu_1d(value)
        if not self._prompt_captured:
            self._prompt_captured = True
            return
        self.committed_blocks.append(tensor)

    def put_draft(self, value: torch.Tensor, **kwargs: Any) -> None:
        self.draft_frames.append(self._to_cpu_1d(value))

    def end(self) -> None:
        return None


def _vram_report(label: str) -> None:
    """Print allocated / reserved / free VRAM at a checkpoint."""
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return
    allocated_gib = torch.cuda.memory_allocated() / 1024**3
    reserved_gib = torch.cuda.memory_reserved() / 1024**3
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    print(
        f"[{label}] allocated={allocated_gib:.2f} GiB  "
        f"reserved={reserved_gib:.2f} GiB  peak={peak_gib:.2f} GiB  "
        f"free={free_gib:.2f}/{total_gib:.2f} GiB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffusionGemma load/frame spike.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/models/diffusiongemma-26B-A4B-it-AWQ-W4A16",
        help="Local directory containing the 4-bit checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Why is the sky blue?",
        help="User prompt for the single test generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation budget; 256 = a single canvas (fastest probe).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    assert model_path.is_dir(), f"model path not found: {model_path}"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; spike requires the GPU.")

    torch.cuda.reset_peak_memory_stats()
    _vram_report("before load")

    # Text-only spike: load the tokenizer directly. AutoProcessor would drag in
    # the Gemma4 image processor (torchvision), which we defer to the multimodal
    # phase.
    print(f"Loading tokenizer from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    is_nf4 = (model_path / "model_nf4.pt").is_file()
    load_start = time.monotonic()
    if is_nf4:
        print("Loading NF4 quantized model ...", flush=True)
        from src.inference.dgemma_nf4 import load_quantized

        model = load_quantized(str(model_path))
    else:
        print("Loading model (bf16 base) ...", flush=True)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            str(model_path),
            dtype="auto",
            device_map="auto",
        ).eval()
    load_seconds = time.monotonic() - load_start
    print(f"Model loaded in {load_seconds:.1f} s", flush=True)
    _vram_report("after load")

    canvas_length = getattr(model.config, "canvas_length", None)
    print(f"config.canvas_length = {canvas_length}")

    chat = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f"prompt tokens = {prompt_len}")

    streamer = CapturingStreamer(tokenizer=tokenizer)

    print("Generating ...", flush=True)
    gen_start = time.monotonic()
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
    )
    gen_seconds = time.monotonic() - gen_start
    _vram_report("after generate")

    sequences = getattr(output, "sequences", output)
    final_ids = sequences[0][prompt_len:]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)

    num_drafts = len(streamer.draft_frames)
    num_blocks = len(streamer.committed_blocks)
    new_tokens = int(final_ids.shape[0])
    tps = new_tokens / gen_seconds if gen_seconds > 0 else 0.0

    print("\n================ SPIKE RESULTS ================")
    print(f"load_seconds        = {load_seconds:.1f}")
    print(f"generate_seconds    = {gen_seconds:.2f}")
    print(f"new_tokens          = {new_tokens}")
    print(f"tokens_per_second   = {tps:.1f}")
    print(f"draft_frames        = {num_drafts}")
    print(f"committed_blocks    = {num_blocks}")

    if num_drafts > 0:
        sample_indices = sorted({0, num_drafts // 2, num_drafts - 1})
        print("\n--- sample draft frames (skip_special_tokens=False) ---")
        for idx in sample_indices:
            frame = streamer.draft_frames[idx]
            text = tokenizer.decode(frame, skip_special_tokens=False)
            preview = text.replace("\n", "\\n")[:160]
            print(f"[draft {idx:>3} | {int(frame.shape[0])} tok] {preview}")

    print("\n--- final text ---")
    print(final_text)
    print("===============================================")

    _seed_resume_probe(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        prompt_len=prompt_len,
        canvas_length=canvas_length,
        draft_frames=streamer.draft_frames,
    )


def _seed_resume_probe(
    *,
    model: Any,
    tokenizer: Any,
    inputs: dict,
    prompt_len: int,
    canvas_length: int | None,
    draft_frames: List[torch.Tensor],
) -> None:
    """Verify the Phase-2 resume hook end to end.

    Confirms that ``generate`` accepts a user-provided starting canvas via
    ``decoder_input_ids`` (even though the encoder forward on canvas 1 sees the
    kwarg before it is popped) together with a reduced ``max_denoising_steps``
    budget, and still streams denoising frames. A handful of positions are
    renoised to random tokens to mimic a user remask.
    """
    if canvas_length is None or len(draft_frames) == 0:
        print("\nRESUME PROBE: skipped (no canvas_length or frames)")
        return

    mid = draft_frames[len(draft_frames) // 2]
    if int(mid.shape[0]) != int(canvas_length):
        print(
            "\nRESUME PROBE: skipped (frame width"
            f" {int(mid.shape[0])} != canvas {canvas_length})"
        )
        return

    seed_canvas = mid.clone()
    vocab_size = int(model.config.text_config.vocab_size)
    stride = max(1, canvas_length // 8)
    for pos in range(0, canvas_length, stride):
        seed_canvas[pos] = torch.randint(0, vocab_size, (1,)).item()
    seed_canvas = seed_canvas.unsqueeze(0).to(model.device)

    resume_streamer = CapturingStreamer(tokenizer=tokenizer)
    print("\nRESUME PROBE: generate with decoder_input_ids seed ...", flush=True)
    try:
        probe_start = time.monotonic()
        r_out = model.generate(
            **inputs,
            max_new_tokens=canvas_length,
            max_denoising_steps=16,
            decoder_input_ids=seed_canvas,
            streamer=resume_streamer,
        )
        probe_seconds = time.monotonic() - probe_start
    except Exception as exc:  # noqa: BLE001
        print(f"RESUME PROBE: FAILED -> {type(exc).__name__}: {exc}")
        return

    r_seq = getattr(r_out, "sequences", r_out)
    r_text = tokenizer.decode(
        r_seq[0][prompt_len:], skip_special_tokens=True
    )
    print(f"resume_seconds      = {probe_seconds:.2f}")
    print(f"resume draft_frames = {len(resume_streamer.draft_frames)}")
    print(f"resume committed    = {len(resume_streamer.committed_blocks)}")
    print("resume final text:")
    print(r_text)
    print("RESUME PROBE: OK")


if __name__ == "__main__":
    main()
