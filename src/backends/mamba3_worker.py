"""Mamba-3 worker: a state-space model, decoding left to right.

Runs in ``.venv-ar``, in plain PyTorch. ``src/inference/mamba3.py``
is this project's own implementation, held to upstream's references,
because upstream's kernels are not built for this card. Generation,
What If substitution and the typed-token probe are the append-only
shell in ``append_only_backend``, shared with SmolLM3; this file is
how Mamba-3 loads.

Float32 on both devices. On the card it costs nothing in speed,
because the Python loop and not memory bandwidth sets the pace, and it
makes decoding a token at a time reproduce the whole-sequence pass
exactly, which is what What If's replays rely on. On a CPU it is the
precision every processor runs well.

The tokenizer is Llama 3.1's, which the checkpoint was trained with,
taken from SmolLM3's pinned repository because Meta's is gated. The
registry names it as this model's companion, the supervisor fetches
it into a cache of its own, and this load refuses a file whose
fingerprint is not Llama 3.1's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.backends.append_only_backend import AppendOnlyBackend
from src.backends.registry import MAMBA3
from src.backends.text_adapter import MAMBA3_TEXT
from src.backends.worker_base import Backend
from src.inference import mamba3
from src.inference.hf_download import (
    download_with_progress,
    fetch_companion_files,
    revision_from_snapshot,
)
from src.inference.mamba3_causal import Mamba3CausalLM
from src.inference.mamba3_tokenizer import (
    TOKENIZER_FILE,
    load_tokenizer,
)

logger = logging.getLogger("mamba3_worker")

# The precision on every device; the module docstring says why.
DTYPE = torch.float32


class Mamba3Backend(AppendOnlyBackend):
    def __init__(self) -> None:
        self.model_info = MAMBA3
        self.text_adapter = MAMBA3_TEXT
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cuda"
        self.load_progress: Optional[Dict[str, Any]] = None
        # The most recent run's prompt, parameters and per-position
        # trace, which What If re-enters. None until a run completes.
        self.last_run_state: Optional[Dict[str, Any]] = None

    def load(self, *, device: str = "cuda") -> None:
        # CPU when CUDA was asked for and is not there, as SmolLM3
        # does, so a GPU-less host degrades instead of failing.
        resolved = (
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.device = resolved
        self.effective_device = resolved
        name = self.model_info.checkpoint
        revision = self.model_info.revision
        companion = self.model_info.companion
        assert revision, "Mamba-3 must load a pinned commit"
        assert companion is not None, "Mamba-3 borrows its tokenizer"
        logger.info("ensuring weights for %s", name)
        snapshot = download_with_progress(
            name,
            revision=revision,
            sink=lambda p: setattr(self, "load_progress", p),
        )
        self.load_progress = None
        self.loaded_revision = revision_from_snapshot(snapshot)
        assert self.loaded_revision is not None, (
            f"cache returned a path naming no commit: {snapshot}"
        )
        paths = fetch_companion_files(
            companion.repo,
            companion.files,
            revision=companion.revision,
        )
        logger.info("loading the tokenizer from %s", companion.repo)
        self.tokenizer = load_tokenizer(
            paths[TOKENIZER_FILE], source=companion.repo
        )
        logger.info(
            "loading model %s on %s (float32)", name, resolved
        )
        model = mamba3.load(
            Path(snapshot), device=resolved, dtype=DTYPE
        )
        self.model = Mamba3CausalLM(model)
        logger.info("Mamba-3 loaded on %s", resolved)


def build_backend() -> Backend:
    return Mamba3Backend()
