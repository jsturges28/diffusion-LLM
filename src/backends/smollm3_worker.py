"""SmolLM3 worker: autoregressive (left-to-right) backend.

Runs in ``.venv-ar`` (Transformers >= 4.53). Loads SmolLM3-3B via
``AutoModelForCausalLM`` and streams a growing token sequence
through the shared worker contract, one frame per new token, so the
existing scrubber/save/overlay tooling works unchanged (as a
left-to-right replay).

Generation, What If substitution and the typed-token probe are the
append-only shell in ``append_only_backend``, which Mamba-3 shares;
this file is how SmolLM3 loads.

Runs on GPU when available and on CPU otherwise, chosen per
activation by the supervisor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.backends.append_only_backend import AppendOnlyBackend
from src.backends.registry import SMOLLM3
from src.backends.text_adapter import SMOLLM3_TEXT
from src.backends.worker_base import Backend
from src.inference.hf_download import (
    download_with_progress,
    revision_from_snapshot,
)
from src.inference.load_progress import (
    load_target_bytes,
    sample_load_progress,
)

logger = logging.getLogger("smollm3_worker")


class Smollm3Backend(AppendOnlyBackend):
    def __init__(self) -> None:
        self.model_info = SMOLLM3
        self.text_adapter = SMOLLM3_TEXT
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cuda"
        self.load_progress: Optional[Dict[str, Any]] = None
        # Prompt, params, and per-position trace of the most recent
        # run, kept so a substitution can re-enter at any position
        # without replaying the whole generation. None until a run
        # completes.
        self.last_run_state: Optional[Dict[str, Any]] = None

    def load(self, *, device: str = "cuda") -> None:
        # Fall back to CPU when CUDA was requested but is unavailable,
        # so a GPU-less host degrades instead of erroring on load.
        resolved = (
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.device = resolved
        self.effective_device = resolved
        name = self.model_info.checkpoint
        # The commit the registry pins, passed to the fetch and to
        # both loads so a saved run names weights that cannot change
        # under it. The chat template is part of that: it ships in the
        # tokenizer files, and it decides what this model is asked.
        revision = self.model_info.revision
        assert revision, "SmolLM3 must load a pinned commit"
        # Fetch weights first (reporting progress via /health) so the
        # first activation shows a download bar; a cache hit is a
        # no-op.
        logger.info("ensuring weights for %s", name)
        snapshot = download_with_progress(
            name,
            revision=revision,
            sink=lambda p: setattr(self, "load_progress", p),
        )
        self.load_progress = None
        # What the cache resolved, read back rather than assumed, so
        # the attestation describes the files on disk. Not compared
        # against the pin: a revision may name a branch or a tag,
        # which resolves to a snapshot directory named for the commit
        # instead, and that is the case where reading it back is worth
        # most. What must hold is that a commit was resolved at all.
        self.loaded_revision = revision_from_snapshot(snapshot)
        assert self.loaded_revision is not None, (
            f"cache returned a path naming no commit: {snapshot}"
        )
        # local_files_only from here down. Returning from
        # download_with_progress means every file is on disk, so any
        # request past this point is transformers revalidating a
        # checkpoint we already have. Offline that fails outright:
        # this environment's transformers asks the Hub API for
        # additional chat templates while building the tokenizer.
        logger.info("loading tokenizer %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, local_files_only=True
        )
        logger.info(
            "loading model %s on %s (bfloat16)", name, resolved
        )
        target = load_target_bytes(
            Path(snapshot), target_dtype=torch.bfloat16
        )
        # The .to() is inside the sampled block because it is half the
        # wait on this model: from_pretrained fills RAM, then the copy
        # fills VRAM, and the sampler follows whichever is climbing.
        with sample_load_progress(
            target_bytes=target,
            sink=lambda p: setattr(self, "load_progress", p),
        ):
            # bfloat16 on both devices halves the ~12 GiB fp32
            # footprint (to ~6 GiB), which matters most for
            # CPU/RAM-constrained hosts.
            model = AutoModelForCausalLM.from_pretrained(
                name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            self.model = model.to(resolved).eval()
        logger.info("SmolLM3 loaded on %s", resolved)


def build_backend() -> Backend:
    return Smollm3Backend()
