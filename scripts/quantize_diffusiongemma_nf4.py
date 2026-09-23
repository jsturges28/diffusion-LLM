"""Quantize DiffusionGemma's MoE experts to NF4 and save.

Loads the bf16 base on CPU, replaces every fused expert stack with
an NF4 ``Experts4bit`` (leaving attention/dense/norms/embeddings/
vision in bf16), and writes a ~16 GB checkpoint that the worker
reloads with ``src.inference.dgemma_nf4``.

Everything is built inside a ``<out>.incomplete`` directory and moved
into place with a single rename at the end, because the previous
version wrote straight into the destination: a Ctrl-C during the 16 GB
``torch.save`` left a directory the menu read as an installed model
and the worker discovered was truncated, minutes into a load. A
manifest is written last, naming the base checkpoint and its revision,
this repository's commit, and the state dict's size and digest, so the
artifact can say what it is and whether it finished.

Run in the DiffusionGemma venv, with the bundled CUDA libs on the
linker path (bitsandbytes needs them):

    LD_LIBRARY_PATH="$PWD/.venv-dgemma/lib/python3.12/\
site-packages/nvidia/cu13/lib:..." \
    .venv-dgemma/bin/python scripts/quantize_diffusiongemma_nf4.py \
        --base ~/models/diffusiongemma-26B-A4B-it-bf16 \
        --out  ~/models/diffusiongemma-26B-A4B-it-nf4

An artifact built before manifests existed is attested in place
instead of rebuilt, which needs no GPU and no base checkpoint:

    .venv/bin/python scripts/quantize_diffusiongemma_nf4.py --adopt \
        --out ~/models/diffusiongemma-26B-A4B-it-nf4
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

# Running a file in ``scripts/`` puts that directory on the path, not
# the repository root, so ``src`` is not importable without this. Same
# bootstrap as ``scripts/measure_frame_payload.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The model class and the quantizer are imported inside
# ``stage_artifact`` rather than here. They need transformers v5 and
# bitsandbytes, which only ``.venv-dgemma`` has, and ``--adopt`` is
# meant to run in ``.venv`` with no GPU: a module-level import would
# make the escape hatch unreachable on exactly the machine that needs
# it, which is how it was first shipped and immediately failed.
from src.inference.artifact_manifest import (  # noqa: E402
    build_manifest,
    file_digest,
    is_complete_artifact,
    promote,
    staging_path,
    write_manifest,
)
from src.inference.hf_download import (  # noqa: E402
    DOWNLOAD_SPACE_RESERVE_BYTES,
    free_bytes,
    revision_from_snapshot,
)

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

# The name recorded in the manifest, so a directory that has been
# moved or renamed still says what it is.
ARTIFACT_NAME = "diffusiongemma-26B-A4B-nf4"

# A rough figure for the early space check only, before the model is
# loaded and the exact size is knowable. The real artifact is ~16 GiB;
# this is deliberately a little over, because the point of an early
# check is to fail before a twenty-minute load rather than to be
# precise. An exact check runs again just before the write.
STATE_DICT_ESTIMATE_BYTES = 18 * 1024**3

assert STATE_DICT_ESTIMATE_BYTES > 0, "the estimate must be a size"


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
    parser.add_argument(
        "--base-revision",
        type=str,
        default=None,
        help=(
            "Hub commit the base was fetched at. Read off the base"
            " path when it is a cache snapshot; give it explicitly"
            " for a base kept in a plain directory, so the artifact"
            " can name what it was built from."
        ),
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help=(
            "Attest an artifact that is already built instead of"
            " building one. Writes the manifest for an existing"
            " --out directory, so a working checkpoint from before"
            " manifests existed does not have to be rebuilt."
        ),
    )
    return parser.parse_args()


def _dir_size_gib(path: Path) -> float:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / 1024**3


def check_space(destination: Path, needed_bytes: int) -> None:
    """Refuse a write that will not fit, with both figures.

    Called twice: once on an estimate before the twenty-minute load,
    and once on the exact size just before the write. The early call
    saves the wait; the late one is the one that is actually right,
    and a pair of checks around a long operation is cheaper than
    trusting either alone.
    """
    assert needed_bytes > 0, "a check needs a size"
    required = needed_bytes + DOWNLOAD_SPACE_RESERVE_BYTES
    available = free_bytes(destination)
    if available < required:
        raise RuntimeError(
            f"This needs about {required / 1024**3:.1f} GiB free"
            f" under {destination}, including a"
            f" {DOWNLOAD_SPACE_RESERVE_BYTES / 1024**3:.1f} GiB"
            f" reserve, and {available / 1024**3:.1f} GiB is"
            " available. Free some space and try again."
        )


def state_dict_bytes(state: dict) -> int:
    """Exactly how much the tensors about to be written occupy.

    Summed from the tensors rather than estimated, which is possible
    here and not possible before the model is loaded. Non-tensor
    entries are ignored: they are metadata, kilobytes against
    gibibytes, and the reserve covers them many times over.
    """
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    assert total > 0, "a state dict of no tensors is not one"
    return total


def resolve_base_revision(
    base: Path, declared: str | None
) -> str | None:
    """What the base checkpoint was, if it can be named.

    An explicit value wins, because the user knows things the path
    does not. Otherwise this reads a cache snapshot path, which is
    free and exact. Neither is a failure: the manifest records null
    and still names the path, the digest and the quantizer commit.
    """
    if declared:
        return declared
    return revision_from_snapshot(str(base))


def stage_artifact(
    *, base: Path, staging: Path, base_revision: str | None
) -> None:
    """Build the whole artifact inside ``staging``.

    Everything happens in here, under a name no reader treats as a
    model, so an interruption at any phase leaves nothing that looks
    installed. The caller promotes it afterwards with one rename.
    """
    # Here, not at module scope: see the note beside the imports.
    from transformers import (  # type: ignore[attr-defined]
        DiffusionGemmaForBlockDiffusion,
    )

    from src.inference.dgemma_nf4 import (
        quantize_experts_inplace,
    )

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

    state = model.state_dict()
    exact_bytes = state_dict_bytes(state)
    print(
        f"State dict is {exact_bytes / 1024**3:.1f} GiB;"
        " checking space",
        flush=True,
    )
    check_space(staging, exact_bytes)

    print("Saving NF4 state dict ...", flush=True)
    save_start = time.monotonic()
    weights = staging / STATE_DICT_NAME
    torch.save(state, str(weights))
    print(
        f"Saved in {time.monotonic() - save_start:.1f}s",
        flush=True,
    )

    copied = []
    for name in COPY_FILES:
        source = base / name
        if source.is_file():
            shutil.copy2(source, staging / name)
            copied.append(name)

    # Last, and only now. Everything the manifest names is on disk,
    # so from this point the directory reads as complete; before it,
    # nothing did.
    print("Digesting the state dict ...", flush=True)
    digest_start = time.monotonic()
    digest = file_digest(weights)
    print(
        f"Digested in {time.monotonic() - digest_start:.1f}s",
        flush=True,
    )
    write_manifest(
        staging,
        build_manifest(
            artifact=ARTIFACT_NAME,
            base_path=str(base),
            base_revision=base_revision,
            state_dict_name=STATE_DICT_NAME,
            state_dict_bytes=weights.stat().st_size,
            state_dict_sha256=digest,
            copied_files=copied,
        ),
    )
    assert is_complete_artifact(staging), (
        "the staged artifact does not read as complete"
    )


def adopt_artifact(
    *, base: Path, out: Path, base_revision: str | None
) -> None:
    """Attest a directory that was built before manifests existed.

    The alternative was rebuilding a 16 GB file to gain a 1 KB record,
    on hardware where the build takes twenty minutes and holds the
    GPU. Adoption is honest about what it can and cannot claim: it
    digests the weights that are actually there, so the size and hash
    describe this directory, and it records the base and quantizer
    commit as the *current* ones, which is a claim the user is making
    rather than something observed.

    It cannot detect a directory that was already truncated. Nothing
    can, after the fact, which is exactly why new builds stage.
    """
    weights = out / STATE_DICT_NAME
    if not weights.is_file():
        raise RuntimeError(
            f"{weights} is missing, so there is nothing to attest."
            " Run without --adopt to build it."
        )
    print(f"Digesting {weights} ...", flush=True)
    digest_start = time.monotonic()
    digest = file_digest(weights)
    print(
        f"Digested in {time.monotonic() - digest_start:.1f}s",
        flush=True,
    )
    present = [
        name for name in COPY_FILES if (out / name).is_file()
    ]
    write_manifest(
        out,
        build_manifest(
            artifact=ARTIFACT_NAME,
            base_path=str(base),
            base_revision=base_revision,
            state_dict_name=STATE_DICT_NAME,
            state_dict_bytes=weights.stat().st_size,
            state_dict_sha256=digest,
            copied_files=present,
        ),
    )
    assert is_complete_artifact(out), (
        "the adopted artifact still does not read as complete"
    )
    print(f"Attested existing artifact at {out}")


def main() -> None:
    args = parse_args()
    base = Path(args.base).expanduser()
    out = Path(args.out).expanduser()
    base_revision = resolve_base_revision(
        base, args.base_revision
    )

    if args.adopt:
        # Deliberately before the CUDA check and the base-directory
        # check: adoption reads the output and needs neither a GPU nor
        # the base still being present, and requiring either would
        # make the escape hatch useless on the machines that need it.
        adopt_artifact(
            base=base, out=out, base_revision=base_revision
        )
        return

    assert base.is_dir(), f"base not found: {base}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for NF4 quantization.")
    if is_complete_artifact(out):
        raise RuntimeError(
            f"{out} already holds a complete artifact. Move it"
            " aside to rebuild."
        )
    if out.exists():
        raise RuntimeError(
            f"{out} exists but is not a complete artifact. Inspect"
            " and remove it, then run this again."
        )

    staging = staging_path(out)
    if staging.exists():
        # Left by an interrupted run. Removed rather than resumed:
        # there is no way to tell how far a killed `torch.save` got,
        # and a partial file that happened to be the right size would
        # be indistinguishable from a good one.
        print(f"Discarding stale staging dir {staging}", flush=True)
        shutil.rmtree(staging)

    # Early, on an estimate, so a full disk is found before the load
    # rather than after it. An exact check runs later.
    #
    # Before the mkdir, not after: refusing here used to leave an
    # empty staging directory behind, which the next run then
    # announced as stale when nothing had ever been staged in it.
    # ``free_bytes`` walks up to an existing ancestor, which is what
    # lets this measure a directory that does not exist yet.
    check_space(staging, STATE_DICT_ESTIMATE_BYTES)

    staging.mkdir(parents=True)
    try:
        stage_artifact(
            base=base,
            staging=staging,
            base_revision=base_revision,
        )
    except BaseException:
        # Including KeyboardInterrupt: a Ctrl-C partway through a
        # 16 GB save is the exact scenario this whole change is
        # about, and leaving the directory behind would recreate the
        # state that used to look installed.
        print(
            f"Failed; discarding {staging}",
            flush=True,
        )
        shutil.rmtree(staging, ignore_errors=True)
        raise

    promote(staging, out)
    print(
        f"Done. Output: {out} ({_dir_size_gib(out):.1f} GiB)"
    )


if __name__ == "__main__":
    main()
