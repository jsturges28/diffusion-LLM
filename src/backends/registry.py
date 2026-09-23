"""Registry of available diffusion models.

Data-only module (no torch/transformers imports) so both the
supervisor and any worker venv can import it. Worker modules are
imported lazily by ``run_worker`` based on the selected id, so a
worker venv never imports another model's dependencies.
"""

from __future__ import annotations

from typing import Dict

from src.backends.protocol import (
    ModelCapabilities,
    ModelInfo,
    ParamOverride,
    ParamSpec,
    ParamType,
    is_hub_checkpoint,
)

DEFAULT_MODEL = "llada"

_SEED_MAX = 2**31 - 1


LLADA = ModelInfo(
    id="llada",
    display_name="LLaDA-8B-Instruct",
    description=(
        "Masked discrete diffusion (semi-autoregressive)."
        " Bidirectional Transformer over a masked canvas."
    ),
    min_vram_gib=17.0,
    worker_module="src.backends.llada_worker",
    venv_python=".venv/bin/python",
    checkpoint="GSAI-ML/LLaDA-8B-Instruct",
    # The commit every run of this model has been made against, taken
    # from the cache that produced them rather than from the Hub, so
    # pinning changes nothing about what loads today. It only stops
    # the repository moving underneath a saved run. This model also
    # executes remote code, which the same commit pins.
    revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
    capabilities=ModelCapabilities(
        family="diffusion",
        generation_shape="iterative_canvas",
        input_mode="chat",
        supports_resume=True,
        supports_cfg=True,
        unresolved_char="\u2591",
        # 17 GiB of bf16 weights. The CPU placement this used to
        # advertise came from the old default rather than from a
        # decision, and it was unreachable and unchecked at the same
        # time: the menu gives diffusion rows a static GPU tag, the
        # Help text only ever claimed CPU for SmolLM3, and the
        # headroom pre-flight skips CPU entirely. Declaring the truth
        # closes a 17 GiB host allocation nothing was measuring.
        supported_devices=("cuda",),
    ),
    param_specs=[
        ParamSpec(
            name="steps",
            label="Steps",
            type=ParamType.INT,
            default=128,
            step=1,
            recommended=(8, 150),
            experimental=(1, 1024),
            help="Number of denoising steps.",
        ),
        ParamSpec(
            name="gen_length",
            label="Gen Length",
            type=ParamType.INT,
            default=160,
            step=1,
            recommended=(16, 160),
            experimental=(1, 1024),
            help="Length of the generated canvas.",
        ),
        ParamSpec(
            name="block_length",
            label="Block Length",
            type=ParamType.INT,
            default=160,
            step=1,
            recommended=(8, 160),
            experimental=(1, 1024),
            help="Semi-autoregressive block size.",
        ),
        ParamSpec(
            name="temperature",
            label="Temperature",
            type=ParamType.FLOAT,
            default=0.0,
            step=0.05,
            recommended=(0.0, 1.0),
            experimental=(0.0, 10.0),
            help="Gumbel sampling temperature.",
        ),
        ParamSpec(
            name="cfg_scale",
            label="CFG Scale",
            type=ParamType.FLOAT,
            default=0.0,
            step=0.1,
            recommended=(0.0, 2.0),
            experimental=(0.0, 20.0),
            help="Classifier-free guidance strength.",
        ),
        ParamSpec(
            name="seed",
            label="Seed",
            type=ParamType.INT,
            default=-1,
            step=1,
            recommended=(-1, _SEED_MAX),
            experimental=(-1, _SEED_MAX),
            help="Random seed; -1 = nondeterministic.",
        ),
        ParamSpec(
            name="remasking",
            label="Remasking",
            type=ParamType.SELECT,
            default="low_confidence",
            options=["low_confidence", "random"],
            help="Remasking strategy.",
        ),
    ],
)


DGEMMA = ModelInfo(
    id="diffusiongemma",
    display_name="DiffusionGemma-26B-A4B",
    description=(
        "Block-autoregressive text diffusion"
        " (encoder-decoder MoE), 4-bit NF4 experts."
        " Denoises 256-token canvases with adaptive"
        " stopping."
    ),
    min_vram_gib=18.0,
    worker_module="src.backends.dgemma_worker",
    venv_python=".venv-dgemma/bin/python",
    checkpoint="~/models/diffusiongemma-26B-A4B-it-nf4",
    capabilities=ModelCapabilities(
        family="diffusion",
        generation_shape="iterative_canvas",
        input_mode="chat",
        supports_resume=True,
        supports_cfg=False,
        # Resume renoises remasked positions instead of hard-masking
        # them, so committed neighbours can move as well.
        remask_renoises=True,
        unresolved_char="\u2591",
        # The NF4 experts run through bitsandbytes, which needs a
        # CUDA compute path. The worker has always refused anything
        # else, but it did so inside load(), by which point the
        # previous model had already been evicted for it.
        supported_devices=("cuda",),
    ),
    param_specs=[
        ParamSpec(
            name="max_new_tokens",
            label="Max Tokens",
            type=ParamType.INT,
            default=256,
            step=64,
            recommended=(64, 512),
            experimental=(64, 2048),
            help="Output budget; canvases of 256 tokens.",
        ),
        ParamSpec(
            name="max_denoising_steps",
            label="Denoising Steps",
            type=ParamType.INT,
            default=48,
            step=1,
            recommended=(4, 64),
            experimental=(1, 256),
            help="Max denoising steps per canvas"
            " (adaptive stopping may use fewer).",
        ),
        ParamSpec(
            name="t_max",
            label="Temp Start",
            type=ParamType.FLOAT,
            default=0.8,
            step=0.05,
            recommended=(0.0, 2.0),
            experimental=(0.0, 5.0),
            help="Initial temperature in the schedule.",
        ),
        ParamSpec(
            name="t_min",
            label="Temp End",
            type=ParamType.FLOAT,
            default=0.4,
            step=0.05,
            recommended=(0.0, 2.0),
            experimental=(0.0, 5.0),
            help="Final temperature in the schedule.",
        ),
        ParamSpec(
            name="seed",
            label="Seed",
            type=ParamType.INT,
            default=-1,
            step=1,
            recommended=(-1, _SEED_MAX),
            experimental=(-1, _SEED_MAX),
            help="Random seed; -1 = nondeterministic.",
        ),
        ParamSpec(
            name="thinking",
            label="Thinking",
            type=ParamType.BOOL,
            default=False,
            help="Enable the step-by-step reasoning"
            " channel.",
        ),
    ],
)


SMOLLM3 = ModelInfo(
    id="smollm3",
    display_name="SmolLM3-3B",
    description=(
        "Autoregressive transformer (left-to-right)."
        " Decoder-only, streamed token-by-token with"
        " per-token sampling confidence. Runs on GPU or CPU."
    ),
    # 3.08B params in bf16 (~6 GiB weights) plus KV cache and
    # activations. Only consulted for the GPU pre-flight; a CPU
    # activation skips it, which is why this model needs no host
    # memory figure to be loadable on a GPU-less machine.
    min_vram_gib=8.0,
    worker_module="src.backends.smollm3_worker",
    venv_python=".venv-ar/bin/python",
    checkpoint="HuggingFaceTB/SmolLM3-3B",
    # As with LLaDA, the commit already in the cache, so pinning is a
    # record of what has been running rather than a change to it.
    revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
    capabilities=ModelCapabilities(
        family="autoregressive",
        generation_shape="append_only",
        input_mode="chat",
        # Left-to-right, so no diffusion remask/resume. Substitution
        # is the autoregressive counterfactual instead: it needs the
        # Alternatives capture, which the frontend gates on.
        supports_resume=False,
        supports_substitution=True,
        supports_cfg=False,
        # The model a GPU-less host can use, so CPU is a placement
        # this one genuinely supports rather than one it inherited.
        supported_devices=("cuda", "cpu"),
    ),
    param_specs=[
        ParamSpec(
            name="max_new_tokens",
            label="Max Tokens",
            type=ParamType.INT,
            default=256,
            step=1,
            # Decoding is sequential, so the time a run takes grows
            # with the count, and that is now the whole reason the
            # recommended ceiling stays modest. It used to be a
            # payload argument: RUNTIME-01 made autoregressive frames
            # append-only on the wire and flat on disk, which took a
            # 2,048-token run from 130 MiB to about 1 MiB.
            recommended=(16, 256),
            experimental=(1, 2048),
            # CPU decoding is slow, so the default budget is lower
            # and the recommended cap is 128 there, shown in the UI
            # rather than applied as a hidden clamp. Experimental
            # still lifts it.
            overrides={
                "cpu": ParamOverride(
                    default=128, recommended=(16, 128)
                )
            },
            help="Number of tokens to generate.",
        ),
        ParamSpec(
            name="temperature",
            label="Temperature",
            type=ParamType.FLOAT,
            default=0.6,
            step=0.05,
            recommended=(0.0, 1.5),
            experimental=(0.0, 10.0),
            help="Sampling temperature; 0 is greedy (argmax).",
        ),
        ParamSpec(
            name="top_p",
            label="Top-p",
            type=ParamType.FLOAT,
            default=0.95,
            step=0.05,
            recommended=(0.0, 1.0),
            experimental=(0.0, 1.0),
            help="Nucleus sampling probability mass.",
        ),
        # Applied before top-p, matching Hugging Face, so the two
        # compose as a hard truncation followed by a nucleus cut
        # within it rather than as competing choices. Unrelated to
        # the fixed five candidates the Alternatives capture
        # records; that count is a separate knob.
        #
        # Off is -1, not Hugging Face's 0. A k of 0 reads as "keep
        # zero tokens", which in a tool built to explain sampling is
        # a worse first impression than matching an upstream default
        # nobody here sees. It also puts this in line with Seed
        # below, which already spends -1 on "unset". The filter
        # disables on anything <= 0, so runs saved with 0 still mean
        # what they meant.
        ParamSpec(
            name="top_k",
            label="Top-k",
            type=ParamType.INT,
            default=-1,
            step=1,
            recommended=(-1, 100),
            experimental=(-1, 1000),
            help="Keep only the k likeliest tokens before"
            " top-p. -1 keeps all of them.",
        ),
        ParamSpec(
            name="seed",
            label="Seed",
            type=ParamType.INT,
            default=-1,
            step=1,
            recommended=(-1, _SEED_MAX),
            experimental=(-1, _SEED_MAX),
            help="Random seed; -1 = nondeterministic.",
        ),
        ParamSpec(
            name="thinking",
            label="Thinking",
            type=ParamType.BOOL,
            default=False,
            help="Enable the extended reasoning channel"
            " (shown in a separate panel).",
        ),
        # On by default: it is what makes the hover popover and What
        # If? substitution work at all, so leaving it off meant the
        # model's two most interesting affordances were invisible
        # until you found the toggle. The capture is a top-k over the
        # logits already computed, and only the frame that introduces
        # a position carries its candidates (see ar_sampler), so the
        # cost is small and the payload grows linearly.
        ParamSpec(
            name="alternatives",
            label="Alternatives",
            type=ParamType.BOOL,
            default=True,
            help="Capture the top competing tokens at each"
            " position, shown on hover and required for"
            " What If substitution (slightly slower).",
        ),
    ],
)


REGISTRY: Dict[str, ModelInfo] = {
    LLADA.id: LLADA,
    DGEMMA.id: DGEMMA,
    SMOLLM3.id: SMOLLM3,
}

# Anything fetched from the Hub names the commit it was fetched at,
# and anything local does not, because a local directory has no commit
# to name and says what it is through its manifest instead. Asserted
# rather than tested only, so a new Hub model cannot be registered
# unpinned: the whole point is that the repository must not be able to
# move underneath a saved run.
for _model in REGISTRY.values():
    if is_hub_checkpoint(_model.checkpoint):
        assert _model.revision, (
            f"{_model.id} loads from the Hub and must pin a revision"
        )
    else:
        assert _model.revision is None, (
            f"{_model.id} is a local artifact and has no Hub revision"
        )
