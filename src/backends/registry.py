"""Registry of available diffusion models.

Data-only module (no torch/transformers imports) so both the
supervisor and any worker venv can import it. Worker modules are
imported lazily by ``run_worker`` based on the selected id, so a
worker venv never imports another model's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from src.backends.environments import environment_names
from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    HubFiles,
    ModelCapabilities,
    ModelInfo,
    ParamOverride,
    ParamSpec,
    ParamType,
    SignalChannel,
    is_hub_checkpoint,
)

# A full git commit. Short ones resolve until a repository grows
# enough to make them ambiguous, so they are not an address.
COMMIT_LENGTH = 40

# The signal channels the models share, declared once. Each says what
# it measures, what it varies over, and where to find it, which is the
# distinction `ROADMAP-03` exists to draw: confidence and entropy are
# both floats on a token record, so location alone cannot tell a
# per-position constant from a per-frame trajectory.

# Every diffusion position is re-decided at every denoising step, so
# what is read off a position varies over frame and position. That is
# the axis pair Analytics could not previously see, having read the
# final frame.
_DIFFUSION_SIGNALS: Tuple[SignalChannel, ...] = (
    SignalChannel(
        name="confidence",
        unit="probability",
        axes=("frame", "position"),
        location="token_record",
        key="c",
        capture="always",
    ),
    SignalChannel(
        name="entropy",
        unit="nats",
        axes=("frame", "position"),
        location="token_record",
        key="e",
        capture="always",
    ),
    SignalChannel(
        name="mean_confidence",
        unit="probability",
        axes=("frame",),
        location="frame_scalar",
        key="mean_conf",
        capture="always",
    ),
    # The autoregressive channel's name over the other axis pair, as
    # entropy is: five candidates per position at every step, rather
    # than once. The only channel with a budget, because it is the
    # only one that could outgrow the run it describes.
    SignalChannel(
        name="alternatives",
        unit="probability",
        axes=("frame", "position"),
        location="sidecar",
        key="candidates",
        capture="opt_in",
        budget_records=CANDIDATE_BUDGET_RECORDS,
    ),
)

# An autoregressive position is decided once and never revisited, so
# its confidence and entropy are per-position constants even though
# they live in the same per-frame records. Candidates are the same,
# and are the one channel here a parameter can switch off.
_AUTOREGRESSIVE_SIGNALS: Tuple[SignalChannel, ...] = (
    SignalChannel(
        name="confidence",
        unit="probability",
        axes=("position",),
        location="token_record",
        key="c",
        capture="always",
    ),
    SignalChannel(
        name="entropy",
        unit="nats",
        axes=("position",),
        location="token_record",
        key="e",
        capture="always",
    ),
    SignalChannel(
        name="mean_confidence",
        unit="probability",
        axes=("frame",),
        location="frame_scalar",
        key="mean_conf",
        capture="always",
    ),
    SignalChannel(
        name="alternatives",
        unit="probability",
        axes=("position",),
        location="sidecar",
        key="alternatives",
        capture="opt_in",
    ),
)

DEFAULT_MODEL = "llada"

_SEED_MAX = 2**31 - 1

# One parameter for both diffusion models, since the capture and its
# budget are shared. On by default, as the autoregressive one is: a
# default LLaDA run's candidates are about 4 MiB, and a run that
# would pass the budget thins to a stride instead of growing.
_DIFFUSION_ALTERNATIVES = ParamSpec(
    name="alternatives",
    label="Alternatives",
    type=ParamType.BOOL,
    default=True,
    help="Capture the five likeliest tokens at each position for"
    " every step, shown on hover. Long runs keep every few"
    " steps instead.",
)


LLADA = ModelInfo(
    id="llada",
    display_name="LLaDA-8B-Instruct",
    description=(
        "Masked discrete diffusion (semi-autoregressive)."
        " Bidirectional Transformer over a masked canvas."
    ),
    min_vram_gib=17.0,
    worker_module="src.backends.llada_worker",
    environment="core",
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
        signals=_DIFFUSION_SIGNALS,
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
        _DIFFUSION_ALTERNATIVES,
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
    environment="dgemma",
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
        # A canvas ends once it is steady and confident, by the two
        # parameters below; the readout above the canvas shows how
        # far it has left to go.
        adaptive_stopping=True,
        unresolved_char="\u2591",
        # The NF4 experts run through bitsandbytes, which needs a
        # CUDA compute path. The worker has always refused anything
        # else, but it did so inside load(), by which point the
        # previous model had already been evicted for it.
        supported_devices=("cuda",),
        signals=_DIFFUSION_SIGNALS,
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
        # The stopping rule, which the checkpoint's own
        # generation_config.json used to set out of sight. The
        # defaults are its values, so a run that leaves them alone
        # stops where every earlier run did; the worker now passes
        # them explicitly, so the rule on screen is the rule the run
        # used. Placed after the temperatures because a Compare label
        # names the first three parameters, and these would push
        # Temp Start out of every DiffusionGemma label.
        ParamSpec(
            name="confidence_threshold",
            label="Stop Entropy",
            type=ParamType.FLOAT,
            default=0.005,
            step=0.001,
            recommended=(0.001, 0.05),
            experimental=(0.0001, 1.0),
            help="Stop a canvas once its mean entropy (nats)"
            " is below this and it has held still for"
            " Steady Steps.",
        ),
        ParamSpec(
            name="stability_threshold",
            label="Steady Steps",
            type=ParamType.INT,
            default=1,
            step=1,
            recommended=(0, 4),
            experimental=(0, 16),
            help="Steps a canvas must stay unchanged before"
            " it can stop; 0 drops the condition.",
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
        _DIFFUSION_ALTERNATIVES,
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
    environment="ar",
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
        signals=_AUTOREGRESSIVE_SIGNALS,
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


# The autoregressive set plus the one state-space signal: what reading
# each token erased from the recurrent state, as a share of it. It was
# allowed in only after passing a check written before it first ran
# (manual item 329); see `mamba3_memory.forgetting`.
_STATE_SPACE_SIGNALS: Tuple[SignalChannel, ...] = (
    *_AUTOREGRESSIVE_SIGNALS,
    SignalChannel(
        name="forgetting",
        unit="fraction",
        axes=("position",),
        location="token_record",
        key="f",
        capture="always",
    ),
)


MAMBA3 = ModelInfo(
    id="mamba3",
    display_name="Mamba-3-1.5B",
    description=(
        "State-space model (left-to-right). A fixed-size recurrent"
        " state that every token decays and writes into. A base"
        " model: it continues text. Runs on GPU or CPU."
    ),
    # 1.49B parameters in float32 is 5.6 GiB of weights, and the
    # probe measured a 6.0 GiB peak on the card with a 256-token
    # prompt. The state is fixed-size, so a longer run does not grow
    # it the way a key-value cache grows.
    min_vram_gib=7.0,
    worker_module="src.backends.mamba3_worker",
    environment="ar",
    checkpoint="state-spaces/mamba3-siso-1.5b",
    revision="5cfc721542ec9ccee768088b2fd6b7e8101219d8",
    # Llama 3.1's tokenizer, which the checkpoint was trained with,
    # from SmolLM3's repository at the commit SmolLM3 runs. Meta's
    # own copy is gated, and this one is the same where ids are
    # decided; the worker refuses it if its fingerprint ever moves.
    companion=HubFiles(
        repo=SMOLLM3.checkpoint,
        revision=SMOLLM3.revision,
        files=("tokenizer.json",),
    ),
    capabilities=ModelCapabilities(
        family="state_space",
        generation_shape="append_only",
        input_mode="completion",
        supports_resume=False,
        # What If replays the prompt and the kept prefix, because a
        # recurrent state cannot be sliced back to a position the way
        # a cache can; in float32 the replay is exact.
        supports_substitution=True,
        supports_cfg=False,
        # CPU decoding cleared the bar set to decide exactly this:
        # 4.3 tokens a second in float32 against 3 (manual item 328).
        supported_devices=("cuda", "cpu"),
        signals=_STATE_SPACE_SIGNALS,
    ),
    # The same sampler, so the same knobs and the same lower CPU
    # budget, less the reasoning switch a base model has no channel
    # for.
    param_specs=[
        spec
        for spec in SMOLLM3.param_specs
        if spec.name != "thinking"
    ],
)


REGISTRY: Dict[str, ModelInfo] = {
    LLADA.id: LLADA,
    DGEMMA.id: DGEMMA,
    SMOLLM3.id: SMOLLM3,
    MAMBA3.id: MAMBA3,
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


def assert_companion_pinned(model: ModelInfo) -> None:
    """A borrowed file is fetched from the Hub by name, so it is
    pinned as a checkpoint is, and to a full commit: a moving donor
    would change what the model reads underneath a saved run."""
    companion = model.companion
    if companion is None:
        return
    assert is_hub_checkpoint(companion.repo), (
        f"{model.id} borrows from {companion.repo!r}, not a Hub repo"
    )
    assert len(companion.revision) == COMMIT_LENGTH, (
        f"{model.id}'s companion is not pinned to a full commit"
    )
    assert all(
        char in "0123456789abcdef" for char in companion.revision
    ), f"{model.id}'s companion revision is not a commit"
    assert companion.files, f"{model.id}'s companion names no files"


for _model in REGISTRY.values():
    assert_companion_pinned(_model)

# Every model names an environment the manifest declares. Asserted at
# import rather than only at launch, because the alternative is to
# find out when a user clicks the model: the interpreter lookup would
# fail after the supervisor had decided what to evict for it.
_DECLARED = environment_names()
for _model in REGISTRY.values():
    assert _model.environment in _DECLARED, (
        f"{_model.id} runs in {_model.environment!r}, which"
        " pyproject.toml does not declare"
    )


# -- How large one run can be --
#
# What a save is held to (`A2-TRUST-02`). The tops of a model's own
# sliders are the product's limit for one run, so its bounds are read
# off them rather than written out: widening a slider widens what a
# save of that model may carry, and a test holds the save's byte
# ceiling above the largest run any model can make.


@dataclass(frozen=True)
class RunBounds:
    """The most one run of a model can hold, field by field."""

    # Frames the run records. An edited run's pre-edit layer is held
    # to the same count, being another run of the same model.
    frames_max: int
    # Positions the run generates, across every canvas.
    positions_max: int
    # Positions one frame shows: one canvas, on a diffusion model.
    frame_positions_max: int


# DiffusionGemma decodes canvas by canvas, each this many tokens, as
# its Max Tokens help says. Must match the checkpoint's own
# ``canvas_length``, which only the worker reads.
DGEMMA_CANVAS_TOKENS = 256


def _slider_top(model: ModelInfo, name: str) -> int:
    """The most ``name`` can be set to on any device: the top of its
    experimental range, or of an override's where that is higher."""
    specs = [spec for spec in model.param_specs if spec.name == name]
    assert len(specs) == 1, f"{model.id} has no one {name} slider"
    spec = specs[0]
    assert spec.experimental is not None, (
        f"{model.id}'s {name} has no experimental range"
    )
    tops = [spec.experimental[1]]
    for override in (spec.overrides or {}).values():
        if override.experimental is not None:
            tops.append(override.experimental[1])
    return int(max(tops))


def _canvas_bounds(*, steps: int, positions: int) -> RunBounds:
    """One canvas denoised: the opening frame, then one per step."""
    assert steps >= 1
    assert positions >= 1
    return RunBounds(
        frames_max=steps + 1,
        positions_max=positions,
        frame_positions_max=positions,
    )


def _canvases_bounds(
    *, steps: int, positions: int, canvas: int
) -> RunBounds:
    """Canvas after canvas: a draft per step, then the commit."""
    assert steps >= 1
    assert canvas >= 1
    canvases = -(-positions // canvas)
    return RunBounds(
        frames_max=canvases * (steps + 1),
        positions_max=canvases * canvas,
        frame_positions_max=canvas,
    )


def _append_bounds(*, positions: int) -> RunBounds:
    """A frame per position, the last one holding the whole run."""
    assert positions >= 1
    return RunBounds(
        frames_max=positions,
        positions_max=positions,
        frame_positions_max=positions,
    )


RUN_BOUNDS: Dict[str, RunBounds] = {
    LLADA.id: _canvas_bounds(
        steps=_slider_top(LLADA, "steps"),
        positions=_slider_top(LLADA, "gen_length"),
    ),
    DGEMMA.id: _canvases_bounds(
        steps=_slider_top(DGEMMA, "max_denoising_steps"),
        positions=_slider_top(DGEMMA, "max_new_tokens"),
        canvas=DGEMMA_CANVAS_TOKENS,
    ),
    SMOLLM3.id: _append_bounds(
        positions=_slider_top(SMOLLM3, "max_new_tokens"),
    ),
    MAMBA3.id: _append_bounds(
        positions=_slider_top(MAMBA3, "max_new_tokens"),
    ),
}

assert set(RUN_BOUNDS) == set(REGISTRY), (
    "every model says how large one run of it can be"
)

# For a model this build does not register, which a save can name
# when it was made under another build: the most any model allows.
RUN_BOUNDS_WIDEST = RunBounds(
    frames_max=max(b.frames_max for b in RUN_BOUNDS.values()),
    positions_max=max(b.positions_max for b in RUN_BOUNDS.values()),
    frame_positions_max=max(
        b.frame_positions_max for b in RUN_BOUNDS.values()
    ),
)


def run_bounds(model_id: str) -> RunBounds:
    """How large one run of ``model_id`` can be."""
    return RUN_BOUNDS.get(model_id, RUN_BOUNDS_WIDEST)
