"""Probe: does our Mamba-3 run the real checkpoint, and is what its
state retains worth drawing?

The unit tests hold `src/inference/mamba3.py` to upstream's two
PyTorch references on random inputs. Three things they cannot show:
that the glue around the recurrence matches upstream's on trained
weights, how fast a plain PyTorch loop is on this project's hardware,
and whether the retention in `src/inference/mamba3_memory.py` says
anything a reader could not have guessed. This answers all three on
the pinned checkpoint. The criteria were fixed before the first run
(manual item 328), so a disappointing number cannot move the bar.

The load report always runs, since nothing else means anything
without it. The rest are chosen with --sections:

- correctness: our recurrence against upstream's parallel form on the
  model's real activations, perplexity on three passages, and greedy
  completions for three prompts.
- speed: prompt and decode throughput, and peak memory.
- retention: the three tests that retired the attention overlay, run
  on what each layer's state keeps and on its output-side attention.
- forgetting: what reading each token erased, the one per-token
  signal a Mamba-3 run would draw, held to its own bar (manual item
  329) before any overlay shows it.

The checkpoint was trained with Llama 3.1's tokenizer, whose own
repository is gated. SmolLM3's, which this project already pins, is
the same tokenizer in everything that decides a text's ids, so the
probe takes that one file from SmolLM3 and checks its fingerprint
against Llama 3.1's before trusting it.

Run it unsandboxed, in the autoregressive environment, once per
device:

    .venv-ar/bin/python scripts/probe_mamba3.py --device cuda \\
        --json ~/mamba3-probe-cuda.json
    .venv-ar/bin/python scripts/probe_mamba3.py --device cpu \\
        --dtype float32 --json ~/mamba3-probe-cpu.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn.functional as F
from torch import Tensor

# Running a file in `scripts/` puts that directory on the path, not
# the repository root. Same bootstrap as the NF4 quantizer.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The tokenizer and the Hub client are imported where they are used,
# so the suite can import this file in `.venv` and drive it with a
# tiny model and a stand-in tokenizer.
from reference.mamba3 import siso_reference as ref  # noqa: E402
from src.backends.registry import SMOLLM3  # noqa: E402
from src.inference import mamba3  # noqa: E402
from src.inference import mamba3_memory as memory  # noqa: E402
from src.inference.artifact_manifest import repo_commit  # noqa: E402
from src.inference.mamba3 import (  # noqa: E402
    CoreInputs,
    LayerState,
    Mamba3LM,
    StepTerms,
)
from src.inference.mamba3_tokenizer import (  # noqa: E402
    LLAMA31_BPE_FINGERPRINT,
    TOKENIZER_FILE,
    Llama31Tokenizer,
    load_tokenizer,
)

MODEL_REPO = "state-spaces/mamba3-siso-1.5b"
MODEL_REVISION = "5cfc721542ec9ccee768088b2fd6b7e8101219d8"
MODEL_FILES = (mamba3.CONFIG_NAME, mamba3.WEIGHTS_NAME)
# SmolLM3's copy of Llama 3.1's tokenizer, at the revision the
# registry already runs. Meta's own is gated; this one is not, and
# differs only in ten reserved special tokens SmolLM3 renamed for its
# chat format, which `load_tokenizer` drops.
TOKENIZER_REPO = SMOLLM3.checkpoint
TOKENIZER_REVISION = SMOLLM3.revision
TOKENIZER_FILES = (TOKENIZER_FILE,)

# The pass criteria, fixed before the first run. Changing one after
# seeing a result would leave the probe deciding nothing.
AGREEMENT_MAX = 1e-3  # relative error between the two cores, fp32
PERPLEXITY_GLUE_ERROR = 40.0  # above this, the glue is wrong
PERPLEXITY_EXPECTED = (10.0, 20.0)
CPU_DECODE_MIN = 3.0  # tokens per second
TOP_POSITIONS = 20
RECENCY_SPEARMAN = 0.9  # above this, a layer keeps only recency
CONTENT_VARIATION = 0.05  # below this, decay ignores content
# The forgetting bar, fixed before its first run (manual item 329).
# It judges the one per-token signal a Mamba-3 run would draw.
FORGETTING_WARM_UP = 8  # tokens the repeated input's state settles in
FORGETTING_FLAT_RATIO = 0.5  # the repeated input's share of variation
CHANCE_FLOOR = 2.0  # multiples of chance before an overlap counts
FORGETTING_TOP_TOKENS = 10  # most-forgetting tokens the report shows

AGREEMENT_TOKENS = 128
RETENTION_TOKENS = 128
SPEED_PROMPT_TOKENS = 256
SPEED_DECODE_TOKENS = 128
COMPLETION_TOKENS = 40
WARM_UP_TOKENS = 8

SECTIONS = ("correctness", "speed", "retention", "forgetting")
DTYPES: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

assert PERPLEXITY_EXPECTED[1] < PERPLEXITY_GLUE_ERROR, "bands"
assert 0 < TOP_POSITIONS < RETENTION_TOKENS, "top of what"
assert WARM_UP_TOKENS < SPEED_PROMPT_TOKENS, "warm-up is short"

# Written for this probe rather than copied, so no passage can be one
# the checkpoint memorised. Plain explanatory prose, which is what its
# training set, FineWeb-Edu, is made of.
PASSAGES = (
    "Bread rises because of yeast, a single-celled fungus that feeds"
    " on the sugars in flour. As the yeast digests those sugars, it"
    " releases carbon dioxide and a small amount of alcohol."
    " Kneading the dough develops gluten, a network of proteins that"
    " forms when flour is mixed with water. The gluten is elastic"
    " enough to stretch around each bubble of gas instead of letting"
    " it escape, so the dough slowly swells. Warmth speeds the"
    " process up, which is why recipes suggest leaving the dough"
    " somewhere warm to prove. When the loaf goes into a hot oven,"
    " the gas expands quickly, and the yeast dies once the dough"
    " passes about sixty degrees Celsius. The heat also sets the"
    " gluten and the starch around the bubbles, fixing the structure"
    " in place. The alcohol evaporates, and the surface browns as"
    " sugars and proteins react with one another. The result is a"
    " firm crust around a soft interior full of small holes, each"
    " one left by a pocket of gas.",
    "The water cycle describes how water moves between the oceans,"
    " the air and the land. Energy from the sun heats the surface of"
    " the sea, and some of the water evaporates, rising into the"
    " atmosphere as an invisible gas. Plants add more water vapour"
    " through their leaves in a process called transpiration. As the"
    " warm, moist air rises, it expands and cools. Cooler air cannot"
    " hold as much vapour, so the vapour condenses onto tiny"
    " particles of dust or salt, forming the droplets that make up"
    " clouds. When the droplets collide and grow heavy enough, they"
    " fall as rain, or as snow if the air is cold. Some of that"
    " water runs off the land into streams and rivers, which carry"
    " it back to the sea. Some soaks into the ground, where it can"
    " remain for years as groundwater before it reaches a spring or"
    " a well. The same water has been moving through this cycle for"
    " billions of years.",
    "The human heart is a muscular pump about the size of a closed"
    " fist. It has four chambers: two upper chambers called atria"
    " and two lower chambers called ventricles. The right side of"
    " the heart receives blood that has already delivered its oxygen"
    " to the body and pumps it to the lungs. There, the blood"
    " releases carbon dioxide and picks up fresh oxygen. The left"
    " side receives this oxygen-rich blood from the lungs and pumps"
    " it out through the aorta to the rest of the body. Valves"
    " between the chambers open and close in sequence, so that blood"
    " flows in only one direction. The familiar sound of a heartbeat"
    " is the noise of these valves closing. Each beat is triggered"
    " by an electrical signal that starts in a small cluster of"
    " cells called the sinoatrial node. At rest, an adult heart"
    " beats roughly sixty to one hundred times a minute, moving"
    " around five litres of blood every minute.",
)
PROMPTS = (
    "The three states of matter are",
    "In 1492, Christopher Columbus",
    "To make a cup of tea, first",
)
# The token the degenerate input repeats: common, and one token in
# Llama 3's vocabulary.
REPEATED_TEXT = " the"


class TextCodec(NamedTuple):
    """The tokenizer, reduced to what the probe asks of it."""

    encode: Callable[[str], List[int]]
    decode: Callable[[Sequence[int]], str]
    vocabulary: int  # every id it can produce, specials included
    end: Optional[int]  # end of text, where a completion stops
    fingerprint: str  # `bpe_fingerprint` of the file it came from


Verdict = Dict[str, str]
Section = Dict[str, Any]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0 when the probe ran, whatever it found; 1 when the checkpoint
    would not load or the tokenizer is not Llama 3.1's; 2 when CUDA
    was asked for and is not there."""
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return _refuse(
            "CUDA is not available to this process. Run it outside"
            " the sandbox, or pass --device cpu --dtype float32."
        )
    checkpoint = _resolve(
        args.checkpoint, MODEL_REPO, MODEL_REVISION, MODEL_FILES
    )
    tokenizer = _resolve(
        args.tokenizer,
        TOKENIZER_REPO,
        TOKENIZER_REVISION,
        TOKENIZER_FILES,
    )
    report = run(
        checkpoint=checkpoint,
        codec=load_codec(tokenizer),
        device=device,
        dtype=DTYPES[args.dtype],
        sections=args.sections,
    )
    report["run"]["tokenizer"] = str(tokenizer)
    print_report(report)
    if args.json is not None:
        write_report(args.json.expanduser(), report)
    if report["load"]["runnable"]:
        return 0
    return 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Mamba-3 SISO on the pinned checkpoint."
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--dtype", choices=tuple(DTYPES), default="bfloat16"
    )
    parser.add_argument(
        "--sections",
        type=_sections,
        default=SECTIONS,
        help=(
            "Comma-separated, from " + ", ".join(SECTIONS) + "."
            " The load report always runs."
        ),
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Also write the whole report here, as JSON.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="A local checkpoint directory, instead of the download.",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=None,
        help="A local tokenizer directory, instead of the download.",
    )
    return parser.parse_args(argv)


def _sections(text: str) -> Tuple[str, ...]:
    """The sections named, in the order they run."""
    named = {part.strip() for part in text.split(",")} - {""}
    unknown = sorted(named - set(SECTIONS))
    if unknown or not named:
        raise argparse.ArgumentTypeError(
            f"choose from {', '.join(SECTIONS)}, not {text!r}"
        )
    return tuple(name for name in SECTIONS if name in named)


def _refuse(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


# -- fetching --


def _resolve(
    local: Optional[Path],
    repo: str,
    revision: str,
    files: Sequence[str],
) -> Path:
    if local is not None:
        return local.expanduser()
    return fetch(repo, revision, files)


def fetch(repo: str, revision: str, files: Sequence[str]) -> Path:
    """`files` from `repo` at a pinned commit, from the cache when
    they are already there, and nothing else from the repository."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo, revision=revision, allow_patterns=list(files)
    )
    return Path(path)


def load_codec(directory: Path) -> TextCodec:
    """Llama 3.1's tokenizer from the file in `directory`, unchecked
    here: the load report judges its fingerprint rather than refusing
    it, so a mismatch is a finding instead of a traceback."""
    tokenizer = load_tokenizer(
        directory / TOKENIZER_FILE,
        source=str(directory),
        required=None,
    )
    return codec_of(tokenizer)


def codec_of(tokenizer: Llama31Tokenizer) -> TextCodec:
    """The probe's view of the shared tokenizer."""

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    return TextCodec(
        tokenizer.encode,
        decode,
        len(tokenizer),
        tokenizer.eos_token_id,
        tokenizer.fingerprint,
    )


# -- running --


def run(
    *,
    checkpoint: Path,
    codec: TextCodec,
    device: torch.device,
    dtype: torch.dtype,
    sections: Sequence[str],
) -> Dict[str, Any]:
    """Every section asked for, after the load report. The report
    stops there if the checkpoint would not load or the tokenizer is
    not the one it was trained with, since every number after would
    describe something else."""
    report: Dict[str, Any] = {
        "run": describe_run(checkpoint, device, dtype)
    }
    _say("load report")
    report["load"] = load_report(checkpoint, codec)
    if not report["load"]["runnable"]:
        return report
    started = time.perf_counter()
    model = mamba3.load(checkpoint, device=device, dtype=dtype)
    report["load"]["seconds"] = time.perf_counter() - started
    for name in sections:
        _say(name)
        with torch.no_grad():
            report[name] = RUNNERS[name](model, codec)
    return report


def describe_run(
    checkpoint: Path, device: torch.device, dtype: torch.dtype
) -> Dict[str, Any]:
    """What a later reader needs to know to reproduce the numbers."""
    return {
        "checkpoint": str(checkpoint),
        "model_revision": MODEL_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "device": _device_name(device),
        "dtype": str(dtype).removeprefix("torch."),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "commit": repo_commit(),
    }


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def _say(section: str) -> None:
    print(f"running: {section}", file=sys.stderr, flush=True)


# -- the load report --


def load_report(checkpoint: Path, codec: TextCodec) -> Section:
    """The checkpoint's names and shapes against the model's, and the
    tokenizer against Llama 3.1's and the config's vocabulary, before
    any weight is used. A tied head may be stored or left out; both
    are complete."""
    path = checkpoint / mamba3.CONFIG_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = mamba3.config_from_json(raw)
    stored = torch.load(
        checkpoint / mamba3.WEIGHTS_NAME,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    wanted = Mamba3LM(config, device="meta").state_dict()
    missing = sorted(set(wanted) - set(stored) - {"lm_head.weight"})
    unexpected = sorted(set(stored) - set(wanted))
    misshapen = _misshapen(wanted, stored)
    vocabulary = raw["vocab_size"]
    loadable = not (missing or unexpected or misshapen)
    llama = codec.fingerprint == LLAMA31_BPE_FINGERPRINT
    return {
        "tensors": len(stored),
        "head_stored": "lm_head.weight" in stored,
        "missing": missing,
        "unexpected": unexpected,
        "misshapen": misshapen,
        "loadable": loadable,
        "tokenizer_fingerprint": codec.fingerprint,
        "runnable": loadable and llama,
        "config_vocabulary": vocabulary,
        "tokenizer_vocabulary": codec.vocabulary,
        "verdicts": [
            _verdict(
                "no missing, unexpected or misshapen tensors",
                loadable,
                f"{len(stored)} tensors stored",
            ),
            _verdict(
                "the tokenizer is Llama 3.1's, by fingerprint",
                llama,
                f"{codec.fingerprint[:16]}, recorded"
                f" {LLAMA31_BPE_FINGERPRINT[:16]}",
            ),
            _verdict(
                "the tokenizer's vocabulary is the config's",
                codec.vocabulary == vocabulary,
                f"{codec.vocabulary} against {vocabulary}",
            ),
        ],
    }


def _misshapen(
    wanted: Dict[str, Tensor], stored: Dict[str, Tensor]
) -> List[str]:
    shared = sorted(set(wanted) & set(stored))
    return [
        name
        for name in shared
        if wanted[name].shape != stored[name].shape
    ]


# -- correctness --


def correctness(model: Mamba3LM, codec: TextCodec) -> Section:
    sample = _encode_exactly(codec, PASSAGES[0], AGREEMENT_TOKENS)
    agreement = core_agreement(model, _on(model, sample))
    perplexities = [
        perplexity(model, _on(model, codec.encode(text)))
        for text in PASSAGES
    ]
    completions = {
        prompt: codec.decode(
            complete(model, codec.encode(prompt), codec.end)
        )
        for prompt in PROMPTS
    }
    return {
        "agreement": agreement,
        "perplexity": perplexities,
        "completions": completions,
        "verdicts": [
            _agreement_verdict(model, agreement),
            _perplexity_verdict(perplexities),
            _verdict(
                "the completions read coherently", None,
                "read them",
            ),
        ],
    }


def parallel_core(
    inputs: CoreInputs, state: LayerState
) -> Tuple[Tensor, LayerState]:
    """Upstream's parallel form, as a core. A quadratic sum rather
    than a loop, so agreeing with it is evidence and not a copy of
    the loop checking itself."""
    out, final = ref.mamba3_siso_fwd_ref(
        inputs.q, inputs.k, inputs.v, inputs.adt, inputs.dt,
        inputs.trap, inputs.q_bias, inputs.k_bias, inputs.angles,
        inputs.d, inputs.z,
        Initial_States=tuple(state),
    )
    return out, LayerState(*final)


def core_agreement(model: Mamba3LM, ids: Tensor) -> Dict[str, Any]:
    """Our recurrence against upstream's parallel form, through the
    same weights and glue, on the same passage."""
    ours, _ = model(ids, model.empty_states(1))
    theirs, _ = model(ids, model.empty_states(1), core=parallel_core)
    same = ours.argmax(dim=-1) == theirs.argmax(dim=-1)
    return {
        "tokens": ids.shape[1],
        "relative_error": _relative(ours, theirs),
        "top1_agreement": float(same.float().mean()),
    }


def perplexity(model: Mamba3LM, ids: Tensor) -> float:
    """exp of the mean loss predicting each token from all before
    it, from the empty state."""
    logits, _ = model(ids, model.empty_states(1))
    loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
    return math.exp(float(loss))


def complete(
    model: Mamba3LM, prompt: List[int], end: Optional[int]
) -> List[int]:
    """A greedy continuation through carried states, one token per
    call: the path a worker's decode loop will take."""
    states = model.empty_states(1)
    ids = _on(model, prompt)
    produced: List[int] = []
    for _ in range(COMPLETION_TOKENS):
        logits, states = model(ids, states)
        token = int(logits[0, -1].argmax())
        if token == end:
            break
        produced.append(token)
        ids = _on(model, [token])
    return produced


def _agreement_verdict(
    model: Mamba3LM, agreement: Dict[str, Any]
) -> Verdict:
    error = agreement["relative_error"]
    criterion = (
        f"our recurrence agrees with upstream's parallel form to a"
        f" relative error under {AGREEMENT_MAX:g}"
    )
    if model.compute_dtype != torch.float32:
        return _verdict(
            criterion, None,
            f"{error:.2e}; the criterion is for a float32 run",
        )
    return _verdict(criterion, error < AGREEMENT_MAX, f"{error:.2e}")


def _perplexity_verdict(perplexities: List[float]) -> Verdict:
    low, high = PERPLEXITY_EXPECTED
    worst = max(perplexities)
    shown = ", ".join(f"{value:.1f}" for value in perplexities)
    expected = all(low <= value <= high for value in perplexities)
    band = "within" if expected else "outside"
    return _verdict(
        f"no passage's perplexity above {PERPLEXITY_GLUE_ERROR:g}",
        worst <= PERPLEXITY_GLUE_ERROR,
        f"{shown}; {band} the expected {low:g} to {high:g}",
    )


# -- speed --


def speed(model: Mamba3LM, codec: TextCodec) -> Section:
    """Throughput of the plain PyTorch loop: the prompt in one call,
    then one token per call with the host reading each token, as a
    streaming worker would."""
    device = model.lm_head.weight.device
    joined = " ".join(PASSAGES)
    prompt = _on(
        model, _encode_exactly(codec, joined, SPEED_PROMPT_TOKENS)
    )
    model(prompt[:, :WARM_UP_TOKENS], model.empty_states(1))
    _reset_peak(device)
    started = _clock(device)
    logits, states = model(prompt, model.empty_states(1))
    prompt_seconds = _clock(device) - started
    token = int(logits[0, -1].argmax())
    started = _clock(device)
    for _ in range(SPEED_DECODE_TOKENS):
        logits, states = model(_on(model, [token]), states)
        token = int(logits[0, -1].argmax())
    decode_rate = SPEED_DECODE_TOKENS / (_clock(device) - started)
    return {
        "prompt_tokens": SPEED_PROMPT_TOKENS,
        "prompt_tokens_per_second": (
            SPEED_PROMPT_TOKENS / prompt_seconds
        ),
        "decode_tokens": SPEED_DECODE_TOKENS,
        "decode_tokens_per_second": decode_rate,
        "peak_vram_mib": _peak_vram_mib(device),
        "peak_host_mib": _peak_host_mib(),
        "verdicts": [_cpu_verdict(device, decode_rate)],
    }


def _cpu_verdict(device: torch.device, rate: float) -> Verdict:
    criterion = (
        f"decoding on CPU reaches {CPU_DECODE_MIN:g} tokens a second"
    )
    detail = f"{rate:.1f} tokens a second on {device.type}"
    if device.type != "cpu":
        return _verdict(criterion, None, detail)
    return _verdict(criterion, rate >= CPU_DECODE_MIN, detail)


def _clock(device: torch.device) -> float:
    """Seconds, once the device has finished what it was given."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mib(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def _peak_host_mib() -> float:
    """The process's peak resident memory since it started, loading
    included. Linux reports it in KiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


# -- retention --


class LayerLens(NamedTuple):
    """What one layer holds of each token after an input's last one.

    `state` and `output` are head-averaged scores over positions:
    each token's share of the state, and of the last output. `alpha`
    is every step's decay per head, (length, heads).
    """

    state: Tensor
    output: Tensor
    alpha: Tensor
    rebuild_error: float  # the weights' state against the stepped one


# The two real inputs; the third, "repeated", is the degenerate one.
PASSAGE_INPUTS = ("passage_a", "passage_b")


def retention(model: Mamba3LM, codec: TextCodec) -> Section:
    """The three tests, on the state and on the output side.

    A lens survives only if it fails none of them. The rebuild error
    comes first: a lens whose weights do not rebuild the state they
    describe is not measuring anything, whatever the tests say.
    """
    lenses = {
        name: read_lenses(model, _on(model, ids))
        for name, ids in _falsification_inputs(codec).items()
    }
    rebuild = _worst_rebuild(lenses)
    alpha = alpha_variation(lenses)
    return {
        "tokens": RETENTION_TOKENS,
        "chance_overlap": TOP_POSITIONS**2 / RETENTION_TOKENS,
        "rebuild_error_max": rebuild,
        "alpha": alpha,
        "state": falsify(lenses, "state"),
        "output": falsify(lenses, "output"),
        "verdicts": [
            _verdict(
                "the retention weights rebuild every stepped state",
                rebuild < AGREEMENT_MAX,
                f"worst relative error {rebuild:.2e}",
            ),
            alpha["verdict"],
        ],
    }


def read_lenses(model: Mamba3LM, ids: Tensor) -> List[LayerLens]:
    """Every layer's lens on one input, run from the empty state,
    which is what the retention weights assume."""
    capture: List[List[StepTerms]] = [
        [] for _ in range(model.config.n_layer)
    ]
    _, states = model(ids, model.empty_states(1), capture=capture)
    return [
        _lens(records, state.ssm[0])
        for records, state in zip(capture, states, strict=True)
    ]


def _lens(records: Sequence[StepTerms], stepped: Tensor) -> LayerLens:
    trace = memory.stack_terms(records)
    last = trace.adt.shape[0] - 1
    weights = memory.retention_weights(trace, last)
    rebuilt = memory.reconstruct_state(trace, weights)
    attention = memory.output_attention(trace, weights, last)
    held = memory.contribution_norms(trace, weights)
    reached = memory.output_norms(trace, attention)
    return LayerLens(
        state=memory.head_average(held).cpu(),
        output=memory.head_average(reached).cpu(),
        alpha=torch.exp(trace.adt).cpu(),
        rebuild_error=_relative(rebuilt, stepped),
    )


def falsify(
    lenses: Dict[str, List[LayerLens]], field: str
) -> Section:
    """The first two tests on one lens, layer by layer.

    Degenerate: the repeated token shares as many top positions with
    the passages as they share with each other, so the pattern is
    about position, not text. Recency: the score climbs with position
    almost perfectly, so the lens says only that recent tokens count.
    Either one in most layers fails the lens.
    """
    # Near chance, about 3 of 20 here, the degenerate test is a coin
    # toss; the report carries `chance_overlap` so a failure there can
    # be told from one caused by positions every input shares.
    layers = [
        _layer_row(lenses, index, field)
        for index in range(len(lenses["passage_a"]))
    ]
    structural = 0
    recent = 0
    for row in layers:
        if row["repeated_overlap"] >= row["passages_overlap"]:
            structural += 1
        if row["position_spearman"] > RECENCY_SPEARMAN:
            recent += 1
    count = len(layers)
    return {
        "layers": layers,
        "verdicts": [
            _verdict(
                f"{field}: real text shares more top positions than"
                " the repeated token does, in most layers",
                2 * structural <= count,
                f"the repeated token matched in {structural} of"
                f" {count} layers",
            ),
            _verdict(
                f"{field}: not only recency, in most layers",
                2 * recent <= count,
                f"Spearman with position above {RECENCY_SPEARMAN:g}"
                f" in {recent} of {count} layers",
            ),
        ],
    }


def _layer_row(
    lenses: Dict[str, List[LayerLens]], index: int, field: str
) -> Dict[str, float]:
    first = getattr(lenses["passage_a"][index], field)
    second = getattr(lenses["passage_b"][index], field)
    repeated = getattr(lenses["repeated"][index], field)
    positions = torch.arange(first.shape[0], dtype=torch.float64)
    with_repeated = memory.top_overlap(
        first, repeated, TOP_POSITIONS
    ) + memory.top_overlap(second, repeated, TOP_POSITIONS)
    recency = memory.spearman(first, positions) + memory.spearman(
        second, positions
    )
    return {
        "passages_overlap": memory.top_overlap(
            first, second, TOP_POSITIONS
        ),
        "repeated_overlap": with_repeated / 2,
        "position_spearman": recency / 2,
    }


def alpha_variation(lenses: Dict[str, List[LayerLens]]) -> Section:
    """The third test: whether forgetting depends on what is read.

    Per head, how much alpha varies across a passage's tokens, as a
    coefficient of variation. Near zero everywhere is a fixed decay,
    which a lens would draw as the same fade on every input.
    """
    per_layer: List[float] = []
    pooled: List[Tensor] = []
    for index in range(len(lenses["passage_a"])):
        variation = torch.cat([
            memory.coefficient_of_variation(lenses[name][index].alpha)
            for name in PASSAGE_INPUTS
        ])
        pooled.append(variation)
        per_layer.append(float(variation.quantile(0.5)))
    median = float(torch.cat(pooled).quantile(0.5))
    return {
        "layer_medians": per_layer,
        "median": median,
        "verdict": _verdict(
            f"decay depends on content: the median coefficient of"
            f" variation of alpha is at least {CONTENT_VARIATION:g}",
            median >= CONTENT_VARIATION,
            f"{median:.4f}",
        ),
    }


def _worst_rebuild(lenses: Dict[str, List[LayerLens]]) -> float:
    worst = 0.0
    for layers in lenses.values():
        for lens in layers:
            worst = max(worst, lens.rebuild_error)
    return worst


def _falsification_inputs(codec: TextCodec) -> Dict[str, List[int]]:
    """Two real passages and one repeated token, all the same length:
    what both retention and forgetting are judged on."""
    return {
        "passage_a": _encode_exactly(
            codec, PASSAGES[0], RETENTION_TOKENS
        ),
        "passage_b": _encode_exactly(
            codec, PASSAGES[1], RETENTION_TOKENS
        ),
        "repeated": _repeated(codec, RETENTION_TOKENS),
    }


def _repeated(codec: TextCodec, length: int) -> List[int]:
    """One token over and over, after whatever the tokenizer puts
    first: an input with no content for a lens to find."""
    lead = codec.encode("")
    unit = codec.encode(REPEATED_TEXT)[len(lead):]
    assert len(unit) == 1, f"{REPEATED_TEXT!r} is not one token"
    assert len(lead) < length, "the lead fills the whole input"
    return lead + unit * (length - len(lead))


# -- forgetting --


def forgetting(model: Mamba3LM, codec: TextCodec) -> Section:
    """Per-token forgetting on the retention inputs, judged by its
    own bar before any overlay draws it."""
    inputs = _falsification_inputs(codec)
    profiles = {
        name: read_forgetting(model, _on(model, ids))
        for name, ids in inputs.items()
    }
    section = judge_forgetting(profiles)
    section["top_tokens"] = _most_forgetting(
        codec, inputs["passage_a"], profiles["passage_a"]
    )
    return section


def read_forgetting(model: Mamba3LM, ids: Tensor) -> Tensor:
    """What reading each token erased, (length,), from one forward."""
    sink: List[Tensor] = []
    core = memory.recording_core(sink)
    model(ids, model.empty_states(1), core=core)
    return memory.forgetting(sink)[0].cpu()


def judge_forgetting(profiles: Dict[str, Tensor]) -> Section:
    """The four tests, on the number a reader would see.

    It varies with content; it is not position; a repeated token
    stays flat once its state settles; and the repeated token does
    not share the passages' top positions as much as they share each
    other's. That last one counts only above twice chance, where the
    comparison stops being a coin toss.
    """
    first = profiles["passage_a"]
    second = profiles["passage_b"]
    repeated = profiles["repeated"]
    positions = torch.arange(first.shape[0], dtype=torch.float64)
    text = (_variation(first) + _variation(second)) / 2
    flat = _variation(repeated[FORGETTING_WARM_UP:])
    trends = [
        memory.spearman(first, positions),
        memory.spearman(second, positions),
    ]
    passages = memory.top_overlap(first, second, TOP_POSITIONS)
    shared = (
        memory.top_overlap(first, repeated, TOP_POSITIONS)
        + memory.top_overlap(second, repeated, TOP_POSITIONS)
    ) / 2
    chance = TOP_POSITIONS**2 / first.shape[0]
    structural = shared >= passages and shared > CHANCE_FLOOR * chance
    verdicts = [
        _verdict(
            "forgetting varies with content: a coefficient of"
            f" variation of at least {CONTENT_VARIATION:g}",
            text >= CONTENT_VARIATION,
            f"{text:.4f}",
        ),
        _verdict(
            "forgetting is not position: |Spearman with position|"
            f" at most {RECENCY_SPEARMAN:g} on each passage",
            all(abs(trend) <= RECENCY_SPEARMAN for trend in trends),
            ", ".join(f"{trend:.3f}" for trend in trends),
        ),
        _verdict(
            "a repeated token stays flat: at most"
            f" {FORGETTING_FLAT_RATIO:g} of real text's variation",
            flat <= FORGETTING_FLAT_RATIO * text,
            f"{flat:.4f} against {text:.4f}",
        ),
        _verdict(
            "real text shares more top positions than a repeated"
            " token does, counted above chance",
            not structural,
            f"repeated {shared:.1f}, passages {passages},"
            f" chance {chance:.1f}",
        ),
    ]
    return {
        "tokens": int(first.shape[0]),
        "chance_overlap": chance,
        "variation_text": text,
        "variation_repeated": flat,
        "position_spearman": trends,
        "passages_overlap": passages,
        "repeated_overlap": shared,
        "informative": all(v["result"] == "pass" for v in verdicts),
        "verdicts": verdicts,
    }


def _variation(profile: Tensor) -> float:
    """A profile's coefficient of variation, as one number."""
    column = profile.double().reshape(-1, 1)
    return float(memory.coefficient_of_variation(column)[0])


def _most_forgetting(
    codec: TextCodec, ids: List[int], profile: Tensor
) -> List[Dict[str, Any]]:
    """The tokens whose reading erased the most, for a person to look
    at: the qualitative half of the evidence."""
    count = min(FORGETTING_TOP_TOKENS, profile.shape[0])
    top = torch.topk(profile, count)
    return [
        {
            "position": int(position),
            "text": codec.decode([ids[int(position)]]),
            "forgetting": round(float(value), 4),
        }
        for value, position in zip(
            top.values.tolist(), top.indices.tolist(), strict=True
        )
    ]


# -- shared --


def _verdict(
    criterion: str, passed: Optional[bool], detail: str
) -> Verdict:
    """`passed` is None where the criterion does not apply to this
    run, or where only a person can judge it."""
    if passed is None:
        result = "not judged"
    elif passed:
        result = "pass"
    else:
        result = "fail"
    return {
        "criterion": criterion,
        "result": result,
        "detail": detail,
    }


def _encode_exactly(
    codec: TextCodec, text: str, length: int
) -> List[int]:
    ids = codec.encode(text)
    assert len(ids) >= length, (
        f"the text is {len(ids)} tokens, fewer than {length}"
    )
    return ids[:length]


def _on(model: Mamba3LM, ids: List[int]) -> Tensor:
    """One batch row of token ids, on the model's device."""
    device = model.lm_head.weight.device
    return torch.tensor([ids], dtype=torch.long, device=device)


def _relative(first: Tensor, second: Tensor) -> float:
    gap = torch.linalg.vector_norm((first - second).double())
    return float(gap / torch.linalg.vector_norm(second.double()))


RUNNERS: Dict[str, Callable[[Mamba3LM, TextCodec], Section]] = {
    "correctness": correctness,
    "speed": speed,
    "retention": retention,
    "forgetting": forgetting,
}
assert tuple(RUNNERS) == SECTIONS, "a runner for every section"


# -- output --


def print_report(report: Dict[str, Any]) -> None:
    """The verdicts, and what a person has to read for themselves."""
    for name in ("load", *SECTIONS):
        if name not in report:
            continue
        print(f"\n== {name} ==")
        for line in LINES[name](report[name]):
            print(line)
        for verdict in _verdicts(report[name]):
            print(
                f"  [{verdict['result']}] {verdict['criterion']}:"
                f" {verdict['detail']}"
            )


def _verdicts(section: Section) -> List[Verdict]:
    found = list(section.get("verdicts", []))
    for lens in ("state", "output"):
        if lens in section:
            found.extend(section[lens]["verdicts"])
    return found


def _load_lines(section: Section) -> List[str]:
    lines = [f"  head stored: {section['head_stored']}"]
    for kind in ("missing", "unexpected", "misshapen"):
        if section[kind]:
            lines.append(f"  {kind}: {', '.join(section[kind])}")
    return lines


def _correctness_lines(section: Section) -> List[str]:
    agreement = section["agreement"]
    lines = [
        f"  top-1 agreement between the cores:"
        f" {agreement['top1_agreement']:.3f}"
        f" over {agreement['tokens']} tokens"
    ]
    for prompt, text in section["completions"].items():
        lines.append(f"  {prompt!r} -> {text!r}")
    return lines


def _speed_lines(section: Section) -> List[str]:
    vram = section["peak_vram_mib"]
    shown = "n/a" if vram is None else f"{vram:.0f} MiB"
    return [
        f"  prompt: {section['prompt_tokens_per_second']:.1f}"
        f" tokens a second over {section['prompt_tokens']}",
        f"  decode: {section['decode_tokens_per_second']:.1f}"
        f" tokens a second over {section['decode_tokens']}",
        f"  peak VRAM {shown}, peak host"
        f" {section['peak_host_mib']:.0f} MiB",
    ]


def _retention_lines(section: Section) -> List[str]:
    lines = [
        f"  top-{TOP_POSITIONS} overlaps, chance is about"
        f" {section['chance_overlap']:.1f}",
        "  layer | state: passages repeated spearman"
        " | output: passages repeated spearman | alpha CV",
    ]
    rows = zip(
        section["state"]["layers"],
        section["output"]["layers"],
        section["alpha"]["layer_medians"],
        strict=True,
    )
    for index, (state, output, variation) in enumerate(rows):
        lines.append(
            f"  {index:5d} | {_row(state)} | {_row(output)}"
            f" | {variation:.4f}"
        )
    return lines


def _row(row: Dict[str, float]) -> str:
    return (
        f"{row['passages_overlap']:4d}"
        f" {row['repeated_overlap']:8.1f}"
        f" {row['position_spearman']:8.3f}"
    )


def _forgetting_lines(section: Section) -> List[str]:
    lines = [
        f"  variation: text {section['variation_text']:.4f},"
        f" repeated token {section['variation_repeated']:.4f}",
        f"  top-{TOP_POSITIONS} overlaps: passages"
        f" {section['passages_overlap']}, repeated"
        f" {section['repeated_overlap']:.1f}, chance"
        f" {section['chance_overlap']:.1f}",
        "  most forgetting in passage A:",
    ]
    for entry in section.get("top_tokens", []):
        lines.append(
            f"  {entry['position']:5d} {entry['text']!r:>14}"
            f" {entry['forgetting']:.4f}"
        )
    return lines


LINES: Dict[str, Callable[[Section], List[str]]] = {
    "load": _load_lines,
    "correctness": _correctness_lines,
    "speed": _speed_lines,
    "retention": _retention_lines,
    "forgetting": _forgetting_lines,
}


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    sys.exit(main())
