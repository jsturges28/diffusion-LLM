"""Shared contracts between the supervisor and model workers.

Defines the generation-parameter schema, model capabilities, and
model descriptor types used to drive the frontend and validate
requests. Kept dependency-light (pydantic + stdlib) so both the
supervisor venv and every worker venv can import it without
pulling in torch or transformers.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel


class ParamType(str, Enum):
    """UI control type for a generation parameter."""

    INT = "int"
    FLOAT = "float"
    SELECT = "select"
    BOOL = "bool"


class ParamOverride(BaseModel):
    """Per-device overrides for a ``ParamSpec``.

    Keyed by device ("cpu" / "cuda") on ``ParamSpec.overrides``. Any
    field left None falls back to the base spec. Lets one parameter
    carry device-specific caps (e.g. a lower CPU token budget) that
    the frontend and worker both honor, instead of a hidden clamp.
    """

    default: Optional[Union[int, float, str, bool]] = None
    recommended: Optional[Tuple[float, float]] = None
    experimental: Optional[Tuple[float, float]] = None


class ParamSpec(BaseModel):
    """One user-facing generation parameter.

    ``recommended`` / ``experimental`` are (low, high) bounds for
    numeric params; ``options`` lists choices for ``SELECT``.
    ``overrides`` optionally narrows the default/bounds per device.
    """

    name: str
    label: str
    type: ParamType
    default: Union[int, float, str, bool]
    step: Optional[float] = None
    options: Optional[List[str]] = None
    recommended: Optional[Tuple[float, float]] = None
    experimental: Optional[Tuple[float, float]] = None
    overrides: Optional[Dict[str, ParamOverride]] = None
    help: Optional[str] = None


class ModelCapabilities(BaseModel):
    """Feature flags a worker advertises to the frontend."""

    # Two orthogonal axes, because one value cannot answer both
    # questions. ``family`` is the architecture class and drives
    # display: the menu's family glyph and the per-class glow
    # settings. ``generation_shape`` is how output arrives and drives
    # behaviour: an iterative canvas has masked positions to remask,
    # scrub and converge (Edit Frames, the Heatmap/Diff overlays,
    # Commit Order, the convergence chart), and an append-only stream
    # has none of that.
    #
    # Neither is derivable from the other, which is the point. A
    # state-space model is its own family and appends like an
    # autoregressive one, so collapsing the two would either erase
    # its family or offer it denoising UI it cannot support.
    #
    # Required rather than defaulted: a model that says nothing would
    # otherwise be filed silently as diffusion on both axes, and that
    # mislabelling is what the split exists to prevent.
    family: Literal[
        "diffusion", "autoregressive", "state_space"
    ]
    generation_shape: Literal[
        "append_only", "iterative_canvas"
    ]
    # How a prompt reaches the model, and a third thing neither axis
    # above can answer: an instruction-tuned model of any family
    # wraps the prompt in a template's role markers, while a base
    # checkpoint continues the text as written and may carry no
    # template at all.
    #
    # Declared rather than inferred because it is user-visible. A base
    # model presented as a chat partner is easy to run and hard to
    # interpret: the prompt box would invite a question and the model
    # would continue it as prose. Required like the axes above, so a
    # base model cannot arrive silently labelled as chat.
    input_mode: Literal["chat", "completion"]
    supports_resume: bool = False
    # Autoregressive counterfactual: replace the token at one
    # position with a captured alternative and regenerate forward
    # ("What If"). Kept separate from ``supports_resume`` because
    # that flag unlocks the diffusion remask/resume UI, whose frame
    # selection and remask controls do not apply here.
    supports_substitution: bool = False
    supports_cfg: bool = False
    # Whether a remasked position is renoised on resume rather than
    # hard-masked. When it is, committed neighbours can shift too, and
    # the edit UI says so. LLaDA hard-masks, DiffusionGemma renoises.
    #
    # A flag because the alternative was the one UI decision still
    # reading a model id, which is what ROADMAP-01 forbids: the note
    # would have gone missing the moment a second renoising model
    # arrived under a different id. Defaults to the hard-masking
    # reading, so a model that says nothing simply shows no note.
    #
    # One boolean beside the three above, deliberately not the start
    # of ROADMAP-03's signal manifest; that finding owns the versioned
    # axes, capture budgets and signal channels.
    remask_renoises: bool = False
    # Character shown for an unresolved token in the UI.
    unresolved_char: str = "\u2591"
    # Placements this model can actually load onto, and the single
    # authority on the question. The supervisor refuses an unsupported
    # activation before it evicts the working model
    # (``_validate_target``), and the menu draws a device toggle only
    # where there is a real choice; a backend that raises inside
    # load() has already cost the user their resident worker by the
    # time it speaks.
    #
    # Required for the same reason as the axes above. "Both devices"
    # was once a convenient default, and it outlived its truth: it
    # left a 17 GiB diffusion model advertising a CPU placement that
    # nothing enforced a memory budget for. A model that needs a GPU
    # now has to say so.
    supported_devices: Tuple[str, ...]


# The declared values of the two axes, so a test can enumerate them
# and a caller can check one without restating the literal.
FAMILIES: Tuple[str, ...] = (
    "diffusion",
    "autoregressive",
    "state_space",
)
GENERATION_SHAPE_APPEND_ONLY = "append_only"
GENERATION_SHAPE_ITERATIVE_CANVAS = "iterative_canvas"
GENERATION_SHAPES: Tuple[str, ...] = (
    GENERATION_SHAPE_APPEND_ONLY,
    GENERATION_SHAPE_ITERATIVE_CANVAS,
)

# -- The on-disk vocabulary --
#
# Saved runs record ``model_type``, which predates the axes above and
# is deliberately kept rather than migrated. Every reader of it asks
# about shape, not family: whether the run has a masked canvas that
# can converge. So the field stays, derived from ``generation_shape``
# when a run is written. A state-space run records "autoregressive"
# here, which is right for every reader that exists even though its
# family is its own.
SAVED_MODEL_TYPE_AUTOREGRESSIVE = "autoregressive"
SAVED_MODEL_TYPE_DIFFUSION = "diffusion"

_SAVED_MODEL_TYPE_BY_SHAPE: Dict[str, str] = {
    GENERATION_SHAPE_APPEND_ONLY: (
        SAVED_MODEL_TYPE_AUTOREGRESSIVE
    ),
    GENERATION_SHAPE_ITERATIVE_CANVAS: (
        SAVED_MODEL_TYPE_DIFFUSION
    ),
}

# A shape with no on-disk spelling would be saved as an empty string
# or crash at save time, which is a poor place to learn about it.
assert len(_SAVED_MODEL_TYPE_BY_SHAPE) == len(
    GENERATION_SHAPES
), "every generation shape needs an on-disk model_type"


def saved_model_type(generation_shape: str) -> str:
    """The ``model_type`` a run of this shape records on disk."""
    assert generation_shape in GENERATION_SHAPES, (
        f"unknown generation shape: {generation_shape}"
    )
    return _SAVED_MODEL_TYPE_BY_SHAPE[generation_shape]


def is_hub_checkpoint(checkpoint: str) -> bool:
    """True when the checkpoint is a Hub repo id, not a local path.

    Repo-id checkpoints (``org/name``) download from the Hub and
    carry a commit; local paths (``~/models/...``) are produced
    offline by the quantize script and attest themselves through a
    manifest.

    Lives here rather than beside its callers because two modules now
    need the same answer: the supervisor, to decide what is
    downloadable, and the registry, to assert that everything fetched
    from the Hub names the commit it was fetched at.
    """
    value = checkpoint.strip()
    if not value:
        return False
    if value.startswith(("~", "/", ".")):
        return False
    return value.count("/") == 1


class ModelInfo(BaseModel):
    """Everything needed to launch and describe one model."""

    id: str
    display_name: str
    description: str = ""
    param_specs: List[ParamSpec]
    capabilities: ModelCapabilities
    # Approximate free VRAM (GiB) required to load the model.
    # The supervisor refuses activation below this. 0 disables
    # the pre-flight check.
    min_vram_gib: float = 0.0
    # Supervisor-only launch config (stripped before the
    # frontend response in the supervisor).
    worker_module: str
    # Which environment from ``[tool.diffusion-llm]`` this model runs
    # in, by name rather than by interpreter path. The path was a
    # fourth place every environment had to be spelled out, alongside
    # its lock, its setup instructions and the agent conventions, and
    # one of them was always going to fall behind the others.
    # ``src.backends.environments`` resolves it, lazily, so a worker
    # that imports this registry never parses the manifest.
    environment: str
    checkpoint: str
    # The Hub commit this model loads, pinning code and weights
    # together. Without it the same app commit, the same parameters
    # and the same displayed seed can load different weights after
    # the repository moves, and a saved run cannot say which it got.
    #
    # ``None`` means the checkpoint is not a Hub artifact. The local
    # quantized directory has no commit to name, and attests itself
    # through the manifest written beside it instead, so a sentinel
    # here would be a value nobody could check. The registry asserts
    # that every Hub checkpoint does declare one.
    revision: Optional[str] = None


# -- WebSocket message type constants (client <-> worker) --

MSG_MODEL_STATUS = "model_status"
MSG_FRAME = "frame"
MSG_DONE = "done"
MSG_ERROR = "error"
MSG_GENERATE = "generate"
MSG_RESUME = "resume"
MSG_SUBSTITUTE = "substitute"
MSG_CANCEL = "cancel"
# Set on a terminal ``done`` frame that ended because the run was
# stopped rather than because it finished.
#
# A flag on ``done`` rather than a fourth terminal type, because a
# stopped run is still a run: it keeps the frames it produced, the
# provenance describing the worker that made them, and the token
# naming it, so it stays scrubbable, editable and savable. What it
# must not do is read as complete, which is the one thing the flag
# changes. Present only when true, matching the rest of this
# protocol, where a field absent means "no" rather than "unknown".
TERMINAL_CANCELLED = "cancelled"
# Resolve a typed string against the loaded vocabulary, for the
# What If typed-token preview. A read-only lookup, not a generation
# request, so it is answered without the generation lock.
MSG_TOKENIZE = "tokenize"
MSG_TOKENIZE_RESULT = "tokenize_result"
# Measure what the model actually gave a token at one position of the
# last run, for the What If typed row. Unlike the pair above this is
# a real forward pass, so it does take the generation lock.
MSG_PROBE = "probe"
MSG_PROBE_RESULT = "probe_result"
# How many tokens a prompt becomes once the chat template has wrapped
# it, for the context-window readout. Kept separate from MSG_TOKENIZE
# rather than folded in as a flag, for two reasons: that path caps at
# a couple of hundred characters because it previews one token, and it
# answers with one object per token, which for an imported file would
# be tens of thousands of objects to deliver a single integer.
MSG_COUNT_PROMPT = "count_prompt"
MSG_COUNT_PROMPT_RESULT = "count_prompt_result"
# Put the retained run back the way generation left it, discarding
# any branch a resume committed.
#
# Sent when an edit session *begins* rather than when one is
# abandoned, which reads backwards until you count the ways a
# session can end. Retry and Exit both roll the browser back and
# tell the worker nothing; so does a run-scoped error; and a reload
# or a closed tab cannot send anything at all, because the client's
# rollback snapshot is memory-only and is not persisted while an
# edit is in progress. Session start is the single point where the
# browser is known to be showing the un-edited run, so rewinding
# there covers every one of those without a message per exit.
MSG_REWIND = "rewind"


# -- Error envelopes --
#
# Every failure used to leave a worker as ``{"type": "error",
# "message": <a sentence>}``, which says what went wrong and nothing
# about who it happened to. The browser had one handler for all of
# them, so a probe rejected because a generation was running tore
# down the whole What If session: a non-terminal auxiliary failure
# treated as if the run had died.
#
# Two fields fix that. ``scope`` says how far the failure reaches, and
# ``code`` names the failure stably, so the client can branch without
# matching on prose that is written for a human to read.
#
# Plain dicts and plain functions, not pydantic models. These are
# built on the error path, which is cold, but they live beside the
# frame path, which is not, and the report rejects validating hot
# frames. Keeping the whole module importable by three venvs with
# deliberately incompatible dependencies is worth more here than
# types the callers already have.

# The connection or the model is gone. Nothing else can be attempted,
# so the session ends: the reducer's business, not a control's.
ERROR_SCOPE_FATAL = "fatal"
# One generation-class operation failed (generate, resume,
# substitute). The socket is fine. An edit session open at the time
# must roll back, because the client truncates the run optimistically
# before the worker answers.
ERROR_SCOPE_RUN = "run"
# One auxiliary request failed (tokenize, count, probe). Concerns
# only whatever asked, and must disturb nothing else.
ERROR_SCOPE_REQUEST = "request"

ERROR_SCOPES: Tuple[str, ...] = (
    ERROR_SCOPE_FATAL,
    ERROR_SCOPE_RUN,
    ERROR_SCOPE_REQUEST,
)

# Stable codes. Add rather than rename: the client branches on these.
ERROR_MODEL_LOAD_FAILED = "model_load_failed"
ERROR_NO_MODEL_ACTIVE = "no_model_active"
ERROR_WORKER_UNREACHABLE = "worker_unreachable"
ERROR_NO_TOKENIZER = "no_tokenizer"
ERROR_BUSY = "busy"
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_GENERATION_FAILED = "generation_failed"
ERROR_UNKNOWN_MESSAGE = "unknown_message"
# The run a stateful request names is not the run the worker holds.
ERROR_STALE_RUN = "stale_run"

# Which scope each request type's failures carry. Generation-class
# requests own the run; the rest own only themselves.
REQUEST_SCOPES: Dict[str, str] = {
    MSG_GENERATE: ERROR_SCOPE_RUN,
    MSG_RESUME: ERROR_SCOPE_RUN,
    MSG_SUBSTITUTE: ERROR_SCOPE_RUN,
    MSG_TOKENIZE: ERROR_SCOPE_REQUEST,
    MSG_COUNT_PROMPT: ERROR_SCOPE_REQUEST,
    MSG_PROBE: ERROR_SCOPE_REQUEST,
    # Request-scoped despite writing run state, because of when it is
    # sent: an edit session opens with one, so a run-scoped refusal
    # would tear down the session in the middle of setting it up. A
    # window that cannot rewind cannot resume either, and that
    # refusal does own the run, so nothing is lost by being quiet
    # here.
    MSG_REWIND: ERROR_SCOPE_REQUEST,
}


def wire_error(
    *,
    message: str,
    code: str,
    scope: str,
    request_type: Optional[str] = None,
    request_id: Optional[int] = None,
) -> Dict[str, object]:
    """Build one error frame.

    ``request_type`` and ``request_id`` are omitted rather than sent
    as null when the failure answers no particular request, so the
    client's "is this mine" test stays a plain presence check and
    cannot mistake a null for an id of zero.
    """
    assert message, "an error frame must say something"
    assert code, "an error frame must carry a code"
    assert scope in ERROR_SCOPES, f"unknown error scope: {scope}"
    frame: Dict[str, object] = {
        "type": MSG_ERROR,
        "message": message,
        "code": code,
        "scope": scope,
    }
    if request_type is not None:
        frame["request_type"] = request_type
    if request_id is not None:
        frame["request_id"] = request_id
    return frame


def request_id_of(data: Dict[str, object]) -> Optional[int]:
    """The client's id for a request, if it sent one.

    ``None`` rather than zero when absent, so an error frame can omit
    the field and the client's ownership test stays a presence check.
    Only the auxiliary requests carry an id today; the generation
    ones are identified by the run they belong to instead.
    """
    raw = data.get("request_id")
    if isinstance(raw, int):
        return raw
    return None


def request_error(
    *,
    message: str,
    code: str,
    request_type: str,
    request_id: Optional[int] = None,
) -> Dict[str, object]:
    """Build an error frame scoped by which request it answers.

    The scope of a failure is a property of the operation, not of the
    site that noticed it, so callers name the request and this decides
    how far the damage reaches. An unrecognised request type is
    treated as run-scoped, which is the cautious reading: doing too
    much cleanup is recoverable, and leaving a half-applied edit on
    screen is not.
    """
    assert request_type, "name the request this answers"
    return wire_error(
        message=message,
        code=code,
        scope=REQUEST_SCOPES.get(request_type, ERROR_SCOPE_RUN),
        request_type=request_type,
        request_id=request_id,
    )
