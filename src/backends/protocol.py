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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class ParamType(str, Enum):
    """UI control type for a generation parameter."""

    INT = "int"
    FLOAT = "float"
    SELECT = "select"
    BOOL = "bool"


class ParamGroup(str, Enum):
    """Presentation group for a generation parameter."""

    GENERAL = "general"
    OUTPUT = "output"
    SAMPLING = "sampling"
    FEATURES = "features"


class ParamProminence(str, Enum):
    """Whether a parameter is repeated in the collapsed summary."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


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
    ``group`` and ``prominence`` describe presentation only. Safe
    defaults keep an unannotated parameter in the expanded General
    group and out of the compact summary.
    """

    name: str
    label: str
    type: ParamType
    default: Union[int, float, str, bool]
    group: ParamGroup = ParamGroup.GENERAL
    prominence: ParamProminence = ParamProminence.SECONDARY
    step: Optional[float] = None
    options: Optional[List[str]] = None
    recommended: Optional[Tuple[float, float]] = None
    experimental: Optional[Tuple[float, float]] = None
    overrides: Optional[Dict[str, ParamOverride]] = None
    help: Optional[str] = None


# What a signal varies over. "canvas" is one value for the whole run,
# and exists because a channel that has neither a frame nor a position
# axis would otherwise be indistinguishable from a channel with no
# declaration at all.
Axis = Literal["frame", "position", "canvas"]

AXES: Tuple[str, ...] = ("frame", "position", "canvas")

# The candidates a run records per position per step. Five for every
# model, so the popover reads the same whichever produced the run;
# the autoregressive sampler's `TOP_K_ALTERNATIVES` is held to it by a
# test.
CANDIDATES_PER_POSITION = 5
# The records a diffusion run's candidates may take: one default LLaDA
# run, 160 positions for 128 steps at five each, captured at every
# step. Past it the capture thins to a stride rather than growing. At
# the 42 bytes a record the ROADMAP measured, this is about 4 MiB,
# where every step at the registry's largest settings would be 212.
CANDIDATE_BUDGET_RECORDS = 160 * 128 * CANDIDATES_PER_POSITION

assert CANDIDATE_BUDGET_RECORDS == 102_400, "the ROADMAP's budget"

# The longest prompt anything here takes, in characters. A run refuses
# one past it, a save refuses one past it, and counting stops at it,
# so the readout never counts a prompt the worker would not run. Here
# rather than in the worker so the supervisor holds saves to the same
# number without importing worker code.
#
# Past every window this app serves by a wide margin, and the one
# bound a model with no window has: Mamba-3 declares none, so nothing
# else would refuse its prompt (`A2-TRUST-02`).
PROMPT_CHARS_MAX = 1_000_000

# What the claim above rests on: even at a generous four characters a
# token, a prompt this long cannot fit the largest window any
# registered model declares.
assert PROMPT_CHARS_MAX // 4 > 200_000, (
    "a prompt at the cap must exceed any real context window"
)


class SignalChannel(BaseModel):
    """One XAI signal, described rather than inferred.

    Signal shape used to be implicit in where the value was stored,
    which is the whole of what `ROADMAP-03` is about. Confidence and
    entropy are both floats on a token record, so nothing told
    "one value per position, the same in every frame" from "a value
    that changes every denoising step". Analytics resolved the
    ambiguity by reading the final frame, which is right for an
    autoregressive run and silently wrong for a diffusion trajectory.

    ``axes`` is what it varies over and ``location`` is where it is
    written, and keeping them apart is the point. The autoregressive
    and diffusion entropy channels share a location and differ only in
    axes, so a reader that knew only the location could not tell a
    trajectory from a constant.
    """

    # Stable identifier, used in the manifest and by a view asking
    # for a channel by name. Not the storage key: several channels
    # live under one-letter keys for payload size, and a name people
    # can read is worth more in a description than in a hot path.
    name: str
    # Nats for entropy, matching the autoregressive sampler, which has
    # reported nats since entropy first appeared there. Two units for
    # one quantity would make the Analytics scale a guess. A fraction
    # is a share of something between 0 and 1 that is not a
    # probability: what reading a token erased from a state.
    unit: Literal["probability", "nats", "fraction"]
    axes: Tuple[Axis, ...]
    location: Literal["token_record", "frame_scalar", "sidecar"]
    # Where to find it: a key on each token record, a metadata key
    # holding one value per frame, or a sidecar filename.
    key: str
    # Whether a run always has it, or only when a parameter asked for
    # it. An ``opt_in`` channel may legitimately be absent from a run
    # that could have captured it, which is a different fact from a
    # model that cannot produce it at all.
    capture: Literal["always", "opt_in"]
    # Records this channel writes for a full run, when that is
    # knowable in advance and large enough to matter. Present so a
    # channel with a real budget has somewhere to say so: per-frame
    # candidate sets run to millions of records at the bounds the
    # registry allows, where entropy is one float per token record.
    budget_records: Optional[int] = None


class ContextPolicyLimits(BaseModel):
    """One device's bounded conversation-context policy.

    These figures are product policy, not claims about a checkpoint's
    theoretical window. The loaded window is read separately and may
    lower the effective budget at request time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "ContextPolicyLimits":
        if self.default_tokens > self.max_tokens:
            raise ValueError(
                "context policy default exceeds its maximum"
            )
        return self


class ContextPolicy(ContextPolicyLimits):
    """Per-model context policy with optional device overrides."""

    status: Literal["provisional"]
    overrides: Dict[str, ContextPolicyLimits] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _known_devices(self) -> "ContextPolicy":
        unknown = set(self.overrides) - {"cpu", "cuda"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown context policy devices: {names}"
            )
        return self

    def limits_for(
        self, device: Optional[str]
    ) -> ContextPolicyLimits:
        """The policy for ``device``, or the model-wide policy."""
        if device is not None:
            override = self.overrides.get(device)
            if override is not None:
                return override
        return ContextPolicyLimits(
            default_tokens=self.default_tokens,
            max_tokens=self.max_tokens,
        )


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
    # One boolean beside the three above. It is not a signal channel:
    # ``signals`` below is where those are declared.
    remask_renoises: bool = False
    # Whether a canvas ends early once it is steady and confident:
    # unchanged for ``stability_threshold`` steps and below
    # ``confidence_threshold`` nats of mean entropy. A model that says
    # so declares those two parameters and ``max_denoising_steps``,
    # which the readout and the Stopping chart read the rule from.
    #
    # A flag rather than a check for the parameters, for the reason
    # ``remask_renoises`` is one: the page should be told what a model
    # does, not left to infer it from which fields happen to exist.
    adaptive_stopping: bool = False
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
    # A deliberately conservative bound for structured conversation
    # requests. This is separate from a checkpoint's architectural
    # context length: policy chooses how much history this product is
    # willing to pack, while the loaded checkpoint supplies a second,
    # independent hard cap when it declares one.
    context_policy: ContextPolicy
    # Which XAI signals this model emits, and in what shape. Declared
    # here so the UI can offer an overlay before any run exists,
    # rather than deciding per token from whether a float happened to
    # be there, which cannot tell "this model does not produce
    # entropy" from "this position has none".
    #
    # Defaulted to empty rather than required, unlike the axes above.
    # An empty tuple is honest for a model whose channels nobody has
    # described yet, and it reads as "infer as before"; a required
    # field would instead force every future model to restate the four
    # channels that are the same everywhere.
    signals: Tuple[SignalChannel, ...] = ()

    @model_validator(mode="after")
    def _context_devices_declared(self) -> "ModelCapabilities":
        undeclared = (
            set(self.context_policy.overrides)
            - set(self.supported_devices)
        )
        if undeclared:
            names = ", ".join(sorted(undeclared))
            raise ValueError(
                "context policy overrides unsupported devices:"
                f" {names}"
            )
        return self


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


class HubFiles(BaseModel):
    """Named files from another Hub repository, at a pinned commit.

    A model's checkpoint is a whole repository, and occasionally it
    also needs a file that lives in somebody else's. Mamba-3 reads
    Llama 3.1's tokenizer out of SmolLM3's repository, because Meta's
    own copy is gated. ``files`` are exact names rather than
    patterns, since each is fetched and checked by name.
    """

    repo: str
    revision: str
    files: Tuple[str, ...]


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
    # Files the model needs from another repository, fetched with the
    # checkpoint and required before the model counts as downloaded.
    # Kept in their own cache, never the donor's, for the reason
    # ``hf_download.companion_cache_dir`` gives.
    companion: Optional[HubFiles] = None


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
# Set on a ``resume`` that carries a stopped branch on from a frame
# rather than editing it, so nothing is remasked. A flag of its own
# rather than an empty position list, so an edit that lost its
# positions on the way is still refused as malformed instead of
# running as a continue. Present only when true.
RESUME_CONTINUE = "continue"
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
# What the machine is doing right now, for the generator's footer
# meter. Purely outbound and purely advisory: the worker volunteers
# these on a timer, nothing requests one, and a client that drops them
# loses a readout rather than a capability.
#
# Which is why it needs no error scope below. The scopes answer "whose
# work does this failure belong to", and a sample belongs to nobody's
# request. A page too old to know this type ignores it, since the
# browser's dispatch has no default branch.
MSG_RESOURCE_SAMPLE = "resource_sample"


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
# Structured-context failures are split so an API client can
# distinguish a malformed transcript from a valid transcript whose
# requested or effective budget cannot hold it.
ERROR_MALFORMED_MESSAGES = "malformed_messages"
ERROR_INVALID_MESSAGE_ROLE = "invalid_message_role"
ERROR_INVALID_MESSAGE_ORDER = "invalid_message_order"
ERROR_CONTEXT_BOUNDS = "context_bounds"

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


def resume_remask_positions(
    data: Dict[str, object], length: int
) -> List[int]:
    """The canvas positions a ``resume`` remasks.

    One rule for every worker that resumes. A continue
    (``RESUME_CONTINUE``) remasks nothing and says so by sending no
    positions; any other resume is an edit and remasks at least one,
    each on a canvas of ``length`` positions. Raises ``ValueError``
    with the message the page shows for a malformed request.
    """
    assert length > 0, "a canvas has positions"
    raw = data.get("remask_positions", [])
    if data.get(RESUME_CONTINUE) is True:
        if not isinstance(raw, list) or len(raw) > 0:
            raise ValueError(
                "A continue remasks nothing, so it sends no"
                " remask_positions."
            )
        return []
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(
            "remask_positions must be a non-empty list."
        )
    positions: List[int] = []
    for item in raw:
        pos = int(item)
        if pos < 0 or pos >= length:
            raise ValueError(
                f"remask position {pos} out of range"
                f" [0, {length})."
            )
        positions.append(pos)
    assert len(positions) > 0, "an edit remasks something"
    return positions


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
