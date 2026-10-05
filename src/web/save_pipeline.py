"""Validate, serialize, publish, and preview saved runs.

This module owns the model-agnostic save pipeline without owning its
HTTP route. It deliberately imports neither FastAPI nor ``server``:
the supervisor supplies its data root, resident-model fallback facts,
host probes, and background scheduler at the boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sized,
    Tuple,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    model_validator,
)

from src.backends.context_pack import (
    IDENTIFIER_CHARS_MAX as CONTEXT_IDENTIFIER_CHARS_MAX,
)
from src.backends.context_pack import MESSAGE_CANDIDATES_MAX
from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    CANDIDATES_PER_POSITION,
    PROMPT_CHARS_MAX,
    SAVED_MODEL_TYPE_DIFFUSION,
    saved_model_type,
)
from src.backends.registry import DEFAULT_MODEL, REGISTRY, run_bounds
from src.inference.kgw_watermark import (
    KGW_EVIDENCE_MIN,
    detection_status,
    detection_z_score,
    green_list_size,
    null_probability,
)
from src.inference.render_gif import history_to_gif
from src.web import run_store
from src.web.save_limits import (
    FREEFORM_JSON_CHARS_MAX,
    IDENTIFIER_CHARS_MAX,
    TOKEN_TEXT_CHARS_MAX,
)


logger = logging.getLogger("diffusion_supervisor")


@dataclass(frozen=True)
class CurrentModelFacts:
    """The resident model facts older snapshots may borrow."""

    device: Optional[str]
    versions: Mapping[str, str]
    tokenizer: Mapping[str, Any]
    context_length: Optional[int]


StringProbe = Callable[[], Optional[str]]
CurrentModelFactsReader = Callable[[], CurrentModelFacts]


@dataclass(frozen=True)
class SavePipelineContext:
    """Supervisor-owned dependencies for one save operation."""

    results_dir: Path
    repo_root: Path
    current_model_facts: CurrentModelFactsReader
    gpu_name: StringProbe
    cpu_name: StringProbe
    git_commit: StringProbe


# Every model on the save boundary refuses fields it does not declare.
# Pydantic's default is to drop them silently, which turns "somebody
# added a signal to the client and forgot the server" into a run saved
# without it and an HTTP 200 saying otherwise.
STRICT = ConfigDict(extra="forbid")
CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"
ASSISTANT_TURN_ID_PATTERN = r"^[0-9]{8}$"
CONVERSATION_TURN_INDEX_MAX = 1_000_000


class RemaskEdit(BaseModel):
    model_config = STRICT

    frame_index: int
    token_positions: List[int]


class TokenRecord(BaseModel):
    """One persisted per-token record for durable overlays."""

    model_config = STRICT

    t: str = Field(max_length=TOKEN_TEXT_CHARS_MAX)
    m: bool
    id: int
    c: Optional[float] = None
    e: Optional[float] = None
    f: Optional[float] = None
    g: Optional[StrictBool] = None
    we: Optional[StrictBool] = None

    @model_validator(mode="after")
    def _watermark_evidence_has_membership(self) -> "TokenRecord":
        if self.we is True and self.g is None:
            raise ValueError(
                "watermark evidence needs green membership"
            )
        return self


class TokenAlternative(BaseModel):
    """One competing candidate token at a single position."""

    model_config = STRICT

    id: int
    t: str = Field(max_length=TOKEN_TEXT_CHARS_MAX)
    p: float
    rank: Optional[int] = None


class CandidateSet(BaseModel):
    """What one diffusion position weighed at one captured step."""

    model_config = STRICT

    h: int
    c: List[TokenAlternative] = Field(min_length=1)


class FrameCandidates(BaseModel):
    """A diffusion run's candidate sets at captured frames."""

    model_config = STRICT

    k: int = Field(ge=1)
    stride: int = Field(ge=1)
    frames: List[int] = Field(min_length=1)
    segments: List[int] = Field(min_length=1)
    sets: List[List[CandidateSet]]

    @model_validator(mode="after")
    def _placeable(self) -> "FrameCandidates":
        _check_frame_candidates(self)
        return self


def _ascending(values: List[int]) -> bool:
    return all(
        earlier < later
        for earlier, later in zip(values, values[1:], strict=False)
    )


def _check_frame_candidates(value: FrameCandidates) -> None:
    """Refuse candidates no reader can place or whose budget grew."""
    if len(value.sets) != len(value.frames):
        raise ValueError("candidates need one list of sets per frame")
    if value.frames[0] < 0 or not _ascending(value.frames):
        raise ValueError("candidate frames must ascend from 0")
    if value.segments[0] != 0 or not _ascending(value.segments):
        raise ValueError("candidate segments must ascend from 0")
    rows = value.k + 1
    for sets in value.sets:
        if any(len(entry.c) > rows for entry in sets):
            raise ValueError(f"a candidate set holds over {rows}")
    records = sum(len(sets) for sets in value.sets) * value.k
    if records > CANDIDATE_BUDGET_RECORDS:
        raise ValueError(
            f"candidates hold {records} records, over the budget of"
            f" {CANDIDATE_BUDGET_RECORDS}"
        )


# Per-frame, per-token stream. A frame may be None when a model
# emitted no token detail for it.
FrameTokens = List[Optional[List[TokenRecord]]]

# A run that only grows, held as one record per position.
RunPositions = List[TokenRecord]


def expand_positions(
    positions: List[TokenRecord],
) -> FrameTokens:
    """Rebuild per-frame token arrays from one flat append run."""
    frames: FrameTokens = []
    for count in range(1, len(positions) + 1):
        frames.append(list(positions[:count]))
    return frames


def expand_position_text(
    positions: List[TokenRecord],
) -> List[str]:
    """Build every rendered text prefix in linear accumulation."""
    texts: List[str] = []
    running: List[str] = []
    for token in positions:
        running.append(token.t)
        texts.append("".join(running))
    return texts


class RunProvenance(BaseModel):
    """What the worker attested when it finished the run.

    This model is deliberately not strict. Workers may gain an
    attestation field ahead of the supervisor without making older
    supervisors unable to save their runs.
    """

    model_id: str
    checkpoint: str = ""
    revision: str = ""
    device: str = "unknown"
    versions: Dict[str, str] = Field(default_factory=dict)
    tokenizer: Dict[str, Any] = Field(default_factory=dict)
    context_length: Optional[int] = None
    context_pack: Optional["ContextPackProvenance"] = None
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)
    watermark: Optional["WatermarkProvenance"] = None


class WatermarkProvenance(BaseModel):
    """The worker-attested KGW contract and detector score."""

    model_config = ConfigDict(
        extra="allow",
        allow_inf_nan=False,
    )

    scheme: Literal["kgw"]
    version: int = Field(ge=1)
    key_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    gamma: float = Field(gt=0.0, lt=1.0)
    delta: float = Field(ge=0.0)
    vocab_size: int = Field(ge=2)
    green_list_size: int = Field(ge=1)
    tokenizer_fingerprint: str = Field(min_length=1, max_length=256)
    seeding_contract: str = Field(min_length=1, max_length=512)
    rng_contract: str = Field(min_length=1, max_length=512)
    exclusions: List[str] = Field(min_length=1, max_length=8)
    status: Literal["insufficient_evidence", "scored"]
    green_count: int = Field(ge=0)
    scored_count: int = Field(ge=0)
    green_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    z_score: float
    p0: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="before")
    @classmethod
    def _secret_never_crosses(cls, value: object) -> object:
        if isinstance(value, dict) and "secret" in value:
            raise ValueError(
                "watermark provenance must not contain a secret"
            )
        return value

    @model_validator(mode="after")
    def _coherent(self) -> "WatermarkProvenance":
        expected_size = green_list_size(
            gamma=self.gamma,
            vocab_size=self.vocab_size,
        )
        if self.green_list_size != expected_size:
            raise ValueError(
                "green list size disagrees with gamma and vocabulary"
            )
        if self.green_count > self.scored_count:
            raise ValueError("green count exceeds scored count")
        expected_rate = (
            self.green_count / self.scored_count
            if self.scored_count > 0
            else 0.0
        )
        if self.green_rate is not None and not math.isclose(
            self.green_rate,
            expected_rate,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "watermark green rate disagrees with its counts"
            )
        expected_p0 = null_probability(
            green_list_size=self.green_list_size,
            vocab_size=self.vocab_size,
        )
        if not math.isclose(
            self.p0,
            expected_p0,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            raise ValueError("watermark p0 is not the exact null")
        expected_status = detection_status(self.scored_count)
        if self.status != expected_status:
            threshold = KGW_EVIDENCE_MIN
            raise ValueError(
                "watermark status disagrees with the "
                f"{threshold}-token evidence threshold"
            )
        expected_score = detection_z_score(
            green_count=self.green_count,
            scored_count=self.scored_count,
            p0=expected_p0,
        )
        if not math.isclose(
            self.z_score,
            expected_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "watermark z-score disagrees with its counts"
            )
        return self


class ContextConversationProvenance(BaseModel):
    """The durable conversation owner attested by the worker."""

    model_config = STRICT

    conversation_id: str = Field(
        max_length=CONTEXT_IDENTIFIER_CHARS_MAX
    )
    conversation_revision: int = Field(ge=1)
    assistant_turn_id: str = Field(
        max_length=CONTEXT_IDENTIFIER_CHARS_MAX
    )


class ContextPackProvenance(BaseModel):
    """The exact bounded suffix the worker supplied to the model."""

    model_config = STRICT

    included_turn_ids: List[str] = Field(
        min_length=1,
        max_length=MESSAGE_CANDIDATES_MAX,
    )
    first_included_index: int = Field(ge=0)
    omitted_turn_count: int = Field(ge=0)
    prompt_token_count: int = Field(ge=1)
    output_reserve: int = Field(ge=1)
    requested_total_budget: int = Field(ge=1)
    effective_total_budget: int = Field(ge=1)
    conversation: Optional[ContextConversationProvenance] = None

    @model_validator(mode="after")
    def _coherent(self) -> "ContextPackProvenance":
        if self.first_included_index != self.omitted_turn_count:
            raise ValueError(
                "context pack index and omitted count disagree"
            )
        if self.omitted_turn_count % 2 != 0:
            raise ValueError(
                "context pack omitted a partial exchange"
            )
        total = self.prompt_token_count + self.output_reserve
        if total > self.effective_total_budget:
            raise ValueError(
                "context pack exceeds its effective budget"
            )
        if self.effective_total_budget > self.requested_total_budget:
            raise ValueError(
                "effective context budget exceeds the request"
            )
        return self


RunProvenance.model_rebuild()


class SaveRunRequest(BaseModel):
    """One complete run at the save boundary."""

    model_config = STRICT

    model: str = Field(
        default=DEFAULT_MODEL, max_length=IDENTIFIER_CHARS_MAX
    )
    prompt: str = Field(max_length=PROMPT_CHARS_MAX)
    params: Dict[str, Any] = Field(default_factory=dict)
    frames: Optional[List[str]] = None
    frame_positions: Optional[RunPositions] = None
    original_frame_positions: Optional[RunPositions] = None
    final_text: str
    elapsed_seconds: Optional[float] = None
    per_frame_elapsed: Optional[List[float]] = None
    frame_tokens: Optional[FrameTokens] = None
    original_frame_tokens: Optional[FrameTokens] = None
    alternatives: Optional[List[Optional[List[TokenAlternative]]]] = (
        None
    )
    canvas_index: Optional[List[int]] = None
    mean_conf: Optional[List[Optional[float]]] = None
    remask_edits: Optional[List[RemaskEdit]] = None
    original_per_frame_elapsed: Optional[List[float]] = None
    original_elapsed_seconds: Optional[float] = None
    original_mean_conf: Optional[List[Optional[float]]] = None
    original_alternatives: Optional[
        List[Optional[List[TokenAlternative]]]
    ] = None
    candidates: Optional[FrameCandidates] = None
    original_candidates: Optional[FrameCandidates] = None
    provenance: Optional[RunProvenance] = None
    run_token: Optional[str] = Field(
        default=None, max_length=IDENTIFIER_CHARS_MAX
    )
    run_id: Optional[str] = Field(
        default=None, max_length=IDENTIFIER_CHARS_MAX
    )
    expected_revision: Optional[int] = Field(default=None, ge=0)
    conversation_id: Optional[str] = Field(
        default=None,
        pattern=CONVERSATION_ID_PATTERN,
    )
    assistant_turn_id: Optional[str] = Field(
        default=None,
        pattern=ASSISTANT_TURN_ID_PATTERN,
    )
    turn_index: Optional[int] = Field(
        default=None,
        ge=1,
        le=CONVERSATION_TURN_INDEX_MAX,
    )
    prompt_len: Optional[int] = Field(default=None, ge=0)
    partial: bool = False

    @model_validator(mode="after")
    def _within_bounds(self) -> "SaveRunRequest":
        _check_run_bounds(self)
        _check_conversation_metadata(self)
        _check_watermark_records(self)
        return self

    def normalized(self) -> "SaveRunRequest":
        """Fill per-frame text for an append-shaped request."""
        if not self.frames and not self.frame_positions:
            raise ValueError(
                "a run must carry frames or frame_positions"
            )
        if self.frames and self.frame_positions:
            raise ValueError(
                "a run carries frames or frame_positions, not both"
            )
        if not self.frame_positions:
            return self
        expanded = self.model_copy(
            update={
                "frames": expand_position_text(self.frame_positions),
            }
        )
        assert expanded.frames, "expansion produced no frames"
        assert len(expanded.frames) == len(self.frame_positions), (
            "one frame per position, or the run is not the run"
        )
        return expanded


# A field as a refusal names it, and what it holds, if anything.
CountedField = Tuple[str, Optional[Sized]]


def _check_conversation_metadata(body: SaveRunRequest) -> None:
    """Require one coherent optional durable-turn location."""
    fields = (
        body.conversation_id,
        body.assistant_turn_id,
        body.turn_index,
    )
    present = tuple(value is not None for value in fields)
    if any(present) and not all(present):
        raise ValueError(
            "conversation_id, assistant_turn_id and turn_index"
            " must be supplied together"
        )
    if not all(present):
        return
    assert body.assistant_turn_id is not None
    assert body.turn_index is not None
    if int(body.assistant_turn_id) != body.turn_index:
        raise ValueError(
            "assistant_turn_id does not match turn_index"
        )
    _check_attested_conversation(body)


def _check_attested_conversation(body: SaveRunRequest) -> None:
    provenance = body.provenance
    if provenance is None or provenance.context_pack is None:
        return
    attested = provenance.context_pack.conversation
    if attested is None:
        return
    if (
        attested.conversation_id != body.conversation_id
        or attested.assistant_turn_id != body.assistant_turn_id
    ):
        raise ValueError(
            "save conversation metadata differs from the worker"
            " attestation"
        )


def _check_watermark_records(body: SaveRunRequest) -> None:
    """Reconcile records, parameters, and worker provenance."""
    records = _live_token_records(body)
    original = _original_token_records(body)
    watermark = (
        body.provenance.watermark
        if body.provenance is not None
        else None
    )
    if watermark is None:
        if records is not None and _has_watermark_fields(records):
            raise ValueError(
                "watermark token fields need watermark provenance"
            )
        if original is not None and _has_watermark_fields(original):
            raise ValueError(
                "original watermark fields need watermark provenance"
            )
        return
    _check_watermark_identity(body, watermark)
    if records is None or not records:
        raise ValueError(
            "watermark provenance needs final token records"
        )
    _check_watermark_record_set(
        records,
        vocab_size=watermark.vocab_size,
        label="watermarked tokens",
    )
    if original is not None:
        _check_watermark_record_set(
            original,
            vocab_size=watermark.vocab_size,
            label="original watermarked tokens",
        )
    scored = sum(record.we is True for record in records)
    green = sum(
        record.we is True and record.g is True for record in records
    )
    if watermark.scored_count != scored:
        raise ValueError(
            "watermark scored count differs from token evidence"
        )
    if watermark.green_count != green:
        raise ValueError(
            "watermark green count differs from token evidence"
        )


def _check_watermark_identity(
    body: SaveRunRequest,
    watermark: WatermarkProvenance,
) -> None:
    provenance = body.provenance
    assert provenance is not None
    entry = REGISTRY.get(provenance.model_id)
    if entry is None or not entry.capabilities.supports_watermark:
        raise ValueError(
            "watermark provenance requires a watermark-capable model"
        )
    if body.model != provenance.model_id:
        raise ValueError(
            "watermark model differs from worker provenance"
        )
    if body.params.get("watermark") is not True:
        raise ValueError(
            "watermark provenance requires watermark=true"
        )
    _check_watermark_parameter(
        body.params,
        "watermark_gamma",
        watermark.gamma,
    )
    _check_watermark_parameter(
        body.params,
        "watermark_delta",
        watermark.delta,
    )
    fingerprint = provenance.tokenizer.get("fingerprint")
    if fingerprint != watermark.tokenizer_fingerprint:
        raise ValueError(
            "watermark tokenizer fingerprint differs from provenance"
        )
    width = provenance.tokenizer.get("model_vocab_size")
    if width != watermark.vocab_size:
        raise ValueError(
            "watermark vocabulary differs from model provenance"
        )


def _check_watermark_parameter(
    params: Mapping[str, Any],
    name: str,
    attested: float,
) -> None:
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a saved number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not math.isclose(
        number,
        attested,
        rel_tol=1e-15,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{name} differs from worker provenance")


def _check_watermark_record_set(
    records: List[TokenRecord],
    *,
    vocab_size: int,
    label: str,
) -> None:
    for record in records:
        if record.id < 0 or record.id >= vocab_size:
            raise ValueError(
                f"{label} contain token id {record.id} outside"
                f" [0, {vocab_size})"
            )
        if record.g is None:
            raise ValueError(f"{label} need membership flags")
        if record.we is None:
            raise ValueError(f"{label} need evidence flags")


def _live_token_records(
    body: SaveRunRequest,
) -> Optional[List[TokenRecord]]:
    """Current final tokens, excluding any original-run snapshot."""
    if body.frame_positions is not None:
        return body.frame_positions
    if body.frame_tokens:
        return body.frame_tokens[-1]
    return None


def _original_token_records(
    body: SaveRunRequest,
) -> Optional[List[TokenRecord]]:
    """The pre-edit layer's final records, when one was retained."""
    if body.original_frame_positions is not None:
        return body.original_frame_positions
    if body.original_frame_tokens:
        for frame in reversed(body.original_frame_tokens):
            if frame:
                return frame
    return None


def _has_watermark_fields(records: List[TokenRecord]) -> bool:
    return any(
        record.g is not None or record.we is not None
        for record in records
    )


def _check_run_bounds(body: SaveRunRequest) -> None:
    """Refuse a save holding more than one model run can contain."""
    bounds = run_bounds(body.model)
    run_text = bounds.positions_max * TOKEN_TEXT_CHARS_MAX
    frame_text = bounds.frame_positions_max * TOKEN_TEXT_CHARS_MAX
    checks: Tuple[Tuple[List[CountedField], int], ...] = (
        (_per_frame(body), bounds.frames_max),
        (_per_position(body), bounds.positions_max),
        (_per_canvas(body), bounds.frame_positions_max),
        (_per_alternative_set(body), CANDIDATES_PER_POSITION + 1),
        ([("final_text", body.final_text)], run_text),
        (_frame_texts(body), frame_text),
        (_carried_through(body), FREEFORM_JSON_CHARS_MAX),
    )
    for fields, limit in checks:
        _check_counts(fields=fields, limit=limit, model=body.model)


def _check_counts(
    *, fields: List[CountedField], limit: int, model: str
) -> None:
    """Refuse the first field holding more than ``limit``."""
    assert limit >= 1, "every bound admits something"
    for name, values in fields:
        if values is None:
            continue
        count = len(values)
        if count > limit:
            raise ValueError(
                f"{name} holds {count:,}, past the {limit:,} one"
                f" {model} run can hold"
            )


def _per_frame(body: SaveRunRequest) -> List[CountedField]:
    """Everything kept one per frame, including the original."""
    fields: List[CountedField] = [
        ("frames", body.frames),
        ("frame_tokens", body.frame_tokens),
        ("original_frame_tokens", body.original_frame_tokens),
        ("per_frame_elapsed", body.per_frame_elapsed),
        (
            "original_per_frame_elapsed",
            body.original_per_frame_elapsed,
        ),
        ("mean_conf", body.mean_conf),
        ("original_mean_conf", body.original_mean_conf),
        ("canvas_index", body.canvas_index),
        ("remask_edits", body.remask_edits),
    ]
    for name, capture in _captures(body):
        fields.append((f"{name}.frames", capture.frames))
        fields.append((f"{name}.segments", capture.segments))
    return fields


def _per_position(body: SaveRunRequest) -> List[CountedField]:
    """Everything kept one per position over the whole run."""
    return [
        ("frame_positions", body.frame_positions),
        ("original_frame_positions", body.original_frame_positions),
        ("alternatives", body.alternatives),
        ("original_alternatives", body.original_alternatives),
    ]


def _per_canvas(body: SaveRunRequest) -> List[CountedField]:
    """What each diffusion frame holds."""
    fields: List[CountedField] = []
    layers = (
        ("frame_tokens", body.frame_tokens),
        ("original_frame_tokens", body.original_frame_tokens),
    )
    for name, layer in layers:
        for index, tokens in enumerate(layer or []):
            fields.append((f"{name}[{index}]", tokens))
    for index, edit in enumerate(body.remask_edits or []):
        field = f"remask_edits[{index}]"
        fields.append((field, edit.token_positions))
    for name, capture in _captures(body):
        for index, sets in enumerate(capture.sets):
            fields.append((f"{name}.sets[{index}]", sets))
    return fields


def _per_alternative_set(
    body: SaveRunRequest,
) -> List[CountedField]:
    """Each position's captured candidates and optional held token."""
    fields: List[CountedField] = []
    layers = (
        ("alternatives", body.alternatives),
        ("original_alternatives", body.original_alternatives),
    )
    for name, layer in layers:
        for index, entries in enumerate(layer or []):
            fields.append((f"{name}[{index}]", entries))
    return fields


def _frame_texts(body: SaveRunRequest) -> List[CountedField]:
    return [
        (f"frames[{index}]", text)
        for index, text in enumerate(body.frames or [])
    ]


def _carried_through(body: SaveRunRequest) -> List[CountedField]:
    """The free-form blocks a save keeps, measured as JSON."""
    provenance: Optional[str] = None
    if body.provenance is not None:
        provenance = body.provenance.model_dump_json()
    return [
        ("params", json.dumps(body.params)),
        ("provenance", provenance),
    ]


def _captures(
    body: SaveRunRequest,
) -> List[Tuple[str, FrameCandidates]]:
    captures: List[Tuple[str, FrameCandidates]] = []
    if body.candidates is not None:
        captures.append(("candidates", body.candidates))
    if body.original_candidates is not None:
        captures.append(
            ("original_candidates", body.original_candidates)
        )
    return captures


def _dump_positions(
    positions: Optional[RunPositions],
) -> List[Dict[str, Any]]:
    """Serialize a flat run, dropping absent confidence."""
    assert positions, "a flat run has at least one position"
    return [
        record.model_dump(exclude_none=True) for record in positions
    ]


def _dump_frame_tokens(
    frames: FrameTokens,
) -> List[Optional[List[Dict[str, Any]]]]:
    """Serialize per-frame token records compactly."""
    dumped: List[Optional[List[Dict[str, Any]]]] = []
    for frame in frames:
        if frame is None:
            dumped.append(None)
            continue
        dumped.append(
            [record.model_dump(exclude_none=True) for record in frame]
        )
    return dumped


def _dump_alternatives(
    positions: List[Optional[List[TokenAlternative]]],
) -> List[Optional[List[Dict[str, Any]]]]:
    """Serialize per-position candidates without losing alignment."""
    dumped: List[Optional[List[Dict[str, Any]]]] = []
    for entry in positions:
        if entry is None:
            dumped.append(None)
            continue
        dumped.append(
            [
                candidate.model_dump(exclude_none=True)
                for candidate in entry
            ]
        )
    return dumped


def _dump_candidates(
    candidates: Optional[FrameCandidates],
) -> Optional[Dict[str, Any]]:
    """Serialize a diffusion candidate capture, when present."""
    if candidates is None:
        return None
    return candidates.model_dump(exclude_none=True)


def _context_metadata(
    prompt_len: Optional[int],
    provenance: Optional[RunProvenance],
    current: CurrentModelFacts,
) -> Dict[str, Any]:
    """The prompt and context-window block, when measurable."""
    packed = (
        provenance.context_pack if provenance is not None else None
    )
    if prompt_len is None and packed is None:
        return {}
    measured = (
        packed.prompt_token_count
        if packed is not None
        else prompt_len
    )
    assert measured is not None
    assert measured >= 0, "prompt length must be non-negative"
    block: Dict[str, Any] = {"prompt_tokens": measured}
    if packed is not None:
        block["context_pack"] = packed.model_dump(exclude_none=True)
    if provenance is not None:
        window = provenance.context_length
    else:
        window = current.context_length
    if window is not None:
        block["context_length"] = window
    return block


def _resources_metadata(
    provenance: Optional[RunProvenance],
) -> Dict[str, Any]:
    """What the run cost its device, when the worker measured it."""
    if provenance is None:
        return {}
    return dict(provenance.resources)


def _watermark_metadata(
    provenance: Optional[RunProvenance],
) -> Dict[str, Any]:
    """The enabled run's worker-attested KGW block."""
    if provenance is None or provenance.watermark is None:
        return {}
    return {
        "watermark": provenance.watermark.model_dump(
            exclude_none=True
        )
    }


def _conversation_metadata(
    body: SaveRunRequest,
) -> Dict[str, Any]:
    """The optional durable conversation location for one run."""
    if body.conversation_id is None:
        return {}
    assert body.assistant_turn_id is not None
    assert body.turn_index is not None
    return {
        "conversation_id": body.conversation_id,
        "assistant_turn_id": body.assistant_turn_id,
        "turn_index": body.turn_index,
    }


_OPTIONAL_METADATA_FIELDS = (
    "elapsed_seconds",
    "per_frame_elapsed",
    "canvas_index",
    "mean_conf",
    "original_per_frame_elapsed",
    "original_elapsed_seconds",
    "original_mean_conf",
)


def _describe_processor(
    provenance: Optional[RunProvenance],
    current: CurrentModelFacts,
    context: SavePipelineContext,
) -> Tuple[str, Optional[str]]:
    """Name the processor that ran the model."""
    if provenance is not None:
        device = provenance.device
    else:
        device = current.device
    if device == "cuda":
        return "GPU", context.gpu_name()
    if device == "cpu":
        return "CPU", context.cpu_name()
    return "Unknown", None


def _attested_model_id(body: SaveRunRequest) -> str:
    """Prefer the worker's model identity over the client's claim."""
    claimed = body.model or DEFAULT_MODEL
    if body.provenance is None:
        return claimed
    attested = body.provenance.model_id
    if not attested:
        return claimed
    if attested != claimed:
        logger.warning(
            "save claims model %s but the run was produced by %s;"
            " recording the latter",
            claimed,
            attested,
        )
    return attested


def _build_metadata(
    body: SaveRunRequest,
    context: SavePipelineContext,
) -> Dict[str, Any]:
    """Assemble the metadata a saved run records."""
    current = context.current_model_facts()
    provenance = body.provenance
    model_id = _attested_model_id(body)
    entry = REGISTRY.get(model_id)
    checkpoint = entry.checkpoint if entry else ""
    if provenance is not None and provenance.checkpoint:
        checkpoint = provenance.checkpoint
    model_type = (
        saved_model_type(entry.capabilities.generation_shape)
        if entry
        else SAVED_MODEL_TYPE_DIFFUSION
    )
    processor, processor_name = _describe_processor(
        provenance, current, context
    )
    metadata: Dict[str, Any] = {
        "backend": model_id,
        "model": checkpoint or model_id,
        "model_type": model_type,
        "processor": processor,
        "processor_name": processor_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": body.prompt,
        "final_text": body.final_text,
        "params": body.params,
    }
    for name in _OPTIONAL_METADATA_FIELDS:
        value = getattr(body, name)
        if value is not None:
            metadata[name] = value
    if body.remask_edits:
        metadata["remask_edits"] = [
            edit.model_dump() for edit in body.remask_edits
        ]
    if body.partial:
        metadata["partial"] = True
    metadata.update(_conversation_metadata(body))
    if body.frame_positions:
        metadata[run_store.FRAME_SHAPE_KEY] = (
            run_store.FRAME_SHAPE_APPEND
        )
    saved_context = _context_metadata(
        body.prompt_len, provenance, current
    )
    if saved_context:
        metadata["context"] = saved_context
    resources = _resources_metadata(provenance)
    if resources:
        metadata["resources"] = resources
    if provenance is not None and provenance.signals:
        metadata[run_store.SIGNALS_KEY] = provenance.signals
    metadata.update(_watermark_metadata(provenance))
    metadata["reproducibility"] = _reproducibility_block(
        body,
        provenance,
        current=current,
        context=context,
    )
    return metadata


def _reproducibility_block(
    body: SaveRunRequest,
    provenance: Optional[RunProvenance],
    *,
    current: CurrentModelFacts,
    context: SavePipelineContext,
) -> Dict[str, Any]:
    """Record the inputs needed to reproduce this run."""
    if provenance is not None:
        versions = dict(provenance.versions)
        tokenizer = dict(provenance.tokenizer)
    else:
        versions = dict(current.versions)
        tokenizer = dict(current.tokenizer)
    return {
        "seed": body.params.get("seed"),
        "gpu": context.gpu_name(),
        "git_commit": context.git_commit(),
        "model_revision": (
            provenance.revision if provenance is not None else ""
        ),
        "versions": versions,
        "tokenizer": tokenizer,
        "attested": provenance is not None,
    }


def _build_bundle(
    body: SaveRunRequest,
    context: SavePipelineContext,
) -> run_store.RunBundle:
    """Turn a request into the serializable run-directory content."""
    return run_store.RunBundle(
        metadata=_build_metadata(body, context),
        final_text=body.final_text,
        frames=list(body.frames),
        frame_tokens=(
            _dump_positions(body.frame_positions)
            if body.frame_positions
            else (
                None
                if body.frame_tokens is None
                else _dump_frame_tokens(body.frame_tokens)
            )
        ),
        original_frame_tokens=(
            _dump_positions(body.original_frame_positions)
            if body.original_frame_positions
            else (
                None
                if body.original_frame_tokens is None
                else _dump_frame_tokens(body.original_frame_tokens)
            )
        ),
        alternatives=(
            None
            if body.alternatives is None
            else _dump_alternatives(body.alternatives)
        ),
        original_alternatives=(
            None
            if body.original_alternatives is None
            else _dump_alternatives(body.original_alternatives)
        ),
        candidates=_dump_candidates(body.candidates),
        original_candidates=_dump_candidates(
            body.original_candidates
        ),
    )


@dataclass(frozen=True)
class RunPreview:
    """Everything an after-reply preview draw needs."""

    root: Path
    run_id: str
    revision: int
    frames: Tuple[str, ...]
    prompt: str
    model_label: Optional[str]
    model_type: str


def _save_run_blocking(
    body: SaveRunRequest,
    context: SavePipelineContext,
) -> Tuple[Dict[str, Any], RunPreview]:
    """Publish a run and capture its revision-guarded preview."""
    body = body.normalized()
    bundle = _build_bundle(body, context)
    run_id, revision = run_store.save(
        context.results_dir,
        bundle,
        model_id=body.model or DEFAULT_MODEL,
        run_id=body.run_id or None,
        expected_revision=body.expected_revision,
        run_token=body.run_token,
    )
    reply = {
        "path": run_store.display_path(
            context.results_dir / run_id, context.repo_root
        ),
        "run_id": run_id,
        "revision": revision,
    }
    preview = _run_preview(
        body,
        bundle.metadata,
        context=context,
        run_id=run_id,
        revision=revision,
    )
    return reply, preview


async def publish_run(
    body: SaveRunRequest,
    context: SavePipelineContext,
) -> Tuple[Dict[str, Any], RunPreview]:
    """Move blocking normalization and publication off the loop."""
    return await asyncio.to_thread(_save_run_blocking, body, context)


def _run_preview(
    body: SaveRunRequest,
    metadata: Dict[str, Any],
    *,
    context: SavePipelineContext,
    run_id: str,
    revision: int,
) -> RunPreview:
    """Describe the preview from the metadata just published."""
    assert body.frames, "the request is normalized before it is saved"
    assert revision >= 1, "a published run has a revision"
    model_id = str(metadata.get("backend", ""))
    entry = REGISTRY.get(model_id)
    return RunPreview(
        root=context.results_dir,
        run_id=run_id,
        revision=revision,
        frames=tuple(body.frames),
        prompt=body.prompt,
        model_label=(
            entry.display_name if entry else model_id or None
        ),
        model_type=str(metadata.get("model_type", "diffusion")),
    )


def _render_run_gif(preview: RunPreview, path: Path) -> None:
    """Draw one run preview to its staged path."""
    history_to_gif(
        list(preview.frames),
        path,
        header_text=preview.prompt,
        model_label=preview.model_label,
        model_type=preview.model_type,
    )


def _draw_preview(preview: RunPreview) -> None:
    """Draw and publish a preview only for its saved revision."""
    staged = run_store.preview_staging_path(
        preview.root, preview.run_id, preview.revision
    )
    try:
        _render_run_gif(preview, staged)
        published = run_store.publish_preview(
            preview.root,
            preview.run_id,
            revision=preview.revision,
            staged=staged,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "GIF rendering failed for %s; run is saved",
            preview.run_id,
        )
        staged.unlink(missing_ok=True)
        return
    if published:
        logger.info(
            "drew the preview for %s at revision %d",
            preview.run_id,
            preview.revision,
        )
    else:
        logger.info(
            "dropped the preview for %s: revision %d was replaced"
            " or deleted while it was drawn",
            preview.run_id,
            preview.revision,
        )


PreviewTask = Callable[[RunPreview], None]
PreviewScheduler = Callable[[PreviewTask, RunPreview], None]


def schedule_preview(
    preview: RunPreview,
    schedule: PreviewScheduler,
) -> None:
    """Queue a preview behind the HTTP reply."""
    schedule(_draw_preview, preview)
