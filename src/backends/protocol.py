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

    # Generation paradigm. Diffusion-only UI (Edit Frames, the
    # Heatmap/Diff overlays, Commit Order, the convergence chart) is
    # gated off for "autoregressive" models, which stream a growing
    # left-to-right sequence rather than denoising a masked canvas.
    model_type: Literal["diffusion", "autoregressive"] = "diffusion"
    supports_resume: bool = False
    # Autoregressive counterfactual: replace the token at one
    # position with a captured alternative and regenerate forward
    # ("What If"). Kept separate from ``supports_resume`` because
    # that flag unlocks the diffusion remask/resume UI, whose frame
    # selection and remask controls do not apply here.
    supports_substitution: bool = False
    supports_cfg: bool = False
    # Character shown for an unresolved token in the UI.
    unresolved_char: str = "\u2591"
    # Placements this model can actually load onto. Declared here so
    # the supervisor can refuse an impossible activation before it
    # evicts the working model; a backend that raises inside load()
    # has already cost the user their resident worker by the time it
    # speaks. Both devices by default, which is true of every model
    # that does not say otherwise.
    #
    # A minimal version of what ROADMAP-01 will fold into a proper
    # device-support axis alongside family and stream shape.
    supported_devices: Tuple[str, ...] = ("cuda", "cpu")


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
    venv_python: str
    checkpoint: str


# -- WebSocket message type constants (client <-> worker) --

MSG_MODEL_STATUS = "model_status"
MSG_FRAME = "frame"
MSG_DONE = "done"
MSG_ERROR = "error"
MSG_GENERATE = "generate"
MSG_RESUME = "resume"
MSG_SUBSTITUTE = "substitute"
MSG_CANCEL = "cancel"
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
