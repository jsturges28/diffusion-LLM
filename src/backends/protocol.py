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
    supports_cfg: bool = False
    # Character shown for an unresolved token in the UI.
    unresolved_char: str = "\u2591"


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
MSG_CANCEL = "cancel"
