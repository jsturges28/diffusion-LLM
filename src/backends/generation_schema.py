"""Stable identities for device-qualified generation schemas.

Only fields that can change accepted generation data are hashed.
Presentation copy and layout metadata are deliberately excluded, so
renaming a label does not strand an already reserved response.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Dict, Sequence

if TYPE_CHECKING:
    from src.backends.protocol import ParamOverride, ParamSpec


GENERATION_CONFIGURATION_CODEC_VERSION = 1
GENERATION_SCHEMA_ID_CHARS = 64
# Bump a model's revision whenever accepted combinations change
# without a corresponding ParamSpec change. LLaDA's divisibility
# rules are the first such model-level contract.
GENERATION_RELATION_REVISIONS = {
    "llada": 1,
}


def generation_schema_id(
    *,
    model_id: str,
    input_mode: str,
    device: str,
    specs: Sequence[ParamSpec],
) -> str:
    """Hash one model/device's canonical generation contract."""
    if not model_id:
        raise ValueError("generation schema needs a model id")
    if input_mode not in {"chat", "completion"}:
        raise ValueError("generation schema input mode is invalid")
    if not device:
        raise ValueError("generation schema needs a device")
    if not specs:
        raise ValueError("generation schema needs parameters")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("generation parameter names must be unique")
    payload: Dict[str, object] = {
        "codec_version": GENERATION_CONFIGURATION_CODEC_VERSION,
        "device": device,
        "input_mode": input_mode,
        "model_id": model_id,
        "parameters": [
            _canonical_spec(spec, device=device)
            for spec in sorted(specs, key=lambda item: item.name)
        ],
    }
    relation_revision = GENERATION_RELATION_REVISIONS.get(model_id)
    if relation_revision is not None:
        payload["relation_revision"] = relation_revision
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    assert len(digest) == GENERATION_SCHEMA_ID_CHARS
    return digest


def generation_schema_ids(
    *,
    model_id: str,
    input_mode: str,
    devices: Sequence[str],
    specs: Sequence[ParamSpec],
) -> Dict[str, str]:
    """Return every supported device's schema identity."""
    if not devices:
        raise ValueError("generation schema needs supported devices")
    if len(devices) != len(set(devices)):
        raise ValueError(
            "supported generation devices must be unique"
        )
    result = {
        device: generation_schema_id(
            model_id=model_id,
            input_mode=input_mode,
            device=device,
            specs=specs,
        )
        for device in devices
    }
    assert set(result) == set(devices)
    return result


def _canonical_spec(
    spec: ParamSpec,
    *,
    device: str,
) -> Dict[str, object]:
    """Select the fields that govern accepted wire values."""
    override = (spec.overrides or {}).get(device)
    return {
        "default": spec.default,
        "experimental": _canonical_bounds(spec.experimental),
        "experimental_only": spec.experimental_only,
        "name": spec.name,
        "options": (
            sorted(spec.options) if spec.options is not None else None
        ),
        "override": _canonical_override(override),
        "recommended": _canonical_bounds(spec.recommended),
        "step": spec.step,
        "type": spec.type.value,
    }


def _canonical_override(
    override: ParamOverride | None,
) -> Dict[str, object] | None:
    if override is None:
        return None
    return {
        "default": override.default,
        "experimental": _canonical_bounds(override.experimental),
        "recommended": _canonical_bounds(override.recommended),
    }


def _canonical_bounds(
    bounds: tuple[float, float] | None,
) -> list[float] | None:
    if bounds is None:
        return None
    low, high = bounds
    if low > high:
        raise ValueError("generation parameter bounds are inverted")
    return [low, high]


assert GENERATION_CONFIGURATION_CODEC_VERSION >= 1
assert hashlib.sha256().digest_size * 2 == GENERATION_SCHEMA_ID_CHARS
assert all(
    isinstance(revision, int)
    for revision in GENERATION_RELATION_REVISIONS.values()
)
assert all(
    not isinstance(revision, bool)
    for revision in GENERATION_RELATION_REVISIONS.values()
)
assert all(
    revision > 0
    for revision in GENERATION_RELATION_REVISIONS.values()
)
