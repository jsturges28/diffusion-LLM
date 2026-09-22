"""Resolve a generation request against a model's parameter schema.

The registry already drove the controls, the defaults and the bounds
the UI shows. Each worker then coerced and clamped the same fields
again, with its own pair of clamp helpers and its own fallback
written out beside every read. LLaDA's had drifted: the registry said
a generation length of 160 in one block, the worker said 128 in
blocks of 32, which is a different decoding regime rather than a
smaller number. Nothing noticed, because the browser always sends
every field.

This module is the one implementation. It takes the specs to resolve
against rather than reaching for the registry, so the shared base
class does not acquire a registry import to use it, and a caller can
resolve against a schema that is not registered at all, which is what
makes the omit-each-field matrix in the tests possible.

Dependency-light on purpose, stdlib and the protocol types only. Three
venvs with deliberately incompatible ``transformers`` versions import
this, the same constraint ``protocol.py`` lives under.

Relational rules stay with their model. Nothing here knows that
LLaDA's generation length must divide by its block length, because
that is one model's arithmetic rather than a property of the schema.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from src.backends.protocol import (
    ParamOverride,
    ParamSpec,
    ParamType,
)

# What a resolved parameter can be, mirroring ParamSpec.default.
ParamValue = Union[int, float, str, bool]


def resolve_params(
    specs: Sequence[ParamSpec],
    data: Dict[str, Any],
    *,
    device: Optional[str],
    experimental: bool,
) -> Dict[str, ParamValue]:
    """Every declared parameter, resolved from one request.

    An omitted field takes the default the UI advertises for this
    device, which is the whole point: a worker fallback that differed
    from the schema was invisible to the browser and visible only to
    an API client or a partially upgraded page.

    Raises ``ValueError`` naming the field for a value that is not a
    number where one is required, or not among a select's options.
    Callers turn that into an invalid-request envelope.
    """
    assert specs, "a model with no parameters cannot generate"
    resolved: Dict[str, ParamValue] = {}
    for spec in specs:
        if spec.name in data:
            given = data[spec.name]
        else:
            given = default_of(spec, device=device)
        resolved[spec.name] = coerce(
            spec,
            given,
            device=device,
            experimental=experimental,
        )
    # Checked on the way out as well as on the way in: the branch per
    # ParamType is the part that could return the wrong shape, and a
    # float where the sampler wants an int surfaces deep inside torch.
    assert len(resolved) == len(specs), (
        "every spec resolves to exactly one value"
    )
    for spec in specs:
        _assert_resolved_type(spec, resolved[spec.name])
    return resolved


def default_of(
    spec: ParamSpec, *, device: Optional[str]
) -> ParamValue:
    """The value a request gets when it omits this field."""
    override = _override_for(spec, device)
    if override is not None and override.default is not None:
        return override.default
    return spec.default


def bounds_of(
    spec: ParamSpec,
    *,
    device: Optional[str],
    experimental: bool,
) -> Optional[Tuple[float, float]]:
    """The range a numeric value is held to, or None if unbounded.

    The device override wins where it declares one, so SmolLM3's lower
    CPU token cap is enforced by the same rule the UI displays rather
    than by a clamp the user cannot see.
    """
    override = _override_for(spec, device)
    if override is not None:
        narrowed = (
            override.experimental
            if experimental
            else override.recommended
        )
        if narrowed is not None:
            return narrowed
    if experimental:
        return spec.experimental
    return spec.recommended


def coerce(
    spec: ParamSpec,
    given: Any,
    *,
    device: Optional[str],
    experimental: bool,
) -> ParamValue:
    """One request value as the type and range the spec declares."""
    if spec.type == ParamType.BOOL:
        return bool(given)
    if spec.type == ParamType.SELECT:
        return _chosen_option(spec, given)
    number = _as_number(spec, given)
    bounds = bounds_of(
        spec, device=device, experimental=experimental
    )
    if bounds is not None:
        number = _clamped(number, bounds)
    if spec.type == ParamType.INT:
        return int(number)
    assert spec.type == ParamType.FLOAT, (
        f"{spec.name} has unhandled type {spec.type}"
    )
    return float(number)


def _override_for(
    spec: ParamSpec, device: Optional[str]
) -> Optional[ParamOverride]:
    """The per-device override this spec declares, if any.

    ``device`` is None before a model has loaded, which is not a
    generation path; answering None keeps that harmless rather than
    resolving against a device nobody picked.
    """
    if not spec.overrides:
        return None
    if device is None:
        return None
    return spec.overrides.get(device)


def _chosen_option(spec: ParamSpec, given: Any) -> str:
    """``given`` if the spec offers it, else a ValueError.

    Replaces a set of valid strings kept beside the worker, which was
    a second copy of the ``options`` the picker is already built from.
    """
    options = spec.options or []
    assert options, f"{spec.name} is a select with no options"
    chosen = str(given)
    if chosen not in options:
        raise ValueError(
            f"{spec.name} must be one of "
            f"{', '.join(options)}"
        )
    return chosen


def _as_number(spec: ParamSpec, given: Any) -> float:
    """``given`` as a float, or a ValueError naming the field.

    The message names the parameter because the old path raised from a
    bare ``float()`` and the user saw which request failed but not
    which field in it.
    """
    try:
        return float(given)
    except (TypeError, ValueError):
        raise ValueError(
            f"{spec.name} must be a number,"
            f" got {given!r}"
        ) from None


def _clamped(
    number: float, bounds: Tuple[float, float]
) -> float:
    low, high = bounds
    assert low <= high, f"inverted bounds: {bounds}"
    return max(low, min(high, number))


def _assert_resolved_type(
    spec: ParamSpec, value: ParamValue
) -> None:
    if spec.type == ParamType.INT:
        assert isinstance(value, int), spec.name
    elif spec.type == ParamType.FLOAT:
        assert isinstance(value, float), spec.name
    elif spec.type == ParamType.BOOL:
        assert isinstance(value, bool), spec.name
    else:
        assert isinstance(value, str), spec.name
