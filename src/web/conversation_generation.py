"""Strict generation snapshots for deferred conversation actions.

The browser freezes one action-local Run settings snapshot. This
module reduces it to the fields needed to reproduce generation,
validates those fields against the registry, and gives the store one
compact payload safe to keep on a pending assistant.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    TypeAlias,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from src.backends.protocol import ParamSpec

from src.backends.generation_schema import (
    GENERATION_CONFIGURATION_CODEC_VERSION,
    GENERATION_SCHEMA_ID_CHARS,
    generation_schema_id,
)

PENDING_GENERATION_KEY = "pending_generation_v1"
GENERATION_CONFIGURATION_FIELDS = frozenset(
    {
        "codec_version",
        "model_id",
        "input_mode",
        "device",
        "schema_id",
        "experimental",
        "parameters",
    }
)
GENERATION_PARAMETER_COUNT_MAX = 64
GENERATION_PARAMETER_NAME_CHARS_MAX = 128
GENERATION_PARAMETER_STRING_CHARS_MAX = 1024
GENERATION_PARAMETER_NUMBER_ABS_MAX = 10**100
GENERATION_DEVICE_CHARS_MAX = 32
GENERATION_CONFIGURATION_BYTES_MAX = 16 * 1024
GENERATION_DEVICE_FORMS = frozenset({"cpu", "cuda"})
GENERATION_SCHEMA_ID_RE = re.compile(r"^[0-9a-f]{64}$")

InputMode: TypeAlias = Literal["chat", "completion"]
GenerationParameterValue: TypeAlias = Union[int, float, str, bool]
GenerationParameters: TypeAlias = Dict[
    str, GenerationParameterValue
]


class GenerationConfigurationPayload(TypedDict):
    """Canonical pending-generation data stored on one turn."""

    codec_version: Literal[1]
    model_id: str
    input_mode: InputMode
    device: str
    schema_id: str
    experimental: bool
    parameters: GenerationParameters


assert (
    len(PENDING_GENERATION_KEY)
    <= GENERATION_PARAMETER_NAME_CHARS_MAX
)
assert GENERATION_CONFIGURATION_CODEC_VERSION == 1
assert GENERATION_PARAMETER_NUMBER_ABS_MAX > 2**63
assert GENERATION_CONFIGURATION_BYTES_MAX < 64 * 1024


def parse_generation_configuration(
    value: object,
    *,
    expected_model_id: str,
    expected_input_mode: str,
) -> GenerationConfigurationPayload:
    """Parse durable data without consulting the live registry."""
    if not isinstance(value, Mapping):
        raise ValueError("generation configuration must be an object")
    _require_exact_fields(value)
    codec_version = _required_codec_version(value)
    model_id = _required_string(value, "model_id")
    input_mode = _required_input_mode(value)
    device = _required_device(value)
    schema_id = _required_schema_id(value)
    experimental = value["experimental"]
    if not isinstance(experimental, bool):
        raise ValueError("experimental must be a boolean")
    _require_identity(
        model_id=model_id,
        input_mode=input_mode,
        expected_model_id=expected_model_id,
        expected_input_mode=expected_input_mode,
    )
    parameters = _parsed_parameters(value["parameters"])
    configuration: GenerationConfigurationPayload = {
        "codec_version": codec_version,
        "model_id": model_id,
        "input_mode": input_mode,
        "device": device,
        "schema_id": schema_id,
        "experimental": experimental,
        "parameters": parameters,
    }
    _require_serialized_bound(configuration)
    return configuration


def validate_generation_configuration(
    value: object,
    *,
    expected_model_id: str,
    expected_input_mode: str,
) -> GenerationConfigurationPayload:
    """Return one canonical registry-backed generation snapshot."""
    from src.backends.registry import REGISTRY

    assert max(
        len(model.param_specs) for model in REGISTRY.values()
    ) <= GENERATION_PARAMETER_COUNT_MAX
    parsed = parse_generation_configuration(
        value,
        expected_model_id=expected_model_id,
        expected_input_mode=expected_input_mode,
    )
    model = REGISTRY.get(parsed["model_id"])
    if model is None:
        raise ValueError(f"unknown model: {parsed['model_id']}")
    if parsed["input_mode"] != model.capabilities.input_mode:
        raise ValueError(
            f"model {parsed['model_id']} uses"
            f" {model.capabilities.input_mode} input,"
            f" not {parsed['input_mode']}"
        )
    if parsed["device"] not in model.capabilities.supported_devices:
        raise ValueError(
            f"model {parsed['model_id']} does not support device"
            f" {parsed['device']}"
        )
    expected_schema_id = registry_generation_schema_id(
        model.id, parsed["device"]
    )
    if parsed["schema_id"] != expected_schema_id:
        raise ValueError(
            "generation schema id does not match the current"
            " model/device schema"
        )
    parameters = _validated_parameters(
        parsed["parameters"],
        specs=model.param_specs,
        device=parsed["device"],
        experimental=parsed["experimental"],
    )
    _validate_model_relations(parsed["model_id"], parameters)
    configuration: GenerationConfigurationPayload = {
        **parsed,
        "parameters": parameters,
    }
    _require_serialized_bound(configuration)
    return configuration


def default_generation_configuration(
    *,
    model_id: str,
    input_mode: str,
) -> GenerationConfigurationPayload:
    """Build the pre-snapshot default for direct store callers."""
    from src.backends.params import resolve_params
    from src.backends.registry import REGISTRY

    model = REGISTRY.get(model_id)
    if model is None:
        raise ValueError(f"unknown model: {model_id}")
    devices = model.capabilities.supported_devices
    if not devices:
        raise ValueError(f"model {model_id} has no supported device")
    device = devices[0]
    resolved = resolve_params(
        model.param_specs,
        {},
        device=device,
        experimental=False,
    )
    parameters = {
        spec.name: resolved[spec.name]
        for spec in model.param_specs
        if not spec.experimental_only
    }
    return validate_generation_configuration(
        {
            "codec_version": (
                GENERATION_CONFIGURATION_CODEC_VERSION
            ),
            "model_id": model_id,
            "input_mode": input_mode,
            "device": device,
            "schema_id": registry_generation_schema_id(
                model.id, device
            ),
            "experimental": False,
            "parameters": parameters,
        },
        expected_model_id=model_id,
        expected_input_mode=input_mode,
    )


def registry_generation_schema_id(
    model_id: str,
    device: str,
) -> str:
    """Read one current registry-backed schema identity."""
    from src.backends.registry import REGISTRY

    model = REGISTRY.get(model_id)
    if model is None:
        raise ValueError(f"unknown model: {model_id}")
    if device not in model.capabilities.supported_devices:
        raise ValueError(
            f"model {model_id} does not support device {device}"
        )
    return generation_schema_id(
        model_id=model.id,
        input_mode=model.capabilities.input_mode,
        device=device,
        specs=model.param_specs,
    )


def generation_configuration_metadata(
    configuration: GenerationConfigurationPayload,
) -> Dict[str, object]:
    """Build the one reserved metadata member for a pending turn."""
    copied: GenerationConfigurationPayload = {
        "codec_version": configuration["codec_version"],
        "model_id": configuration["model_id"],
        "input_mode": configuration["input_mode"],
        "device": configuration["device"],
        "schema_id": configuration["schema_id"],
        "experimental": configuration["experimental"],
        "parameters": dict(configuration["parameters"]),
    }
    return {PENDING_GENERATION_KEY: copied}


def pending_generation_configuration(
    metadata: Mapping[str, object],
    *,
    expected_model_id: str,
    expected_input_mode: str,
) -> GenerationConfigurationPayload | None:
    """Read and validate the reserved pending metadata member."""
    if PENDING_GENERATION_KEY not in metadata:
        return None
    if len(metadata) != 1:
        raise ValueError(
            "pending generation metadata has unrelated fields"
        )
    return parse_generation_configuration(
        metadata[PENDING_GENERATION_KEY],
        expected_model_id=expected_model_id,
        expected_input_mode=expected_input_mode,
    )


def reject_pending_generation_metadata(
    metadata: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Keep the reserved field off user and completed turn data."""
    if PENDING_GENERATION_KEY in metadata:
        raise ValueError(
            f"{label} cannot contain {PENDING_GENERATION_KEY}"
        )


def _require_exact_fields(value: Mapping[object, object]) -> None:
    fields = set(value.keys())
    if fields != GENERATION_CONFIGURATION_FIELDS:
        missing = sorted(GENERATION_CONFIGURATION_FIELDS - fields)
        unknown = sorted(
            str(field)
            for field in fields - GENERATION_CONFIGURATION_FIELDS
        )
        if missing:
            raise ValueError(
                "generation configuration is missing "
                + ", ".join(missing)
            )
        raise ValueError(
            "generation configuration has unknown fields: "
            + ", ".join(unknown)
        )


def _required_codec_version(
    value: Mapping[object, object],
) -> Literal[1]:
    found = value["codec_version"]
    if isinstance(found, bool) or not isinstance(found, int):
        raise ValueError("codec_version must be an integer")
    if found != GENERATION_CONFIGURATION_CODEC_VERSION:
        raise ValueError(
            f"unsupported generation configuration codec {found}"
        )
    return 1


def _required_string(
    value: Mapping[object, object],
    name: str,
) -> str:
    found = value[name]
    if not isinstance(found, str) or found == "":
        raise ValueError(f"{name} must be a non-empty string")
    maximum = (
        GENERATION_DEVICE_CHARS_MAX
        if name == "device"
        else GENERATION_PARAMETER_NAME_CHARS_MAX
    )
    if len(found) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return found


def _required_device(
    value: Mapping[object, object],
) -> str:
    found = _required_string(value, "device")
    if found not in GENERATION_DEVICE_FORMS:
        raise ValueError("device must be cpu or cuda")
    return found


def _required_schema_id(
    value: Mapping[object, object],
) -> str:
    found = value["schema_id"]
    if not isinstance(found, str):
        raise ValueError("schema_id must be a string")
    if len(found) != GENERATION_SCHEMA_ID_CHARS:
        raise ValueError(
            "schema_id must be 64 lowercase hexadecimal characters"
        )
    if GENERATION_SCHEMA_ID_RE.fullmatch(found) is None:
        raise ValueError(
            "schema_id must be 64 lowercase hexadecimal characters"
        )
    return found


def _required_input_mode(
    value: Mapping[object, object],
) -> InputMode:
    found = _required_string(value, "input_mode")
    if found == "chat":
        return "chat"
    if found == "completion":
        return "completion"
    raise ValueError("input_mode must be chat or completion")


def _require_identity(
    *,
    model_id: str,
    input_mode: InputMode,
    expected_model_id: str,
    expected_input_mode: str,
) -> None:
    if model_id != expected_model_id:
        raise ValueError(
            "generation configuration model_id does not match"
            " the reserved model"
        )
    if input_mode != expected_input_mode:
        raise ValueError(
            "generation configuration input_mode does not match"
            " the reserved input mode"
        )


def _validated_parameters(
    value: Mapping[str, GenerationParameterValue],
    *,
    specs: list[ParamSpec],
    device: str,
    experimental: bool,
) -> GenerationParameters:
    from src.backends.params import resolve_params

    raw: Dict[str, object] = dict(value)
    included = [
        spec
        for spec in specs
        if experimental or not spec.experimental_only
    ]
    _require_parameter_fields(raw, included)
    _require_parameter_types(raw, included)
    try:
        resolved = resolve_params(
            specs,
            raw,
            device=device,
            experimental=experimental,
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    parameters: GenerationParameters = {}
    for spec in included:
        canonical = resolved[spec.name]
        if canonical != raw[spec.name]:
            raise ValueError(
                f"{spec.name} is outside the active {device} bounds"
            )
        parameters[spec.name] = canonical
    return parameters


def _parsed_parameters(value: object) -> GenerationParameters:
    if not isinstance(value, Mapping):
        raise ValueError("generation parameters must be an object")
    if len(value) > GENERATION_PARAMETER_COUNT_MAX:
        raise ValueError(
            "generation parameters exceed "
            f"{GENERATION_PARAMETER_COUNT_MAX} fields"
        )
    raw = _copy_parameter_mapping(value)
    parameters: GenerationParameters = {}
    for name, parameter in raw.items():
        parameters[name] = _parsed_parameter_value(name, parameter)
    return parameters


def _copy_parameter_mapping(
    value: Mapping[object, object],
) -> Dict[str, object]:
    copied: Dict[str, object] = {}
    for name, parameter in value.items():
        if not isinstance(name, str) or name == "":
            raise ValueError(
                "generation parameter names must be non-empty strings"
            )
        if len(name) > GENERATION_PARAMETER_NAME_CHARS_MAX:
            raise ValueError(
                "generation parameter name exceeds "
                f"{GENERATION_PARAMETER_NAME_CHARS_MAX} characters"
            )
        copied[name] = parameter
    return copied


def _parsed_parameter_value(
    name: str,
    value: object,
) -> GenerationParameterValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > GENERATION_PARAMETER_STRING_CHARS_MAX:
            raise ValueError(f"{name} exceeds the string limit")
        return value
    if isinstance(value, int):
        if abs(value) > GENERATION_PARAMETER_NUMBER_ABS_MAX:
            raise ValueError(f"{name} exceeds the numeric limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if abs(value) > GENERATION_PARAMETER_NUMBER_ABS_MAX:
            raise ValueError(f"{name} exceeds the numeric limit")
        return value
    raise ValueError(
        f"{name} must be a boolean, number, or string"
    )


def _require_parameter_fields(
    raw: Mapping[str, object],
    specs: list[ParamSpec],
) -> None:
    expected = {spec.name for spec in specs}
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(
            "generation parameters are missing " + ", ".join(missing)
        )
    if unknown:
        raise ValueError(
            "unknown generation parameters: " + ", ".join(unknown)
        )


def _require_parameter_types(
    raw: Mapping[str, object],
    specs: list[ParamSpec],
) -> None:
    by_name = {spec.name: spec for spec in specs}
    for name, value in raw.items():
        _require_parameter_type(by_name[name], value)


def _require_parameter_type(
    spec: ParamSpec,
    value: object,
) -> None:
    parameter_type = getattr(spec.type, "value", spec.type)
    if parameter_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{spec.name} must be a boolean")
        return
    if parameter_type == "select":
        if not isinstance(value, str):
            raise ValueError(f"{spec.name} must be a string")
        return
    if isinstance(value, bool):
        raise ValueError(f"{spec.name} must be a number, not boolean")
    if parameter_type == "int":
        if not isinstance(value, int):
            raise ValueError(f"{spec.name} must be an integer")
        return
    if not isinstance(value, (int, float)):
        raise ValueError(f"{spec.name} must be a number")


def _validate_model_relations(
    model_id: str,
    parameters: GenerationParameters,
) -> None:
    if model_id != "llada":
        return
    from src.inference.llada_schedule import block_schedule

    steps = parameters["steps"]
    gen_length = parameters["gen_length"]
    block_length = parameters["block_length"]
    assert isinstance(steps, int)
    assert isinstance(gen_length, int)
    assert isinstance(block_length, int)
    block_schedule(
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
    )


def _require_serialized_bound(
    configuration: GenerationConfigurationPayload,
) -> None:
    try:
        encoded = json.dumps(
            configuration,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "generation configuration is not valid JSON"
        ) from exc
    if len(encoded) > GENERATION_CONFIGURATION_BYTES_MAX:
        raise ValueError(
            "generation configuration exceeds "
            f"{GENERATION_CONFIGURATION_BYTES_MAX} bytes"
        )
