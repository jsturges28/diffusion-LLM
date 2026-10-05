"""One resolver answers for every model's generation parameters.

Strategy: the schema is data, so all of this runs with no checkpoint,
no GPU and no worker. The centre of the file is one generated matrix
over every registered model, every device it declares and every
parameter it has: fill a request with values that differ from the
defaults, drop one field, and prove the dropped field came back as the
value the UI advertises while the rest kept what was sent. That is the
shape the report asked for, and it is the shape that would have caught
the bug: a worker fallback could disagree with the registry forever,
because the browser always sends every field.

Around it sit the cases a matrix cannot express: the boundary where a
value stops being accepted, the two vocabularies a select has to keep
apart, and the line between what the schema decides and what a model's
own arithmetic decides.

Passing proves the registry is the single authority for defaults,
device overrides, types, options and bounds, and that LLaDA's
divisibility rules stayed with LLaDA. Failing on the drift test
specifically means the defaults have come apart again.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pytest

from src.backends.llada_worker import LladaBackend
from src.backends.smollm3_worker import Smollm3Backend
from src.backends.generation_schema import (
    GENERATION_RELATION_REVISIONS,
    generation_schema_id,
    generation_schema_ids,
)
from src.backends.params import (
    bounds_of,
    default_of,
    resolve_params,
)
from src.backends.protocol import (
    ParamGroup,
    ParamOverride,
    ParamProminence,
    ParamSpec,
    ParamType,
)
from src.backends.registry import REGISTRY

EXPECTED_SCHEMA_IDS = {
    "llada": {
        "cuda": (
            "2d3a3566f1dfd65f124e5620418fab8c"
            "b43678bd908619c66dc17fd30696ab96"
        ),
    },
    "diffusiongemma": {
        "cuda": (
            "c7b5b65876458c0af62e5389e661ea668"
            "6b79ab0c06394336d48885c46fdfd35"
        ),
    },
    "smollm3": {
        "cuda": (
            "c05b20b6bed4e37dfa875e2e7d61b9ee"
            "53bcfbaf3146f5694a743fa77b09f001"
        ),
        "cpu": (
            "404e9a296b05898b3ac549df29bdf747"
            "5b0028e1448c9c071ab9f9b13aaa9424"
        ),
    },
    "mamba3": {
        "cuda": (
            "0a7fa05b3bc0ca05ae2001f648ef5b407"
            "6e6fff60e37e96eb81d90c7f6c29241"
        ),
        "cpu": (
            "866224aa6f60b08299a3d5ff1da7555a"
            "f6b4f1b05cb1794742ba35ad2be87515"
        ),
    },
}


def _spec(model_id: str, name: str) -> ParamSpec:
    for spec in REGISTRY[model_id].param_specs:
        if spec.name == name:
            return spec
    raise AssertionError(f"{model_id} has no {name!r}")


def _defaults(model_id: str, device: str) -> Dict[str, Any]:
    return {
        spec.name: default_of(spec, device=device)
        for spec in REGISTRY[model_id].param_specs
    }


def _other_than_default(spec: ParamSpec, device: str) -> Any:
    """A legal value for ``spec`` that is not its default.

    The matrix needs every sent value to differ from the default, or
    "took the default" and "kept what was sent" would be the same
    assertion and the test would prove nothing.
    """
    default = default_of(spec, device=device)
    if spec.type == ParamType.BOOL:
        return not default
    if spec.type == ParamType.SELECT:
        options = spec.options or []
        others = [one for one in options if one != default]
        assert others, f"{spec.name} offers only its default"
        return others[0]
    bounds = bounds_of(
        spec, device=device, experimental=False
    )
    assert bounds is not None, f"{spec.name} declares no bounds"
    low, high = bounds
    middle = (low + high) / 2
    if spec.type == ParamType.INT:
        middle = int(middle)
    assert middle != default, (
        f"{spec.name} midpoint equals its default"
    )
    return middle


def _full_request(model_id: str, device: str) -> Dict[str, Any]:
    return {
        spec.name: _other_than_default(spec, device)
        for spec in REGISTRY[model_id].param_specs
    }


def _matrix() -> List[Tuple[str, str, str]]:
    """Every (model, device, parameter) the registry declares."""
    cases: List[Tuple[str, str, str]] = []
    for model_id, info in REGISTRY.items():
        for device in info.capabilities.supported_devices:
            for spec in info.param_specs:
                cases.append((model_id, device, spec.name))
    assert cases, "the matrix must not be empty"
    return cases


def _schema_id(model_id: str, device: str) -> str:
    model = REGISTRY[model_id]
    return generation_schema_id(
        model_id=model.id,
        input_mode=model.capabilities.input_mode,
        device=device,
        specs=model.param_specs,
    )


# -- the matrix --


def test_every_model_device_has_one_stable_schema_identity() -> None:
    """All model families expose every declared device exactly."""
    for model_id, model in REGISTRY.items():
        first = generation_schema_ids(
            model_id=model.id,
            input_mode=model.capabilities.input_mode,
            devices=model.capabilities.supported_devices,
            specs=model.param_specs,
        )
        second = generation_schema_ids(
            model_id=model.id,
            input_mode=model.capabilities.input_mode,
            devices=model.capabilities.supported_devices,
            specs=model.param_specs,
        )

        assert first == second, model_id
        assert first == EXPECTED_SCHEMA_IDS[model_id]
        assert set(first) == set(
            model.capabilities.supported_devices
        )
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in first.values()
        )


def test_schema_identity_changes_for_parameter_evolution() -> None:
    """Every generation-relevant ParamSpec change changes identity."""
    model = REGISTRY["llada"]
    original = _schema_id(model.id, "cuda")
    added = [
        *model.param_specs,
        ParamSpec(
            name="new_limit",
            label="New Limit",
            type=ParamType.INT,
            default=1,
            recommended=(1, 2),
        ),
    ]
    removed = model.param_specs[:-1]
    changed = list(model.param_specs)
    changed[0] = changed[0].model_copy(
        update={"default": changed[0].default + 1}
    )

    for specs in (added, removed, changed):
        evolved = generation_schema_id(
            model_id=model.id,
            input_mode=model.capabilities.input_mode,
            device="cuda",
            specs=specs,
        )
        assert evolved != original


def test_relational_generation_contracts_have_schema_revisions(
) -> None:
    """Rules outside ParamSpec still invalidate old snapshots."""
    assert GENERATION_RELATION_REVISIONS["llada"] == 1
    assert set(GENERATION_RELATION_REVISIONS) <= set(REGISTRY)


def test_schema_identity_is_device_qualified() -> None:
    """SmolLM3's CPU override is part of CPU identity only."""
    model = REGISTRY["smollm3"]

    assert _schema_id(model.id, "cpu") != _schema_id(
        model.id, "cuda"
    )


def test_only_selected_device_override_changes_identity() -> None:
    """A CPU policy change does not strand an equivalent GPU run."""
    model = REGISTRY["smollm3"]
    specs = list(model.param_specs)
    index = next(
        position
        for position, spec in enumerate(specs)
        if spec.name == "max_new_tokens"
    )
    original = specs[index]
    overrides = dict(original.overrides or {})
    cpu = overrides["cpu"]
    assert isinstance(cpu.default, int)
    overrides["cpu"] = cpu.model_copy(
        update={"default": cpu.default + 1}
    )
    specs[index] = original.model_copy(
        update={"overrides": overrides}
    )

    changed_cpu = generation_schema_id(
        model_id=model.id,
        input_mode=model.capabilities.input_mode,
        device="cpu",
        specs=specs,
    )
    unchanged_gpu = generation_schema_id(
        model_id=model.id,
        input_mode=model.capabilities.input_mode,
        device="cuda",
        specs=specs,
    )

    assert changed_cpu != _schema_id(model.id, "cpu")
    assert unchanged_gpu == _schema_id(model.id, "cuda")


def test_presentation_copy_does_not_change_schema_identity() -> None:
    """A label-only release does not invalidate pending generation."""
    model = REGISTRY["llada"]
    specs = list(model.param_specs)
    specs[0] = specs[0].model_copy(
        update={
            "label": "Renamed Steps",
            "help": "Different presentation copy.",
        }
    )

    changed = generation_schema_id(
        model_id=model.id,
        input_mode=model.capabilities.input_mode,
        device="cuda",
        specs=specs,
    )

    assert changed == _schema_id(model.id, "cuda")


@pytest.mark.parametrize(
    ("model_id", "device", "omitted"), _matrix()
)
def test_an_omitted_field_takes_the_advertised_default(
    model_id: str, device: str, omitted: str
) -> None:
    """The case the browser never exercises and an API client always
    might. One field at a time, so a default that is wrong for exactly
    one parameter cannot hide behind six that are right."""
    info = REGISTRY[model_id]
    sent = _full_request(model_id, device)
    del sent[omitted]

    resolved = resolve_params(
        info.param_specs,
        sent,
        device=device,
        experimental=False,
    )

    assert resolved[omitted] == default_of(
        _spec(model_id, omitted), device=device
    )
    for name, value in sent.items():
        assert resolved[name] == value, name


@pytest.mark.parametrize(
    ("model_id", "device"),
    [
        (model_id, device)
        for model_id, info in REGISTRY.items()
        for device in info.capabilities.supported_devices
    ],
)
def test_an_empty_request_is_all_defaults(
    model_id: str, device: str
) -> None:
    """The matrix drops one field; this drops all of them, which is
    what a client sending only a prompt does."""
    resolved = resolve_params(
        REGISTRY[model_id].param_specs,
        {},
        device=device,
        experimental=False,
    )

    assert resolved == _defaults(model_id, device)


# -- the drift this replaces --


def test_llada_no_longer_defaults_to_a_different_shape() -> None:
    """The named regression. The worker used to fall back to a
    generation length of 128 in blocks of 32 while the registry
    advertised 160 in one block, so an omitted field did not shorten
    the run, it moved it to four-block semi-autoregressive decoding
    with a different remasking pool at every step."""
    resolved = resolve_params(
        REGISTRY["llada"].param_specs,
        {},
        device="cuda",
        experimental=False,
    )

    assert resolved["gen_length"] == 160
    assert resolved["block_length"] == 160
    # The property that actually differed, stated rather than implied.
    assert (
        resolved["gen_length"] // resolved["block_length"] == 1
    )


def test_the_advertised_default_is_what_gets_resolved() -> None:
    """Paired with the test above: that one pins two numbers, this one
    pins the relationship, so changing a registry default cannot
    silently diverge from what a request resolves to."""
    for model_id, info in REGISTRY.items():
        for device in info.capabilities.supported_devices:
            resolved = resolve_params(
                info.param_specs,
                {},
                device=device,
                experimental=False,
            )
            for spec in info.param_specs:
                assert resolved[spec.name] == default_of(
                    spec, device=device
                ), f"{model_id}/{device}/{spec.name}"


# -- device overrides --


def test_the_cpu_override_lowers_the_default() -> None:
    """SmolLM3 is the only model declaring an override, and it is the
    reason the resolver takes a device at all."""
    on_gpu = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {},
        device="cuda",
        experimental=False,
    )
    on_cpu = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {},
        device="cpu",
        experimental=False,
    )

    assert on_gpu["max_new_tokens"] == 256
    assert on_cpu["max_new_tokens"] == 128


def test_the_cpu_override_lowers_the_ceiling_too() -> None:
    """A default the user can raise past its own cap would be a
    pointless override. The report calls the alternative a hidden
    clamp: the UI shows this bound, so the worker must hold to it."""
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {"max_new_tokens": 2048},
        device="cpu",
        experimental=False,
    )

    assert resolved["max_new_tokens"] == 128


def test_experimental_still_lifts_the_cpu_cap() -> None:
    """What the registry comment beside the override claims, and
    nothing checked. The CPU override narrows the recommended range
    only, so the experimental range stays the model's, and a user who
    accepts slow decoding is not held to the cautious ceiling."""
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {"max_new_tokens": 1024},
        device="cpu",
        experimental=True,
    )

    assert resolved["max_new_tokens"] == 1024


def test_an_override_may_narrow_the_experimental_range_too() -> None:
    """The branch no registered model reaches. Built here rather than
    added to the registry, because the resolver takes the specs it is
    given and a schema nothing ships is exactly what that buys.

    Without this the override's experimental bounds could be ignored
    entirely and every other test would still pass."""
    spec = ParamSpec(
        name="budget",
        label="Budget",
        type=ParamType.INT,
        default=100,
        recommended=(1, 200),
        experimental=(1, 4000),
        overrides={
            "cpu": ParamOverride(
                default=50,
                recommended=(1, 60),
                experimental=(1, 80),
            )
        },
    )

    assert bounds_of(
        spec, device="cpu", experimental=True
    ) == (1, 80)
    assert bounds_of(
        spec, device="cpu", experimental=False
    ) == (1, 60)
    assert bounds_of(
        spec, device="cuda", experimental=True
    ) == (1, 4000)
    resolved = resolve_params(
        [spec], {"budget": 3000}, device="cpu", experimental=True
    )
    assert resolved["budget"] == 80


def test_an_unloaded_device_resolves_the_base_spec() -> None:
    """``effective_device`` is None until a model loads. Not a
    generation path, but answering with the base default beats
    resolving against a device nobody picked."""
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {},
        device=None,
        experimental=False,
    )

    assert resolved["max_new_tokens"] == 256


# -- the boundary --


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        (16, 16),
        (17, 17),
        (15, 16),
        (256, 256),
        (257, 256),
        (100000, 256),
        (-5, 16),
    ],
)
def test_a_numeric_value_is_held_to_its_bounds(
    sent: int, expected: int
) -> None:
    """Both sides of both ends, since a clamp with min and max
    transposed passes any test that only pushes one way."""
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {"max_new_tokens": sent},
        device="cuda",
        experimental=False,
    )

    assert resolved["max_new_tokens"] == expected


def test_experimental_widens_what_is_accepted() -> None:
    """The same request, twice, differing only in the flag. A value
    past the recommended ceiling is the whole point of the toggle."""
    request = {"max_new_tokens": 1024}
    specs = REGISTRY["smollm3"].param_specs

    recommended = resolve_params(
        specs, request, device="cuda", experimental=False
    )
    experimental = resolve_params(
        specs, request, device="cuda", experimental=True
    )

    assert recommended["max_new_tokens"] == 256
    assert experimental["max_new_tokens"] == 1024


def test_a_seed_is_held_to_its_declared_range() -> None:
    """A behaviour change worth stating. The workers used to pass the
    seed through as a bare int, so the one parameter with bounds in
    the registry and no enforcement anywhere was the one feeding
    ``torch.manual_seed``."""
    resolved = resolve_params(
        REGISTRY["llada"].param_specs,
        {"seed": 2**40},
        device="cuda",
        experimental=False,
    )

    assert resolved["seed"] == 2**31 - 1


# -- the types --


@pytest.mark.parametrize("sent", ["abc", None, "", [1]])
def test_a_value_that_is_not_a_number_is_refused(
    sent: Any,
) -> None:
    """Refused rather than coerced, and the message names the field:
    the old path raised from a bare ``float()``, so the user learned
    which request failed but not which part of it."""
    with pytest.raises(ValueError, match="max_new_tokens"):
        resolve_params(
            REGISTRY["smollm3"].param_specs,
            {"max_new_tokens": sent},
            device="cuda",
            experimental=False,
        )


def test_a_numeric_string_is_still_accepted() -> None:
    """The negative space of the test above. Query strings and older
    clients send numbers as text, and the previous ``float()`` took
    them, so refusing them now would be a regression dressed as
    strictness."""
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {"max_new_tokens": "64"},
        device="cuda",
        experimental=False,
    )

    assert resolved["max_new_tokens"] == 64


@pytest.mark.parametrize(
    "sent",
    [10**1000, 1e101, "1e10000", float("inf")],
)
def test_huge_numeric_values_are_refused_without_overflow(
    sent: Any,
) -> None:
    """Worker boundaries make hostile magnitudes a ValueError."""
    with pytest.raises(ValueError, match="number|finite|limit"):
        resolve_params(
            REGISTRY["smollm3"].param_specs,
            {"max_new_tokens": sent},
            device="cuda",
            experimental=True,
        )


def test_an_integer_parameter_resolves_to_an_int() -> None:
    """A float reaching a sampler that indexes with it fails much
    later and much less clearly."""
    resolved = resolve_params(
        REGISTRY["llada"].param_specs,
        {"steps": 64.7},
        device="cuda",
        experimental=False,
    )

    assert resolved["steps"] == 64
    assert isinstance(resolved["steps"], int)


def test_boolean_parameters_keep_actual_booleans() -> None:
    resolved = resolve_params(
        REGISTRY["smollm3"].param_specs,
        {"thinking": True, "alternatives": False},
        device="cuda",
        experimental=False,
    )

    assert resolved["thinking"] is True
    assert resolved["alternatives"] is False


@pytest.mark.parametrize(
    ("model_id", "parameter"),
    [
        (model_id, spec.name)
        for model_id, info in REGISTRY.items()
        for spec in info.param_specs
        if spec.type == ParamType.BOOL
    ],
)
@pytest.mark.parametrize(
    "sent",
    ["false", "true", 0, 1, None],
)
def test_every_boolean_parameter_refuses_non_booleans(
    model_id: str,
    parameter: str,
    sent: Any,
) -> None:
    info = REGISTRY[model_id]
    device = info.capabilities.supported_devices[0]

    with pytest.raises(ValueError, match=parameter):
        resolve_params(
            info.param_specs,
            {parameter: sent},
            device=device,
            experimental=True,
        )


@pytest.mark.parametrize(
    "backend",
    [
        LladaBackend(),
        Smollm3Backend(),
    ],
    ids=["llada", "append_only"],
)
def test_every_worker_boundary_refuses_textual_experimental(
    backend: Any,
) -> None:
    with pytest.raises(ValueError, match="experimental"):
        backend._validate_generate(
            {
                "prompt": "hi",
                "experimental": "false",
            }
        )


@pytest.mark.parametrize("model_id", ["llada", "diffusiongemma"])
def test_a_diffusion_run_captures_candidates_unless_told_not_to(
    model_id: str,
) -> None:
    """On by default, as the autoregressive parameter is, because a
    run that would outgrow the budget thins rather than grows; and
    the workers read it unguarded, so it must always resolve."""
    specs = REGISTRY[model_id].param_specs

    omitted = resolve_params(
        specs, {}, device="cuda", experimental=False
    )
    refused = resolve_params(
        specs,
        {"alternatives": False},
        device="cuda",
        experimental=False,
    )

    assert omitted["alternatives"] is True
    assert refused["alternatives"] is False


# -- the options --


def test_a_select_refuses_a_value_it_does_not_offer() -> None:
    """The list lived twice: once as the picker's options and once as
    a set beside the worker. Either copy could have grown alone."""
    with pytest.raises(ValueError, match="remasking"):
        resolve_params(
            REGISTRY["llada"].param_specs,
            {"remasking": "confidence"},
            device="cuda",
            experimental=False,
        )


@pytest.mark.parametrize(
    "strategy", ["low_confidence", "random"]
)
def test_a_select_accepts_every_option_it_offers(
    strategy: str,
) -> None:
    """Both halves of the same rule, so a validator that refused
    everything would fail here rather than pass the test above."""
    resolved = resolve_params(
        REGISTRY["llada"].param_specs,
        {"remasking": strategy},
        device="cuda",
        experimental=False,
    )

    assert resolved["remasking"] == strategy


# -- what the schema does not decide --


def test_the_resolver_leaves_relational_rules_alone() -> None:
    """The report says to keep model-specific relational checks in the
    worker, so the resolver has to accept a pair the sampler will
    reject. If it silently repaired them instead, the worker's error
    would become unreachable and the user would get a run at a shape
    they did not ask for."""
    resolved = resolve_params(
        REGISTRY["llada"].param_specs,
        {"gen_length": 100, "block_length": 30},
        device="cuda",
        experimental=False,
    )

    assert resolved["gen_length"] == 100
    assert resolved["block_length"] == 30
    assert resolved["gen_length"] % resolved["block_length"]


def test_the_worker_still_enforces_them() -> None:
    """The other side of the split, through the real backend, so the
    two tests together prove the rule moved nowhere."""
    backend = LladaBackend()

    with pytest.raises(ValueError, match="divisible"):
        backend._validate_generate(
            {
                "prompt": "hi",
                "gen_length": 100,
                "block_length": 30,
            }
        )


def test_the_worker_still_requires_a_prompt() -> None:
    """The prompt is not a declared parameter, so the resolver knows
    nothing about it and each worker keeps its own check."""
    backend = LladaBackend()

    with pytest.raises(ValueError, match="prompt"):
        backend._validate_generate({"prompt": "   "})


# -- presentation metadata --


def test_presentation_metadata_defaults_are_safe() -> None:
    """An older or third-party spec stays reachable in General.

    It does not silently join the collapsed summary, where an
    unreviewed parameter would crowd out the deliberate readout.
    """
    spec = ParamSpec(
        name="new_parameter",
        label="New Parameter",
        type=ParamType.INT,
        default=1,
    )

    assert spec.group == ParamGroup.GENERAL
    assert spec.prominence == ParamProminence.SECONDARY


def test_registered_parameters_have_meaningful_groups() -> None:
    """Every shipped control opted into presentation metadata."""
    for model_id, info in REGISTRY.items():
        for spec in info.param_specs:
            assert spec.group != ParamGroup.GENERAL, (
                f"{model_id}/{spec.name} still uses the safe fallback"
            )


def test_each_model_has_a_bounded_collapsed_summary() -> None:
    """A summary stays useful only while it is selective."""
    for model_id, info in REGISTRY.items():
        primary = [
            spec
            for spec in info.param_specs
            if spec.prominence == ParamProminence.PRIMARY
        ]

        assert 1 <= len(primary) <= 3, (
            f"{model_id} exposes {len(primary)} summary controls"
        )


def test_kgw_parameters_belong_only_to_append_models() -> None:
    expected = {
        "watermark",
        "watermark_gamma",
        "watermark_delta",
    }
    for model_id, info in REGISTRY.items():
        names = {spec.name for spec in info.param_specs}
        if model_id in {"smollm3", "mamba3"}:
            assert expected <= names
        else:
            assert expected.isdisjoint(names)


def test_kgw_controls_are_experimental_signals() -> None:
    for model_id in ("smollm3", "mamba3"):
        specs = {
            spec.name: spec
            for spec in REGISTRY[model_id].param_specs
        }
        for name in (
            "watermark",
            "watermark_gamma",
            "watermark_delta",
        ):
            assert specs[name].group == ParamGroup.SIGNALS
            assert specs[name].experimental_only is True
        assert specs["watermark"].default is False
        assert specs["watermark_gamma"].default == pytest.approx(0.25)
        assert specs["watermark_delta"].default == pytest.approx(2.0)


def test_watermark_membership_signal_is_append_only() -> None:
    for model_id, info in REGISTRY.items():
        names = {
            signal.name for signal in info.capabilities.signals
        }
        if model_id in {"smollm3", "mamba3"}:
            assert "watermark_membership" in names
            assert "watermark_evidence" in names
        else:
            assert "watermark_membership" not in names
            assert "watermark_evidence" not in names
