"""Tests for the registry's declaration of adaptive stopping.

Strategy: read the registry directly. A model that says it stops a
canvas adaptively promises the page three parameters to read the rule
from, and the readout, the Stopping chart and the frames endpoint all
rely on that promise rather than checking for it. The rest pins the
numbers that only the checkpoint can confirm, with where they came
from, and the ordering a Compare label depends on.

Passing proves the flag and its parameters travel together, that
DiffusionGemma's defaults are the rule its runs used before the rule
was a parameter, that no legal value can make the rule unreachable or
meaningless, and that DiffusionGemma's Compare labels read as before.
"""

from __future__ import annotations

from typing import Dict

import pytest

from src.backends.protocol import ParamSpec, ParamType
from src.backends.registry import DGEMMA, REGISTRY

# The parameters a model that stops adaptively must declare, with the
# type the readout reads each one as.
RULE_PARAMS: Dict[str, ParamType] = {
    "confidence_threshold": ParamType.FLOAT,
    "stability_threshold": ParamType.INT,
    "max_denoising_steps": ParamType.INT,
}


STOPPING_MODELS = sorted(
    model_id
    for model_id, entry in REGISTRY.items()
    if entry.capabilities.adaptive_stopping
)


def _specs(model_id: str) -> Dict[str, ParamSpec]:
    return {
        spec.name: spec for spec in REGISTRY[model_id].param_specs
    }


@pytest.mark.parametrize("model_id", STOPPING_MODELS)
def test_a_model_that_stops_adaptively_declares_its_rule(
    model_id: str,
) -> None:
    specs = _specs(model_id)
    for name, kind in RULE_PARAMS.items():
        assert name in specs, f"{model_id} lacks {name}"
        assert specs[name].type == kind, name


def test_only_diffusiongemma_stops_adaptively() -> None:
    """LLaDA runs a fixed schedule and the others stop at an end
    token, so a readout of distance to a stop would be about a rule
    they do not have."""
    assert STOPPING_MODELS == ["diffusiongemma"]


def test_the_defaults_are_the_checkpoints_rule() -> None:
    """Read from the checkpoint's generation_config.json on
    2026-10-01, which every earlier run stopped by. Tests cannot read
    the checkpoint, a 13 GiB local directory, so the numbers are
    pinned here with their source."""
    specs = _specs("diffusiongemma")

    assert specs["confidence_threshold"].default == 0.005
    assert specs["stability_threshold"].default == 1


def test_no_legal_entropy_makes_the_rule_unreachable() -> None:
    """A mean entropy can never be below zero, and transformers
    refuses a threshold of zero outright."""
    spec = _specs("diffusiongemma")["confidence_threshold"]
    assert spec.recommended is not None
    assert spec.experimental is not None

    assert spec.recommended[0] > 0
    assert spec.experimental[0] > 0


def test_no_legal_steady_steps_is_negative() -> None:
    """Zero is legal and drops the condition; below that means
    nothing, and transformers refuses it."""
    spec = _specs("diffusiongemma")["stability_threshold"]
    assert spec.recommended is not None
    assert spec.experimental is not None

    assert spec.recommended[0] == 0
    assert spec.experimental[0] == 0


@pytest.mark.parametrize(
    "name", ["confidence_threshold", "stability_threshold"]
)
def test_each_default_sits_inside_its_ranges(name: str) -> None:
    spec = _specs("diffusiongemma")[name]
    assert spec.recommended is not None
    assert spec.experimental is not None
    low, high = spec.recommended
    wide_low, wide_high = spec.experimental

    assert low <= float(spec.default) <= high
    assert wide_low <= low
    assert high <= wide_high


def test_compare_labels_still_name_the_same_parameters() -> None:
    """A Compare label names a run's first three parameters, so the
    rule sits after the temperatures rather than beside the step
    budget it relates to."""
    names = [spec.name for spec in DGEMMA.param_specs]
    labelled = ["max_new_tokens", "max_denoising_steps", "t_max"]

    assert names[:3] == labelled
    assert names.index("confidence_threshold") > names.index("t_min")
