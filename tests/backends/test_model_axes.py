"""The model axes are orthogonal, and the disk form derives from one.

Strategy: pure schema work over `ModelCapabilities`, so this needs no
model, no GPU and no worker. Two halves. The first walks the registry
and asserts every model answers all three axes, which is what stops a
new entry inheriting a family it never chose. The second builds
capability objects for combinations no registered model has, because
the whole reason `ROADMAP-01` split one value into three is that the
combinations it could not express are the ones coming next: a GPU-only
append-only model, and a family that is neither of today's two.

Passing proves the axes can be set independently, that a shape decides
the `model_type` a run records on disk while the family does not, and
that every declared shape has a disk spelling. Failing means either an
axis has quietly acquired a default, or a shape was added without
teaching the save path how to write it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.backends.protocol import (
    FAMILIES,
    GENERATION_SHAPE_APPEND_ONLY,
    GENERATION_SHAPE_ITERATIVE_CANVAS,
    GENERATION_SHAPES,
    SAVED_MODEL_TYPE_AUTOREGRESSIVE,
    SAVED_MODEL_TYPE_DIFFUSION,
    ModelCapabilities,
    saved_model_type,
)
from src.backends.registry import REGISTRY


def _capabilities(
    *,
    family: str,
    shape: str,
    devices: tuple,
    resume: bool = False,
) -> ModelCapabilities:
    """One capability set, with only the axes under test named."""
    return ModelCapabilities(
        family=family,
        generation_shape=shape,
        supported_devices=devices,
        supports_resume=resume,
    )


# -- what the registry declares --


def test_every_model_answers_all_three_axes() -> None:
    """A model that answered none of them used to be filed as
    diffusion on both axes and CPU-capable on the third, which is
    three wrong answers from saying nothing."""
    assert REGISTRY, "the registry must not be empty"
    for model_id, info in REGISTRY.items():
        capabilities = info.capabilities
        assert capabilities.family in FAMILIES, model_id
        assert (
            capabilities.generation_shape in GENERATION_SHAPES
        ), model_id
        assert capabilities.supported_devices, model_id


def test_every_declared_device_is_one_we_know() -> None:
    """A typo here reads as "supports nothing", and the supervisor
    would refuse every activation of the model with a message naming
    a device the user never picked."""
    for model_id, info in REGISTRY.items():
        for device in info.capabilities.supported_devices:
            assert device in ("cuda", "cpu"), (
                f"{model_id} declares {device!r}"
            )


def test_some_model_can_still_run_without_a_gpu() -> None:
    """The axes made every diffusion model GPU-only, which is honest,
    but a host with no GPU must keep one model it can load or the app
    has nothing to show such a user."""
    cpu_capable = [
        model_id
        for model_id, info in REGISTRY.items()
        if "cpu" in info.capabilities.supported_devices
    ]

    assert cpu_capable, "no model is loadable without a GPU"


# -- the axes are independent --


def test_a_gpu_only_append_only_model_is_expressible() -> None:
    """The combination that motivated the split and that no registered
    model has: a state-space model appends like an autoregressive one
    and needs a GPU like a diffusion one. Under a single `model_type`
    it had to lie on one axis to be honest on the other."""
    capabilities = _capabilities(
        family="state_space",
        shape=GENERATION_SHAPE_APPEND_ONLY,
        devices=("cuda",),
    )

    assert capabilities.family == "state_space"
    assert (
        capabilities.generation_shape
        == GENERATION_SHAPE_APPEND_ONLY
    )
    assert "cpu" not in capabilities.supported_devices


def test_a_cpu_capable_append_only_model_is_expressible() -> None:
    """The one shape/device pairing that does exist today, asserted
    beside its opposite so the pair reads as a matrix rather than as
    one case that happens to work."""
    capabilities = _capabilities(
        family="autoregressive",
        shape=GENERATION_SHAPE_APPEND_ONLY,
        devices=("cuda", "cpu"),
    )

    assert "cpu" in capabilities.supported_devices


@pytest.mark.parametrize("resume", [True, False])
def test_an_iterative_canvas_works_either_way_on_resume(
    resume: bool,
) -> None:
    """Resume is a capability of the model, not of the shape. Both
    diffusion models happen to support it, so nothing today would
    catch a reader that inferred one from the other."""
    capabilities = _capabilities(
        family="diffusion",
        shape=GENERATION_SHAPE_ITERATIVE_CANVAS,
        devices=("cuda",),
        resume=resume,
    )

    assert capabilities.supports_resume is resume
    assert (
        capabilities.generation_shape
        == GENERATION_SHAPE_ITERATIVE_CANVAS
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_any_family_may_take_any_shape(family: str) -> None:
    """The negative space: no family is wired to a shape. If one were,
    adding a model would mean discovering the coupling at runtime."""
    for shape in GENERATION_SHAPES:
        capabilities = _capabilities(
            family=family, shape=shape, devices=("cuda",)
        )
        assert capabilities.family == family
        assert capabilities.generation_shape == shape


# -- nothing may be omitted --


@pytest.mark.parametrize(
    "missing", ["family", "generation_shape", "supported_devices"]
)
def test_an_omitted_axis_is_refused(missing: str) -> None:
    """The point of making these required. A default is how LLaDA came
    to advertise a CPU placement nobody decided on, so saying nothing
    must fail at construction rather than at activation."""
    fields = {
        "family": "diffusion",
        "generation_shape": GENERATION_SHAPE_ITERATIVE_CANVAS,
        "supported_devices": ("cuda",),
    }
    del fields[missing]

    with pytest.raises(ValidationError):
        ModelCapabilities(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "transformer"),
        ("generation_shape", "diffusion"),
    ],
)
def test_an_unknown_axis_value_is_refused(
    field: str, value: str
) -> None:
    """The two axes have overlapping vocabularies in the reader's head
    ("diffusion" is a family, never a shape), so the schema has to be
    the thing that keeps them apart."""
    fields = {
        "family": "diffusion",
        "generation_shape": GENERATION_SHAPE_ITERATIVE_CANVAS,
        "supported_devices": ("cuda",),
    }
    fields[field] = value

    with pytest.raises(ValidationError):
        ModelCapabilities(**fields)


# -- the disk form follows the shape --


def test_the_disk_form_names_every_shape() -> None:
    """A shape with no spelling would be written as whatever the
    mapping returned for a miss, and Analytics would gate the wrong
    charts off a value nobody chose."""
    written = {
        saved_model_type(shape) for shape in GENERATION_SHAPES
    }

    assert written == {
        SAVED_MODEL_TYPE_AUTOREGRESSIVE,
        SAVED_MODEL_TYPE_DIFFUSION,
    }


def test_an_appending_run_records_the_autoregressive_form() -> None:
    """Shape, not family: a state-space run records the same disk
    value a SmolLM3 run does, because every reader of the field is
    asking whether there is a masked canvas to converge."""
    assert (
        saved_model_type(GENERATION_SHAPE_APPEND_ONLY)
        == SAVED_MODEL_TYPE_AUTOREGRESSIVE
    )
    assert (
        saved_model_type(GENERATION_SHAPE_ITERATIVE_CANVAS)
        == SAVED_MODEL_TYPE_DIFFUSION
    )


def test_the_family_cannot_reach_the_disk_form() -> None:
    """The derivation takes a shape, so a caller that passes a family
    fails loudly instead of writing "state_space" into a field three
    readers compare against "autoregressive"."""
    with pytest.raises(AssertionError):
        saved_model_type("state_space")


def test_every_registered_model_writes_a_known_form() -> None:
    """Paired with the mapping test above: that one proves the table
    is complete, this one proves the registry only ever asks it for
    shapes it holds."""
    for model_id, info in REGISTRY.items():
        written = saved_model_type(
            info.capabilities.generation_shape
        )
        assert written in (
            SAVED_MODEL_TYPE_AUTOREGRESSIVE,
            SAVED_MODEL_TYPE_DIFFUSION,
        ), model_id
