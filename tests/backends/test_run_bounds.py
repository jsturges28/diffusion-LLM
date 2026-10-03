"""How large one run of each model can be (`A2-TRUST-02`).

A save is held to the most one run of its model can hold, and that
is read off the tops of the model's own sliders. Strategy: compute
each model's bounds independently from its registered sliders and
the shape its generation takes, and compare. Passing proves the
bounds follow the sliders, so widening one widens what a save may
carry, and that a model the build does not know is held to the most
any model allows rather than refused or left unbounded.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

from src.backends import registry
from src.backends.protocol import ModelInfo, ParamOverride
from src.backends.registry import (
    DGEMMA,
    DGEMMA_CANVAS_TOKENS,
    LLADA,
    MAMBA3,
    REGISTRY,
    RUN_BOUNDS,
    RUN_BOUNDS_WIDEST,
    SMOLLM3,
    RunBounds,
    run_bounds,
)


def _top(model: ModelInfo, name: str) -> int:
    """The top of a slider's experimental range, read directly."""
    for spec in model.param_specs:
        if spec.name == name:
            assert spec.experimental is not None
            return int(spec.experimental[1])
    raise AssertionError(f"{model.id} has no {name} slider")


def test_llada_holds_an_opening_frame_and_one_a_step() -> None:
    bounds = run_bounds(LLADA.id)

    assert bounds.frames_max == _top(LLADA, "steps") + 1
    assert bounds.positions_max == _top(LLADA, "gen_length")
    assert bounds.frame_positions_max == bounds.positions_max


def test_diffusiongemma_holds_a_draft_a_step_and_a_commit() -> None:
    bounds = run_bounds(DGEMMA.id)
    budget = _top(DGEMMA, "max_new_tokens")
    canvases = -(-budget // DGEMMA_CANVAS_TOKENS)
    per_canvas = _top(DGEMMA, "max_denoising_steps") + 1

    assert canvases >= 2, "the budget spans more than one canvas"
    assert bounds.frames_max == canvases * per_canvas
    assert bounds.positions_max >= budget
    assert bounds.frame_positions_max == DGEMMA_CANVAS_TOKENS


@pytest.mark.parametrize("model", [SMOLLM3, MAMBA3])
def test_an_appending_model_holds_a_frame_a_position(
    model: ModelInfo,
) -> None:
    bounds = run_bounds(model.id)
    budget = _top(model, "max_new_tokens")

    assert bounds.frames_max == budget
    assert bounds.positions_max == budget
    assert bounds.frame_positions_max == budget


def test_every_registered_model_has_its_own_bounds() -> None:
    assert set(RUN_BOUNDS) == set(REGISTRY)
    for model_id in REGISTRY:
        assert run_bounds(model_id) is RUN_BOUNDS[model_id]


def test_a_model_this_build_does_not_know_gets_the_widest() -> None:
    assert "retired-model" not in REGISTRY
    widest = run_bounds("retired-model")

    assert widest == RUN_BOUNDS_WIDEST
    for bounds in RUN_BOUNDS.values():
        assert bounds.frames_max <= widest.frames_max
        assert bounds.positions_max <= widest.positions_max
        assert (
            bounds.frame_positions_max <= widest.frame_positions_max
        )


def _with_override(
    model: ModelInfo, name: str, experimental: Tuple[float, float]
) -> ModelInfo:
    """``model`` with one slider given a device override."""
    specs = []
    for spec in model.param_specs:
        if spec.name == name:
            override = ParamOverride(experimental=experimental)
            spec = spec.model_copy(
                update={"overrides": {"cuda": override}}
            )
        specs.append(spec)
    return model.model_copy(update={"param_specs": specs})


@pytest.mark.parametrize(
    ("override_top", "expected"),
    [(4096.0, 4096), (512.0, None)],
)
def test_a_device_override_can_only_raise_the_top(
    override_top: float, expected: Optional[int]
) -> None:
    """The bound is the most the slider reaches on any device, so an
    override above the base range raises it and one below does not
    lower it."""
    model = _with_override(
        SMOLLM3, "max_new_tokens", (1.0, override_top)
    )
    top = registry._slider_top(model, "max_new_tokens")

    if expected is None:
        assert top == _top(SMOLLM3, "max_new_tokens")
    else:
        assert top == expected


def test_bounds_cannot_be_changed_after_import() -> None:
    bounds = run_bounds(LLADA.id)
    assert isinstance(bounds, RunBounds)
    with pytest.raises(AttributeError):
        bounds.frames_max = 1  # type: ignore[misc]
