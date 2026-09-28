"""Retention weights rebuild the state they claim to describe.

Strategy: step the real recurrence from an empty state with capture
on, then rebuild the state from nothing but the retention weights and
the captured keys and values, and require the two to match. The same
for the output: the dual-form attention times the values must give
the ungated output the recurrence produced. Passing proves the weights
are exact rather than plausible, which is the claim the retention lens
is built on. Hand-made decays check the formula's edges, and the
statistics the falsification reads get their own boundary cases.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from src.inference.mamba3 import (
    CoreInputs,
    LayerState,
    StepTerms,
    recur_sequence,
)
from src.inference.mamba3_memory import (
    LayerTrace,
    coefficient_of_variation,
    contribution_norms,
    head_average,
    output_attention,
    output_norms,
    reconstruct_state,
    retention_weights,
    spearman,
    stack_terms,
    top_overlap,
)

LENGTH, HEADS, QK_DIM, V_DIM, ANGLES = 10, 4, 16, 8, 4
# Rebuilding sums ten fp32 terms that the recurrence accumulated
# step by step, so the two agree to rounding, not bit for bit.
TOLERANCE = 1e-5


def _close(first: torch.Tensor, second: torch.Tensor) -> None:
    torch.testing.assert_close(
        first, second, rtol=TOLERANCE, atol=TOLERANCE
    )


def _draw(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=gen)


def _inputs(seed: int, length: int = LENGTH) -> CoreInputs:
    """One batch row with no gate and no skip, so the recurrence's
    output is the state read by the query and nothing else."""
    gen = torch.Generator().manual_seed(seed)
    dt = F.softplus(_draw(gen, 1, HEADS, length))
    rate = F.softplus(_draw(gen, 1, HEADS, length)) + 1e-4
    return CoreInputs(
        q=_draw(gen, 1, length, 1, QK_DIM),
        k=_draw(gen, 1, length, 1, QK_DIM),
        v=_draw(gen, 1, length, HEADS, V_DIM),
        adt=-rate * dt,
        dt=dt,
        trap=_draw(gen, 1, HEADS, length),
        q_bias=_draw(gen, HEADS, QK_DIM),
        k_bias=_draw(gen, HEADS, QK_DIM),
        angles=_draw(gen, 1, length, HEADS, ANGLES),
        d=torch.zeros(HEADS),
        z=None,
    )


def _empty() -> LayerState:
    return LayerState(
        torch.zeros(1, HEADS, ANGLES),
        torch.zeros(1, HEADS, V_DIM, QK_DIM),
        torch.zeros(1, HEADS, QK_DIM),
        torch.zeros(1, HEADS, V_DIM),
    )


def _prefix(inputs: CoreInputs, count: int) -> CoreInputs:
    """The first `count` tokens of a sequence, in every field."""
    return inputs._replace(
        q=inputs.q[:, :count],
        k=inputs.k[:, :count],
        v=inputs.v[:, :count],
        adt=inputs.adt[..., :count],
        dt=inputs.dt[..., :count],
        trap=inputs.trap[..., :count],
        angles=inputs.angles[:, :count],
    )


def _run(
    inputs: CoreInputs,
) -> Tuple[torch.Tensor, LayerState, LayerTrace]:
    sink: List[StepTerms] = []
    output, state = recur_sequence(inputs, _empty(), sink)
    return output, state, stack_terms(sink)


def _trace(
    adt: torch.Tensor, dt: torch.Tensor, gate: torch.Tensor
) -> LayerTrace:
    """A trace with chosen decays and write weights, for the formula's
    hand cases; the vectors do not enter the weights."""
    length, heads = adt.shape
    ones = torch.ones(length, heads, QK_DIM)
    return LayerTrace(
        adt, dt, gate, ones, ones, torch.ones(length, heads, V_DIM),
        torch.ones(length, heads),
    )


# -- the weights are exact --


@pytest.mark.parametrize("step", [0, 4, LENGTH - 1])
def test_the_weights_rebuild_the_stepped_state(step: int) -> None:
    """The claim the lens rests on. The state after `step` is exactly
    the sum of each earlier token's value times its rotated key,
    weighted by its retention; any other weights rebuild a different
    state."""
    inputs = _inputs(1)
    _, _, trace = _run(inputs)
    _, state, _ = _run(_prefix(inputs, step + 1))

    rebuilt = reconstruct_state(trace, retention_weights(trace, step))

    _close(rebuilt, state.ssm[0])


def test_the_attention_rebuilds_the_output() -> None:
    """The output side: attention times values gives what the
    recurrence emitted, which is what makes it the dual form."""
    inputs = _inputs(2)
    output, _, trace = _run(inputs)
    step = LENGTH - 1

    attention = output_attention(
        trace, retention_weights(trace, step), step
    )

    rebuilt = torch.einsum("sh,shp->hp", attention, trace.v)
    _close(rebuilt, output[0, step])


def test_the_norms_are_the_norms_of_each_term() -> None:
    """Each token's term in the state is rank one, so its Frobenius
    norm is the product of the three factors' sizes."""
    _, _, trace = _run(_inputs(3))
    weights = retention_weights(trace, LENGTH - 1)

    norms = contribution_norms(trace, weights)

    token, head = 3, 2
    term = weights[token, head] * torch.outer(
        trace.v[token, head], trace.k[token, head]
    )
    expected = torch.linalg.matrix_norm(term)
    _close(norms[token, head], expected)


def test_the_output_norms_scale_by_each_value() -> None:
    _, _, trace = _run(_inputs(4))
    step = LENGTH - 1
    attention = output_attention(
        trace, retention_weights(trace, step), step
    )

    norms = output_norms(trace, attention)

    expected = attention.abs() * trace.v.norm(dim=-1)
    _close(norms, expected)


# -- the formula's edges --


def test_without_decay_every_write_is_kept_whole() -> None:
    dt = torch.full((5, 2), 0.5)
    gate = torch.full((5, 2), 0.25)
    trace = _trace(torch.zeros(5, 2), dt, gate)

    weights = retention_weights(trace, 4)

    # gamma is 0.5 * 0.25; the next token rewrites 0.5 * 0.75.
    expected = torch.full((5, 2), 0.125 + 0.375)
    expected[4] = 0.125
    _close(weights, expected)


def test_a_constant_decay_gives_geometric_weights() -> None:
    rate = -0.3
    trace = _trace(
        torch.full((6, 1), rate),
        torch.full((6, 1), 1.0),
        torch.full((6, 1), 0.5),
    )

    weights = retention_weights(trace, 5)[:5, 0]

    ratios = weights[1:] / weights[:-1]
    _close(ratios, torch.full((4,), math.exp(-rate)))


def test_the_newest_token_has_only_its_first_write() -> None:
    """Its second write belongs to a step that has not happened."""
    _, _, trace = _run(_inputs(5))

    weights = retention_weights(trace, LENGTH - 1)

    newest = trace.dt[-1] * trace.gate[-1]
    _close(weights[-1], newest)


def test_a_long_run_of_decay_does_not_underflow() -> None:
    """Products of a thousand decays are sums of logarithms here, so
    the newest weight stays exact and nothing turns into NaN."""
    trace = _trace(
        torch.full((1000, 1), -50.0),
        torch.full((1000, 1), 1.0),
        torch.full((1000, 1), 0.5),
    )

    weights = retention_weights(trace, 999)

    assert bool(torch.isfinite(weights).all())
    assert float(weights[-1, 0]) == 0.5
    assert float(weights[0, 0]) == 0.0


def test_a_step_outside_the_trace_is_refused() -> None:
    _, _, trace = _run(_inputs(6))

    with pytest.raises(AssertionError):
        retention_weights(trace, LENGTH)


# -- the statistics the falsification reads --


def test_head_average_weights_every_head_equally() -> None:
    """A head with large numbers must not outvote a head with small
    ones; each is normalised to sum to one first."""
    scores = torch.tensor([[1.0, 100.0], [3.0, 300.0]])

    averaged = head_average(scores)

    _close(averaged, torch.tensor([0.25, 0.75]))
    _close(averaged.sum(), torch.tensor(1.0))


def test_top_overlap_counts_shared_positions() -> None:
    first = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    reversed_order = torch.flip(first, dims=[0])

    assert top_overlap(first, first, 3) == 3
    assert top_overlap(first, reversed_order, 2) == 0
    assert top_overlap(first, reversed_order, 3) == 1


def test_spearman_at_its_bounds() -> None:
    rising = torch.arange(6.0)

    assert spearman(rising, rising * 10) == pytest.approx(1.0)
    assert spearman(rising, -rising) == pytest.approx(-1.0)


def test_spearman_gives_ties_their_average_rank() -> None:
    """[1, 1, 2] ranks as [0.5, 0.5, 2]; against [0, 1, 2] that is a
    correlation of sqrt(3) / 2."""
    tied = torch.tensor([1.0, 1.0, 2.0])

    value = spearman(tied, torch.arange(3.0))

    assert value == pytest.approx(math.sqrt(3) / 2)


def test_spearman_of_a_constant_is_undefined() -> None:
    assert math.isnan(spearman(torch.ones(4), torch.arange(4.0)))


def test_the_coefficient_of_variation() -> None:
    values = torch.tensor([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])

    variation = coefficient_of_variation(values)

    # Column two: mean 4, population deviation sqrt(8 / 3).
    expected = torch.tensor([0.0, math.sqrt(8.0 / 3.0) / 4.0])
    _close(variation, expected)
