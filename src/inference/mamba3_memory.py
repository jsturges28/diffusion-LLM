"""What a Mamba-3 layer's state retains of each token, exactly.

A state-space layer forgets by multiplication. Every step scales the
whole state by alpha = exp(A * dt) and adds the new token, so the
weight an earlier token still carries in the state is not an estimate
or an attribution method: it is a product of decays, and this module
computes it. That is the lens this project wants Mamba-3 for, and why
the tests hold it to a state actually stepped, not to a formula.

The trapezoid rule adds one wrinkle. Each token is written twice: on
its own step, weighted gamma = dt * gate, and again on the next step,
weighted alpha * dt * (1 - gate), from the key and value the layer
carried over. So token s's weight in the state after step t is

    w[s, t] = (gamma[s] + dt[s+1] * (1 - gate[s+1]))
              * prod(alpha[s+1..t])

for s < t, and gamma[t] for s = t, whose second write has not happened
yet. That bracket is exactly the `scale` upstream's parallel reference
builds.

**Everything here assumes a run from the empty state**, which is how
the probe runs; a carried state contributes terms no trace can see.
The exception is `forgetting`, the per-token signal: what reading a
token erased is 1 - alpha for that token alone, whatever came before,
so it needs no trace and holds from any state.

The products are sums of logarithms: `StepTerms.adt` is log(alpha), so
a thousand decays cannot underflow before the subtraction that makes
them relative. The sums are float64 for the same reason.

The statistics at the bottom are what the retention falsification
reads, kept here so they are tested rather than trusted.
"""

from __future__ import annotations

import functools
import math
from typing import List, NamedTuple, Sequence, Set, Tuple

import torch
from torch import Tensor

from src.inference.mamba3 import (
    Core,
    CoreInputs,
    LayerState,
    StepTerms,
    recur_sequence,
)


class LayerTrace(NamedTuple):
    """One batch row of one layer's `StepTerms`, stacked over steps.

    adt, dt, gate and state_norm are (length, heads); q and k are
    (length, heads, d_state), rotated; v is (length, heads, headdim).
    """

    adt: Tensor
    dt: Tensor
    gate: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    state_norm: Tensor


assert LayerTrace._fields == StepTerms._fields, (
    "a trace stacks step terms field for field"
)


def stack_terms(
    records: Sequence[StepTerms], row: int = 0
) -> LayerTrace:
    """One layer's captured steps as tensors over time, one row."""
    assert len(records) > 0, "no steps to stack"

    def column(field: str) -> Tensor:
        steps = [getattr(step, field)[row] for step in records]
        return torch.stack(steps)

    return LayerTrace(*(column(field) for field in StepTerms._fields))


def retention_weights(trace: LayerTrace, t: int) -> Tensor:
    """w[s] for s in 0..t: token s's exact weight in the state after
    step t, per head, as (t + 1, heads)."""
    assert 0 <= t < trace.adt.shape[0], f"step {t} is not traced"
    log_decay = torch.cumsum(trace.adt[: t + 1].double(), dim=0)
    decay = torch.exp(log_decay[t] - log_decay)
    dt = trace.dt[: t + 1].double()
    gate = trace.gate[: t + 1].double()
    second_write = dt[1:] * (1.0 - gate[1:])
    carried = torch.cat([second_write, torch.zeros_like(dt[:1])])
    weights = (dt * gate + carried) * decay
    assert bool((weights >= 0).all()), "negative retention weight"
    return weights.to(trace.adt.dtype)


def reconstruct_state(trace: LayerTrace, weights: Tensor) -> Tensor:
    """The state rebuilt from its tokens: the sum of w[s] v[s] k[s]^T
    per head, as (heads, headdim, d_state). It equals the stepped
    state exactly when the weights are right."""
    count = weights.shape[0]
    return torch.einsum(
        "sh,shp,shn->hpn",
        weights.double(),
        trace.v[:count].double(),
        trace.k[:count].double(),
    ).to(trace.v.dtype)


def contribution_norms(trace: LayerTrace, weights: Tensor) -> Tensor:
    """The Frobenius norm of each token's term in the state, (count,
    heads). Exact: the term is rank one, and the rotation that made
    each key preserves its length."""
    count = weights.shape[0]
    values = torch.linalg.vector_norm(trace.v[:count], dim=-1)
    keys = torch.linalg.vector_norm(trace.k[:count], dim=-1)
    return weights * values * keys


def output_attention(
    trace: LayerTrace, weights: Tensor, t: int
) -> Tensor:
    """How much of each earlier token's value reaches the output at
    step t, signed, (t + 1, heads): the recurrence's dual form. The
    ungated output before the skip is the sum of these times v."""
    assert weights.shape[0] == t + 1, "weights must end at step t"
    scores = torch.einsum("shn,hn->sh", trace.k[: t + 1], trace.q[t])
    return weights * scores


def output_norms(trace: LayerTrace, attention: Tensor) -> Tensor:
    """The norm of each token's term in the step's output, (count,
    heads): |attention| times the length of that token's value."""
    count = attention.shape[0]
    values = torch.linalg.vector_norm(trace.v[:count], dim=-1)
    return attention.abs() * values


# -- forgetting, one number per token --


def forgetting(adts: Sequence[Tensor]) -> Tensor:
    """What reading each token erased from the state, (batch, length).

    Every step scales each head's whole state by alpha = exp(A * dt)
    before writing, so 1 - alpha is the share of that head's memory
    the token wiped out. Exact, and it needs no recurrence: only the
    `adt` each layer computed for the token. Averaged over heads and
    layers it is one number per token, which is what a reader can
    look at. `adts` holds one (batch, heads, length) tensor per layer,
    in layer order, as `recording_core` collects them.
    """
    assert len(adts) > 0, "no layer's decay was recorded"
    stacked = torch.stack([adt.float() for adt in adts])
    # A is clamped below zero and dt is positive: log(alpha) < 0.
    assert bool((stacked <= 0).all()), "a decay above one"
    kept = torch.exp(stacked).mean(dim=(0, 2))
    return 1.0 - kept


def recording_core(sink: List[Tensor]) -> Core:
    """The recurrence, keeping each layer's `adt` as it runs, in layer
    order: all `forgetting` needs, without the per-step capture that
    keeps whole query, key and value vectors."""
    return functools.partial(_record_decay, sink=sink)


def _record_decay(
    inputs: CoreInputs, state: LayerState, *, sink: List[Tensor]
) -> Tuple[Tensor, LayerState]:
    sink.append(inputs.adt)
    return recur_sequence(inputs, state)


# -- the statistics the retention falsification reads --


def head_average(scores: Tensor) -> Tensor:
    """(positions, heads) to (positions,): each head normalised to sum
    to one first, so no head dominates by its scale alone."""
    totals = scores.sum(dim=0, keepdim=True)
    assert bool((totals > 0).all()), "a head retained nothing"
    return (scores / totals).mean(dim=1)


def top_positions(scores: Tensor, count: int) -> Set[int]:
    """The positions of the `count` largest scores."""
    assert 0 < count <= scores.shape[0], "count outside the scores"
    return set(torch.topk(scores, count).indices.tolist())


def top_overlap(first: Tensor, second: Tensor, count: int) -> int:
    """How many of their top `count` positions two scorings share."""
    assert first.shape == second.shape, "compare equal lengths"
    left = top_positions(first, count)
    right = top_positions(second, count)
    return len(left & right)


def spearman(first: Tensor, second: Tensor) -> float:
    """Rank correlation, ties given their average rank. NaN when
    either side is constant, since no ordering exists to correlate."""
    assert first.dim() == 1, "rank one vector at a time"
    assert first.shape == second.shape, "compare equal lengths"
    assert first.shape[0] >= 2, "a correlation needs two points"
    a = _average_ranks(first)
    b = _average_ranks(second)
    a, b = a - a.mean(), b - b.mean()
    spread = torch.sqrt((a * a).sum() * (b * b).sum())
    if float(spread) == 0.0:
        return math.nan
    return float((a * b).sum() / spread)


def coefficient_of_variation(values: Tensor) -> Tensor:
    """Standard deviation over mean down each column, (columns,).

    Near zero means the quantity barely moves with the input, which is
    what a decay that ignores content would look like.
    """
    assert values.shape[0] >= 2, "variation needs two rows"
    mean = values.mean(dim=0)
    assert bool((mean > 0).all()), "defined for positive quantities"
    return values.std(dim=0, unbiased=False) / mean


def _average_ranks(values: Tensor) -> Tensor:
    data = values.double()
    order = torch.argsort(data, stable=True)
    ranks = torch.empty_like(data)
    ranks[order] = torch.arange(data.shape[0], dtype=torch.float64)
    unique, inverse = torch.unique(data, return_inverse=True)
    sums = torch.zeros(unique.shape[0], dtype=torch.float64)
    sums.index_add_(0, inverse, ranks)
    counts = torch.zeros(unique.shape[0], dtype=torch.float64)
    counts.index_add_(0, inverse, torch.ones_like(ranks))
    return (sums / counts)[inverse]
