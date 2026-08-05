"""First-time token reveals, shared by every sampler.

A frame's ``revealed`` field lists the positions that became resolved
in that frame and had not been resolved in any earlier frame of the
same canvas. Two features read it, and both need that exact shape.

The birth glow animates each listed position once. A signal that
merely said "resolved right now" would re-fire on every frame for
every settled token, and one that said "changed since the last
frame" would flicker on DiffusionGemma, whose draft tokens churn
until they stabilize. Monotonicity is what lets a glow mean "this
token was just born" instead of "this token exists".

Tokens per second counts the same list, so the two features can
never disagree about how many tokens a step produced.

Deliberately torch-free: the samplers hand over plain booleans, which
keeps this unit testable without a model or a GPU.
"""

from __future__ import annotations

from typing import AbstractSet, List, Sequence


def newly_revealed(
    resolved: Sequence[bool],
    seen: AbstractSet[int],
) -> List[int]:
    """Positions resolved now and not resolved in any earlier frame.

    Pure. The caller owns ``seen`` and folds the result back in
    itself, so one canvas's history never leaks into the next and
    the sampler keeps a single place where that state changes.
    """
    fresh: List[int] = []
    for index, is_resolved in enumerate(resolved):
        if is_resolved and index not in seen:
            fresh.append(index)

    assert len(fresh) <= len(resolved), (
        "more reveals than positions"
    )
    for index in fresh:
        assert index not in seen, "a seen position leaked through"
    return fresh
