"""The candidates a diffusion run records, held to a budget.

A diffusion position is re-decided at every denoising step, so what
the model was weighing there is a trajectory, one set of candidates
per step, where an autoregressive position has one set for good. That
is what makes a diffusion capture large: frames times positions times
five, which is 4 MiB for a default LLaDA run and 212 at the largest
settings the registry allows (the ROADMAP measured 42 bytes a record).

So the capture holds to `CANDIDATE_BUDGET_RECORDS`. It keeps every
step until the next one would pass the budget, then drops every other
step it kept and doubles its stride, and carries on at that stride.
The kept steps are therefore always evenly spaced, and the policy
needs no advance knowledge of how many steps a run will take, which
DiffusionGemma cannot give: it stops a canvas when the canvas settles,
and chains as many canvases as the output needs. The final step is
kept whatever the stride, since the finished canvas is the frame a
reader lands on first. A default LLaDA run is exactly the budget, so
it is captured at every step.

The record is decoded only at the end. Steps are held as small
tensors of ids and probabilities, and `flush` decodes each distinct
id once, which is also where the wire message is built: the whole
capture reaches the page once, when the run ends.

One narrow slice travels sooner. DiffusionGemma's frames arrive about
a second apart, so as each leaves for the page it carries the sets of
the positions that changed on it (`position_sets`), decoded, for the
page to cycle until the next frame lands: a median of about 10 KiB a
frame on saved runs. LLaDA's frames arrive about twenty-five a
second, too quickly for a cycle to read, and carry none.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch

from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    CANDIDATES_PER_POSITION,
)
from src.inference.logit_signals import Candidates

MESSAGE_TYPE = "candidates"
# The likeliest candidates round as the autoregressive popover's do.
# The held token's own row, appended only when it falls outside them,
# is exact, because that is the row whose probability rounds to zero.
PROBABILITY_PLACES = 4


class StepCandidates(NamedTuple):
    """One step's reading, on the CPU."""

    frame: int
    held: torch.Tensor  # (positions,), the token each position holds
    candidates: Candidates

    @property
    def positions(self) -> int:
        return int(self.held.shape[0])


class _Kept(NamedTuple):
    ordinal: int  # which step this was, counting only steps offered
    step: StepCandidates


class CandidateCapture:
    """Every step's candidates until the budget binds, then a stride.

    Steps are offered in frame order. Records are counted as positions
    times ``k`` per step, the unit the budget is written in.
    """

    def __init__(
        self,
        *,
        k: int = CANDIDATES_PER_POSITION,
        budget_records: int = CANDIDATE_BUDGET_RECORDS,
    ) -> None:
        assert k > 0, "at least one candidate per position"
        assert budget_records >= k, "room for one position at least"
        self.k = k
        self.budget_records = budget_records
        self.stride = 1
        self._kept: List[_Kept] = []
        self._latest: Optional[_Kept] = None
        self._offered = 0

    def offer(self, step: StepCandidates) -> None:
        """Consider one step, keeping it if the stride lands on it."""
        latest = self._latest
        if latest is not None:
            assert step.frame > latest.step.frame, "frames in order"
        width = step.candidates.ids.shape[1]
        assert width == self.k, "k candidates per position"
        cost = step.positions * self.k
        assert cost <= self.budget_records, "one step fits the budget"
        entry = _Kept(self._offered, step)
        self._offered += 1
        self._latest = entry
        if entry.ordinal % self.stride != 0:
            return
        while self.records() + cost > self.budget_records:
            self._decimate()
            if entry.ordinal % self.stride != 0:
                return
        self._kept.append(entry)

    def records(self) -> int:
        """Records the kept steps hold, in the budget's unit."""
        positions = sum(kept.step.positions for kept in self._kept)
        return positions * self.k

    def frames(self) -> List[int]:
        """The frames kept so far, in order."""
        return [kept.step.frame for kept in self._kept]

    def finish(self) -> None:
        """Keep the final step if the stride skipped it.

        When there is no room it takes the place of the last step
        kept, rather than halving the whole sample for one frame, so
        the spacing breaks only at the very end.
        """
        latest = self._latest
        if latest is None:
            return
        if self._kept and self._kept[-1].ordinal == latest.ordinal:
            return
        cost = latest.step.positions * self.k
        while self.records() + cost > self.budget_records:
            self._kept.pop()
        self._kept.append(latest)
        assert self.records() <= self.budget_records, "within budget"

    def flush(
        self, decode: Callable[[int], str]
    ) -> Optional[Dict[str, Any]]:
        """The wire message for the whole run, or None when no step
        was offered. ``decode`` turns an id into the text a reader
        sees; it is called once per distinct id."""
        self.finish()
        if not self._kept:
            return None
        texts = _decode_once(self._kept, decode)
        sets = [_step_sets(kept.step, texts) for kept in self._kept]
        return {
            "type": MESSAGE_TYPE,
            "k": self.k,
            "stride": self.stride,
            "frames": self.frames(),
            "sets": sets,
        }

    def _decimate(self) -> None:
        self.stride *= 2
        self._kept = [
            kept for kept in self._kept
            if kept.ordinal % self.stride == 0
        ]


def step_candidates(
    frame: int, held: torch.Tensor, candidates: Candidates
) -> StepCandidates:
    """A step's reading moved to the CPU, where the capture keeps it:
    a few kilobytes a step, against a card that needs the room."""
    assert frame >= 0, "frame indices start at zero"
    assert held.dim() == 1, "one held token per position"
    moved = Candidates(
        *(part.detach().to("cpu") for part in candidates)
    )
    assert moved.ids.shape[0] == held.shape[0], "a set per position"
    return StepCandidates(frame, held.detach().to("cpu"), moved)


def _decode_once(
    kept: List[_Kept], decode: Callable[[int], str]
) -> Dict[int, str]:
    distinct: set[int] = set()
    for entry in kept:
        distinct.update(entry.step.candidates.ids.flatten().tolist())
        distinct.update(entry.step.held.tolist())
    return {token: decode(token) for token in sorted(distinct)}


def position_sets(
    step: StepCandidates,
    positions: List[int],
    decode: Callable[[int], str],
) -> Dict[str, Any]:
    """The sets of `positions` in one step, as a frame carries them
    for the page to cycle while the run streams.

    The rows are the end-of-run message's, for those positions alone,
    so a position cycles live through exactly what its popover shows
    once the run has ended. ``decode`` is called once per distinct id
    here; a caller decoding every frame passes one that caches.
    """
    for position in positions:
        assert 0 <= position < step.positions, "a position in range"
    assert len(set(positions)) == len(positions), "each position once"
    index = torch.tensor(positions, dtype=torch.long)
    ids = step.candidates.ids[index].tolist()
    probs = step.candidates.probs[index].tolist()
    held = step.held[index].tolist()
    held_probs = step.candidates.held_probs[index].tolist()
    held_ranks = step.candidates.held_ranks[index].tolist()
    distinct = set(held)
    for row in ids:
        distinct.update(row)
    texts = {token: decode(token) for token in sorted(distinct)}
    sets: List[Dict[str, Any]] = []
    for at, token in enumerate(held):
        sets.append(_position_set(
            token=token,
            ids=ids[at],
            probs=probs[at],
            held_prob=held_probs[at],
            held_rank=held_ranks[at],
            texts=texts,
        ))
    return {"positions": list(positions), "sets": sets}


def _step_sets(
    step: StepCandidates, texts: Dict[int, str]
) -> List[Dict[str, Any]]:
    """Every position's set, in canvas order."""
    ids = step.candidates.ids.tolist()
    probs = step.candidates.probs.tolist()
    held = step.held.tolist()
    held_probs = step.candidates.held_probs.tolist()
    held_ranks = step.candidates.held_ranks.tolist()
    sets: List[Dict[str, Any]] = []
    for position, token in enumerate(held):
        sets.append(_position_set(
            token=token,
            ids=ids[position],
            probs=probs[position],
            held_prob=held_probs[position],
            held_rank=held_ranks[position],
            texts=texts,
        ))
    return sets


def _position_set(
    *,
    token: int,
    ids: List[int],
    probs: List[float],
    held_prob: float,
    held_rank: int,
    texts: Dict[int, str],
) -> Dict[str, Any]:
    """One position's set: the token it held, and the candidates,
    with the held token appended with its rank when they omit it."""
    rows: List[Dict[str, Any]] = []
    for candidate, probability in zip(ids, probs, strict=True):
        rows.append({
            "id": candidate,
            "t": texts[candidate],
            "p": round(probability, PROBABILITY_PLACES),
        })
    if token not in ids:
        rows.append({
            "id": token,
            "t": texts[token],
            "p": held_prob,
            "rank": held_rank,
        })
    return {"h": token, "c": rows}
