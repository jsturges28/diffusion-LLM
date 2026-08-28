"""Tests that a resumed LLaDA branch reports real confidence.

Strategy: a stub model returning fixed logits stands in for the
checkpoint, so ``streaming_resume`` runs for real on CPU with no
weights. The assertions read the protocol frames the browser would
receive, because the defect was visible there and nowhere else.

A resume used to hand every surviving token a confidence of 1.0.
The heatmap on an edited run therefore showed uniform certainty for
the whole inherited prefix, and its mean confidence was an average
over invented numbers, which is worse than a missing reading: it
looks like a measurement. Passing proves each survivor keeps the
probability it was actually revealed at, that a position the user
threw away does not keep the old one, and that two attempts at the
same edit agree even with a generation's worth of random draws in
between.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
import torch

from src.inference.checkpoint import rng_capture
from src.inference.streaming_sampler import (
    MASK_ID,
    resume_reveal_conf,
    streaming_resume,
)

GEN_LENGTH = 4
PROMPT_LEN = 2
VOCAB = 8

# The recorded frame every test branches from: three positions
# written, the last still masked, each with its own confidence so a
# survivor can be told apart from a fabricated 1.0.
BASE_IDS = [5, 6, 7, MASK_ID]
BASE_CONF = [0.2, 0.4, 0.9, 0.0]

assert len(BASE_IDS) == GEN_LENGTH, "the frame fills the canvas"
assert max(BASE_CONF) < 1.0, (
    "no recorded value may equal the fabricated one"
)


class _StubTokenizer:
    def batch_decode(
        self,
        tokens: torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> List[str]:
        return ["".join("x" for _ in tokens[0])]

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join(str(int(i)) for i in ids)


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _StubModel:
    """Returns the same logits every step, biased toward token 3."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def __call__(
        self, x: torch.Tensor, attention_mask: Any = None
    ) -> _Output:
        batch, seq = x.shape
        logits = torch.zeros(batch, seq, VOCAB)
        logits[:, :, 3] = 2.0
        return _Output(logits)


def _resume(
    *,
    remask_positions: List[int],
    remaining_steps: int = 1,
    temperature: float = 0.0,
    base_rng: Any = None,
) -> List[Dict[str, Any]]:
    async def collect() -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        generator = streaming_resume(
            _StubModel(),
            _StubTokenizer(),
            base_tokens=torch.tensor(
                [BASE_IDS], dtype=torch.long
            ),
            base_conf=torch.tensor(
                BASE_CONF, dtype=torch.float
            ),
            base_rng=base_rng,
            prompt_ids=torch.zeros(
                (1, PROMPT_LEN), dtype=torch.long
            ),
            attention_mask=torch.ones(
                (1, PROMPT_LEN + GEN_LENGTH), dtype=torch.long
            ),
            remask_positions=remask_positions,
            remaining_steps=remaining_steps,
            gen_length=GEN_LENGTH,
            temperature=temperature,
        )
        async for item in generator:
            frames.append(item)
        return frames

    return asyncio.run(collect())


def _conf_at(frame: Dict[str, Any], index: int) -> Any:
    return frame["tokens"][index].get("c")


# -- the pure helper --


def test_a_survivor_keeps_the_value_it_was_revealed_at() -> None:
    conf = resume_reveal_conf(
        torch.tensor(BASE_CONF), [1], GEN_LENGTH
    )
    assert float(conf[0]) == pytest.approx(0.2)
    assert float(conf[2]) == pytest.approx(0.9)


def test_a_remasked_position_keeps_nothing() -> None:
    conf = resume_reveal_conf(
        torch.tensor(BASE_CONF), [2], GEN_LENGTH
    )
    assert float(conf[2]) == 0.0


def test_the_recorded_confidence_is_not_written_through() -> None:
    """The checkpoint outlives the resume, and a failed attempt has
    to leave it resumable again."""
    base = torch.tensor(BASE_CONF)
    _ = resume_reveal_conf(base, [0, 1], GEN_LENGTH)
    assert base.tolist() == pytest.approx(BASE_CONF)


# -- what the browser receives --


def test_the_first_resumed_frame_carries_real_confidence() -> None:
    frames = _resume(remask_positions=[1])
    first = frames[0]

    assert _conf_at(first, 0) == 0.2
    assert _conf_at(first, 2) == 0.9


def test_a_survivor_is_never_reported_as_certain() -> None:
    """The specific fabrication this replaced."""
    frames = _resume(remask_positions=[1])
    for index in (0, 2):
        assert _conf_at(frames[0], index) != 1.0


def test_a_remasked_position_reports_no_confidence() -> None:
    frames = _resume(remask_positions=[1])
    token = frames[0]["tokens"][1]

    assert token["m"] is True
    assert "c" not in token


def test_the_mean_averages_measurements_not_inventions() -> None:
    """0.55 is the mean of the two survivors. Before this it was
    1.0, because every survivor had been assigned that value."""
    frames = _resume(remask_positions=[1])

    assert frames[0]["mean_conf"] == 0.55


# -- reproducibility --


def test_one_edit_repeats_across_intervening_random_work() -> None:
    """The audit's own verification: resume the same frame twice
    with random work in between and compare frame by frame."""
    torch.manual_seed(11)
    state = rng_capture()

    first = _resume(
        remask_positions=[1, 3],
        remaining_steps=2,
        temperature=1.0,
        base_rng=state,
    )
    _ = torch.rand(4096)
    torch.manual_seed(404)
    second = _resume(
        remask_positions=[1, 3],
        remaining_steps=2,
        temperature=1.0,
        base_rng=state,
    )

    assert len(first) > 1, "the comparison needs frames to compare"
    assert first == second


def test_a_different_state_reaches_a_different_branch() -> None:
    """Guards the test above from going vacuous.

    If the stub ever stopped depending on the generator, two
    resumes would match for reasons having nothing to do with the
    checkpoint, and the reproducibility claim would pass while
    proving nothing.
    """
    torch.manual_seed(11)
    one = rng_capture()
    torch.manual_seed(9999)
    other = rng_capture()

    assert _resume(
        remask_positions=[1, 3],
        remaining_steps=2,
        temperature=1.0,
        base_rng=one,
    ) != _resume(
        remask_positions=[1, 3],
        remaining_steps=2,
        temperature=1.0,
        base_rng=other,
    )


def test_without_a_checkpoint_the_run_still_resumes() -> None:
    """A frame past the budget has no random state. It re-enters on
    whatever the process holds, which is what every resume did
    before checkpoints existed, rather than refusing to run."""
    frames = _resume(
        remask_positions=[1], remaining_steps=1, base_rng=None
    )

    assert len(frames) > 0
    assert frames[0]["tokens"][0]["c"] == 0.2
