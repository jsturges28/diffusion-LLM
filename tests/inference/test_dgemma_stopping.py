"""Tests that DiffusionGemma's stopping rule reaches ``generate``.

Strategy: a stub model whose ``generate`` records the keyword
arguments it was called with, driven through both entry points the
worker calls, ``streaming_generate`` and ``streaming_resume``, on CPU
with no checkpoint. The rule used to live only in the checkpoint's
generation_config.json, out of sight; the page now draws a readout
against it, so the numbers it draws against have to be the numbers
the run was given.

Passing proves a run and a resume each hand ``generate`` the two
thresholds they were called with, that leaving them out passes the
checkpoint's own values rather than nothing, that the sampler's
defaults and the registry's cannot drift apart, and that a rule no
canvas could ever satisfy is refused before it reaches the model.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

from src.backends.registry import DGEMMA
from src.backends.text_adapter import DGEMMA_TEXT
from src.inference.checkpoint import DgemmaFrame, FrameCheckpoint
from src.inference.dgemma_sampler import (
    CONFIDENCE_THRESHOLD_DEFAULT,
    STABILITY_THRESHOLD_DEFAULT,
    stopping_kwargs,
    streaming_generate,
    streaming_resume,
)

CANVAS_LENGTH = 4
CANVAS_IDS = [11, 12, 13, 14]

assert len(CANVAS_IDS) == CANVAS_LENGTH, "the canvas is full"


class _StubTokenizer:
    def decode(
        self, ids: Any, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, torch.Tensor):
            values = ids.tolist()
        else:
            values = list(ids)
        return "".join(
            chr(ord("a") + (int(i) % 26)) for i in values
        )


class _RecordingModel:
    """Commits one canvas and keeps what ``generate`` was asked."""

    config = SimpleNamespace(
        canvas_length=CANVAS_LENGTH,
        text_config=SimpleNamespace(vocab_size=64),
    )

    def __init__(self) -> None:
        self.device = "cpu"
        self.asked: List[Dict[str, Any]] = []

    def generate(self, *, streamer: Any, **kwargs: Any) -> Any:
        self.asked.append(kwargs)
        canvas = torch.tensor([CANVAS_IDS], dtype=torch.long)
        streamer.put(torch.zeros((1, 2), dtype=torch.long))
        streamer.put_draft(value=canvas)
        streamer.put(canvas)
        streamer.end()
        return canvas


class _StubAdapter:
    """DiffusionGemma's text conventions over a fixed two-token
    prompt, so both entry points run without a chat template."""

    def build_inputs(
        self, tokenizer: Any, model: Any, prompt: str, *,
        thinking: bool,
    ) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.zeros((1, 2), dtype=torch.long)}

    def sanitize(self, text: str) -> str:
        return DGEMMA_TEXT.sanitize(text)

    def split_channels(self, raw: str) -> Tuple[str, str]:
        return DGEMMA_TEXT.split_channels(raw)


def _checkpoint() -> FrameCheckpoint:
    return FrameCheckpoint(
        ids=torch.tensor(CANVAS_IDS, dtype=torch.long),
        canvas_index=0,
        rng=None,
        extra=DgemmaFrame(
            seen_revealed=frozenset(range(CANVAS_LENGTH)),
        ),
    )


def _generate(model: _RecordingModel, **rule: Any) -> None:
    async def drain() -> None:
        async for _ in streaming_generate(
            model,
            _StubTokenizer(),
            _StubAdapter(),
            "a prompt",
            **rule,
        ):
            pass

    asyncio.run(drain())


def _resume(model: _RecordingModel, **rule: Any) -> None:
    async def drain() -> None:
        async for _ in streaming_resume(
            model,
            _StubTokenizer(),
            _StubAdapter(),
            prompt="a prompt",
            base=_checkpoint(),
            remask_positions=[1],
            remaining_steps=3,
            **rule,
        ):
            pass

    asyncio.run(drain())


def test_a_generation_hands_its_rule_to_generate() -> None:
    model = _RecordingModel()

    _generate(
        model, confidence_threshold=0.02, stability_threshold=2
    )

    assert len(model.asked) == 1
    assert model.asked[0]["confidence_threshold"] == 0.02
    assert model.asked[0]["stability_threshold"] == 2


def test_a_resume_hands_its_rule_to_generate() -> None:
    """The edit path too, or an edit would stop its canvas by the
    checkpoint's rule while the readout drew the run's."""
    model = _RecordingModel()

    _resume(
        model, confidence_threshold=0.03, stability_threshold=0
    )

    assert len(model.asked) == 1
    assert model.asked[0]["confidence_threshold"] == 0.03
    assert model.asked[0]["stability_threshold"] == 0


def test_a_rule_left_out_is_the_checkpoints_own() -> None:
    """Explicit even when defaulted, so the call never falls back on
    whatever the checkpoint's config happens to say."""
    for drive in (_generate, _resume):
        model = _RecordingModel()

        drive(model)

        assert model.asked[0]["confidence_threshold"] == 0.005
        assert model.asked[0]["stability_threshold"] == 1


def test_the_entropy_reaches_generate_as_a_float() -> None:
    """transformers refuses an int here, so an integer that slipped
    through would fail the run inside generate."""
    model = _RecordingModel()

    _generate(model, confidence_threshold=1, stability_threshold=1)

    assert isinstance(model.asked[0]["confidence_threshold"], float)


def test_the_sampler_and_registry_defaults_agree() -> None:
    specs = {spec.name: spec for spec in DGEMMA.param_specs}

    assert (
        specs["confidence_threshold"].default
        == CONFIDENCE_THRESHOLD_DEFAULT
    )
    assert (
        specs["stability_threshold"].default
        == STABILITY_THRESHOLD_DEFAULT
    )


@pytest.mark.parametrize(
    ("entropy", "steps"),
    [(0.0, 1), (-0.01, 1), (0.005, -1), (0.005, True)],
)
def test_an_unreachable_rule_is_refused(
    entropy: float, steps: Any
) -> None:
    """Zero entropy can never be undercut, a negative count means
    nothing, and a flag is not a count."""
    with pytest.raises(AssertionError):
        stopping_kwargs(
            confidence_threshold=entropy, stability_threshold=steps
        )


def test_the_boundary_rule_is_accepted() -> None:
    """The smallest legal values: any positive entropy, no steadiness
    required at all."""
    assert stopping_kwargs(
        confidence_threshold=0.0001, stability_threshold=0
    ) == {"confidence_threshold": 0.0001, "stability_threshold": 0}
