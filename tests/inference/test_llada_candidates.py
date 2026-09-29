"""A LLaDA run's candidates reach the page once, just before done.

Strategy: a stub model with seeded logits and a stub tokenizer let
``streaming_generate`` and ``streaming_resume`` run for real on CPU.
The assertions read the messages the browser would receive, and a
spy on the step function reads what the sampler asked it for.

Passing proves the sampler's half of the contract. Candidates arrive
once, immediately before done, covering frames 1 through the last
(frame 0 has no forward pass behind it). Each position's set marks
the token that frame shows, which is the guess for a masked position
and the token for a settled one. A run that did not ask gets no
message and asks the step for nothing; a resume captures at its own
frame indices; and a stopped run sends none, the stopping point the
ROADMAP records.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest
import torch

from src.inference import streaming_sampler
from src.inference.streaming_sampler import (
    MASK_ID,
    streaming_generate,
    streaming_resume,
)

VOCAB = 16
PROMPT_LEN = 3
GEN_LENGTH = 4
STEPS = 4


class _StubTokenizer:
    """Every id decodes to its own digits, so a token's text names
    it, and the chat template is the prompt itself."""

    def apply_chat_template(
        self, messages: List[Dict[str, str]], **_: Any
    ) -> str:
        return messages[0]["content"]

    def __call__(self, texts: List[str], **_: Any) -> Dict[str, Any]:
        ids = torch.arange(1, PROMPT_LEN + 1).unsqueeze(0)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join(str(int(i)) for i in ids)

    def batch_decode(
        self, tokens: torch.Tensor, skip_special_tokens: bool = False
    ) -> List[str]:
        return [" ".join(str(int(i)) for i in tokens[0])]


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _StubModel:
    """Different logits at every call, the same across runs."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, x: torch.Tensor, attention_mask: Any = None
    ) -> _Output:
        generator = torch.Generator().manual_seed(99 + self.calls)
        self.calls += 1
        logits = torch.randn(
            x.shape[0], x.shape[1], VOCAB, generator=generator
        )
        return _Output(logits)


def _collect(
    generator: Any,
    *,
    cancel: Optional[threading.Event] = None,
    stop_after: int = -1,
) -> List[Dict[str, Any]]:
    """Every message, setting ``cancel`` once the frame numbered
    ``stop_after`` arrives; -1 names no frame."""

    async def run() -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        async for message in generator:
            messages.append(message)
            if message.get("index") == stop_after:
                assert cancel is not None, "a stop needs an event"
                cancel.set()
        return messages

    return asyncio.run(run())


def _generate(
    alternatives: bool,
    cancel: Optional[threading.Event] = None,
    stop_after: int = -1,
) -> List[Dict[str, Any]]:
    generator = streaming_generate(
        _StubModel(),
        _StubTokenizer(),
        "hello",
        steps=STEPS,
        gen_length=GEN_LENGTH,
        block_length=GEN_LENGTH,
        alternatives=alternatives,
        cancel_event=cancel,
    )
    return _collect(generator, stop_after=stop_after, cancel=cancel)


def _resume(alternatives: bool) -> List[Dict[str, Any]]:
    generator = streaming_resume(
        _StubModel(),
        _StubTokenizer(),
        base_tokens=torch.tensor([[5, 6, MASK_ID, MASK_ID]]),
        base_conf=torch.tensor([0.5, 0.5, 0.0, 0.0]),
        base_rng=None,
        prompt_ids=torch.arange(1, PROMPT_LEN + 1).unsqueeze(0),
        attention_mask=torch.ones(
            (1, PROMPT_LEN + GEN_LENGTH), dtype=torch.long
        ),
        remask_positions=[1],
        remaining_steps=2,
        gen_length=GEN_LENGTH,
        alternatives=alternatives,
    )
    return _collect(generator)


def _of_type(
    messages: List[Dict[str, Any]], kind: str
) -> List[Dict[str, Any]]:
    return [m for m in messages if m["type"] == kind]


# -- delivery --


def test_candidates_arrive_once_just_before_done() -> None:
    messages = _generate(alternatives=True)
    closing = [m["type"] for m in messages[-2:]]

    assert closing == ["candidates", "done"]
    assert len(_of_type(messages, "candidates")) == 1


def test_candidates_cover_every_step_but_the_opening_frame() -> None:
    """Frame 0 is the all-masked canvas before any forward pass, so
    there is nothing it was weighing."""
    message = _of_type(_generate(alternatives=True), "candidates")[0]

    assert message["frames"] == list(range(1, STEPS + 1))
    assert message["stride"] == 1
    assert len(message["sets"]) == STEPS
    assert all(len(sets) == GEN_LENGTH for sets in message["sets"])


def test_a_run_without_alternatives_sends_none() -> None:
    assert _of_type(_generate(alternatives=False), "candidates") == []


def test_a_run_without_alternatives_asks_the_step_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost argument for the parameter: off means the step never
    walks the logits for candidates at all."""
    asked: List[int] = []
    real_step = streaming_sampler.diffusion_step

    def spy(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs["top_k"])
        return real_step(*args, **kwargs)

    monkeypatch.setattr(streaming_sampler, "diffusion_step", spy)
    _generate(alternatives=False)
    off = list(asked)
    asked.clear()
    _generate(alternatives=True)

    assert off == [0] * STEPS
    assert asked == [5] * STEPS


def test_a_stopped_run_sends_no_candidates() -> None:
    """The worker writes a stopped run's terminal frame, after the
    sampler has returned, so nothing is left to flush them."""
    cancel = threading.Event()
    messages = _generate(True, cancel=cancel, stop_after=1)

    assert _of_type(messages, "candidates") == []
    assert _of_type(messages, "done") == []


# -- what each set says --


def test_each_set_marks_the_token_its_frame_shows() -> None:
    """The marked row is the token on screen: a settled position's
    id, or the guess a masked position displays, which the stub
    tokenizer spells as the id's digits."""
    messages = _generate(alternatives=True)
    frames = {m["index"]: m for m in _of_type(messages, "frame")}
    message = _of_type(messages, "candidates")[0]

    for frame, sets in zip(
        message["frames"], message["sets"], strict=True
    ):
        for token, entry in zip(
            frames[frame]["tokens"], sets, strict=True
        ):
            shown = int(token["t"]) if token["m"] else token["id"]
            assert entry["h"] == shown


def test_candidate_text_is_the_raw_decode() -> None:
    message = _of_type(_generate(alternatives=True), "candidates")[0]

    for sets in message["sets"]:
        for entry in sets:
            for row in entry["c"]:
                assert row["t"] == str(row["id"])


def test_a_resume_captures_at_its_own_frame_indices() -> None:
    """The client places a resume's frames after the point it
    branched from, so the capture counts from the resume's frame 1."""
    messages = _resume(alternatives=True)
    message = _of_type(messages, "candidates")[0]

    assert message["frames"] == [1, 2]
    assert messages[-1]["type"] == "done"
    assert messages[-2] is message


def test_a_resume_without_alternatives_sends_none() -> None:
    assert _of_type(_resume(alternatives=False), "candidates") == []
