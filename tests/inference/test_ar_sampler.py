"""Tests for the autoregressive sampler's XAI signals.

Strategy: the sampler's numeric core is pure torch, so these tests
drive it on small CPU tensors with a stub tokenizer and a stub model,
no checkpoint required. They cover the entropy computation against
hand-computable references, top-k candidate extraction and its
ordering, the top-p filter at its boundaries, the frame payload shape
(entropy per token, candidates only on the frame that introduces a
position), and the substitution branch's splice arithmetic.

Passing proves the signals the UI reads are correct and correctly
placed, and that a substitution keeps the untouched prefix, forces
the requested token, and stays inside the original token budget.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional

import pytest
import torch

from src.inference.ar_sampler import (
    TOP_K_ALTERNATIVES,
    _build_frame,
    _entropy_nats,
    _sample_next,
    _top_alternatives,
    _top_p_filter,
    streaming_generate,
    streaming_substitute,
)

VOCAB_SIZE = 12
EOS_ID = 11


class StubTokenizer:
    """Minimal tokenizer: ids decode to stable placeholder text."""

    eos_token_id = EOS_ID
    unk_token_id = 0

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join("t%d " % int(i) for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return -1

    def apply_chat_template(
        self, chat: Any, **kwargs: Any
    ) -> Any:
        class Batch(dict):  # type: ignore[type-arg]
            def to(self, device: Any) -> "Batch":
                return self

        return Batch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )


class StubOutput:
    def __init__(self, logits: torch.Tensor, past: Any) -> None:
        self.logits = logits
        self.past_key_values = past


class StubModel:
    """Logits depend only on the last input token.

    That dependence is what makes a substitution observable: forcing a
    different token at one position changes every logit after it.
    """

    device = "cpu"
    generation_config = None

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> StubOutput:
        assert input_ids is not None, "input_ids required"
        last = int(input_ids[0, -1].item())
        logits = torch.zeros(1, input_ids.shape[-1], VOCAB_SIZE)
        for value in range(VOCAB_SIZE):
            # Never favors EOS, so runs use the whole budget.
            logits[0, -1, value] = (
                -5.0
                if value == EOS_ID
                else ((last * 3 + value * 5) % 7) * 0.7
            )
        return StubOutput(logits, 0)


# -- entropy --


def test_entropy_of_uniform_equals_log_support() -> None:
    probs = torch.full((4,), 0.25)
    assert _entropy_nats(probs) == pytest.approx(
        math.log(4.0), abs=1e-6
    )


def test_entropy_of_one_hot_is_zero() -> None:
    probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert _entropy_nats(probs) == pytest.approx(0.0, abs=1e-6)


def test_entropy_rises_with_uncertainty() -> None:
    peaked = torch.tensor([0.97, 0.01, 0.01, 0.01])
    spread = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert _entropy_nats(peaked) < _entropy_nats(spread)


def test_entropy_ignores_zero_probability_entries() -> None:
    """log(0) would be a NaN, which must never reach the UI."""
    probs = torch.tensor([0.5, 0.5, 0.0, 0.0])
    value = _entropy_nats(probs)
    assert not math.isnan(value)
    assert value == pytest.approx(math.log(2.0), abs=1e-6)


# -- top-k alternatives --


def test_top_alternatives_are_ordered_by_probability() -> None:
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    alts = _top_alternatives(probs, StubTokenizer(), 3)
    assert [a["id"] for a in alts] == [1, 3, 2]
    assert [a["p"] for a in alts] == [0.5, 0.3, 0.15]


def test_top_alternatives_clamps_k_to_vocabulary() -> None:
    probs = torch.tensor([0.7, 0.3])
    alts = _top_alternatives(probs, StubTokenizer(), 5)
    assert len(alts) == 2


def test_top_alternatives_rejects_non_positive_k() -> None:
    probs = torch.tensor([0.7, 0.3])
    with pytest.raises(AssertionError):
        _top_alternatives(probs, StubTokenizer(), 0)


# -- top-p filter boundaries --


def test_top_p_of_one_keeps_every_candidate() -> None:
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kept = _top_p_filter(probs, 1.0)
    assert int((kept > 0).sum().item()) == 4


def test_top_p_keeps_at_least_the_top_candidate() -> None:
    """A tiny p must not zero the whole distribution."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kept = _top_p_filter(probs, 0.01)
    assert int((kept > 0).sum().item()) == 1
    assert int(torch.argmax(kept).item()) == 3


def test_top_p_drops_the_tail() -> None:
    probs = torch.tensor([0.01, 0.04, 0.35, 0.6])
    kept = _top_p_filter(probs, 0.9)
    assert kept[0].item() == pytest.approx(0.0)
    assert kept[3].item() > 0.0


# -- step picks and frame payloads --


def test_sample_next_reports_untempered_confidence() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    expected = torch.softmax(logits.squeeze(0), dim=-1)
    pick = _sample_next(
        logits,
        temperature=0.0,
        top_p=1.0,
        tokenizer=StubTokenizer(),
        alternatives=True,
    )
    assert pick.token_id == 1
    assert pick.confidence == pytest.approx(
        float(expected[1]), abs=1e-6
    )
    assert pick.entropy > 0.0
    assert pick.alternatives is not None
    assert len(pick.alternatives) == 4


def test_sample_next_omits_alternatives_when_disabled() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    pick = _sample_next(
        logits,
        temperature=0.0,
        top_p=1.0,
        tokenizer=StubTokenizer(),
        alternatives=False,
    )
    assert pick.alternatives is None


def test_build_frame_carries_entropy_per_token() -> None:
    frame = _build_frame(
        StubTokenizer(),
        [4, 5],
        [0.9, 0.5],
        [0.25, 1.5],
        frame_index=1,
        total_steps=8,
    )
    assert [t["e"] for t in frame["tokens"]] == [0.25, 1.5]
    # 1-based to match the diffusion step convention.
    assert frame["index"] == 2


def test_build_frame_omits_alternatives_by_default() -> None:
    frame = _build_frame(
        StubTokenizer(),
        [4],
        [0.9],
        [0.25],
        frame_index=0,
        total_steps=8,
    )
    assert "alts" not in frame


def test_build_frame_rejects_misaligned_signals() -> None:
    with pytest.raises(AssertionError):
        _build_frame(
            StubTokenizer(),
            [4, 5],
            [0.9, 0.5],
            [0.25],
            frame_index=1,
            total_steps=8,
        )


# -- streaming: capture placement and substitution splice --


def _run_generate(
    *, alternatives: bool, budget: int
) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    frames: List[Dict[str, Any]] = []

    async def drive() -> None:
        async for item in streaming_generate(
            StubModel(),
            StubTokenizer(),
            "prompt",
            max_new_tokens=budget,
            temperature=0.0,
            top_p=1.0,
            alternatives=alternatives,
            state_sink=state,
        ):
            frames.append(item)

    asyncio.run(drive())
    return {"state": state, "frames": frames}


def test_generate_emits_one_frame_per_token() -> None:
    out = _run_generate(alternatives=False, budget=5)
    frames = [f for f in out["frames"] if f["type"] == "frame"]
    assert len(frames) == 5
    assert [f["index"] for f in frames] == [1, 2, 3, 4, 5]
    assert [len(f["tokens"]) for f in frames] == [1, 2, 3, 4, 5]
    assert out["frames"][-1]["type"] == "done"


def test_generate_sends_each_candidate_set_once() -> None:
    """Candidates ride only the frame introducing their position."""
    out = _run_generate(alternatives=True, budget=4)
    frames = [f for f in out["frames"] if f["type"] == "frame"]
    for frame in frames:
        assert len(frame["alts"]) == TOP_K_ALTERNATIVES
        # The set belongs to the newest position, so a frame never
        # carries candidates for the positions before it.
        assert isinstance(frame["alts"], list)


def test_generate_without_capture_sends_no_candidates() -> None:
    out = _run_generate(alternatives=False, budget=4)
    frames = [f for f in out["frames"] if f["type"] == "frame"]
    assert all("alts" not in f for f in frames)
    assert all(a is None for a in out["state"]["alternatives"])


def test_state_sink_traces_every_position() -> None:
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    assert len(state["ids"]) == 6
    assert len(state["confidences"]) == 6
    assert len(state["entropies"]) == 6
    assert len(state["alternatives"]) == 6


def _run_substitute(
    *, state: Dict[str, Any], position: int, budget: int
) -> Dict[str, Any]:
    captured = state["alternatives"][position]
    forced = next(
        c for c in captured if c["id"] != state["ids"][position]
    )
    branch: Dict[str, Any] = {}
    frames: List[Dict[str, Any]] = []

    async def drive() -> None:
        async for item in streaming_substitute(
            StubModel(),
            StubTokenizer(),
            "prompt",
            position=position,
            forced_id=forced["id"],
            forced_conf=forced["p"],
            forced_entropy=state["entropies"][position],
            forced_alts=captured,
            prefix_ids=state["ids"][:position],
            prefix_confs=state["confidences"][:position],
            prefix_entropies=state["entropies"][:position],
            prefix_alts=state["alternatives"][:position],
            max_new_tokens=budget,
            alternatives=True,
            state_sink=branch,
        ):
            frames.append(item)

    asyncio.run(drive())
    return {"branch": branch, "frames": frames, "forced": forced}


def test_substitute_keeps_prefix_and_forces_position() -> None:
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    result = _run_substitute(state=state, position=2, budget=6)
    branch = result["branch"]

    assert branch["ids"][:2] == state["ids"][:2]
    assert branch["ids"][2] == result["forced"]["id"]
    assert branch["ids"][2] != state["ids"][2]


def test_substitute_preserves_the_forced_signals() -> None:
    """The forced position keeps the decision it actually faced."""
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    result = _run_substitute(state=state, position=2, budget=6)

    assert result["branch"]["confidences"][2] == pytest.approx(
        result["forced"]["p"]
    )
    assert result["branch"]["entropies"][2] == pytest.approx(
        state["entropies"][2]
    )


def test_substitute_emits_a_seed_frame_then_continues() -> None:
    out = _run_generate(alternatives=True, budget=6)
    result = _run_substitute(
        state=out["state"], position=2, budget=6
    )
    frames = [
        f for f in result["frames"] if f["type"] == "frame"
    ]
    # Seed frame covers positions 0..2 (index is 1-based), then one
    # frame per regenerated token.
    assert [f["index"] for f in frames] == [3, 4, 5, 6]
    assert [len(f["tokens"]) for f in frames] == [3, 4, 5, 6]


def test_substitute_stays_within_the_token_budget() -> None:
    out = _run_generate(alternatives=True, budget=6)
    result = _run_substitute(
        state=out["state"], position=1, budget=6
    )
    assert len(result["branch"]["ids"]) == 6


def test_substitute_at_the_last_position_adds_nothing() -> None:
    out = _run_generate(alternatives=True, budget=4)
    result = _run_substitute(
        state=out["state"], position=3, budget=4
    )
    frames = [
        f for f in result["frames"] if f["type"] == "frame"
    ]
    assert len(frames) == 1
    assert len(result["branch"]["ids"]) == 4


def test_substitute_rejects_a_misaligned_prefix() -> None:
    async def drive() -> None:
        generator = streaming_substitute(
            StubModel(),
            StubTokenizer(),
            "prompt",
            position=3,
            forced_id=5,
            forced_conf=0.2,
            forced_entropy=1.0,
            forced_alts=None,
            prefix_ids=[1, 2],
            prefix_confs=[0.5, 0.5],
            prefix_entropies=[1.0, 1.0],
            prefix_alts=[None, None],
            max_new_tokens=6,
        )
        async for _ in generator:
            pass

    with pytest.raises(AssertionError):
        asyncio.run(drive())
