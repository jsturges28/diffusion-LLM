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

from src.inference import ar_sampler
from src.inference.ar_sampler import (
    AR_CACHE_BYTES_MAX,
    TOP_K_ALTERNATIVES,
    _build_frame,
    _cache_record,
    _candidates_hold,
    _entropy_nats,
    _reuse_cache,
    _sample_next,
    _sliced_cache,
    _token_rank,
    _top_alternatives,
    _top_k_filter,
    _top_p_filter,
    probe_token,
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
    different token at one position changes every logit after it. It
    also means a prefill and a decode ending on the same token agree
    exactly, so a test can check *which* call shape was used without
    the answer changing underneath it. A real model in bf16 is the
    opposite on both counts, which is the whole reason the cache is
    worth retaining.

    ``calls`` records the width of each pass and whether it was handed
    a cache, which is how the cached path is observed at all.
    """

    device = "cpu"
    generation_config = None

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> StubOutput:
        assert input_ids is not None, "input_ids required"
        self.calls.append(
            {
                "width": int(input_ids.shape[-1]),
                "cached": past_key_values is not None,
            }
        )
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


class StubCache:
    """A DynamicCache in miniature: the legacy tuple format, and a
    from_legacy_cache that builds a new object rather than mutating
    one. That asymmetry is the property under test, since transformers
    implements ``crop`` in place and the retained cache has to survive
    a probe."""

    def __init__(self, layers: Any) -> None:
        self.layers = tuple(layers)

    @classmethod
    def from_legacy_cache(cls, layers: Any) -> "StubCache":
        return cls(layers)

    def to_legacy_cache(self) -> Any:
        return self.layers


def _stub_cache(length: int, layers: int = 2) -> StubCache:
    return StubCache(
        tuple(
            (
                torch.zeros(1, 2, length, 4),
                torch.zeros(1, 2, length, 4),
            )
            for _ in range(layers)
        )
    )


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


# -- rank and the chosen entry --


def test_rank_of_the_likeliest_is_one() -> None:
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    assert _token_rank(probs, 0.5) == 1


def test_rank_of_the_least_likely_is_the_vocabulary() -> None:
    """The boundary at the far end, where nothing is behind it."""
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    assert _token_rank(probs, 0.05) == 4


def test_tied_tokens_share_the_better_rank() -> None:
    """Two tokens at one probability are genuinely not ordered, so
    neither is demoted for it. The count is strictly greater."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert _token_rank(probs, 0.25) == 1


def test_candidates_hold_finds_a_listed_token() -> None:
    captured = [{"id": 3, "p": 0.5}, {"id": 7, "p": 0.2}]
    assert _candidates_hold(captured, 7) is True
    assert _candidates_hold(captured, 4) is False


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


def test_top_k_keeps_exactly_k_and_renormalizes() -> None:
    probs = torch.tensor([0.05, 0.15, 0.3, 0.5])
    kept = _top_k_filter(probs, 2)
    assert int((kept > 0).sum().item()) == 2
    assert kept[0].item() == pytest.approx(0.0)
    assert kept[1].item() == pytest.approx(0.0)
    assert float(kept.sum().item()) == pytest.approx(1.0)


def test_top_k_off_changes_nothing() -> None:
    """Both spellings of "off" disable it.

    -1 is the UI's default. 0 was the default before that and is
    still what older saved runs carry, so it has to keep meaning
    the same thing or replaying one would silently truncate.
    """
    probs = torch.tensor([0.05, 0.15, 0.3, 0.5])
    assert torch.equal(_top_k_filter(probs, -1), probs)
    assert torch.equal(_top_k_filter(probs, 0), probs)


def test_top_k_wider_than_the_vocabulary_changes_nothing() -> None:
    """The boundary on the other side: k at or past the vocabulary."""
    probs = torch.tensor([0.05, 0.15, 0.3, 0.5])
    assert torch.equal(_top_k_filter(probs, 4), probs)
    assert torch.equal(_top_k_filter(probs, 99), probs)


def test_top_k_runs_before_top_p() -> None:
    """Order matters, and it follows Hugging Face's.

    Top-k renormalizes over the k it kept, so a nucleus applied
    afterwards is measured against that inflated distribution and
    bites harder. Here k=3 drops 0.1 and rescales the rest, after
    which a 0.75 nucleus admits only two. Reversing the filters
    leaves three, so this pins the order down rather than merely
    exercising both functions.
    """
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

    composed = _top_p_filter(_top_k_filter(probs, 3), 0.75)
    assert int((composed > 0).sum().item()) == 2
    assert composed[3].item() == pytest.approx(
        0.5714, abs=1e-4
    )
    assert composed[2].item() == pytest.approx(
        0.4286, abs=1e-4
    )

    reversed_order = _top_k_filter(
        _top_p_filter(probs, 0.75), 3
    )
    assert int((reversed_order > 0).sum().item()) == 3


# -- step picks and frame payloads --


def test_sample_next_reports_untempered_confidence() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    expected = torch.softmax(logits.squeeze(0), dim=-1)
    pick = _sample_next(
        logits,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
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


def test_sample_next_leaves_a_pick_inside_the_set_alone() -> None:
    """The common case, and the one that must not grow a row: greedy
    takes the likeliest token, which the captured set always lists."""
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    pick = _sample_next(
        logits,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        tokenizer=StubTokenizer(),
        alternatives=True,
    )
    assert pick.alternatives is not None
    assert len(pick.alternatives) == 4
    assert all("rank" not in a for a in pick.alternatives)


def test_sample_next_appends_a_pick_from_outside_the_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm temperature can reach past the captured candidates, and
    the popover then has a chosen token it cannot explain.

    The draw is pinned rather than seeded: every filter the sampler
    applies is order-preserving, so there is no distribution that
    makes a low-ranked token certain, and a seeded multinomial would
    be a fact about this torch build rather than about the sampler.
    """
    monkeypatch.setattr(ar_sampler, "TOP_K_ALTERNATIVES", 2)
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, count: torch.tensor([0]),
    )
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    expected = torch.softmax(logits.squeeze(0), dim=-1)

    pick = _sample_next(
        logits,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        tokenizer=StubTokenizer(),
        alternatives=True,
    )

    assert pick.token_id == 0
    assert pick.alternatives is not None
    # Appended, not substituted for the second: the captured set is a
    # statement about what the model preferred, and dropping a member
    # of it to make room would be a different statement.
    assert len(pick.alternatives) == 3
    assert [a["id"] for a in pick.alternatives[:2]] == [1, 3]
    chosen = pick.alternatives[2]
    assert chosen["id"] == 0
    assert chosen["rank"] == 4
    assert chosen["p"] == pytest.approx(
        float(expected[0]), abs=1e-9
    )


def test_the_appended_probability_is_not_rounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This entry is routinely the one that rounds away: four places
    floor at 0.0001 and a wild pick sits below that."""
    monkeypatch.setattr(ar_sampler, "TOP_K_ALTERNATIVES", 2)
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, count: torch.tensor([0]),
    )

    pick = _sample_next(
        torch.tensor([[0.0, 5.0, 1.0, 2.0]]),
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        tokenizer=StubTokenizer(),
        alternatives=True,
    )

    assert pick.alternatives is not None
    appended = pick.alternatives[2]["p"]
    assert appended != round(appended, 4)


def test_sample_next_omits_alternatives_when_disabled() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    pick = _sample_next(
        logits,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
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


def test_build_frame_reveals_only_the_newest_position() -> None:
    # Decoding is left to right, so growing the sequence to n tokens
    # births position n-1 and cannot disturb anything before it.
    frame = _build_frame(
        StubTokenizer(),
        [4, 5, 6],
        [0.9, 0.5, 0.7],
        [0.25, 1.5, 0.8],
        frame_index=2,
        total_steps=8,
    )
    assert frame["revealed"] == [2]


def test_build_frame_reveals_nothing_on_an_empty_sequence() -> None:
    frame = _build_frame(
        StubTokenizer(),
        [],
        [],
        [],
        frame_index=0,
        total_steps=8,
    )
    assert frame["revealed"] == []


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


def test_the_done_frame_reports_the_prompts_length() -> None:
    """The templated length this run built, which is what a saved run
    records. Measured here rather than counted by the client, so the
    figure cannot describe a prompt that was edited after the run."""
    out = _run_generate(alternatives=False, budget=3)
    done = out["frames"][-1]
    assert done["type"] == "done"
    # The stub templates every prompt to three ids.
    assert done["prompt_len"] == 3


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
    *,
    state: Dict[str, Any],
    position: int,
    budget: int,
    typed: bool = False,
    forced_id: Optional[int] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    captured = state["alternatives"][position]
    if forced_id is None:
        forced = next(
            c for c in captured if c["id"] != state["ids"][position]
        )
    else:
        # A token from outside the captured set, which is what a
        # typed entry usually is. No recorded probability exists for
        # it, so it takes the typed path by construction.
        assert not any(c["id"] == forced_id for c in captured)
        forced = {"id": forced_id, "p": None}
        typed = True
    branch: Dict[str, Any] = {}
    frames: List[Dict[str, Any]] = []
    model = StubModel()

    async def drive() -> None:
        async for item in streaming_substitute(
            model,
            StubTokenizer(),
            "prompt",
            position=position,
            forced_id=forced["id"],
            # None is the typed case: no recorded probability, so
            # the sampler has to measure the real one.
            forced_conf=None if typed else forced["p"],
            forced_entropy=state["entropies"][position],
            forced_alts=captured,
            prefix_ids=state["ids"][:position],
            prefix_confs=state["confidences"][:position],
            prefix_entropies=state["entropies"][:position],
            prefix_alts=state["alternatives"][:position],
            max_new_tokens=budget,
            alternatives=True,
            state_sink=branch,
            cache=cache,
        ):
            frames.append(item)

    asyncio.run(drive())
    return {
        "branch": branch,
        "frames": frames,
        "forced": forced,
        # The pass that reads the forced position's distribution, and
        # the one the retained cache is supposed to shorten.
        "first_call": model.calls[0] if model.calls else None,
    }


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


def test_typed_substitution_measures_the_real_probability() -> None:
    """A typed token reports what the model actually gave it.

    Checked against a token that *was* captured, so the recorded
    probability is available as an independent oracle: the measured
    value has to reproduce it. That is what proves the probe reads
    the distribution at the forced position rather than at the one
    before or after it, which is the mistake this arrangement of the
    prefill exists to avoid. The recorded value is rounded to four
    places, hence the tolerance.
    """
    out = _run_generate(alternatives=True, budget=6)
    result = _run_substitute(
        state=out["state"], position=2, budget=6, typed=True
    )

    measured = result["branch"]["confidences"][2]
    assert measured == pytest.approx(
        result["forced"]["p"], abs=1e-4
    )
    assert 0.0 < measured <= 1.0


def test_a_typed_token_joins_the_forced_position() -> None:
    """A forced token the model never offered still has to appear in
    that position's candidates, or the branch reads as a run whose
    chosen token is absent from its own popover.

    The rank comes from the probe rather than from the row's place in
    the list, which is the whole reason it is stored: this entry is
    sixth in the list and further back than that in the distribution.
    """
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    captured = state["alternatives"][2]
    outsider = next(
        i
        for i in range(VOCAB_SIZE)
        if not any(c["id"] == i for c in captured)
    )

    result = _run_substitute(
        state=state, position=2, budget=6, forced_id=outsider
    )
    alts = result["branch"]["alternatives"][2]

    assert len(alts) == len(captured) + 1
    assert [a["id"] for a in alts[:-1]] == [
        c["id"] for c in captured
    ]
    appended = alts[-1]
    assert appended["id"] == outsider
    assert appended["rank"] > len(captured)
    assert appended["p"] == pytest.approx(
        result["branch"]["confidences"][2]
    )


def test_a_captured_forced_token_adds_no_row() -> None:
    """The negative space: substituting a token the position already
    listed must leave its candidate set exactly as it was."""
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    captured = state["alternatives"][2]

    result = _run_substitute(state=state, position=2, budget=6)
    alts = result["branch"]["alternatives"][2]

    assert len(alts) == len(captured)
    assert all("rank" not in a for a in alts)


def test_the_seed_frame_carries_the_appended_row() -> None:
    """The client stores its candidates from this frame, so the row it
    shows and the row the saved run keeps have to be one list."""
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    captured = state["alternatives"][2]
    outsider = next(
        i
        for i in range(VOCAB_SIZE)
        if not any(c["id"] == i for c in captured)
    )

    result = _run_substitute(
        state=state, position=2, budget=6, forced_id=outsider
    )
    seed = result["frames"][0]

    assert len(seed["tokens"]) == 3
    assert seed["alts"] is not None
    assert seed["alts"][-1]["id"] == outsider
    assert seed["alts"] == result["branch"]["alternatives"][2]


def test_typed_substitution_still_keeps_the_prefix() -> None:
    """Moving the prefill boundary must not move the splice."""
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    result = _run_substitute(
        state=state, position=2, budget=6, typed=True
    )
    branch = result["branch"]

    assert branch["ids"][:2] == state["ids"][:2]
    assert branch["ids"][2] == result["forced"]["id"]
    assert len(branch["ids"]) == 6


def test_substitute_at_the_first_position_has_no_prefix() -> None:
    """The boundary: position 0 prefills the prompt and nothing else.

    Worth its own test because the probe builds its input by
    concatenating the prompt with the kept prefix, and at position 0
    that prefix is empty.
    """
    out = _run_generate(alternatives=True, budget=6)
    result = _run_substitute(
        state=out["state"], position=0, budget=6, typed=True
    )
    branch = result["branch"]

    assert branch["ids"][0] == result["forced"]["id"]
    assert len(branch["ids"]) == 6
    assert 0.0 < branch["confidences"][0] <= 1.0


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


# -- probe_token --


def test_probe_reproduces_a_recorded_probability() -> None:
    """The probe and the run must agree about the same position.

    Probing a token the run captured gives an independent oracle:
    the recorded probability is what the model gave it, so the
    measurement has to reproduce it. Disagreement would mean the
    probe is reading the distribution one position off, which is the
    only interesting way for this to be wrong. Recorded values are
    rounded to four places, hence the tolerance.
    """
    state = _run_generate(alternatives=True, budget=6)["state"]
    captured = state["alternatives"][2]
    result = probe_token(
        model=StubModel(),
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=state["ids"][:2],
        token_id=captured[0]["id"],
        thinking=False,
    )
    assert result["probability"] == pytest.approx(
        captured[0]["p"], abs=1e-4
    )


def test_probe_ranks_the_likeliest_token_first() -> None:
    """The token the run chose is the one the model preferred.

    Greedy sampling picks the argmax, so the run's own token at a
    position must come back as rank 1. This pins the rank's
    orientation: an off-by-one or an inverted comparison would show
    up as 2 or as the vocabulary size.
    """
    state = _run_generate(alternatives=True, budget=6)["state"]
    result = probe_token(
        model=StubModel(),
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=state["ids"][:2],
        token_id=state["ids"][2],
        thinking=False,
    )
    assert result["rank"] == 1


def test_probe_ranks_a_rejected_token_behind_it() -> None:
    """The negative space: a token the model dispreferred ranks late.

    The stub is built to never favor EOS, so it is the least likely
    token at every position, which makes its rank the vocabulary
    size. Together with the test above this brackets the rank at both
    ends rather than only confirming it can produce 1.
    """
    state = _run_generate(alternatives=True, budget=6)["state"]
    result = probe_token(
        model=StubModel(),
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=state["ids"][:2],
        token_id=EOS_ID,
        thinking=False,
    )
    assert result["rank"] == VOCAB_SIZE
    assert result["vocab_size"] == VOCAB_SIZE


def test_probe_at_the_first_position_has_no_prefix() -> None:
    """Position 0 prefills the prompt alone, an empty-prefix path."""
    result = probe_token(
        model=StubModel(),
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=[],
        token_id=3,
        thinking=False,
    )
    assert 0.0 < result["probability"] <= 1.0
    assert 1 <= result["rank"] <= VOCAB_SIZE


def test_probe_rejects_a_token_outside_the_output() -> None:
    """An operating error, so it raises rather than asserting.

    The bound is the model's output width, which the caller cannot
    check for itself, so a bad id has to be turned away here with a
    message the client can show.
    """
    with pytest.raises(ValueError):
        probe_token(
            model=StubModel(),
            tokenizer=StubTokenizer(),
            prompt="prompt",
            prefix_ids=[],
            token_id=VOCAB_SIZE,
            thinking=False,
        )


def test_probe_agrees_with_a_typed_substitution() -> None:
    """The popover's promise and the run's report are one number.

    The strip shows the probe's figure before the substitution runs,
    and the branch shows its own measurement afterwards. If these
    could diverge, the UI would quote odds the run then contradicts,
    so this pins them to the same distribution.
    """
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    result = _run_substitute(
        state=state, position=2, budget=6, typed=True
    )
    probed = probe_token(
        model=StubModel(),
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=state["ids"][:2],
        token_id=result["forced"]["id"],
        thinking=False,
    )
    assert probed["probability"] == pytest.approx(
        result["branch"]["confidences"][2], abs=1e-9
    )


# -- the retained KV cache --
#
# What can be settled here is the arithmetic of the reuse: which
# positions are sliced, which call shape is issued, and that every
# disagreement falls back rather than answering. What cannot is
# whether the numbers improve, because that is a bf16 property and
# every model here is a stub in float32. HANDOFF.md carries the
# bit-equality check for hardware.


def test_a_slice_leaves_the_retained_cache_whole() -> None:
    """The reuse has to be non-destructive: transformers implements
    ``crop`` in place, and the next probe may want a later position
    than this one did."""
    retained = _stub_cache(8)
    sliced = _sliced_cache(retained, 5)
    assert sliced is not None
    assert sliced.to_legacy_cache()[0][0].shape[-2] == 5
    assert retained.to_legacy_cache()[0][0].shape[-2] == 8
    assert len(sliced.to_legacy_cache()) == 2


def test_a_slice_is_a_view_and_not_a_copy() -> None:
    """The point of slicing rather than copying: pointer work, not
    tens of megabytes of device bandwidth per probe."""
    retained = _stub_cache(8)
    sliced = _sliced_cache(retained, 5)
    assert sliced is not None
    assert (
        sliced.to_legacy_cache()[0][0].data_ptr()
        == retained.to_legacy_cache()[0][0].data_ptr()
    )


def test_an_unfamiliar_cache_declines_to_slice() -> None:
    """A cache that does not speak the legacy format is a prefill,
    not a crash: the fast path is optional by construction."""
    assert _sliced_cache(object(), 3) is None
    assert _sliced_cache(None, 3) is None


def _cache_for(prefix: List[int], prompt_len: int = 3) -> Any:
    return {
        "past": _stub_cache(prompt_len + len(prefix)),
        "prompt_len": prompt_len,
        "ids": list(prefix),
    }


def test_reuse_slices_to_one_short_of_the_position() -> None:
    """The call-shape rule. The run sampled position n from a decode
    of the token at n-1, so the cache covers prompt + n - 1 and that
    token is the one forwarded. Slicing to n instead would make this a
    prefill wearing a cache, and reintroduce the rounding step the
    whole commit exists to remove."""
    cache = _cache_for([5, 6, 7, 8])
    reused = _reuse_cache(cache, [5, 6], 3)
    assert reused is not None
    assert reused.to_legacy_cache()[0][0].shape[-2] == 4


def test_reuse_declines_at_the_first_position() -> None:
    """Position 0 was itself a prefill during the run, so there is
    nothing to decode against and reproducing it means prefilling."""
    cache = _cache_for([5, 6])
    assert _reuse_cache(cache, [], 3) is None


def test_reuse_declines_on_a_prefix_mismatch() -> None:
    """The guard that keeps this an optimization rather than a
    liability: a cache from another run would answer confidently and
    wrongly."""
    cache = _cache_for([5, 6, 7])
    assert _reuse_cache(cache, [5, 9], 3) is None


def test_reuse_declines_on_a_different_prompt_length() -> None:
    """Same ids against a different prompt is a different sequence,
    and the slice offset would be wrong by the difference."""
    cache = _cache_for([5, 6, 7])
    assert _reuse_cache(cache, [5, 6], 4) is None


def test_reuse_declines_past_what_the_cache_covers() -> None:
    """The boundary at the far end: a prefix longer than the run the
    cache came from."""
    cache = _cache_for([5, 6])
    assert _reuse_cache(cache, [5, 6, 7], 3) is None


def test_reuse_declines_without_a_cache() -> None:
    assert _reuse_cache(None, [5, 6], 3) is None


def test_a_cache_is_kept_with_what_it_covers() -> None:
    """Kept with its ids and prompt length, because a probe has to
    prove the cache describes the prefix it is asking about before
    trusting a number that came out of it."""
    record = _cache_record(_stub_cache(7), 3, [5, 6, 7, 8])
    assert record is not None
    assert record["prompt_len"] == 3
    assert record["ids"] == [5, 6, 7, 8]


def test_an_oversized_cache_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded residency. Dropped rather than trimmed: a run large
    enough to pass the ceiling is one whose cache was never worth
    holding for the session.

    The ceiling is lowered rather than the cache raised to it. At the
    real bound a stub would have to allocate half a gigabyte to prove
    a comparison, and a test that measures nothing should not be the
    most expensive one in the file.
    """
    cache = _stub_cache(8)
    monkeypatch.setattr(ar_sampler, "AR_CACHE_BYTES_MAX", 16)
    assert _cache_record(cache, 3, [5]) is None
    monkeypatch.setattr(
        ar_sampler, "AR_CACHE_BYTES_MAX", AR_CACHE_BYTES_MAX
    )
    assert _cache_record(cache, 3, [5]) is not None


def test_an_unreadable_cache_is_dropped() -> None:
    assert _cache_record(object(), 3, [5]) is None
    assert _cache_record(None, 3, [5]) is None


def test_a_probe_decodes_one_token_against_the_cache() -> None:
    """End to end: with a cache the probe issues a one-token decode
    instead of a pass over the whole prefix. The stub's logits depend
    only on the last input token, so the answer is unchanged and the
    call shape is the only thing that moved."""
    model = StubModel()
    measured = probe_token(
        model=model,
        tokenizer=StubTokenizer(),
        prompt="prompt",
        prefix_ids=[5, 6],
        token_id=3,
        thinking=False,
        cache=_cache_for([5, 6]),
    )
    assert len(model.calls) == 1
    assert model.calls[0] == {"width": 1, "cached": True}
    assert 0.0 < measured["probability"] <= 1.0


def test_a_probe_without_a_cache_prefills_the_prefix() -> None:
    """The fallback's shape, and the same number out of it, which is
    what makes the fallback safe to take at any time."""
    cached = StubModel()
    plain = StubModel()
    args: Dict[str, Any] = {
        "tokenizer": StubTokenizer(),
        "prompt": "prompt",
        "prefix_ids": [5, 6],
        "token_id": 3,
        "thinking": False,
    }

    with_cache = probe_token(
        model=cached, cache=_cache_for([5, 6]), **args
    )
    without = probe_token(model=plain, **args)

    assert plain.calls[0] == {"width": 5, "cached": False}
    assert with_cache["probability"] == pytest.approx(
        without["probability"], abs=1e-9
    )
    assert with_cache["rank"] == without["rank"]


def test_a_substitution_reuses_the_cache_for_its_prefix() -> None:
    """The substitution's dominant cost is the prefix pass it makes
    before producing a single new token. With the run's cache that
    becomes one decode, and the branch is otherwise identical."""
    out = _run_generate(alternatives=True, budget=6)
    state = out["state"]
    plain = _run_substitute(state=state, position=2, budget=6)
    cached = _run_substitute(
        state=state,
        position=2,
        budget=6,
        cache=_cache_for(state["ids"]),
    )

    assert cached["first_call"] == {"width": 1, "cached": True}
    assert plain["first_call"] == {"width": 5, "cached": False}
    assert cached["branch"]["ids"] == plain["branch"]["ids"]
    assert cached["branch"]["confidences"] == pytest.approx(
        plain["branch"]["confidences"]
    )


def test_a_run_keeps_a_cache_it_can_read() -> None:
    """The end of the plumbing: what _stream_tokens finished with has
    to reach the sink, or none of the above is ever exercised."""
    state = _run_generate(alternatives=True, budget=4)["state"]
    assert "cache" in state


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
