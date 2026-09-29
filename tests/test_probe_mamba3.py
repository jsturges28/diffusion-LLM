"""The Mamba-3 probe runs end to end, and judges by its stated bar.

Strategy: the probe itself is hardware work, but everything around
the model is ordinary code. A tiny random checkpoint saved under
upstream's names, and a word-level stand-in for the tokenizer, drive
`main` through every section on CPU in seconds. The downloads are
checked for their pins with the Hub client replaced, and the verdict
logic is fed hand-made lenses whose right answers are known. The
tokenizer the probe shares with the worker is tested on its own, in
tests/inference/test_mamba3_tokenizer.py. Passing proves the probe
cannot pass a broken load or a tokenizer the checkpoint was not
trained with, that its criteria are the ones written down before the
first run, and that each retention test fails the lens it exists to
catch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import pytest
import torch

from scripts import probe_mamba3 as probe
from src.backends.registry import SMOLLM3
from src.inference.mamba3 import Mamba3LM, config_from_json
from src.inference.mamba3_tokenizer import (
    BEGIN_OF_TEXT,
    END_OF_TEXT,
    Llama31Tokenizer,
)

TINY: Dict[str, Any] = {
    "d_model": 32,
    "d_intermediate": 64,
    "n_layer": 2,
    "vocab_size": 50,
    "ssm_cfg": {
        "layer": "Mamba3",
        "d_state": 16,
        "expand": 2,
        "headdim": 8,
        "rope_fraction": 0.5,
    },
    "pad_vocab_size_multiple": 16,
}
BEGIN = 1
HEADS = 3


def _codec(
    fingerprint: str = probe.LLAMA31_BPE_FINGERPRINT,
) -> probe.TextCodec:
    """Words to ids by their letters, after a begin-of-text id, the
    shape Llama's tokenizer has."""
    size = TINY["vocab_size"]

    def encode(text: str) -> List[int]:
        words = text.split()
        ids = [sum(map(ord, word)) % (size - 2) + 2 for word in words]
        return [BEGIN] + ids

    def decode(ids: Any) -> str:
        return " ".join(f"w{token}" for token in ids)

    return probe.TextCodec(encode, decode, size, None, fingerprint)


def _checkpoint(root: Path, extra: bool = False) -> Path:
    """A tiny random model saved as upstream saves its checkpoint."""
    torch.manual_seed(0)
    model = Mamba3LM(config_from_json(TINY))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.3 * torch.randn_like(parameter))
    weights = dict(model.state_dict())
    if extra:
        weights["backbone.extra"] = torch.zeros(1)
    root.mkdir()
    (root / "config.json").write_text(json.dumps(TINY))
    torch.save(weights, root / "pytorch_model.bin")
    return root


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *options: str,
    extra: bool = False,
    codec: probe.TextCodec | None = None,
) -> tuple[int, Dict[str, Any]]:
    def no_fetch(*_: Any) -> Path:
        raise AssertionError("fetched what it was given")

    chosen = _codec() if codec is None else codec
    monkeypatch.setattr(probe, "fetch", no_fetch)
    monkeypatch.setattr(probe, "load_codec", lambda _: chosen)
    report = tmp_path / "report.json"
    code = probe.main([
        "--device", "cpu",
        "--dtype", "float32",
        "--checkpoint", str(_checkpoint(tmp_path / "model", extra)),
        "--tokenizer", str(tmp_path),
        "--json", str(report),
        *options,
    ])
    return code, json.loads(report.read_text())


def _results(section: Dict[str, Any]) -> List[str]:
    return [verdict["result"] for verdict in section["verdicts"]]


# -- end to end --


def test_every_section_runs_on_a_tiny_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report = _run(tmp_path, monkeypatch)

    assert code == 0
    assert set(report) == {"run", "load", *probe.SECTIONS}
    assert _results(report["load"]) == ["pass", "pass", "pass"]
    agreement = report["correctness"]["agreement"]
    assert agreement["relative_error"] < probe.AGREEMENT_MAX
    assert _results(report["correctness"])[0] == "pass"
    retention = report["retention"]
    assert retention["rebuild_error_max"] < probe.AGREEMENT_MAX
    assert len(retention["state"]["layers"]) == TINY["n_layer"]
    assert report["speed"]["peak_vram_mib"] is None
    forgetting = report["forgetting"]
    shown = len(forgetting["top_tokens"])
    assert len(forgetting["verdicts"]) == 4
    assert shown == probe.FORGETTING_TOP_TOKENS


def test_sections_choose_what_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report = _run(tmp_path, monkeypatch, "--sections", "speed")

    assert code == 0
    assert set(report) == {"run", "load", "speed"}


def test_an_unexpected_tensor_stops_the_probe_at_the_load_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report = _run(tmp_path, monkeypatch, extra=True)

    assert code == 1
    assert set(report) == {"run", "load"}
    assert report["load"]["unexpected"] == ["backbone.extra"]
    assert _results(report["load"])[0] == "fail"


def test_a_tokenizer_that_is_not_llamas_stops_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every number after the load report would describe a model fed
    ids it was never trained on, so none of them is computed."""
    stranger = _codec(fingerprint="0" * 64)

    code, report = _run(tmp_path, monkeypatch, codec=stranger)

    assert code == 1
    assert set(report) == {"run", "load"}
    assert report["load"]["loadable"] is True
    assert _results(report["load"])[1] == "fail"


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["--sections", "speed,nonsense"])


# -- the downloads --


def test_downloads_are_pinned_and_fetch_only_the_named_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SmolLM3's repository also holds 6 GB of weights, so the file
    list is what keeps the tokenizer download to one file. Its
    revision is the registry's, the one the SmolLM3 worker runs."""
    import huggingface_hub

    calls: List[Dict[str, Any]] = []

    def record(repo: str, **options: Any) -> str:
        calls.append({"repo": repo, **options})
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", record)

    probe.fetch(
        probe.TOKENIZER_REPO,
        probe.TOKENIZER_REVISION,
        probe.TOKENIZER_FILES,
    )

    assert calls == [{
        "repo": "HuggingFaceTB/SmolLM3-3B",
        "revision": SMOLLM3.revision,
        "allow_patterns": ["tokenizer.json"],
    }]
    assert probe.MODEL_REVISION == (
        "5cfc721542ec9ccee768088b2fd6b7e8101219d8"
    )


# -- the probe's view of the shared tokenizer --


class _Pieces(NamedTuple):
    ids: List[int]


class _BytePairs:
    """Word lengths as ids, standing in for a `tokenizers` model."""

    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _Pieces:
        return _Pieces([len(word) for word in sequence.split()])

    def decode(self, ids: List[int]) -> str:
        return ",".join(str(token) for token in ids)


def test_the_codec_wraps_the_shared_tokenizer() -> None:
    """Encoding keeps begin-of-text, decoding drops the specials, and
    the fingerprint the load report judges is the tokenizer's own.
    The tokenizer itself is tested in test_mamba3_tokenizer.py."""
    begin, end = BEGIN_OF_TEXT, END_OF_TEXT
    data = {
        "normalizer": None,
        "pre_tokenizer": None,
        "decoder": None,
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        "added_tokens": [
            {"id": begin, "content": "<|begin_of_text|>"},
            {"id": end, "content": "<|end_of_text|>"},
        ],
    }
    tokenizer = Llama31Tokenizer(_BytePairs(), data, source="test")

    codec = probe.codec_of(tokenizer)

    assert codec.encode("ab c") == [begin, 2, 1]
    assert codec.decode([begin, 0, end]) == "0"
    assert codec.vocabulary == end + 1
    assert codec.end == end
    assert codec.fingerprint == tokenizer.fingerprint


def test_a_stranger_tokenizer_is_reported_not_refused(
    tmp_path: Path,
) -> None:
    """The worker refuses a file that is not Llama 3.1's; the probe
    must load it anyway, so the load report can say so as a verdict
    instead of the run ending in a traceback."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    built = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    built.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    built.save(str(tmp_path / "tokenizer.json"))

    codec = probe.load_codec(tmp_path)

    assert codec.fingerprint != probe.LLAMA31_BPE_FINGERPRINT


# -- the bar --


def test_the_criteria_are_the_ones_registered_in_advance() -> None:
    """Item 328 in docs/MANUAL_VERIFICATION.md states these numbers.
    Moving one has to be deliberate, and this makes it visible."""
    assert probe.AGREEMENT_MAX == 1e-3
    assert probe.PERPLEXITY_GLUE_ERROR == 40.0
    assert probe.PERPLEXITY_EXPECTED == (10.0, 20.0)
    assert probe.CPU_DECODE_MIN == 3.0
    assert probe.TOP_POSITIONS == 20
    assert probe.RECENCY_SPEARMAN == 0.9
    assert probe.CONTENT_VARIATION == 0.05
    assert probe.FORGETTING_WARM_UP == 8
    assert probe.FORGETTING_FLAT_RATIO == 0.5
    assert probe.CHANCE_FLOOR == 2.0


def test_criteria_that_do_not_apply_to_a_run_are_not_judged() -> None:
    """The CPU bar only means something on CPU, and the agreement bar
    only in float32, where rounding cannot account for a gap."""
    cuda = probe._cpu_verdict(torch.device("cuda"), 1.0)
    slow = probe._cpu_verdict(torch.device("cpu"), 2.9)
    fast = probe._cpu_verdict(torch.device("cpu"), 3.0)
    config = config_from_json(TINY)
    agreement = {"relative_error": 0.5}
    half = Mamba3LM(config, dtype=torch.bfloat16)
    full = Mamba3LM(config, dtype=torch.float32)

    assert cuda["result"] == "not judged"
    assert slow["result"] == "fail"
    assert fast["result"] == "pass"
    half_verdict = probe._agreement_verdict(half, agreement)
    full_verdict = probe._agreement_verdict(full, agreement)
    assert half_verdict["result"] == "not judged"
    assert full_verdict["result"] == "fail"


def test_only_a_perplexity_above_the_glue_line_fails() -> None:
    low = probe._perplexity_verdict([3.3, 8.1])
    high = probe._perplexity_verdict([12.0, 41.0])

    assert low["result"] == "pass"
    assert "outside the expected" in low["detail"]
    assert high["result"] == "fail"


# -- the retention tests, on lenses with known answers --

LENGTH = 40


def _lens(
    scores: torch.Tensor, alpha: torch.Tensor
) -> probe.LayerLens:
    return probe.LayerLens(scores, scores, alpha, 0.0)


def _lenses(
    first: torch.Tensor,
    second: torch.Tensor,
    repeated: torch.Tensor,
    alpha: torch.Tensor,
) -> Dict[str, List[probe.LayerLens]]:
    return {
        "passage_a": [_lens(first, alpha)] * 3,
        "passage_b": [_lens(second, alpha)] * 3,
        "repeated": [_lens(repeated, alpha)] * 3,
    }


def _varied_alpha() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return 0.5 + 0.4 * torch.rand(LENGTH, HEADS, generator=gen)


def test_a_lens_that_is_only_recency_fails_both_tests() -> None:
    """The same climb on every input: the repeated token matches the
    passages exactly, and the scores rank by position alone."""
    climb = torch.linspace(0.1, 1.0, LENGTH)
    lenses = _lenses(climb, climb, climb, _varied_alpha())

    verdicts = probe.falsify(lenses, "state")["verdicts"]

    assert [v["result"] for v in verdicts] == ["fail", "fail"]


def test_a_lens_that_follows_the_text_passes_both_tests() -> None:
    """Both passages keep the same early tokens, which the repeated
    token does not, and neither ranks by position."""
    gen = torch.Generator().manual_seed(1)
    shared = 0.1 * torch.rand(LENGTH, generator=gen)
    shared[: probe.TOP_POSITIONS] += 1.0
    repeated = torch.flip(shared, dims=[0])
    lenses = _lenses(shared, shared * 2, repeated, _varied_alpha())

    verdicts = probe.falsify(lenses, "state")["verdicts"]

    assert [v["result"] for v in verdicts] == ["pass", "pass"]


def test_a_fixed_decay_fails_the_content_test() -> None:
    climb = torch.linspace(0.1, 1.0, LENGTH)
    fixed = torch.full((LENGTH, HEADS), 0.9)

    blind = probe.alpha_variation(_lenses(climb, climb, climb, fixed))
    moved = probe.alpha_variation(
        _lenses(climb, climb, climb, _varied_alpha())
    )

    assert blind["verdict"]["result"] == "fail"
    assert math.isclose(blind["median"], 0.0, abs_tol=1e-6)
    assert moved["verdict"]["result"] == "pass"


# -- the forgetting tests, on profiles with known answers --

TOKENS = 128
CRITERIA = ("content", "position", "flat", "degenerate")


def _noise(seed: int) -> torch.Tensor:
    """Forgetting that follows the text: its own value at each token,
    with no trend and no shared structure."""
    gen = torch.Generator().manual_seed(seed)
    return 0.1 + 0.05 * torch.rand(TOKENS, generator=gen)


def _flat() -> torch.Tensor:
    """A repeated token: some movement while its state settles, then
    the same value at every token after."""
    profile = torch.full((TOKENS,), 0.12)
    profile[: probe.FORGETTING_WARM_UP] = torch.linspace(0.3, 0.13, 8)
    return profile


def _spikes(positions: List[int]) -> torch.Tensor:
    """High forgetting at exactly these positions, so they are the
    profile's top ones, over a faint wobble that keeps it varied."""
    profile = _noise(9) * 0.01 + 0.1
    profile[positions] = 1.0
    return profile


def _results_by_name(section: Dict[str, Any]) -> Dict[str, str]:
    results = _results(section)
    return dict(zip(CRITERIA, results, strict=True))


def test_forgetting_that_follows_the_text_passes() -> None:
    section = probe.judge_forgetting({
        "passage_a": _noise(1),
        "passage_b": _noise(2),
        "repeated": _flat(),
    })

    assert _results(section) == ["pass"] * 4
    assert section["informative"] is True


def test_forgetting_that_barely_moves_fails() -> None:
    """A tint nobody could tell from uniform says nothing about the
    text, however cleanly it passes the other tests."""
    section = probe.judge_forgetting({
        "passage_a": 0.1 + 0.02 * _noise(1),
        "passage_b": 0.1 + 0.02 * _noise(2),
        "repeated": _flat(),
    })

    assert _results_by_name(section)["content"] == "fail"


def test_forgetting_that_only_tracks_position_fails() -> None:
    ramp = torch.linspace(0.1, 0.2, TOKENS)

    section = probe.judge_forgetting({
        "passage_a": ramp,
        "passage_b": ramp,
        "repeated": _flat(),
    })

    assert _results_by_name(section)["position"] == "fail"
    assert section["informative"] is False


def test_a_repeated_token_that_moves_like_text_fails() -> None:
    """Variation a contentless input shows too cannot be about the
    content."""
    section = probe.judge_forgetting({
        "passage_a": _noise(1),
        "passage_b": _noise(2),
        "repeated": _noise(3),
    })

    assert _results_by_name(section)["flat"] == "fail"


def test_structure_every_input_shares_fails() -> None:
    """The attention-sink shape: the same positions on top whatever
    the input, far above chance."""
    same = list(range(20))

    section = probe.judge_forgetting({
        "passage_a": _spikes(same),
        "passage_b": _spikes(same),
        "repeated": _spikes(same),
    })

    assert _results_by_name(section)["degenerate"] == "fail"


def test_a_coin_toss_below_the_chance_floor_is_no_failure() -> None:
    """The repeated token shares 3 top positions with each passage
    while the passages share none, so "as many" holds, but 3 of 20 is
    what chance gives. Without the floor this would fail."""
    first = list(range(0, 20))
    second = list(range(40, 60))
    repeated = [17, 18, 19, 57, 58, 59, *range(100, 114)]

    section = probe.judge_forgetting({
        "passage_a": _spikes(first),
        "passage_b": _spikes(second),
        "repeated": _spikes(repeated),
    })

    assert section["passages_overlap"] == 0
    assert section["repeated_overlap"] == 3
    assert _results_by_name(section)["degenerate"] == "pass"
