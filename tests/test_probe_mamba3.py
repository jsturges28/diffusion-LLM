"""The Mamba-3 probe runs end to end, and judges by its stated bar.

Strategy: the probe itself is hardware work, but everything around
the model is ordinary code. A tiny random checkpoint saved under
upstream's names, and a word-level stand-in for the tokenizer, drive
`main` through every section on CPU in seconds. The downloads are
checked for their pins with the Hub client replaced, the tokenizer
fingerprint on hand-made files and on SmolLM3's real one where it is
cached, and the verdict logic is fed hand-made lenses whose right
answers are known. Passing proves the probe cannot pass a broken load
or a tokenizer the checkpoint was not trained with, that its criteria
are the ones written down before the first run, and that each
retention test fails the lens it exists to catch.
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


# -- the tokenizer fingerprint --


def _tokenizer_file(strings: bool = False) -> Dict[str, Any]:
    """The parts of a tokenizer.json the fingerprint reads, small.
    `strings` writes the merges the way Meta's file does."""
    merges: List[Any] = [["a", "b"], ["ab", "c"]]
    if strings:
        merges = ["a b", "ab c"]
    begin, end = probe.BEGIN_OF_TEXT, probe.END_OF_TEXT
    return {
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
            "merges": merges,
        },
        "added_tokens": [
            {"id": begin, "content": "<|begin_of_text|>"},
            {"id": end, "content": "<|end_of_text|>"},
            {"id": end + 1, "content": "<|reserved_0|>"},
        ],
    }


def test_the_fingerprint_reads_both_merge_formats_alike() -> None:
    """Meta's file writes merges as "a b" strings and SmolLM3's as
    lists; the same rules must give the same fingerprint."""
    strings = probe.bpe_fingerprint(_tokenizer_file(True))
    lists = probe.bpe_fingerprint(_tokenizer_file(False))

    assert strings == lists


def test_the_fingerprint_changes_with_anything_moving_an_id() -> None:
    base = probe.bpe_fingerprint(_tokenizer_file())
    reordered = _tokenizer_file()
    reordered["model"]["merges"].reverse()
    renumbered = _tokenizer_file()
    renumbered["model"]["vocab"]["abc"] = 5
    renamed_end = _tokenizer_file()
    renamed_end["added_tokens"][1]["content"] = "<|im_end|>"

    for changed in (reordered, renumbered, renamed_end):
        assert probe.bpe_fingerprint(changed) != base


def test_the_fingerprint_ignores_tokens_never_emitted() -> None:
    """SmolLM3 renamed ten reserved tokens for its chat format. The
    codec drops every added token, so they must not count."""
    base = probe.bpe_fingerprint(_tokenizer_file())
    renamed = _tokenizer_file()
    renamed["added_tokens"][2]["content"] = "<think>"

    assert probe.bpe_fingerprint(renamed) == base


class _Pieces(NamedTuple):
    ids: List[int]


class _BytePairs:
    """Word lengths as ids, recording whether specials were asked for.
    Stands in for `tokenizers.Tokenizer`, whose installed version here
    cannot read SmolLM3's file."""

    def __init__(self) -> None:
        self.asked_for_specials: List[bool] = []

    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _Pieces:
        self.asked_for_specials.append(add_special_tokens)
        return _Pieces([len(word) for word in sequence.split()])

    def decode(self, ids: List[int]) -> str:
        return ",".join(str(token) for token in ids)


def test_the_codec_adds_begin_of_text_and_no_other_special() -> None:
    """Begin-of-text comes from the codec, once, and the byte pairs
    are never asked for specials of their own; decoding drops every
    id past the vocabulary, since only the specials live there."""
    pairs = _BytePairs()
    data = _tokenizer_file()

    codec = probe.codec_from(pairs, data)

    begin, end = probe.BEGIN_OF_TEXT, probe.END_OF_TEXT
    assert codec.encode("ab c") == [begin, 2, 1]
    assert pairs.asked_for_specials == [False]
    assert codec.decode([begin, 3, end]) == "3"
    assert codec.end == end
    assert codec.vocabulary == end + 2
    assert codec.fingerprint == probe.bpe_fingerprint(data)


def test_the_pinned_smollm3_tokenizer_is_llamas() -> None:
    """Holds the recorded fingerprint to the real file, on any
    machine that has run SmolLM3 and so has it cached."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        probe.TOKENIZER_REPO,
        probe.TOKENIZER_FILE,
        revision=probe.TOKENIZER_REVISION,
    )
    if not isinstance(path, str):
        pytest.skip("SmolLM3's tokenizer is not cached here")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = probe.LLAMA31_BPE_FINGERPRINT

    assert probe.bpe_fingerprint(data) == recorded
    assert probe._id_count(data) == 128256


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
