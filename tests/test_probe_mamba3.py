"""The Mamba-3 probe runs end to end, and judges by its stated bar.

Strategy: the probe itself is hardware work, but everything around
the model is ordinary code. A tiny random checkpoint saved under
upstream's names, and a word-level stand-in for the gated tokenizer,
drive `main` through every section on CPU in seconds. The downloads
are checked for their pins with the Hub client replaced, and the
verdict logic is fed hand-made lenses whose right answers are known.
Passing proves the probe cannot pass a broken load, that its criteria
are the ones written down before the first run, and that each
retention test fails the lens it exists to catch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from scripts import probe_mamba3 as probe
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


def _codec() -> probe.TextCodec:
    """Words to ids by their letters, after a begin-of-text id, the
    shape Llama's tokenizer has."""
    size = TINY["vocab_size"]

    def encode(text: str) -> List[int]:
        words = text.split()
        ids = [sum(map(ord, word)) % (size - 2) + 2 for word in words]
        return [BEGIN] + ids

    def decode(ids: Any) -> str:
        return " ".join(f"w{token}" for token in ids)

    return probe.TextCodec(encode, decode, size, None)


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
) -> tuple[int, Dict[str, Any]]:
    def no_fetch(*_: Any) -> Path:
        raise AssertionError("fetched what it was given")

    monkeypatch.setattr(probe, "fetch", no_fetch)
    monkeypatch.setattr(probe, "load_codec", lambda _: _codec())
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
    assert _results(report["load"]) == ["pass", "pass"]
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


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["--sections", "speed,nonsense"])


# -- the downloads --


def test_downloads_are_pinned_and_fetch_only_the_named_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Llama's repository also holds 16 GB of weights, so the file
    list is what keeps the tokenizer download to three small files."""
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
        "repo": "meta-llama/Llama-3.1-8B",
        "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "allow_patterns": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    }]
    assert probe.MODEL_REVISION == (
        "5cfc721542ec9ccee768088b2fd6b7e8101219d8"
    )


def test_a_gated_tokenizer_is_refused_with_what_to_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    def gated(repo: str, **_: Any) -> str:
        raise GatedRepoError(f"{repo} is gated")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    checkpoint = _checkpoint(tmp_path / "model")

    code = probe.main([
        "--device", "cpu", "--checkpoint", str(checkpoint),
    ])

    assert code == 2
    advice = capsys.readouterr().err
    assert "meta-llama/Llama-3.1-8B is gated" in advice
    assert ".venv-ar/bin/hf auth login" in advice


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
