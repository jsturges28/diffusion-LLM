"""The Mamba-3 worker loads the registry's pins and runs end to end.

Strategy: a tiny random checkpoint written to disk the way upstream
writes one, with the Hub replaced by functions that hand back paths
and record what they were asked, drives `load()` on CPU in well under
a second. The same tiny model then runs through the real append-only
handlers and the real sampler, with a character-level tokenizer in
place of Llama 3.1's, whose begin-of-text id is outside a 50-id
vocabulary.

Passing proves the worker asks for the pinned weights and the pinned
companion tokenizer, refuses a tokenizer that is not Llama 3.1's
before it spends time on the weights, and that a run, a What If
branch and a probe all work on a model whose "cache" is a list of
recurrent states, with every streamed token carrying what reading
that token erased.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest
import torch

from src.backends import mamba3_worker
from src.backends.mamba3_worker import Mamba3Backend, build_backend
from src.backends.protocol import MSG_PROBE_RESULT
from src.backends.registry import MAMBA3, SMOLLM3
from src.backends.text_adapter import (
    MAMBA3_TEXT,
    CompletionTextAdapter,
)
from src.backends.worker_base import provenance_envelope
from src.inference import mamba3
from src.inference.mamba3_causal import Mamba3CausalLM
from src.inference.mamba3_memory import forgetting, recording_core
from src.inference.mamba3_tokenizer import (
    TOKENIZER_FILE,
    Encoding,
    TokenizerMismatchError,
)
from tests.backends.test_smollm3_substitute import (
    _StubStreamer,
    _StubWebSocket,
)
from tests.inference.test_mamba3_memory import TINY_MODEL, _tiny_model
from tests.inference.test_mamba3_tokenizer import _tokenizer_file

BEGIN = 1
PROMPT = "a quiet river"
# The lowest max_new_tokens outside experimental mode.
BUDGET = 16
BRANCH_POSITION = 3
# Frames carry per-token values rounded to four places, as they carry
# confidence and entropy; the retained trace keeps them whole.
WIRE_PLACES = 4
# Half the last place, plus room for float32 noise between reading a
# token on its own call and reading the whole sequence at once.
WIRE_TOLERANCE = 0.5 * 10.0 ** -WIRE_PLACES + 1e-5
TRACE_TOLERANCE = 1e-5


class _CharTokenizer:
    """Llama31Tokenizer's surface over the tiny model's ids: one id
    per character after a begin-of-text id. No end-of-text, so every
    run spends its whole budget and a test knows its length."""

    name_or_path = "test/characters"
    vocab_size = int(TINY_MODEL["vocab_size"])
    is_fast = True
    bos_token_id = BEGIN
    eos_token_id = None
    fingerprint = "test-fingerprint"

    def __len__(self) -> int:
        return self.vocab_size

    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> List[int]:
        ids = [2 + ord(char) % 40 for char in text]
        if add_special_tokens:
            return [BEGIN, *ids]
        return ids

    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join(f"<{int(token)}>" for token in ids)

    def __call__(
        self, text: str, return_tensors: str = "pt"
    ) -> Encoding:
        encoding = Encoding()
        encoding["input_ids"] = torch.tensor(
            [self.encode(text)], dtype=torch.long
        )
        return encoding


# -- loading --


def _snapshot(tmp_path: Path) -> Path:
    """The tiny model where the Hub cache puts the pinned commit."""
    root = (
        tmp_path
        / "models--state-spaces--mamba3-siso-1.5b"
        / "snapshots"
        / str(MAMBA3.revision)
    )
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(TINY_MODEL), encoding="utf-8"
    )
    torch.save(_tiny_model().state_dict(), root / "pytorch_model.bin")
    return root


def _install_hub(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: Path,
    tokenizer_path: Path,
) -> Dict[str, Any]:
    """Both fetches, answered from disk, recording what each was
    asked. CUDA is reported absent, as it is in the sandbox."""
    asked: Dict[str, Any] = {}

    def download(name: str, *, revision: str, sink: Any) -> str:
        asked["weights"] = (name, revision)
        return str(snapshot)

    def companion(
        repo: str, files: Sequence[str], *, revision: str
    ) -> Dict[str, Path]:
        asked["companion"] = (repo, tuple(files), revision)
        return {TOKENIZER_FILE: tokenizer_path}

    monkeypatch.setattr(
        mamba3_worker, "download_with_progress", download
    )
    monkeypatch.setattr(
        mamba3_worker, "fetch_companion_files", companion
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return asked


def _load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Any]:
    """A load asked for CUDA, with the tokenizer load recorded and
    answered by the character tokenizer."""
    tokenizer_path = tmp_path / TOKENIZER_FILE
    asked = _install_hub(
        monkeypatch, _snapshot(tmp_path), tokenizer_path
    )
    loads: List[Any] = []

    def load_tokenizer(path: Path, **options: Any) -> _CharTokenizer:
        loads.append((path, options))
        return _CharTokenizer()

    monkeypatch.setattr(
        mamba3_worker, "load_tokenizer", load_tokenizer
    )
    backend = Mamba3Backend()
    backend.load(device="cuda")
    asked["tokenizer"] = loads
    asked["tokenizer_path"] = tokenizer_path
    asked["backend"] = backend
    return asked


def test_load_asks_for_the_pinned_weights_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both pins, and the fingerprint check left at its default. The
    tokenizer is loaded without `required`, so Llama 3.1's
    fingerprint is what a file has to match."""
    asked = _load(tmp_path, monkeypatch)

    assert asked["weights"] == (MAMBA3.checkpoint, MAMBA3.revision)
    assert asked["companion"] == (
        SMOLLM3.checkpoint,
        (TOKENIZER_FILE,),
        SMOLLM3.revision,
    )
    assert asked["tokenizer"] == [
        (asked["tokenizer_path"], {"source": SMOLLM3.checkpoint})
    ]
    assert asked["backend"].loaded_revision == MAMBA3.revision


def test_load_falls_back_to_the_cpu_in_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA asked for and absent: the model lands on the CPU, and in
    float32, the precision every device runs this model in."""
    backend = _load(tmp_path, monkeypatch)["backend"]

    assert backend.effective_device == "cpu"
    assert isinstance(backend.model, Mamba3CausalLM)
    for weight in backend.model.model.parameters():
        assert weight.dtype == torch.float32
        assert weight.device.type == "cpu"


def test_a_tokenizer_that_is_not_llamas_stops_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real check on a real file. A stranger's ids would give
    plausible nonsense, so the load stops, and stops before the
    weights rather than after minutes of reading them."""
    tokenizer_path = tmp_path / TOKENIZER_FILE
    tokenizer_path.write_text(
        json.dumps(_tokenizer_file()), encoding="utf-8"
    )
    _install_hub(monkeypatch, _snapshot(tmp_path), tokenizer_path)

    def no_weights(*_: Any, **__: Any) -> None:
        raise AssertionError("read weights past a refused tokenizer")

    monkeypatch.setattr(mamba3, "load", no_weights)
    backend = Mamba3Backend()

    with pytest.raises(TokenizerMismatchError):
        backend.load(device="cpu")
    assert backend.model is None


def test_the_worker_is_a_completion_model() -> None:
    """What the process entry point builds, before anything loads."""
    backend = build_backend()

    assert isinstance(backend, Mamba3Backend)
    assert backend.model_info is MAMBA3
    assert backend.text_adapter is MAMBA3_TEXT
    assert isinstance(MAMBA3_TEXT, CompletionTextAdapter)
    assert backend.model is None


def test_a_saved_run_says_what_made_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The envelope a save carries: this model at its commit, where it
    ran, the tokenizer by content, the forgetting channel described,
    and no context window claimed for a model that has none."""
    backend = _load(tmp_path, monkeypatch)["backend"]

    envelope = provenance_envelope(backend)

    assert envelope["model_id"] == MAMBA3.id
    assert envelope["revision"] == MAMBA3.revision
    assert envelope["device"] == "cpu"
    assert envelope["tokenizer"]["fingerprint"] == "test-fingerprint"
    names = [channel["name"] for channel in envelope["signals"]]
    assert "forgetting" in names
    assert "context_length" not in envelope


# -- the handlers, on the real sampler --


def _ready_backend() -> Mamba3Backend:
    """A worker as `load()` leaves it, on the CPU, tiny."""
    backend = Mamba3Backend()
    backend.model = Mamba3CausalLM(_tiny_model())
    backend.tokenizer = _CharTokenizer()
    backend.device = "cpu"
    backend.effective_device = "cpu"
    return backend


def _generate(backend: Mamba3Backend) -> List[Dict[str, Any]]:
    """A greedy run with candidates captured; its token frames."""
    ws = _StubWebSocket()
    stream = _StubStreamer()
    payload = {
        "prompt": PROMPT,
        "max_new_tokens": BUDGET,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "alternatives": True,
    }
    asyncio.run(
        backend.handle_generate(
            ws,  # type: ignore[arg-type]
            payload,
            asyncio.Event(),  # type: ignore[arg-type]
            stream,  # type: ignore[arg-type]
        )
    )
    assert ws.sent == [], ws.sent
    return _token_frames(stream.frames)


def _substitute(
    backend: Mamba3Backend, position: int, token_id: int
) -> List[Dict[str, Any]]:
    ws = _StubWebSocket()
    stream = _StubStreamer()
    payload = {
        "position": position,
        "token_id": token_id,
        "run_token": backend.run_token,
    }
    asyncio.run(
        backend.handle_substitute(
            ws,  # type: ignore[arg-type]
            payload,
            asyncio.Event(),  # type: ignore[arg-type]
            stream,  # type: ignore[arg-type]
        )
    )
    assert ws.sent == [], ws.sent
    return _token_frames(stream.frames)


def _token_frames(
    frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "frame"]


def _read_whole(
    backend: Mamba3Backend, ids: List[int]
) -> List[float]:
    """Forgetting for every token of `ids` read in one pass from the
    empty state: the definition, with no sampler involved."""
    model = backend.model.model
    decays: List[torch.Tensor] = []
    with torch.no_grad():
        model(
            torch.tensor([ids]),
            model.empty_states(1),
            core=recording_core(decays),
        )
    values: List[float] = forgetting(decays)[0].tolist()
    return values


def test_a_run_streams_every_token_with_its_forgetting() -> None:
    backend = _ready_backend()

    frames = _generate(backend)

    state = backend.last_run_state
    assert state is not None
    assert len(frames) == BUDGET
    assert [frame["token"]["id"] for frame in frames] == state["ids"]
    values = [frame["token"]["f"] for frame in frames]
    for value in values:
        assert 0.0 <= value <= 1.0
    assert values == [
        round(signal["f"], WIRE_PLACES) for signal in state["signals"]
    ]


def test_each_value_is_what_reading_that_token_erased() -> None:
    """The alignment the read-then-emit loop exists for, checked
    against the definition: the value kept for generated token i is
    what a single pass over prompt and output gives position
    prompt + i. One position off, it would be its neighbour's, which
    the second assertion shows is a different number."""
    backend = _ready_backend()
    _generate(backend)
    state = backend.last_run_state
    assert state is not None
    prompt = backend.tokenizer.encode(PROMPT)
    kept = [signal["f"] for signal in state["signals"]]

    read = _read_whole(backend, prompt + state["ids"])

    assert kept == pytest.approx(
        read[len(prompt):], abs=TRACE_TOLERANCE
    )
    shifted = read[len(prompt) - 1:-1]
    assert kept != pytest.approx(shifted, abs=TRACE_TOLERANCE)


def test_a_what_if_branch_reads_its_forced_token_first() -> None:
    """The branch replays the prompt and the kept prefix, reads the
    forced token, and streams it with its own value before anything
    after it. Every value is the definition's, over the branch."""
    backend = _ready_backend()
    _generate(backend)
    state = backend.last_run_state
    assert state is not None
    forced = next(
        candidate["id"]
        for candidate in state["alternatives"][BRANCH_POSITION]
        if candidate["id"] != state["ids"][BRANCH_POSITION]
    )

    frames = _substitute(backend, BRANCH_POSITION, forced)

    assert frames[0]["token"]["id"] == forced
    prompt = backend.tokenizer.encode(PROMPT)
    branch = state["ids"][:BRANCH_POSITION] + [
        frame["token"]["id"] for frame in frames
    ]
    read = _read_whole(backend, prompt + branch)
    values = [frame["token"]["f"] for frame in frames]
    start = len(prompt) + BRANCH_POSITION
    assert values == pytest.approx(read[start:], abs=WIRE_TOLERANCE)
    assert backend.last_run_state is state


def test_a_probe_answers_through_the_replay() -> None:
    """A list of states cannot be sliced back like a cache, so the
    probe prefills the prompt and the kept prefix. In float32 that is
    the run again: the run's own greedy token measures to its
    recorded probability, and ranks first."""
    backend = _ready_backend()
    _generate(backend)
    state = backend.last_run_state
    assert state is not None
    assert state.get("cache") is None
    position = 5
    ws = _StubWebSocket()

    asyncio.run(
        backend.handle_probe(
            ws,  # type: ignore[arg-type]
            {
                "position": position,
                "token_id": state["ids"][position],
                "run_token": backend.run_token,
                "request_id": 1,
            },
        )
    )

    reply = ws.sent[-1]
    assert reply["type"] == MSG_PROBE_RESULT
    assert reply["probability"] == pytest.approx(
        state["confidences"][position], rel=1e-5
    )
    assert reply["rank"] == 1
    assert reply["vocab_size"] == backend.model.config.vocab_size


# -- the registry entry --


def test_the_entry_declares_a_state_space_completion_model() -> None:
    capabilities = MAMBA3.capabilities

    assert capabilities.family == "state_space"
    assert capabilities.generation_shape == "append_only"
    assert capabilities.input_mode == "completion"
    assert capabilities.supports_substitution is True
    assert capabilities.supported_devices == ("cuda", "cpu")
    assert MAMBA3.environment == SMOLLM3.environment


def test_the_companion_is_smollm3s_tokenizer_at_its_pin() -> None:
    """The file SmolLM3 itself loads, so the two cannot drift onto
    different tokenizers by one pin moving without the other."""
    companion = MAMBA3.companion

    assert companion is not None
    assert companion.repo == SMOLLM3.checkpoint
    assert companion.revision == SMOLLM3.revision
    assert companion.files == (TOKENIZER_FILE,)


def test_the_knobs_are_smollm3s_less_the_reasoning_switch() -> None:
    """Same sampler, same knobs, including the lower CPU budget; a
    base model has no reasoning channel for `thinking` to select."""
    names = [spec.name for spec in MAMBA3.param_specs]
    budget = next(
        spec for spec in MAMBA3.param_specs
        if spec.name == "max_new_tokens"
    )

    assert "thinking" not in names
    assert names == [
        spec.name for spec in SMOLLM3.param_specs
        if spec.name != "thinking"
    ]
    assert "cpu" in budget.overrides
