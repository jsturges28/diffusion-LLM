"""Tests that a cached model loads without asking the Hub anything.

Strategy: run each worker's ``load`` with the weight fetch, the
progress sampler, and both ``from_pretrained`` calls replaced by
stubs, then assert on the keyword arguments the loaders were handed.
No model, no GPU, no network.

The bug this pins was found by TRUST-02's offline pass. With the
network off, every page rendered but two of the three models refused
to activate: SmolLM3 failed resolving `huggingface.co` while fetching
additional chat templates, and LLaDA hung for minutes resolving its
remote code. Both were fully downloaded at the time. The cache check
was never the problem; `download_with_progress` already answers it
and returns a local path without touching the network. The workers
then handed `from_pretrained` the repo *name* with no
`local_files_only`, so transformers went back to the Hub to
revalidate a checkpoint already sitting on disk.

Passing proves every load call is pinned to local files. What makes
that safe is the ordering asserted here too: the flag is only set on
calls made after `download_with_progress` has returned, and returning
means every file is present.

DiffusionGemma is deliberately absent. Its checkpoint is a local
directory rather than a Hub id, which is why it was the one model
that did activate offline, and it has no Hub call to pin.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List

import pytest

from src.backends import llada_worker, smollm3_worker
from src.backends.llada_worker import LladaBackend
from src.backends.smollm3_worker import Smollm3Backend

SNAPSHOT = "/cache/models--org--model/snapshots/abc123"


class _LoadRecorder:
    """Collects what each stubbed loader was asked for."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.order: List[str] = []

    def loader(self, label: str) -> Any:
        def _load(name: str, **kwargs: Any) -> Any:
            self.order.append(label)
            self.calls.append(
                {"label": label, "name": name, **kwargs}
            )
            return _FakeLoaded()
        return _load

    def kwargs_for(self, label: str) -> Dict[str, Any]:
        for call in self.calls:
            if call["label"] == label:
                return call
        raise AssertionError(f"{label} was never loaded")


class _FakeLoaded:
    """Enough of a model/tokenizer for ``load`` to finish."""

    padding_side = "left"

    def eval(self) -> "_FakeLoaded":
        return self

    def to(self, device: Any) -> "_FakeLoaded":
        del device
        return self


def _install(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    recorder: _LoadRecorder,
    *,
    tokenizer_attr: str,
    model_attr: str,
) -> None:
    """Replace the fetch, the samplers, and both loaders."""

    def fake_download(repo_id: str, *, sink: Any) -> str:
        del sink
        recorder.order.append("download")
        return SNAPSHOT

    @contextlib.contextmanager
    def fake_sampler(**kwargs: Any) -> Iterator[None]:
        del kwargs
        yield

    monkeypatch.setattr(
        module, "download_with_progress", fake_download
    )
    monkeypatch.setattr(
        module, "load_target_bytes", lambda path, **kw: 0
    )
    monkeypatch.setattr(
        module, "sample_load_progress", fake_sampler
    )
    monkeypatch.setattr(
        module,
        tokenizer_attr,
        type(
            "Tok",
            (),
            {"from_pretrained": staticmethod(
                recorder.loader("tokenizer")
            )},
        ),
    )
    monkeypatch.setattr(
        module,
        model_attr,
        type(
            "Mdl",
            (),
            {"from_pretrained": staticmethod(
                recorder.loader("model")
            )},
        ),
    )


def _load_llada(
    monkeypatch: pytest.MonkeyPatch,
) -> _LoadRecorder:
    recorder = _LoadRecorder()
    _install(
        monkeypatch,
        llada_worker,
        recorder,
        tokenizer_attr="AutoTokenizer",
        model_attr="AutoModel",
    )
    LladaBackend().load(device="cpu")
    return recorder


def _load_smollm3(
    monkeypatch: pytest.MonkeyPatch,
) -> _LoadRecorder:
    recorder = _LoadRecorder()
    _install(
        monkeypatch,
        smollm3_worker,
        recorder,
        tokenizer_attr="AutoTokenizer",
        model_attr="AutoModelForCausalLM",
    )
    Smollm3Backend().load(device="cpu")
    return recorder


# -- LLaDA, which hung offline --


def test_llada_loads_its_tokenizer_from_local_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _load_llada(monkeypatch)

    assert (
        recorder.kwargs_for("tokenizer")["local_files_only"]
        is True
    )


def test_llada_loads_its_model_from_local_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _load_llada(monkeypatch)

    assert (
        recorder.kwargs_for("model")["local_files_only"] is True
    )


def test_llada_still_asks_for_its_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning to local files must not disable the remote code this
    checkpoint needs; the code ships inside the snapshot."""
    recorder = _load_llada(monkeypatch)

    assert (
        recorder.kwargs_for("tokenizer")["trust_remote_code"]
        is True
    )
    assert (
        recorder.kwargs_for("model")["trust_remote_code"] is True
    )


# -- SmolLM3, which failed offline --


def test_smollm3_loads_its_tokenizer_from_local_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact call that failed: this environment's transformers
    asks the Hub API for additional chat templates while building the
    tokenizer, which offline is a name-resolution error."""
    recorder = _load_smollm3(monkeypatch)

    assert (
        recorder.kwargs_for("tokenizer")["local_files_only"]
        is True
    )


def test_smollm3_loads_its_model_from_local_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _load_smollm3(monkeypatch)

    assert (
        recorder.kwargs_for("model")["local_files_only"] is True
    )


# -- the ordering the flag depends on --


@pytest.mark.parametrize(
    "loader", [_load_llada, _load_smollm3], ids=["llada", "smollm3"]
)
def test_the_fetch_runs_before_anything_is_pinned(
    monkeypatch: pytest.MonkeyPatch, loader: Any
) -> None:
    """local_files_only is only honest after the download.

    Pinning before the weights are fetched would turn a first
    activation into a failure instead of a download, so the order is
    the safety property, not an implementation detail.
    """
    recorder = loader(monkeypatch)

    assert recorder.order[0] == "download"
    assert "tokenizer" in recorder.order
    assert "model" in recorder.order


@pytest.mark.parametrize(
    "loader", [_load_llada, _load_smollm3], ids=["llada", "smollm3"]
)
def test_every_hub_load_is_pinned(
    monkeypatch: pytest.MonkeyPatch, loader: Any
) -> None:
    """Negative space over the whole load rather than call by call:
    one unpinned call is all it takes to reach the network, so the
    assertion is that none of them is."""
    recorder = loader(monkeypatch)

    assert len(recorder.calls) == 2
    for call in recorder.calls:
        assert call.get("local_files_only") is True, (
            f"{call['label']} can still reach the Hub"
        )
