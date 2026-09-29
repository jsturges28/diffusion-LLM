"""The shared append-only shell serves models with or without
reasoning.

Strategy: a minimal subclass of `AppendOnlyBackend` whose registry
entry either declares SmolLM3's `thinking` parameter or leaves it
out, the way a completion model's will, with the sampler replaced by
a recorder. SmolLM3's handler tests, in test_smollm3_substitute.py,
already pin substitution and the probe through this same class.
Passing proves a model with no `thinking` parameter generates, is
handed `thinking=False`, and keeps a run a later substitution can
re-enter, while a model that declares it keeps its value.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.backends import append_only_backend
from src.backends.append_only_backend import AppendOnlyBackend
from src.backends.registry import SMOLLM3
from src.backends.text_adapter import SMOLLM3_TEXT


class _Backend(AppendOnlyBackend):
    def __init__(self, *, declares_thinking: bool) -> None:
        specs = [
            spec
            for spec in SMOLLM3.param_specs
            if declares_thinking or spec.name != "thinking"
        ]
        self.model_info = SMOLLM3.model_copy(
            update={"param_specs": specs}
        )
        self.text_adapter = SMOLLM3_TEXT
        # No model or tokenizer: the prompt check skips without them.
        self.model = None
        self.tokenizer = None
        self.effective_device = "cpu"
        self.last_run_state = None

    def load(self, *, device: str = "cuda") -> None:
        raise NotImplementedError


class _StubWebSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


class _StubStreamer:
    async def run(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
    ) -> bool:
        async for _ in generator:
            pass
        return True


def _install(
    monkeypatch: pytest.MonkeyPatch, calls: List[Dict[str, Any]]
) -> None:
    async def fake_generate(
        *_args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        calls.append(kwargs)
        sink = kwargs["state_sink"]
        sink["ids"] = [5]
        sink["confidences"] = [0.5]
        sink["entropies"] = [0.1]
        sink["alternatives"] = [None]
        yield {"type": "done", "final_text": "x"}

    monkeypatch.setattr(
        append_only_backend, "streaming_generate", fake_generate
    )


def _generate(backend: _Backend, **payload: Any) -> _StubWebSocket:
    ws = _StubWebSocket()
    request: Dict[str, Any] = {"prompt": "Once upon"}
    request.update(payload)
    asyncio.run(
        backend.handle_generate(
            ws,  # type: ignore[arg-type]
            request,
            threading.Event(),
            _StubStreamer(),  # type: ignore[arg-type]
        )
    )
    return ws


def test_a_model_without_thinking_generates_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _install(monkeypatch, calls)
    backend = _Backend(declares_thinking=False)

    ws = _generate(backend)

    assert ws.sent == []
    assert calls[0]["thinking"] is False
    assert backend.last_run_state is not None
    assert backend.last_run_state["thinking"] is False
    assert backend.last_run_state["ids"] == [5]


def test_a_model_with_thinking_keeps_its_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _install(monkeypatch, calls)
    backend = _Backend(declares_thinking=True)

    _generate(backend, thinking=True)

    assert calls[0]["thinking"] is True
    assert backend.last_run_state is not None
    assert backend.last_run_state["thinking"] is True
