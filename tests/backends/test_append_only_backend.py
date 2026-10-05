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
from src.inference.kgw_key import WatermarkKey


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


class _BackendTokenizer:
    """Raw-tokenizer stand-in that records special-token policy."""

    class _Inner:
        @staticmethod
        def to_str() -> str:
            return '{"test":"tokenizer"}'

    backend_tokenizer = _Inner()

    def __init__(self) -> None:
        self.add_special_tokens: List[bool] = []

    def encode(
        self, text: str, *, add_special_tokens: bool
    ) -> List[int]:
        self.add_special_tokens.append(add_special_tokens)
        return [ord(character) % 256 for character in text]


class _BackendModel:
    """Output width only. Calling it would fail the detector test."""

    class _Config:
        vocab_size = 256

    config = _Config()

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("detector must not run a model forward")


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


def _detector_backend() -> _Backend:
    backend = _Backend(declares_thinking=True)
    backend.tokenizer = _BackendTokenizer()
    backend.model = _BackendModel()
    return backend


def _detect(backend: _Backend, **payload: Any) -> _StubWebSocket:
    ws = _StubWebSocket()
    request: Dict[str, Any] = {
        "text": "a" * 60,
        "request_id": 7,
    }
    request.update(payload)
    asyncio.run(
        backend.handle_detect_watermark(
            ws,  # type: ignore[arg-type]
            request,
        )
    )
    return ws


def test_pasted_text_detector_uses_raw_tokenizer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()
    monkeypatch.setattr(
        append_only_backend,
        "load_key",
        lambda: WatermarkKey(bytes(range(32)), "0123456789abcdef"),
    )

    ws = _detect(backend, z_threshold=3.5)

    reply = ws.sent[0]
    assert reply["type"] == "detect_watermark_result"
    assert reply["request_id"] == 7
    assert reply["token_count"] == 60
    assert reply["scored_count"] == 59
    assert reply["key_id"] == "0123456789abcdef"
    assert reply["model_id"] == "smollm3"
    assert reply["gamma"] == pytest.approx(0.25)
    assert reply["vocab_size"] == 256
    assert reply["display_threshold"] == pytest.approx(3.5)
    assert backend.tokenizer.add_special_tokens == [False]


def test_pasted_text_detector_loads_but_never_creates_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()

    def missing() -> WatermarkKey:
        raise FileNotFoundError("no key")

    monkeypatch.setattr(append_only_backend, "load_key", missing)

    ws = _detect(backend)

    assert ws.sent[0]["code"] == "watermark_key_missing"
    assert ws.sent[0]["scope"] == "request"


def test_pasted_text_detector_refuses_wrong_expected_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()
    monkeypatch.setattr(
        append_only_backend,
        "load_key",
        lambda: WatermarkKey(bytes(range(32)), "0123456789abcdef"),
    )

    ws = _detect(backend, expected_key_id="ffffffffffffffff")

    assert ws.sent[0]["code"] == "watermark_key_mismatch"
    assert ws.sent[0]["request_id"] == 7


def test_pasted_text_detector_refuses_text_past_bound() -> None:
    backend = _detector_backend()

    ws = _detect(
        backend,
        text="x"
        * (append_only_backend.WATERMARK_DETECT_TEXT_MAX_CHARS + 1),
    )

    assert ws.sent[0]["code"] == "invalid_request"
    assert "limit" in ws.sent[0]["message"]


def test_pasted_text_detector_refuses_token_count_past_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()
    monkeypatch.setattr(
        append_only_backend,
        "load_key",
        lambda: WatermarkKey(bytes(range(32)), "0123456789abcdef"),
    )

    ws = _detect(
        backend,
        text="x"
        * (append_only_backend.WATERMARK_DETECT_TOKENS_MAX + 1),
    )

    assert ws.sent[0]["code"] == "invalid_request"
    assert "tokens" in ws.sent[0]["message"]


def test_pasted_text_detector_reports_insecure_key_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()

    def insecure() -> WatermarkKey:
        raise PermissionError("key mode is 0644")

    monkeypatch.setattr(append_only_backend, "load_key", insecure)

    ws = _detect(backend)

    assert ws.sent[0]["code"] == "watermark_key_state"
    assert ws.sent[0]["scope"] == "request"
    assert "0644" in ws.sent[0]["message"]


def test_generation_reports_key_state_before_beginning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _detector_backend()

    def insecure() -> WatermarkKey:
        raise PermissionError("key mode is 0644")

    monkeypatch.setattr(
        append_only_backend,
        "load_or_create_key",
        insecure,
    )

    ws = _generate(
        backend,
        experimental=True,
        watermark=True,
    )

    assert ws.sent[0]["code"] == "invalid_request"
    assert "key state" in ws.sent[0]["message"]
    assert backend.run_counter == 0


@pytest.mark.parametrize("gamma", [0.0, 1.0, float("nan"), True])
def test_pasted_text_detector_refuses_invalid_gamma(
    gamma: Any,
) -> None:
    backend = _detector_backend()

    ws = _detect(backend, gamma=gamma)

    assert ws.sent[0]["code"] == "invalid_request"
    assert "gamma" in ws.sent[0]["message"].lower()
