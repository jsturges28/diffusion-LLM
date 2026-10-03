"""A worker starts its load from a lifespan (`A2-DEPS-01`).

Strategy: build the app around a backend whose load only notes the
device it was given, enter it through FastAPI's test client, and
poll ``/health`` until the load reports in. Warnings are recorded
around building and entering, so a lifecycle deprecation from either
step fails.

Passing proves that building the app loads nothing, that entering it
starts exactly one load on the requested device, that ``/health``
reports the load the lifespan started (the two share one load
state), and that the app registers no deprecated hook.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Dict, List

from fastapi import WebSocket
from fastapi.testclient import TestClient

from src.backends.protocol import ModelCapabilities, ModelInfo
from src.backends.worker_app import create_worker_app
from src.backends.worker_base import Backend, FrameStreamer

# The stub's load returns at once, so a status still "loading" after
# this many polls means the load never started or never reported.
HEALTH_POLLS_MAX = 500
HEALTH_POLL_SECONDS = 0.01


def _model_info() -> ModelInfo:
    return ModelInfo(
        id="stub",
        display_name="Stub",
        param_specs=[],
        capabilities=ModelCapabilities(
            family="diffusion",
            generation_shape="iterative_canvas",
            input_mode="chat",
            supported_devices=("cuda", "cpu"),
        ),
        worker_module="none",
        environment="none",
        checkpoint="none",
    )


class _RecordedLoad(Backend):
    """A backend whose load notes the device it was given and does
    nothing else."""

    def __init__(self) -> None:
        self.model_info = _model_info()
        self.effective_device = "cpu"
        self.devices: List[str] = []

    def load(self, *, device: str = "cuda") -> None:
        self.devices.append(device)

    async def handle_generate(
        self,
        ws: WebSocket,
        data: Dict[str, Any],
        cancel_event: threading.Event,
        stream: FrameStreamer,
    ) -> None:
        raise AssertionError("nothing here generates")


def _settled_status(client: TestClient) -> str:
    """Poll ``/health`` until the load stops reporting "loading"."""
    for _ in range(HEALTH_POLLS_MAX):
        status = str(client.get("/health").json()["status"])
        if status != "loading":
            return status
        time.sleep(HEALTH_POLL_SECONDS)
    return "loading"


def _is_lifecycle_deprecation(
    warning: warnings.WarningMessage,
) -> bool:
    if not issubclass(warning.category, DeprecationWarning):
        return False
    text = str(warning.message)
    return "on_event" in text or "lifespan" in text


def test_entering_the_app_starts_one_load_on_its_device() -> None:
    backend = _RecordedLoad()
    app = create_worker_app(backend, device="cpu")

    assert backend.devices == [], "building the app loaded"
    with TestClient(app) as client:
        status = _settled_status(client)

    assert status == "ready", status
    assert backend.devices == ["cpu"], backend.devices


def test_building_and_entering_the_app_warns_nothing() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = create_worker_app(_RecordedLoad(), device="cpu")
        with TestClient(app) as client:
            status = _settled_status(client)

    lifecycle = [
        str(warning.message).strip()
        for warning in caught
        if _is_lifecycle_deprecation(warning)
    ]
    assert status == "ready", status
    assert lifecycle == [], lifecycle
