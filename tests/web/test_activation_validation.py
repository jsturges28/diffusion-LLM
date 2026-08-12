"""A switch that cannot work must not cost you the model you have.

Strategy: stand up a manager with a resident, ready worker (the same
fakes `tests/web/test_worker_lifecycle.py` uses), then ask it to
activate a target that cannot run, one impossibility per test. The
assertion is always the same shape and it is the finding's whole
point: the request fails, and the worker that was already loaded is
still loaded, still the active model, still serving.

What passing proves. Activation used to stop the resident worker
first and inspect the target afterwards, so choosing a model with no
interpreter installed, no local checkpoint, or no chance of fitting
in VRAM unloaded a working model to discover something that needed no
VRAM to discover. Recovery then meant another slow load, and on the
generator the visible run had already been discarded by the client on
its way into the same request.

The post-eviction `_preflight_vram` is deliberately still there and
still authoritative. This is the cheaper check in front of it, not a
replacement: it answers "could this ever fit", where that one answers
"did the memory actually come back".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.backends.registry import REGISTRY
from src.web import server
from src.web.server import ModelManager
from tests.web.test_worker_lifecycle import (
    READY,
    FakeProcess,
)

RESIDENT = "llada"


class Harness:
    """A manager holding one ready worker, ready to be disturbed."""

    def __init__(self) -> None:
        self.processes: List[FakeProcess] = []

        def spawn(
            command: Any, *, cwd: Any, env: Any
        ) -> FakeProcess:
            made = FakeProcess(pid=2000 + len(self.processes))
            self.processes.append(made)
            return made

        async def probe(url: str) -> Optional[Dict[str, Any]]:
            return READY

        self.manager = ModelManager(
            spawn=spawn,  # type: ignore[arg-type]
            probe=probe,
            start_timeout_s=0.05,
            stop_timeout_s=0.01,
            kill_timeout_s=0.01,
            health_poll_s=0.001,
            progress_poll_s=0.001,
            # The real one polls nvidia-smi for eight seconds. Every
            # case here reaches it, so leaving it alone would put a
            # minute of sleeping in the suite.
            vram_settle_timeout_s=0.01,
        )

    def resident_holds_vram(self) -> bool:
        """Whether a worker is currently occupying the GPU."""
        return any(p.alive for p in self.processes)

    async def make_resident(self) -> None:
        await self.manager.activate(RESIDENT, device="cuda")
        task = self.manager._monitor_task
        if task is not None:
            await task

    def assert_undisturbed(self) -> None:
        """The resident worker is exactly as it was."""
        assert self.manager.active_id == RESIDENT
        assert self.manager.active_device == "cuda"
        assert self.manager.load_state == "ready"
        assert self.manager.is_serving(RESIDENT)
        assert len(self.processes) == 1
        assert self.processes[0].alive
        assert self.processes[0].calls == []


@pytest.fixture(autouse=True)
def _plenty_of_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "_gpu_name", lambda: "Fake GPU"
    )
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: 99.0
    )


@pytest.fixture(autouse=True)
def _interpreters_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three venvs are not installed where tests run."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)


def _resident() -> Harness:
    harness = Harness()
    asyncio.run(harness.make_resident())
    harness.processes[0].calls.clear()
    return harness


# -- each impossibility, and what it costs --


def test_a_missing_interpreter_evicts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check that was one line below the eviction."""
    harness = _resident()
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(
            harness.manager.activate("smollm3", device="cpu")
        )

    harness.assert_undisturbed()


def test_an_unsupported_device_evicts_nothing() -> None:
    """DiffusionGemma's NF4 experts need CUDA. The worker has always
    refused CPU, but from inside load(), by which point the resident
    model was already gone."""
    harness = _resident()

    with pytest.raises(RuntimeError, match="cannot run on CPU"):
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cpu"
            )
        )

    harness.assert_undisturbed()


def test_a_missing_local_checkpoint_evicts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The README calls out that the optional model stays listed
    even when activating it will fail. Listing it is fine; charging
    a loaded model for the discovery is not."""
    harness = _resident()
    monkeypatch.setattr(Path, "is_dir", lambda self: False)

    with pytest.raises(
        RuntimeError, match="checkpoint not found"
    ):
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cuda"
            )
        )

    harness.assert_undisturbed()


def test_a_model_that_cannot_fit_evicts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counted against free plus what unloading would return, so it
    only refuses the genuinely hopeless case."""
    harness = _resident()
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: 0.5
    )

    with pytest.raises(
        RuntimeError, match="Not enough GPU memory"
    ):
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cuda"
            )
        )

    harness.assert_undisturbed()


def test_the_refusal_says_the_model_is_still_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user reading "not enough memory" would reasonably assume
    they had just lost their model, which is what used to happen."""
    harness = _resident()
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: 0.5
    )

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cuda"
            )
        )

    assert "still loaded" in str(caught.value)


# -- the negative space: a switch that can work, does --


def test_a_valid_switch_still_replaces_the_worker() -> None:
    """The checks must refuse the impossible without refusing the
    ordinary, which is every switch the app actually performs."""
    harness = _resident()

    async def scenario() -> None:
        await harness.manager.activate("smollm3", device="cpu")
        task = harness.manager._monitor_task
        if task is not None:
            await task

    asyncio.run(scenario())

    assert harness.manager.active_id == "smollm3"
    assert harness.manager.load_state == "ready"
    assert not harness.processes[0].alive
    assert harness.processes[1].alive


def test_a_reclaimable_resident_lets_the_next_model_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the estimate counts reclaimable VRAM. With LLaDA's
    17 GiB resident, almost nothing is free, and DiffusionGemma still
    fits once that memory comes back.

    The stub releases the memory when the worker dies, because both
    checks run in this test and they are supposed to see different
    readings: the estimate before eviction, the real one after.
    """
    harness = _resident()
    required = REGISTRY["diffusiongemma"].min_vram_gib
    resident = REGISTRY[RESIDENT].min_vram_gib
    free_while_resident = required - resident + 1.0

    def free_vram() -> float:
        if harness.resident_holds_vram():
            return free_while_resident
        return free_while_resident + resident

    monkeypatch.setattr(server, "_free_vram_gib", free_vram)

    async def scenario() -> None:
        await harness.manager.activate(
            "diffusiongemma", device="cuda"
        )
        task = harness.manager._monitor_task
        if task is not None:
            await task

    asyncio.run(scenario())

    assert harness.manager.active_id == "diffusiongemma"


def test_an_unreadable_gpu_does_not_block_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host where nvidia-smi says nothing must not become a host
    where no GPU model can be selected. The post-eviction check makes
    the same call."""
    harness = _resident()
    monkeypatch.setattr(
        server, "_free_vram_gib", lambda: None
    )

    async def scenario() -> None:
        await harness.manager.activate(
            "diffusiongemma", device="cuda"
        )
        task = harness.manager._monitor_task
        if task is not None:
            await task

    asyncio.run(scenario())

    assert harness.manager.active_id == "diffusiongemma"


def test_reselecting_the_resident_model_does_nothing() -> None:
    """No spawn, no eviction, no validation cost: the model asked
    for is the one already running."""
    harness = _resident()

    asyncio.run(
        harness.manager.activate(RESIDENT, device="cuda")
    )

    harness.assert_undisturbed()


# -- how a refusal reaches the browser --


def test_a_refusal_is_its_own_kind_of_error() -> None:
    """Separate from a fault so the route can answer with the
    reason. Everything the validation raises has to carry the type,
    or the endpoint falls through to the generic 500."""
    harness = _resident()

    with pytest.raises(server.ActivationRefused):
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cpu"
            )
        )


def test_a_refusal_is_still_a_runtime_error() -> None:
    """Callers that predate the type, including the endpoint's
    generic handler, must keep catching it."""
    assert issubclass(server.ActivationRefused, RuntimeError)


def test_the_post_eviction_check_refuses_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_preflight_vram` runs after the resident model is gone, so
    its refusal is a different situation with the same shape, and it
    should not be the one error that reads as a crash."""
    harness = _resident()

    def free_vram() -> float:
        # Plenty while resident (so the estimate passes), nothing
        # once it is gone (so the real check refuses).
        return 99.0 if harness.resident_holds_vram() else 0.5

    monkeypatch.setattr(server, "_free_vram_gib", free_vram)

    with pytest.raises(server.ActivationRefused):
        asyncio.run(
            harness.manager.activate(
                "diffusiongemma", device="cuda"
            )
        )


# -- the declared devices are real --


def test_the_registry_agrees_with_what_the_workers_accept() -> None:
    """DiffusionGemma is the only CUDA-only model, and the other two
    genuinely run on CPU (SmolLM3 is the model a GPU-less host uses).
    A wrong entry here refuses a switch that would have worked."""
    assert REGISTRY[
        "diffusiongemma"
    ].capabilities.supported_devices == ("cuda",)
    for model_id in ("llada", "smollm3"):
        supported = REGISTRY[
            model_id
        ].capabilities.supported_devices
        assert "cpu" in supported
        assert "cuda" in supported
