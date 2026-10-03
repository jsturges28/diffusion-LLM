"""Two supervisors on one machine, and only one model between them.

Strategy: build two real ``ModelManager`` objects in one process, both
pointed at one lease file, and drive activation on each. That is a
fair stand-in for two supervisors because the thing under test is a
flock, which lives on the open file description rather than on the
process: each manager opens the path itself, so they contend exactly
as two processes would. The spawn and probe are faked, as everywhere
else in these tests, since none of this is about a model loading.

The lease path is the test's own. The suite-wide fixture in
``tests/conftest.py`` already keeps every test off the real one, and
these pass an explicit path on top so a failure here names a file
belonging to this module.

Why the ordering assertions matter as much as the refusal: a refusal
that arrived after the resident worker was evicted would have cost the
user a working model to tell them they could not have a new one, which
is the failure mode the four-phase activation exists to prevent. So
the resident-untouched checks are not decoration.

Passing proves a second supervisor cannot load while a first holds the
claim, that its refusal names who to go to, that the loser's own
resident model and pages are untouched, that switching models inside
one supervisor is not self-competition, that a peer is refused for the
whole of another supervisor's switch, that a switch which cannot
launch gives the machine back, that eviction hands the claim over, and
that a slow finalize cannot take the claim from a newer activation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.web import model_manager
from src.web.model_lease import LEASE_FILE_NAME, PrimaryModelLease
from src.web.model_manager import ActivationRefused, ModelManager
from tests.web.test_worker_lifecycle import READY, FakeProcess

GPU_MODEL = "llada"
CPU_MODEL = "smollm3"


class Supervisor:
    """One manager, with its own claim on the shared lease file."""

    def __init__(self, lease_file: Path) -> None:
        self.processes: List[FakeProcess] = []
        # Set to make the next launch fail at the spawn, the way a
        # missing interpreter would after everything else has passed.
        self.fail_spawn = False

        def spawn(
            command: Any, *, cwd: Any, env: Any
        ) -> FakeProcess:
            if self.fail_spawn:
                raise OSError("the test refused this spawn")
            made = FakeProcess(pid=3000 + len(self.processes))
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
            vram_settle_timeout_s=0.01,
        )
        # Replaced rather than configured, because the manager builds
        # its own from the environment and this test wants both
        # supervisors on one file it can point at.
        self.manager._residency = PrimaryModelLease(lease_file)

    async def load(self, model: str, device: str) -> None:
        await self.manager.activate(model, device=device)
        task = self.manager._monitor_task
        if task is not None:
            await task

    @property
    def holds_lease(self) -> bool:
        return self.manager._residency.held

    @property
    def workers_alive(self) -> int:
        return sum(1 for p in self.processes if p.alive)


@pytest.fixture()
def lease_file(tmp_path: Path) -> Path:
    return tmp_path / LEASE_FILE_NAME


@pytest.fixture(autouse=True)
def _plenty_of_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_manager, "gpu_name", lambda: "Fake GPU"
    )
    monkeypatch.setattr(model_manager, "free_vram_gib", lambda: 99.0)


@pytest.fixture(autouse=True)
def _interpreters_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """The three venvs are not installed where tests run."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)


# -- one machine, one model --


def test_the_first_supervisor_takes_the_claim(
    lease_file: Path,
) -> None:
    first = Supervisor(lease_file)

    asyncio.run(first.load(GPU_MODEL, "cuda"))

    assert first.holds_lease is True
    assert first.manager.is_serving(GPU_MODEL)


def test_the_second_supervisor_is_refused(
    lease_file: Path,
) -> None:
    """The finding itself. Before this, both would have passed their
    own VRAM pre-flight and launched into a device sized for one."""
    first = Supervisor(lease_file)
    second = Supervisor(lease_file)

    async def both() -> None:
        await first.load(GPU_MODEL, "cuda")
        with pytest.raises(ActivationRefused):
            await second.load(GPU_MODEL, "cuda")

    asyncio.run(both())

    assert second.holds_lease is False
    assert second.workers_alive == 0


def test_the_refusal_says_who_to_go_to(lease_file: Path) -> None:
    """A refusal that only said no would leave the user hunting for
    which of their windows is holding the card."""
    first = Supervisor(lease_file)
    second = Supervisor(lease_file)

    async def both() -> str:
        await first.load(GPU_MODEL, "cuda")
        try:
            await second.load(GPU_MODEL, "cuda")
        except ActivationRefused as exc:
            return str(exc)
        raise AssertionError("the second was not refused")

    message = asyncio.run(both())

    assert "pid" in message
    assert GPU_MODEL.lower() in message.lower() or "model" in message
    assert "close" in message.lower()


def test_a_refused_supervisor_keeps_its_own_worker(
    lease_file: Path,
) -> None:
    """The half of the verification clause about remaining usable.

    A supervisor that already has a model and is refused a *switch*
    must still be serving the model it had. Losing it to a refusal
    would be the four-phase ordering failing.
    """
    first = Supervisor(lease_file)
    second = Supervisor(lease_file)

    async def sequence() -> None:
        # The second one gets there first and keeps a model.
        await second.load(GPU_MODEL, "cuda")
        # Now the first tries, holding nothing, and is refused.
        with pytest.raises(ActivationRefused):
            await first.load(GPU_MODEL, "cuda")

    asyncio.run(sequence())

    assert second.manager.is_serving(GPU_MODEL)
    assert second.workers_alive == 1


def test_a_supervisor_without_a_lease_loads_nothing(
    tmp_path: Path,
) -> None:
    """The refuse policy where a person meets it (`A2-TRUST-01`).

    Refused before anything spawns, and in words that name the file
    that could not be made and how to get one, rather than loading
    without the one-model guarantee the app promises. The lease sits
    under a regular file, which no user, root included, can open.
    """
    not_a_directory = tmp_path / "a-file"
    not_a_directory.write_text("", encoding="utf-8")
    lease = not_a_directory / LEASE_FILE_NAME
    unguarded = Supervisor(lease)

    with pytest.raises(ActivationRefused) as raised:
        asyncio.run(unguarded.load(GPU_MODEL, "cuda"))

    assert str(lease) in str(raised.value)
    assert "XDG_RUNTIME_DIR" in str(raised.value)
    assert unguarded.processes == []
    assert unguarded.holds_lease is False


def test_a_cpu_model_takes_the_claim_too(lease_file: Path) -> None:
    """The lease guards residency, not the GPU.

    The finding's own argument for preferring a lease over the VRAM
    pre-flight is that the pre-flight cannot govern CPU RAM, so a
    CPU-placed model has to count.
    """
    first = Supervisor(lease_file)
    second = Supervisor(lease_file)

    async def both() -> None:
        await first.load(CPU_MODEL, "cpu")
        with pytest.raises(ActivationRefused):
            await second.load(CPU_MODEL, "cpu")

    asyncio.run(both())

    assert first.holds_lease is True


# -- one supervisor is not its own rival --


def test_switching_models_is_not_self_competition(
    lease_file: Path,
) -> None:
    """The obvious way to get this wrong. A supervisor that holds the
    claim and evicts its own worker to load another must not then
    refuse itself the claim it is still holding."""
    only = Supervisor(lease_file)

    async def switch() -> None:
        await only.load(GPU_MODEL, "cuda")
        await only.load(CPU_MODEL, "cpu")

    asyncio.run(switch())

    assert only.manager.is_serving(CPU_MODEL)
    assert only.holds_lease is True


def test_a_switch_updates_what_the_claim_says(
    lease_file: Path,
) -> None:
    """So a refused peer names the model that is loaded now, rather
    than whichever one this supervisor happened to load first."""
    only = Supervisor(lease_file)
    onlooker = PrimaryModelLease(lease_file)

    async def switch() -> None:
        await only.load(GPU_MODEL, "cuda")
        await only.load(CPU_MODEL, "cpu")

    asyncio.run(switch())
    assert onlooker.acquire({"pid": 1}) is False

    owner = onlooker.owner()
    assert owner is not None
    assert owner["model"] == CPU_MODEL
    assert owner["device"] == "cpu"


def _pause_after_stop(
    supervisor: Supervisor,
) -> Tuple[asyncio.Event, asyncio.Event]:
    """Hold a switch just after its old worker is finalized.

    Where a switch used to give the claim up and, a moment later, take
    it back. Nothing awaits between the two, so two managers in one
    process never meet there by chance, while two supervisors in two
    processes can. Pausing there opens the gap on purpose.

    Installed after the first load, since every activation stops
    whatever came before it, even when nothing did.
    """
    paused = asyncio.Event()
    resume = asyncio.Event()
    original = supervisor.manager._stop_locked

    async def stop_then_wait(**kwargs: Any) -> None:
        await original(**kwargs)
        paused.set()
        await resume.wait()

    supervisor.manager._stop_locked = (  # type: ignore[method-assign]
        stop_then_wait
    )
    return paused, resume


def test_a_peer_is_refused_throughout_a_switch(
    lease_file: Path,
) -> None:
    """A switch used to release the claim as it finalized its old
    worker, so a peer arriving before it claimed again got the
    machine, and the switch, its model already evicted, was then
    refused. The user paid for an eviction and got nothing."""
    switching = Supervisor(lease_file)
    peer = Supervisor(lease_file)

    async def sequence() -> Tuple[Any, Any]:
        await switching.load(GPU_MODEL, "cuda")
        paused, resume = _pause_after_stop(switching)
        switch = asyncio.create_task(
            switching.load(CPU_MODEL, "cpu")
        )
        await paused.wait()
        refused = None
        try:
            await peer.load(GPU_MODEL, "cuda")
        except ActivationRefused as exc:
            refused = exc
        resume.set()
        (switched,) = await asyncio.gather(
            switch, return_exceptions=True
        )
        return refused, switched

    refused, switched = asyncio.run(sequence())

    assert refused is not None, "the peer took the machine mid-switch"
    assert switched is None, f"the switch was refused: {switched}"
    assert switching.manager.is_serving(CPU_MODEL)
    assert switching.holds_lease is True
    assert peer.processes == []


def test_a_switch_that_cannot_launch_frees_the_claim(
    lease_file: Path,
) -> None:
    """The other half of keeping the claim through a switch. A switch
    that evicts its model and then cannot start the next one leaves
    nothing resident, and holding the machine for nothing would lock
    every other launcher out until this one exits."""
    only = Supervisor(lease_file)
    peer = Supervisor(lease_file)

    async def sequence() -> None:
        await only.load(GPU_MODEL, "cuda")
        only.fail_spawn = True
        with pytest.raises(OSError):
            await only.load(CPU_MODEL, "cpu")
        await peer.load(GPU_MODEL, "cuda")

    asyncio.run(sequence())

    assert only.holds_lease is False
    assert peer.holds_lease is True
    assert peer.manager.is_serving(GPU_MODEL)


# -- handing it over --


def test_stopping_hands_the_claim_on(lease_file: Path) -> None:
    """Unloading in one window is what frees the other, which is the
    instruction the refusal message gives."""
    first = Supervisor(lease_file)
    second = Supervisor(lease_file)

    async def sequence() -> None:
        await first.load(GPU_MODEL, "cuda")
        await first.manager.stop()
        await second.load(GPU_MODEL, "cuda")

    asyncio.run(sequence())

    assert first.holds_lease is False
    assert second.holds_lease is True
    assert second.manager.is_serving(GPU_MODEL)


def test_a_superseded_finalize_does_not_give_the_claim_away(
    lease_file: Path,
) -> None:
    """The hazard hidden in the terminal path.

    ``_finalize`` guards its state clearing because a slow termination
    can finish after a newer activation has taken over the manager's
    fields. The claim needs the same guard: released unconditionally,
    a lingering finalize would hand the machine to a peer while this
    supervisor still has a worker coming up.
    """
    only = Supervisor(lease_file)

    async def sequence() -> None:
        await only.load(GPU_MODEL, "cuda")
        superseded = only.processes[0]
        # A second worker takes over the manager's fields, the way a
        # switch does, leaving the first one's finalize to arrive
        # late.
        await only.load(CPU_MODEL, "cpu")
        await only.manager._finalize(superseded, error=None)

    asyncio.run(sequence())

    assert only.holds_lease is True
    assert only.manager.is_serving(CPU_MODEL)
