"""A worker says where it actually is, and every run carries it.

Strategy: two halves, both without a model. First, `FrameStreamer`
against a stub socket, checking that a terminal frame acquires the
envelope and an ordinary frame does not, through both of the paths a
`done` can leave by. Second, `provenance_envelope` against a stub
backend, checking it reports the loaded placement rather than the
requested one.

What passing proves is the missing link in `DATA-04`. The supervisor
records the device it *asked* for. Two of the three backends fall back
to CPU when CUDA was requested and is unavailable, and nothing
downstream could tell: a run that took four minutes on a CPU was saved
as a GPU run, which is a misleading record precisely in the fields
that exist to make timings comparable.

The other half, that the save prefers the run's envelope over the
supervisor's current state, is in `tests/web/test_run_provenance.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from src.backends.protocol import ModelCapabilities, ModelInfo
from src.backends.worker_base import (
    FrameStreamer,
    library_versions,
    provenance_envelope,
)
from src.inference.hf_download import revision_from_snapshot

# Any full sha; nothing here depends on it resolving.
SHA = "08b83a6feb34df1a6011b80c3c00c7563e963b07"

ENVELOPE: Dict[str, Any] = {
    "model_id": "stub",
    "device": "cpu",
    "versions": {"torch": "0.0.0"},
    "tokenizer": {},
}


class _StubSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.sent.append(payload)


async def _frames(
    *frames: Dict[str, Any],
) -> AsyncGenerator[Dict[str, Any], None]:
    for frame in frames:
        yield frame


def _streamer(socket: _StubSocket) -> FrameStreamer:
    return FrameStreamer(
        socket,  # type: ignore[arg-type]
        provenance=lambda: dict(ENVELOPE),
    )


# -- what a terminal frame carries --


def test_a_done_frame_carries_the_envelope() -> None:
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.run(
            _frames(
                {"type": "frame", "text": "a"},
                {"type": "done", "final_text": "a"},
            ),
            0.0,
        )
    )

    assert socket.sent[-1]["provenance"] == ENVELOPE


def test_an_ordinary_frame_does_not() -> None:
    """Once per run, not once per step. A diffusion run emits
    hundreds of frames and the envelope re-describes the same
    worker every time."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.run(
            _frames(
                {"type": "frame", "text": "a"},
                {"type": "done", "final_text": "a"},
            ),
            0.0,
        )
    )

    assert "provenance" not in socket.sent[0]


def test_a_worker_sent_done_carries_it_too() -> None:
    """The path a guided edit takes. Its terminal frame is built by
    the worker rather than the sampler, and it used to be assembled
    by hand at each call site with whatever fields that site
    remembered."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.send_done(
            {"type": "done", "final_text": "stopped here"}, 0.0
        )
    )

    assert socket.sent[-1]["provenance"] == ENVELOPE
    assert socket.sent[-1]["final_text"] == "stopped here"


def test_a_worker_sent_done_is_timed_like_any_other() -> None:
    """The LLaDA site omitted elapsed and the DiffusionGemma one
    included it. Routing both through here settles it."""
    socket = _StubSocket()
    stream = _streamer(socket)

    asyncio.run(
        stream.send_done({"type": "done"}, 0.0)
    )

    assert isinstance(socket.sent[-1]["elapsed"], float)


def test_send_done_refuses_a_non_terminal_frame() -> None:
    """It stamps provenance, so sending an ordinary frame through
    it would attach the envelope to every step."""
    socket = _StubSocket()
    stream = _streamer(socket)

    with pytest.raises(AssertionError):
        asyncio.run(
            stream.send_done({"type": "frame"}, 0.0)
        )


def test_a_streamer_without_provenance_stamps_nothing() -> None:
    """The default. Tests and any future caller that has no backend
    to attest for must still be able to stream."""
    socket = _StubSocket()
    stream = FrameStreamer(socket)  # type: ignore[arg-type]

    asyncio.run(
        stream.run(_frames({"type": "done"}), 0.0)
    )

    assert "provenance" not in socket.sent[-1]


# -- what the envelope says --


class _StubBackend:
    """A loaded backend, minus the model."""

    def __init__(
        self,
        effective_device: str,
        loaded_revision: Optional[str] = None,
    ) -> None:
        self.model_info = ModelInfo(
            id="stub",
            display_name="Stub",
            param_specs=[],
            # The axes are required, so even a stub says what it is;
            # provenance reads none of them.
            capabilities=ModelCapabilities(
                family="diffusion",
                generation_shape="iterative_canvas",
                input_mode="chat",
                supported_devices=("cuda", "cpu"),
            ),
            worker_module="none",
            venv_python="none",
            checkpoint="org/stub-checkpoint",
        )
        self.effective_device = effective_device
        self.loaded_revision = loaded_revision
        self.tokenizer = None
        self.model = None


def test_the_envelope_reports_where_the_model_landed() -> None:
    """The finding in one assertion. The supervisor would say cuda
    here, because cuda is what it asked for."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert envelope["device"] == "cpu"


def test_a_backend_that_has_not_loaded_says_unknown() -> None:
    """Not a device. A blank must not read as a placement, or the
    default would quietly become whatever "" happens to compare
    equal to downstream."""
    backend = _StubBackend("cpu")
    backend.effective_device = None

    envelope = provenance_envelope(
        backend  # type: ignore[arg-type]
    )

    assert envelope["device"] == "unknown"


def test_the_envelope_names_the_model_and_checkpoint() -> None:
    envelope = provenance_envelope(
        _StubBackend("cuda")  # type: ignore[arg-type]
    )

    assert envelope["model_id"] == "stub"
    assert envelope["checkpoint"] == "org/stub-checkpoint"


def test_the_envelope_reports_this_worker_s_libraries() -> None:
    """Read in the worker, not the supervisor: the three venvs hold
    deliberately incompatible versions, so the supervisor's own
    imports describe the wrong environment."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert envelope["versions"] == library_versions()
    assert "torch" in envelope["versions"]


def test_the_envelope_omits_an_unreadable_context_window() -> None:
    """Omitted rather than null, matching /health, so a consumer's
    "is there a ceiling" test stays a plain key check."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert "context_length" not in envelope


# -- the commit, which is the other half of "which model" --


def test_the_envelope_names_the_commit_it_loaded() -> None:
    """The checkpoint name alone does not identify weights. This is
    the field that makes the name mean one thing, and it is the
    worker's to report because only the worker saw the files."""
    envelope = provenance_envelope(
        _StubBackend("cpu", SHA)  # type: ignore[arg-type]
    )

    assert envelope["revision"] == SHA
    # Beside the name, not instead of it: the name is what a reader
    # recognises, and the pair is what makes the record usable.
    assert envelope["checkpoint"] == "org/stub-checkpoint"


def test_a_local_checkpoint_omits_the_commit() -> None:
    """Omitted rather than empty, for the same reason the window is:
    a local directory has no commit, and a blank string would be a
    value a reader has to learn to ignore."""
    envelope = provenance_envelope(
        _StubBackend("cpu")  # type: ignore[arg-type]
    )

    assert "revision" not in envelope


# -- reading the commit back off the cache layout --


def test_a_snapshot_path_yields_its_commit() -> None:
    """The reason no network call is needed: the cache already names
    the resolved commit in the path it hands back."""
    resolved = revision_from_snapshot(
        f"/cache/models--org--model/snapshots/{SHA}"
    )

    assert resolved == SHA


def test_a_trailing_separator_does_not_hide_the_commit() -> None:
    resolved = revision_from_snapshot(
        f"/cache/models--org--model/snapshots/{SHA}/"
    )

    assert resolved == SHA


def test_a_branch_resolves_to_the_commit_behind_it() -> None:
    """Why the worker reads this back instead of recording what it
    asked for.

    A revision may name a branch or a tag; the cache lays the result
    out under the commit that name resolved to. Recording the request
    would then attest "main", which is not a fact about the run: it
    describes wherever the branch points when someone reads it.
    """
    resolved = revision_from_snapshot(
        f"/cache/models--org--model/snapshots/{SHA}"
    )

    assert resolved == SHA
    assert resolved != "main"


@pytest.mark.parametrize(
    "path",
    [
        # The local quantized checkpoint, which is the real case.
        "/home/user/models/diffusiongemma-nf4",
        # A cache path stopping one level short.
        "/cache/models--org--model/snapshots",
        # Nothing to read at all.
        "",
        "/",
    ],
)
def test_a_path_that_names_no_commit_reads_as_none(
    path: str,
) -> None:
    """Negative space. Returning something for these would put a
    directory name in a field that means "commit"."""
    assert revision_from_snapshot(path) is None
