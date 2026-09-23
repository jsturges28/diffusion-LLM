"""Every signal says what it measures and what it varies over.

Strategy: read the real registry and the real provenance envelope,
since the point of the manifest is that there is one description and
it is the one shipped. Then the four shapes the finding's Verification
clause names, each against a real channel rather than an invented one,
plus a fifth the views cannot draw.

What passing proves: no view has to infer a signal's shape from where
the value is stored. Confidence and entropy are both floats on a token
record, so location alone could not tell "one value per position, the
same in every frame" from "a value that changes every denoising step".
Analytics resolved that by reading the final frame, which is correct
for an autoregressive run and silently wrong for a diffusion
trajectory: it showed whatever the last step happened to hold.

The axes-versus-location split is the property under test. Two
channels here share a location and differ only in axes, and if that
distinction were dropped the manifest would describe storage, which
the file layout already does.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from src.backends.protocol import AXES, SignalChannel
from src.backends.protocol import ModelInfo
from src.backends.registry import DGEMMA, LLADA, REGISTRY, SMOLLM3

# The four shapes the finding asks for fixtures over, and the real
# channel that has each. Named so a failure says which shape broke.
SHAPES: Dict[str, Tuple[str, ...]] = {
    "one value per position": ("position",),
    "position by frame": ("frame", "position"),
    "one value per frame": ("frame",),
}


def _channels(model: ModelInfo) -> Dict[str, SignalChannel]:
    return {
        channel.name: channel
        for channel in model.capabilities.signals
    }


# -- every model describes its signals --


@pytest.mark.parametrize(
    "model", sorted(REGISTRY.values(), key=lambda m: m.id),
    ids=lambda m: str(m.id),
)
def test_every_model_declares_its_signals(
    model: ModelInfo,
) -> None:
    channels = _channels(model)

    assert channels, f"{model.id} declares none"
    # Confidence and entropy are the two every model emits; a model
    # that stopped declaring one would still emit the float, and the
    # overlay would go back to guessing.
    assert "confidence" in channels
    assert "entropy" in channels


@pytest.mark.parametrize(
    "model", sorted(REGISTRY.values(), key=lambda m: m.id),
    ids=lambda m: str(m.id),
)
def test_every_axis_is_a_declared_one(
    model: ModelInfo,
) -> None:
    """Negative space on the vocabulary. A typo in an axis name would
    otherwise read as a shape no view supports, which is a real state
    and would mask the mistake as a missing feature."""
    for channel in _channels(model).values():
        assert channel.axes, f"{channel.name} varies over nothing"
        for axis in channel.axes:
            assert axis in AXES, f"{channel.name}: {axis!r}"


@pytest.mark.parametrize(
    "model", sorted(REGISTRY.values(), key=lambda m: m.id),
    ids=lambda m: str(m.id),
)
def test_no_channel_is_declared_twice(
    model: ModelInfo,
) -> None:
    names = [
        channel.name for channel in model.capabilities.signals
    ]

    assert len(names) == len(set(names))


# -- the four shapes, each on a real channel --


def test_a_diffusion_position_varies_by_frame() -> None:
    """The shape that did not previously exist as a description.
    Every LLaDA position is re-decided at every denoising step, so its
    entropy is a trajectory and not a constant."""
    channels = _channels(LLADA)

    assert channels["entropy"].axes == SHAPES["position by frame"]
    assert channels["confidence"].axes == SHAPES["position by frame"]


def test_an_autoregressive_position_is_decided_once() -> None:
    """The pair to the test above, and the whole reason the axes are
    declared rather than inferred: this entropy lives in exactly the
    same place as LLaDA's and means something different."""
    channels = _channels(SMOLLM3)

    per_position = SHAPES["one value per position"]
    assert channels["entropy"].axes == per_position
    assert channels["entropy"].location == "token_record"
    assert (
        channels["entropy"].location
        == _channels(LLADA)["entropy"].location
    )
    assert (
        channels["entropy"].axes
        != _channels(LLADA)["entropy"].axes
    )


def test_a_per_frame_scalar_is_described() -> None:
    """`mean_conf` has been computed by all three samplers and
    persisted per frame since long before this manifest, entirely
    undescribed. It is the "one value per frame" shape."""
    for model in (LLADA, DGEMMA, SMOLLM3):
        channel = _channels(model)["mean_confidence"]
        assert channel.axes == SHAPES["one value per frame"]
        assert channel.location == "frame_scalar"
        assert channel.key == "mean_conf"


def test_an_opt_in_channel_says_it_is_optional() -> None:
    """The fourth shape is an absence: a channel a run could have
    captured and did not. That is a different fact from a model that
    cannot produce it, and only the declaration can tell them
    apart."""
    alternatives = _channels(SMOLLM3)["alternatives"]

    assert alternatives.capture == "opt_in"
    assert alternatives.location == "sidecar"
    # And the diffusion models do not claim it at all.
    assert "alternatives" not in _channels(LLADA)
    assert "alternatives" not in _channels(DGEMMA)


def test_everything_else_is_always_captured() -> None:
    """The negative of the test above. A channel wrongly marked
    opt_in would have its absence excused on runs that should always
    carry it."""
    for model in (LLADA, DGEMMA, SMOLLM3):
        for channel in _channels(model).values():
            if channel.name == "alternatives":
                continue
            assert channel.capture == "always", channel.name


# -- units, so a scale is not a guess --


def test_entropy_is_in_nats_everywhere() -> None:
    """Two units for one quantity would make the Analytics colour
    scale a guess, and the autoregressive sampler has reported nats
    since entropy first appeared there."""
    for model in (LLADA, DGEMMA, SMOLLM3):
        assert _channels(model)["entropy"].unit == "nats"


def test_confidence_is_a_probability_everywhere() -> None:
    for model in (LLADA, DGEMMA, SMOLLM3):
        assert _channels(model)["confidence"].unit == "probability"


# -- the keys point at where the value really is --


def test_the_declared_keys_are_the_ones_written() -> None:
    """A description pointing at the wrong key is worse than none:
    a reader would look up a field that is always absent and conclude
    the channel was never captured."""
    for model in (LLADA, DGEMMA, SMOLLM3):
        channels = _channels(model)
        assert channels["confidence"].key == "c"
        assert channels["entropy"].key == "e"
        assert channels["mean_confidence"].key == "mean_conf"


# -- what travels with a run --


def test_the_envelope_carries_the_declared_channels() -> None:
    """A saved run outlives the registry entry that made it, so the
    description has to travel with the frames rather than be looked
    up later from whatever model happens to be resident."""
    from tests.backends.test_worker_provenance import _StubBackend
    from src.backends.worker_base import provenance_envelope

    backend = _StubBackend("cpu")
    backend.model_info = LLADA

    envelope = provenance_envelope(backend)

    assert "signals" in envelope
    names = {channel["name"] for channel in envelope["signals"]}
    assert "entropy" in names
    for channel in envelope["signals"]:
        assert "axes" in channel
        assert "unit" in channel


def test_a_model_declaring_none_omits_the_key() -> None:
    """Omitted rather than empty, matching every other optional field
    on the envelope, so a reader's key check is enough and an older
    worker's runs read as "infer as before"."""
    from tests.backends.test_worker_provenance import _StubBackend
    from src.backends.worker_base import provenance_envelope

    envelope = provenance_envelope(_StubBackend("cpu"))

    assert "signals" not in envelope


# -- a shape no view here can draw --


def test_a_canvas_wide_channel_is_expressible() -> None:
    """The fifth case, which no model ships. `canvas` exists so a
    channel with neither a frame nor a position axis can still be
    described, and the views can then say they have no chart for it
    instead of drawing one from nothing."""
    channel = SignalChannel(
        name="write_intensity",
        unit="nats",
        axes=("canvas",),
        location="frame_scalar",
        key="write_intensity",
        capture="opt_in",
    )

    assert channel.axes == ("canvas",)
    assert "canvas" in AXES


def test_a_budget_is_recorded_only_where_it_matters() -> None:
    """Present so a channel with a real budget has somewhere to
    declare it. None of the four shipped channels needs one: entropy
    is a float per token record, where per-frame candidate sets run to
    millions of records at the bounds the registry allows."""
    for model in (LLADA, DGEMMA, SMOLLM3):
        for channel in _channels(model).values():
            assert channel.budget_records is None

    budgeted = SignalChannel(
        name="candidates",
        unit="probability",
        axes=("frame", "position"),
        location="sidecar",
        key="candidates",
        capture="opt_in",
        budget_records=102_400,
    )

    assert budgeted.budget_records == 102_400
