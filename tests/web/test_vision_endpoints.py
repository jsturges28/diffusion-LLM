"""The tokeniser view's two endpoints answer, and refuse, correctly.

Strategy: drive both through `TestClient`, checking the contract the
page depends on and every way a request can be wrong. Passing proves a
reader gets the checkpoint's own geometry, that a bad request is
refused with a reason rather than a default, and that an uncached
encoder is reported as a temporary condition rather than a crash.

The arithmetic itself is not retested here; `test_vision_geometry.py`
holds that against the library. What this file owns is the boundary:
what crosses the wire, and what happens when the input is hostile.

One property is worth stating because it is the reason this feature
sits in the supervisor at all: **neither endpoint touches the model
manager.** No activation, no residency claim, no eviction. A reader
comparing encoders must not cost someone their loaded model, and the
test at the bottom checks that by asserting the resident model is
untouched across a request.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.inference.vision_encoders import ENCODERS, declared
from src.web.server import app

GEOMETRY = "/api/vision/geometry"
LIST = "/api/vision/encoders"


@pytest.fixture()
def client():
    with TestClient(app) as ready:
        yield ready


def _geometry(client, **params: Any):
    return client.get(GEOMETRY, params=params)


def _cached_encoder_id() -> str:
    """An encoder whose config is on disk, or skip.

    The geometry endpoint has to read a real configuration, and the
    suite does not download. Skipping is honest: the endpoint's
    refusals are all tested without one.
    """
    from src.inference.vision_encoders import is_cached

    for encoder in declared():
        if is_cached(encoder):
            return encoder.id
    pytest.skip("no encoder configuration is cached")


# -- what the list says --


def test_the_list_names_every_declared_encoder(client) -> None:
    body = client.get(LIST).json()

    listed = {item["id"] for item in body["encoders"]}
    assert listed == set(ENCODERS)


def test_the_list_says_whether_each_is_cached(client) -> None:
    """The page offers a download from this, so the field has to be
    present and boolean rather than absent when false."""
    body = client.get(LIST).json()

    for item in body["encoders"]:
        assert isinstance(item["cached"], bool)


def test_the_list_carries_the_pin(client) -> None:
    """A reader writing a number down should be able to see which
    commit produced it."""
    body = client.get(LIST).json()

    for item in body["encoders"]:
        assert len(item["revision"]) == 40


def test_the_list_keeps_declaration_order(client) -> None:
    """Cheapest first, so the encoder a small machine can run is the
    one a reader meets first."""
    body = client.get(LIST).json()

    ids = [item["id"] for item in body["encoders"]]
    assert ids == [encoder.id for encoder in declared()]


# -- what the geometry says --


def test_a_good_request_answers(client) -> None:
    response = _geometry(
        client, encoder=_cached_encoder_id(), width=1920, height=1080
    )

    assert response.status_code == 200


def test_the_answer_carries_both_halves(client) -> None:
    """The page draws from the encoder's fixed geometry and this
    image's placement, and needs them apart: the first labels the
    legend, the second positions the grid."""
    body = _geometry(
        client, encoder=_cached_encoder_id(), width=1920, height=1080
    ).json()

    assert set(body) == {"encoder", "image"}
    for key in ("tile", "patch", "scale", "patch_side",
                "unseen_edge", "tokens_per_tile"):
        assert key in body["encoder"], key
    for key in ("fitted_width", "fitted_height", "tile_rows",
                "tile_cols", "total_tokens", "aspect_changed"):
        assert key in body["image"], key


def test_the_tile_grid_covers_the_fitted_image(client) -> None:
    """The property that makes a drawn grid truthful, checked at the
    boundary as well as in the geometry's own tests, because the page
    trusts these two numbers together."""
    body = _geometry(
        client, encoder=_cached_encoder_id(), width=1920, height=1080
    ).json()

    tile = body["encoder"]["tile"]
    image = body["image"]
    assert image["tile_rows"] * tile == image["fitted_height"]
    assert image["tile_cols"] * tile == image["fitted_width"]


def test_the_total_counts_the_thumbnail(client) -> None:
    body = _geometry(
        client, encoder=_cached_encoder_id(), width=1024, height=768
    ).json()

    per_tile = body["encoder"]["tokens_per_tile"]
    image = body["image"]
    assert image["total_tokens"] == (
        image["tile_count"] + 1
    ) * per_tile


def test_resolution_is_free_and_shape_costs(client) -> None:
    """The page's headline, checked through the wire so the claim a
    reader sees is the claim the server makes."""
    encoder_id = _cached_encoder_id()

    icon = _geometry(
        client, encoder=encoder_id, width=64, height=64
    ).json()["image"]["total_tokens"]
    square = _geometry(
        client, encoder=encoder_id, width=1000, height=1000
    ).json()["image"]["total_tokens"]
    banner = _geometry(
        client, encoder=encoder_id, width=1500, height=300
    ).json()["image"]["total_tokens"]

    assert icon == square
    assert banner < square


def test_the_two_encoders_disagree_about_the_same_image(
    client,
) -> None:
    """The reason both are declared. If they answered alike the page
    would have nothing to compare."""
    from src.inference.vision_encoders import is_cached

    ready = [
        encoder.id for encoder in declared() if is_cached(encoder)
    ]
    if len(ready) < 2:
        pytest.skip("both encoder configs are needed to compare")

    totals = {
        encoder_id: _geometry(
            client, encoder=encoder_id, width=1024, height=1024
        ).json()["image"]["total_tokens"]
        for encoder_id in ready
    }

    assert len(set(totals.values())) == len(totals), totals


# -- and what it refuses --


def test_an_unknown_encoder_is_a_404(client) -> None:
    response = _geometry(client, encoder="nope", width=10, height=10)

    assert response.status_code == 404
    assert "nope" in response.json()["error"]


def test_a_404_lists_what_is_known(client) -> None:
    """So a caller can correct itself, and a reader reading the
    response learns the ids rather than guessing."""
    body = _geometry(
        client, encoder="nope", width=10, height=10
    ).json()

    assert set(body["known"]) == set(ENCODERS)


@pytest.mark.parametrize("width,height", [
    (0, 100), (100, 0), (-1, 100), (100, -1),
    (200_000, 100), (100, 200_000),
])
def test_a_dimension_outside_the_bounds_is_a_400(
    client, width: int, height: int
) -> None:
    """Both directions and both axes. Zero and negative because a
    browser reporting `naturalWidth` on a broken image sends zero, and
    the absurd end because the arithmetic stops describing anything
    there."""
    response = _geometry(
        client, encoder=_cached_encoder_id(),
        width=width, height=height,
    )

    assert response.status_code == 400
    assert "outside" in response.json()["error"]


def test_the_boundary_itself_is_accepted(client) -> None:
    """The other side of the bound, so it is not simply tight enough
    to refuse everything."""
    encoder_id = _cached_encoder_id()

    for width, height in ((1, 1), (100_000, 100_000)):
        response = _geometry(
            client, encoder=encoder_id, width=width, height=height
        )
        assert response.status_code == 200, (width, height)


@pytest.mark.parametrize("params", [
    {"width": 100, "height": 100},
    {"encoder": "smolvlm-2b", "height": 100},
    {"encoder": "smolvlm-2b", "width": 100},
])
def test_a_missing_parameter_is_a_422(
    client, params: Dict[str, Any]
) -> None:
    """FastAPI's own validation, asserted so a later signature change
    cannot make a missing parameter default to something."""
    assert client.get(GEOMETRY, params=params).status_code == 422


@pytest.mark.parametrize("value", ["wide", "12.5", ""])
def test_a_non_integer_dimension_is_a_422(client, value: str) -> None:
    response = client.get(GEOMETRY, params={
        "encoder": "smolvlm-2b", "width": value, "height": "100",
    })

    assert response.status_code == 422


def test_an_uncached_encoder_reports_a_503(
    client, monkeypatch
) -> None:
    """The offline first run. A temporary condition the page can
    render, not a 500: the request was well formed and will work once
    the config is on disk."""
    from src.inference.vision_encoders import EncoderUnavailable
    from src.web import server

    def refuse(encoder, **kwargs):
        raise EncoderUnavailable("nothing cached")

    monkeypatch.setattr(server, "vision_load_geometry", refuse)

    response = _geometry(
        client, encoder="smolvlm-2b", width=100, height=100
    )

    assert response.status_code == 503
    assert "nothing cached" in response.json()["error"]


# -- and what it must not disturb --


def test_the_resident_model_is_untouched(client) -> None:
    """Why this lives in the supervisor and not in a worker.

    Inspecting an image must not evict anyone's model or take the
    residency lease, so a reader can compare encoders while a
    generation is running.
    """
    from src.web.server import manager

    before = (manager.active_id, manager.active_device)

    _geometry(
        client, encoder=_cached_encoder_id(), width=800, height=600
    )
    client.get(LIST)

    assert (manager.active_id, manager.active_device) == before
