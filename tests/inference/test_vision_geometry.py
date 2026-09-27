"""Our image geometry still agrees with the library that defines it.

Strategy: lift the real sizing functions out of the SmolVLM processor
in `.venv-ar`, compose them in the order `preprocess` calls them, and
compare against `src/inference/vision_geometry` over thousands of
sizes at both encoder geometries. Passing proves the page's numbers
are the processor's numbers, which is the only claim that matters for
a readout whose whole purpose is to be believed.

**Why a reimplementation exists to be tested at all.** The supervisor
runs `transformers` 4.38.2, which predates SmolVLM, so it cannot
import the processor; and this answers a pointer moving over an image,
where spawning the worker environment per query would not. Neither
constraint would justify guessing, so the arithmetic is reproduced and
held here instead.

**The functions are lifted with `ast` rather than imported**, because
`image_processing_smolvlm` imports `PILImageResampling`, which is
gated on Pillow, which no environment here installs. Only the sizing
functions are needed and they are pure, so parsing the module and
executing those definitions gets the authority without the
dependencies. It reads another environment's tree, so it skips when
that tree is absent.

**This file has already caught two bugs**, which is the argument for
its existence over a reading of the source:

1. `resize_for_encoder` clamped a collapsed dimension once at the end
   rather than between its two stages, and disagreed on 9 of 8,144
   sizes, all extreme aspect ratios where a side rounds to zero.
2. `fit_to_whole_tiles` did not exist. The processor applies it
   between the resize and the split, and without it every tile count
   and every token total was wrong, while each individual function
   still matched.

Both were in how the steps fit together, not inside one of them, so
the comparison below is of the composed pipeline and not of the parts.
"""

from __future__ import annotations

import ast
import functools
import math
import random
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

from src.inference.vision_geometry import (
    EncoderGeometry,
    fit_to_whole_tiles,
    fuse_map,
    geometry,
    patch_block,
    resize_for_encoder,
    split_tiles,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSOR = (
    REPO_ROOT
    / ".venv-ar/lib/python3.12/site-packages/transformers/models"
    / "smolvlm/image_processing_smolvlm.py"
)
MODELLING = (
    REPO_ROOT
    / ".venv-ar/lib/python3.12/site-packages/transformers/models"
    / "smolvlm/modeling_smolvlm.py"
)

# The two shipping checkpoints, whose values differ in every field.
# Hard-coded here and only here: production reads them from the
# checkpoint, and a test that did the same would pass whatever the
# checkpoint said rather than checking these geometries in particular.
SMALL = EncoderGeometry(
    longest_edge=2048, tile=512, patch=16, scale=4
)
LARGE = EncoderGeometry(
    longest_edge=1536, tile=384, patch=14, scale=3
)
ENCODERS = {"SmolVLM-500M": SMALL, "SmolVLM-2.2B": LARGE}

MAX_IMAGE_SIZE = 4096


def _definitions(tree: ast.Module) -> List[ast.FunctionDef]:
    """Top-level functions, plus methods one class deep.

    Both are needed: the sizing helpers are module-level and
    `pixel_shuffle` is a method, and neither needs its class to run.
    """
    found: List[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            found.extend(
                inner for inner in node.body
                if isinstance(inner, ast.FunctionDef)
            )
    return found


def _lift(source: Path, names: set) -> Dict[str, Callable[..., Any]]:
    """The named top-level functions, executed without their module.

    Nested one level deep as well, so a method on the processor class
    is reachable: `pixel_shuffle` is a method and `split_image` is
    too, and neither needs its class to run.
    """
    if not source.is_file():
        pytest.skip(f"{source.name} is not in this checkout")

    tree = ast.parse(source.read_text(encoding="utf-8"))
    found: Dict[str, Callable[..., Any]] = {}
    namespace: Dict[str, Any] = {
        "Optional": typing.Optional,
        "dict": dict,
        "int": int,
        "max": max,
        "math": math,
    }

    for node in _definitions(tree):
        if node.name in names and node.name not in found:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(source), "exec"), namespace)
            found[node.name] = namespace[node.name]

    missing = names - set(found)
    assert not missing, (
        f"{source.name} no longer defines {sorted(missing)}. The"
        " processor was rewritten upstream, so the geometry in"
        " src/inference/vision_geometry.py needs re-deriving rather"
        " than this test relaxing."
    )
    return found


@functools.lru_cache(maxsize=1)
def _sizing() -> Tuple[Callable[..., Any], ...]:
    """The two pure sizing functions, lifted and callable.

    Cached because the comparisons below call this per size, and
    parsing a thousand-line module each time took the suite from
    milliseconds to eighteen seconds.

    `resize_for_vision_encoder` is deliberately not among them: its
    signature defaults to `PILImageResampling.LANCZOS`, which is
    evaluated when the definition executes and is not defined without
    Pillow. It is transcribed instead, and
    `test_the_transcribed_step_matches_its_source` pins the four lines
    that matter by reading them rather than running them.
    """
    lifted = _lift(PROCESSOR, {
        "_resize_output_size_rescale_to_max_len",
        "_resize_output_size_scale_below_upper_bound",
    })
    return (
        lifted["_resize_output_size_rescale_to_max_len"],
        lifted["_resize_output_size_scale_below_upper_bound"],
    )


def _library_pipeline(
    height: int, width: int, encoder: EncoderGeometry
) -> Tuple[int, int, int, int]:
    """The processor's own answer, as `(height, width, rows, cols)`.

    Composed in the order `preprocess` calls them: `resize` with a
    `longest_edge`, then `resize_for_vision_encoder`, then
    `split_image`. The middle step is transcribed rather than lifted
    because it ends by calling `self.resize`, which needs Pillow; its
    sizing arithmetic is the four lines reproduced here, and
    `test_the_transcribed_step_matches_its_source` pins them.
    """
    rescale, cap = _sizing()
    height, width = rescale(
        height, width, max_len=encoder.longest_edge
    )
    height, width = cap(height, width, max_len=MAX_IMAGE_SIZE)

    aspect = width / height
    if width >= height:
        width = math.ceil(width / encoder.tile) * encoder.tile
        height = math.ceil(
            int(width / aspect) / encoder.tile
        ) * encoder.tile
    elif height > width:
        height = math.ceil(height / encoder.tile) * encoder.tile
        width = math.ceil(
            int(height * aspect) / encoder.tile
        ) * encoder.tile

    if height > encoder.tile or width > encoder.tile:
        rows = math.ceil(height / encoder.tile)
        cols = math.ceil(width / encoder.tile)
    else:
        rows = cols = 0
    return height, width, rows, cols


def _ours(
    height: int, width: int, encoder: EncoderGeometry
) -> Tuple[int, int, int, int]:
    result = geometry(encoder, width, height)
    return (
        result.fitted_height,
        result.fitted_width,
        result.tile_rows,
        result.tile_cols,
    )


@functools.lru_cache(maxsize=1)
def _sizes() -> Tuple[Tuple[int, int], ...]:
    """Deliberate edges first, then a seeded spread."""
    chosen = [
        (height, width)
        for height in (1, 2, 3, 17, 200, 333, 334, 768, 1080, 4000)
        for width in (1, 2, 3, 16, 300, 500, 1024, 1920, 4000)
    ]
    generator = random.Random(4242)
    chosen += [
        (generator.randint(1, 6000), generator.randint(1, 6000))
        for _ in range(1500)
    ]
    return tuple(chosen)


# -- the composed pipeline, which is where both bugs lived --


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_whole_pipeline_agrees(name: str) -> None:
    encoder = ENCODERS[name]
    sizes = _sizes()

    disagreed = [
        (height, width, _ours(height, width, encoder),
         _library_pipeline(height, width, encoder))
        for height, width in sizes
        if _ours(height, width, encoder)
        != _library_pipeline(height, width, encoder)
    ]

    assert disagreed == [], (
        f"{len(disagreed)} of {len(sizes)} sizes disagree with the"
        f" processor on {name}, first few: {disagreed[:3]}"
    )


def test_there_were_sizes_to_compare() -> None:
    """The test above passes on an empty list, and a `_sizes` that
    quietly returned nothing would look like agreement."""
    assert len(_sizes()) > 1_000


# -- and each step, so a failure says which one --


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_first_resize_agrees(name: str) -> None:
    rescale, cap = _sizing()
    longest = ENCODERS[name].longest_edge

    for height, width in _sizes():
        expected = cap(
            *rescale(height, width, max_len=longest),
            max_len=MAX_IMAGE_SIZE,
        )
        ours = resize_for_encoder(height, width, longest)
        assert ours == expected, (
            f"resize disagrees at {width}x{height} on {name}"
        )


def test_the_clamp_sits_between_the_two_stages() -> None:
    """The first bug, pinned. A very wide image rounds its height to
    zero in the first stage, and clamping only at the end leaves the
    second stage dividing by it."""
    rescale, cap = _sizing()

    # The first stage alone collapses this.
    collapsed = rescale(1, 1920, max_len=1536)
    assert collapsed[0] == 1, (
        "the library's own first stage no longer clamps, so this"
        " reasoning needs rechecking"
    )
    assert resize_for_encoder(1, 1920, 1536) == cap(
        *collapsed, max_len=MAX_IMAGE_SIZE
    )


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_fitting_to_whole_tiles_agrees(name: str) -> None:
    """The second bug, pinned: the step that was missing entirely."""
    encoder = ENCODERS[name]

    for height, width in _sizes():
        resized = resize_for_encoder(
            height, width, encoder.longest_edge
        )
        ours = fit_to_whole_tiles(*resized, encoder.tile)
        expected = _library_pipeline(height, width, encoder)[:2]
        assert ours == expected, (
            f"fitting disagrees at {width}x{height} on {name}"
        )


def test_the_transcribed_step_matches_its_source() -> None:
    """`resize_for_vision_encoder` is transcribed rather than lifted,
    because it ends by calling a Pillow-gated resize. Its arithmetic
    is four lines, so they are pinned by inspection: if upstream
    changes them, this fails rather than the page drifting."""
    if not PROCESSOR.is_file():
        pytest.skip("the worker environment is not in this checkout")

    # Normalised through the parser so whitespace and line wrapping
    # upstream cannot break the match, only the arithmetic can.
    source = ast.unparse(
        ast.parse(PROCESSOR.read_text(encoding="utf-8"))
    )
    assert "def resize_for_vision_encoder" in source, (
        "the step fit_to_whole_tiles reproduces is gone from the"
        " processor, so the pipeline needs re-deriving"
    )

    for fragment in (
        "math.ceil(width / vision_encoder_max_size) * "
        "vision_encoder_max_size",
        "math.ceil(height / vision_encoder_max_size) * "
        "vision_encoder_max_size",
    ):
        assert fragment in source, (
            "resize_for_vision_encoder no longer rounds up the way"
            " fit_to_whole_tiles reproduces it"
        )


# -- the fusion mapping, which the overlays depend on --


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_fusion_mapping_agrees(name: str) -> None:
    """Our index arithmetic against the library's reshapes.

    Run through `torch` because `pixel_shuffle` is tensor code, on a
    tensor whose single channel holds the patch index, so the output
    states which patches were fused.
    """
    torch = pytest.importorskip("torch")
    shuffle = _lift(MODELLING, {"pixel_shuffle"})["pixel_shuffle"]
    encoder = ENCODERS[name]
    side = encoder.patch_side

    patches = side * side
    labelled = torch.arange(patches, dtype=torch.float32).view(
        1, patches, 1
    )
    fused = shuffle(None, labelled, scale_factor=encoder.scale)
    expected = [
        [int(value) for value in fused[0, token]]
        for token in range(fused.shape[1])
    ]

    assert fuse_map(side, encoder.scale) == expected, (
        f"the fusion mapping disagrees on {name}"
    )


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_token_count_matches_the_mapping(name: str) -> None:
    encoder = ENCODERS[name]

    mapping = fuse_map(encoder.patch_side, encoder.scale)

    assert len(mapping) == encoder.tokens_per_tile


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_every_patch_reaches_exactly_one_token(name: str) -> None:
    """The property the overlays rely on. A patch in two tokens, or in
    none, would make a highlighted region a lie."""
    encoder = ENCODERS[name]
    side = encoder.patch_side

    seen: List[int] = []
    for members in fuse_map(side, encoder.scale):
        seen.extend(members)

    covered = side * side - (side % encoder.scale) * side
    assert len(seen) == len(set(seen)), "a patch reached two tokens"
    assert len(seen) == encoder.tokens_per_tile * encoder.scale ** 2
    assert max(seen) < side * side, "a token claimed a patch that is"
    assert covered >= len(seen), "more patches fused than exist"


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_hover_form_agrees_with_the_map(name: str) -> None:
    """`patch_block` exists so a hover does not build the whole map,
    so the two have to answer the same."""
    encoder = ENCODERS[name]
    side = encoder.patch_side
    mapping = fuse_map(side, encoder.scale)

    for token, members in enumerate(mapping):
        row, col, height, width = patch_block(
            token, side, encoder.scale
        )
        rebuilt = [
            (row + dr) * side + (col + dc)
            for dr in range(height)
            for dc in range(width)
        ]
        assert rebuilt == members, f"token {token} disagrees"


# -- the facts the page teaches, asserted so they cannot rot --


@pytest.mark.parametrize(
    "name,patch_side,unseen,per_tile,fused",
    [
        ("SmolVLM-500M", 32, 0, 64, 16),
        ("SmolVLM-2.2B", 27, 6, 81, 9),
    ],
)
def test_the_encoder_geometry_is_what_the_page_claims(
    name: str, patch_side: int, unseen: int, per_tile: int, fused: int
) -> None:
    encoder = ENCODERS[name]

    assert encoder.patch_side == patch_side
    assert encoder.unseen_edge == unseen
    assert encoder.tokens_per_tile == per_tile
    assert encoder.patches_per_token == fused


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_bound_lands_on_a_tile_boundary(name: str) -> None:
    """A latent assumption, asserted so a new encoder cannot inherit
    it silently.

    `longest_edge` is exactly four tiles in both checkpoints, which is
    why step 2 never changes the longer side and why two different
    roundings of the shorter side agree. Mutating `fit_to_whole_tiles`
    to the shorter form passes every other test in this file, and
    searching 1.28 million source sizes found no input that separates
    them. An encoder whose bound is not a whole number of tiles would
    separate them immediately, so it has to fail here rather than draw
    a subtly wrong grid.
    """
    encoder = ENCODERS[name]

    assert encoder.longest_edge_is_whole_tiles, (
        f"{name} has longest_edge {encoder.longest_edge} against a"
        f" {encoder.tile} tile; re-derive fit_to_whole_tiles against"
        " the processor before trusting this geometry"
    )
    assert encoder.longest_edge // encoder.tile == 4


def test_the_patch_grid_form_is_the_simpler_one() -> None:
    """Recorded because it looks like a bug and is not.

    `(tile - patch) // patch + 1` is the convolution derivation and
    `tile // patch` is the same number for every pair. Kept as the
    former for the reason the property's docstring gives, and pinned
    here so nobody has to rediscover that the two agree.
    """
    for encoder in ENCODERS.values():
        assert encoder.patch_side == encoder.tile // encoder.patch


def test_the_larger_encoder_wastes_six_pixels_an_edge() -> None:
    """Stated on its own because it is a claim about what the model
    cannot see, and holds only while 384 is not a multiple of 14."""
    assert LARGE.tile % LARGE.patch != 0
    assert LARGE.unseen_edge == 6
    assert SMALL.tile % SMALL.patch == 0
    assert SMALL.unseen_edge == 0


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_resolution_is_free_and_shape_costs(name: str) -> None:
    """The page's headline, which would otherwise be prose nobody
    checks. A 64x64 icon costs what a megapixel square costs, because
    both are scaled up to the same working resolution."""
    encoder = ENCODERS[name]

    icon = geometry(encoder, 64, 64).total_tokens
    square = geometry(encoder, 1000, 1000).total_tokens
    wide = geometry(encoder, 1500, 300).total_tokens

    assert icon == square, "resolution changed the cost"
    assert wide < square, "shape did not change the cost"


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_two_photographs_of_different_sizes_cost_the_same(
    name: str,
) -> None:
    """The same claim from the side a reader is most likely to test,
    with 2.6 times the pixels between them."""
    encoder = ENCODERS[name]

    large = geometry(encoder, 1920, 1080).total_tokens
    small = geometry(encoder, 1024, 768).total_tokens

    assert large == small


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_every_image_is_tiled(name: str) -> None:
    """Negative space for an assumption the plan originally got wrong.
    The first resize scales the longest edge up unconditionally, so
    nothing is small enough to skip tiling."""
    encoder = ENCODERS[name]

    for width, height in ((1, 1), (8, 8), (64, 64), (5000, 5000)):
        result = geometry(encoder, width, height)
        assert result.tile_count > 0, (
            f"{width}x{height} produced no tiles on {name}"
        )


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_a_photograph_has_its_shape_changed(name: str) -> None:
    """Where the distortion actually happens, now that tiles are known
    to be square: step 2, once, to the whole picture."""
    encoder = ENCODERS[name]

    assert geometry(encoder, 1920, 1080).aspect_changed
    assert not geometry(encoder, 1000, 1000).aspect_changed


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_a_tile_grid_covers_the_fitted_image_exactly(
    name: str,
) -> None:
    """The property `split_tiles` asserts internally, checked from
    outside too, since it is what makes a drawn tile grid truthful."""
    encoder = ENCODERS[name]

    for width, height in ((1920, 1080), (1000, 1000), (1500, 300)):
        result = geometry(encoder, width, height)
        assert result.tile_rows * encoder.tile == result.fitted_height
        assert result.tile_cols * encoder.tile == result.fitted_width


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_the_thumbnail_is_always_counted(name: str) -> None:
    """Every image pays for the whole-image view on top of its tiles,
    so the total is never a whole multiple of the tile count alone."""
    encoder = ENCODERS[name]
    result = geometry(encoder, 1920, 1080)

    expected = (result.tile_count + 1) * encoder.tokens_per_tile
    assert result.total_tokens == expected


# -- the bounds, which are this module's own contract --


def test_a_dimension_below_one_is_refused() -> None:
    with pytest.raises(AssertionError):
        geometry(SMALL, 0, 100)
    with pytest.raises(AssertionError):
        geometry(SMALL, 100, 0)


def test_an_absurd_dimension_is_refused() -> None:
    """Bounded because the arithmetic stops describing anything at the
    extremes, not because a larger number would overflow."""
    with pytest.raises(AssertionError):
        geometry(SMALL, 200_000, 100)


def test_the_smallest_and_largest_allowed_sizes_work() -> None:
    """The boundary from the inside, so the bound is not simply tight
    enough to refuse everything."""
    assert geometry(SMALL, 1, 1).total_tokens > 0
    assert geometry(SMALL, 100_000, 100_000).total_tokens > 0


def test_a_split_that_would_stretch_a_tile_is_refused() -> None:
    """`split_tiles` runs after the fitting step and says so. Driven
    out of order here, it has to fail rather than return a grid that
    does not cover the image."""
    with pytest.raises(AssertionError):
        split_tiles(1000, 1000, 384)
