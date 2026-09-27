"""How a vision-language model turns an image into tokens.

Everything a reader needs to see that happen, derived from the
encoder's own configuration and the image's width and height. Nothing
here touches pixels, which is the finding this module rests on: the
SmolVLM processor's sizing functions accept an image and read only
`get_image_size(image)`, and every step after that is integer
arithmetic. So an image's geometry is knowable without the image, and
the page that draws it can keep the picture in the browser.

Torch-free, Pillow-free and transformers-free, deliberately. The
supervisor runs an older `transformers` than the one implementing
SmolVLM, so importing the real processor is not available here; and
an interactive view needs an answer in milliseconds, where spawning
another interpreter would not. What keeps it honest instead is
`tests/inference/test_vision_geometry.py`, which lifts the real
functions out of the worker environment's source and compares the
composed pipeline against this one.

**That test has already earned its place twice**, which is why it is
worth keeping even though it reads another environment's files. The
first draft of `resize_for_encoder` clamped a collapsed dimension once
at the end rather than between its two stages, and disagreed with the
library on 9 of 8,144 sizes at extreme aspect ratios. Then the test
turned up a step missing altogether: `fit_to_whole_tiles` below, which
the processor applies between the resize and the split. Both bugs were
in how the steps fit together rather than inside any one of them,
which is why the test compares the composition.

The pipeline, in the order a reader meets it:

1. `resize_for_encoder` sets the longest edge to the configured bound.
   **Unconditionally**, so a small image is scaled up rather than left
   alone, and no image is too small to be tiled.
2. `fit_to_whole_tiles` rounds each side up to a whole number of
   tiles, **independently of the other**. This is the step that
   changes the aspect ratio: a 16:9 photo becomes 4:3 before any cut.
3. `split_tiles` divides what is now a multiple of the tile, so every
   crop is exactly one tile square and none is stretched.
4. The encoder reads a tile as a grid of fixed-size patches. Where the
   tile side is not a multiple of the patch size, the remainder
   completes no patch and is never seen.
5. A square block of neighbouring patches fuses into one token, so the
   token count is the patch count over the square of the scale factor.
6. On top of the tiles, the whole image is squashed into one more tile
   and contributes one more block of tokens.

The consequence worth knowing, because it is the opposite of what a
reader expects: **resolution is free and shape is what costs**. Steps
1 and 2 normalise every image to the same working resolution, so a
64x64 icon and a megapixel square cost the same, while a wide banner
costs a third of either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

# The processor's own absolute ceiling, above which it will not scale
# whatever the configuration asks. Named because this module has to
# reproduce it exactly, not because it is a choice made here.
MAX_IMAGE_SIZE = 4096

# What a caller may ask about. Far above any photograph, and bounded
# because the arithmetic stops describing anything at the extremes:
# the processor collapses a dimension to zero there and survives only
# by clamping, so an answer about a 1x100000 strip would be fiction.
MIN_DIMENSION = 1
MAX_DIMENSION = 100_000

assert MIN_DIMENSION >= 1, "a zero dimension has no geometry"
assert MAX_DIMENSION > MAX_IMAGE_SIZE, (
    "an image below the processor's ceiling could never be scaled"
    " down to it, which would make the cap unreachable"
)


@dataclass(frozen=True)
class EncoderGeometry:
    """The parts of an encoder's configuration this module reads.

    Read from the checkpoint rather than declared anywhere, because a
    hand-maintained copy drifts from what the checkpoint loads, and
    this feature exists to tell a reader the truth about it. The two
    SmolVLM checkpoints differ in every one of these fields.
    """

    # Longest edge the whole image is scaled to, before any splitting.
    longest_edge: int
    # The square side of one tile, and the encoder's own input size.
    tile: int
    # Side of one patch, in tile pixels.
    patch: int
    # How many patches on a side fuse into a single token.
    scale: int

    def __post_init__(self) -> None:
        assert self.longest_edge > 0, "longest_edge must be positive"
        assert self.tile > 0, "tile must be positive"
        assert self.patch > 0, "patch must be positive"
        assert self.scale > 0, "scale must be positive"
        assert self.patch <= self.tile, (
            "a patch larger than the tile leaves no grid"
        )
        assert self.scale <= self.patch_side, (
            "fusing more patches than a row holds leaves no tokens"
        )

    @property
    def patch_side(self) -> int:
        """Patches along one edge of a tile.

        A convolution of this kernel and this stride, so trailing
        pixels that do not complete a patch are never covered.

        Written as the convolution it is, though that is provably
        equal to `tile // patch` for every pair:
        checked exhaustively, and worth saying so because the
        difference looks like an off-by-one waiting to be tidied.
        """
        return (self.tile - self.patch) // self.patch + 1

    @property
    def longest_edge_is_whole_tiles(self) -> bool:
        """Whether step 1's bound lands on a tile boundary.

        True for both shipping checkpoints, where `longest_edge` is
        exactly four tiles. It matters more than it looks: when it
        holds, step 2 cannot change the longer side, so recomputing
        the shorter side from the aspect reproduces what step 1
        already derived. When it does not hold, that recomputation
        starts to differ, which is why `fit_to_whole_tiles` follows
        the library's form rather than the shorter one that happens to
        agree here.
        """
        return self.longest_edge % self.tile == 0

    @property
    def unseen_edge(self) -> int:
        """Tile pixels on the right and bottom that no patch covers.

        Zero when the tile side divides by the patch size. It is 6 on
        SmolVLM-2.2B, whose 384px tile is not a multiple of 14, and
        that is a fact about what the model cannot see.
        """
        return self.tile - self.patch_side * self.patch

    @property
    def token_side(self) -> int:
        """Tokens along one edge, so a token has a grid position."""
        return self.patch_side // self.scale

    @property
    def tokens_per_tile(self) -> int:
        return self.token_side * self.token_side

    @property
    def patches_per_token(self) -> int:
        return self.scale * self.scale


@dataclass(frozen=True)
class ImageGeometry:
    """What one image costs, and where each piece of it sits."""

    # As handed in, before anything was done to it.
    source_width: int
    source_height: int
    # After step 1, which preserves the aspect ratio.
    resized_width: int
    resized_height: int
    # After step 2, which does not. Always a multiple of the tile.
    fitted_width: int
    fitted_height: int
    # The tile grid over the fitted image.
    tile_rows: int
    tile_cols: int
    tokens_per_tile: int
    # Tiles plus the one whole-image thumbnail every image gets.
    total_tokens: int

    @property
    def tile_count(self) -> int:
        return self.tile_rows * self.tile_cols

    @property
    def aspect_changed(self) -> bool:
        """Whether step 2 altered the shape of the picture.

        True for most photographs, and the honest place to say the
        model is not seeing the framing you chose. A 1920x1080 image
        arrives at the encoder as 4:3.
        """
        return (
            self.resized_width * self.fitted_height
            != self.fitted_width * self.resized_height
        )


def resize_for_encoder(
    height: int, width: int, longest_edge: int
) -> Tuple[int, int]:
    """Step 1, reproducing the processor's two stages exactly.

    Returns `(height, width)`, matching the library rather than this
    module's usual width-first order. Stated because the differential
    test compares call for call, and a silent transposition here would
    pass every square image.

    The clamp between the stages is why this is one function and not
    two. A very wide image rounds its height to zero in the first
    stage, and the second divides by it.
    """
    assert height >= MIN_DIMENSION, "height below the minimum"
    assert width >= MIN_DIMENSION, "width below the minimum"
    assert longest_edge > 0, "longest_edge must be positive"

    aspect = width / height
    if width >= height:
        width = longest_edge
        height = int(width / aspect)
        # The processor rounds the derived side up to an even number.
        if height % 2 != 0:
            height += 1
    else:
        height = longest_edge
        width = int(height * aspect)
        if width % 2 != 0:
            width += 1

    # Between the stages, not after. See the docstring above.
    height = max(height, 1)
    width = max(width, 1)

    aspect = width / height
    if width >= height and width > MAX_IMAGE_SIZE:
        width = MAX_IMAGE_SIZE
        height = int(width / aspect)
    elif height > width and height > MAX_IMAGE_SIZE:
        height = MAX_IMAGE_SIZE
        width = int(height * aspect)

    height = max(height, 1)
    width = max(width, 1)
    assert height <= MAX_IMAGE_SIZE, "height above the ceiling"
    assert width <= MAX_IMAGE_SIZE, "width above the ceiling"
    return height, width


def fit_to_whole_tiles(
    height: int, width: int, tile: int
) -> Tuple[int, int]:
    """Step 2: round each side up to a whole number of tiles.

    The step this module originally missed. Each side is rounded
    independently, which is what changes the aspect ratio, and the
    longer side is rounded first so the shorter one is derived from
    the already-rounded aspect rather than from the original.
    """
    assert height >= 1, "height must be positive"
    assert width >= 1, "width must be positive"
    assert tile > 0, "tile must be positive"

    aspect = width / height
    if width >= height:
        width = math.ceil(width / tile) * tile
        height = math.ceil(int(width / aspect) / tile) * tile
    else:
        height = math.ceil(height / tile) * tile
        width = math.ceil(int(height * aspect) / tile) * tile

    assert height % tile == 0, "height is not a whole number of tiles"
    assert width % tile == 0, "width is not a whole number of tiles"
    return height, width


def split_tiles(
    height: int, width: int, tile: int
) -> Tuple[int, int]:
    """Step 3, as `(rows, cols)`.

    Only a count, unlike the library's version which also returns the
    crops, because after step 2 every crop is exactly one tile square.
    That is asserted rather than assumed: it is the property that says
    no tile is stretched, and it holds only while step 2 runs first.

    Zero rows and columns when the image fits a single tile, which is
    how the library signals it did not split rather than reporting one
    tile. Unreachable through `geometry` below, since step 1 scales
    every image up to the bound, but the case is real if the steps are
    ever driven separately.
    """
    assert height >= 1, "height must be positive"
    assert width >= 1, "width must be positive"
    assert tile > 0, "tile must be positive"

    if height <= tile and width <= tile:
        return 0, 0

    rows = math.ceil(height / tile)
    cols = math.ceil(width / tile)

    assert rows * tile == height, (
        "rows do not divide the height exactly, so a crop would be"
        " stretched; fit_to_whole_tiles has to run first"
    )
    assert cols * tile == width, (
        "columns do not divide the width exactly, so a crop would be"
        " stretched; fit_to_whole_tiles has to run first"
    )
    return rows, cols


def fuse_map(patch_side: int, scale: int) -> List[List[int]]:
    """Step 5: which patch indices land in each token.

    Index arithmetic rather than the library's four reshapes and two
    permutes, because the answer a reader needs is "which part of the
    picture is this token", and a rectangle states that where a
    sequence of views does not. The two agree exactly, which the
    differential test checks at every geometry either encoder uses.

    Patch indices are row-major over the `patch_side` grid, tokens are
    row-major over the fused grid.
    """
    assert patch_side > 0, "patch_side must be positive"
    assert scale > 0, "scale must be positive"
    assert scale <= patch_side, "cannot fuse more patches than exist"

    token_side = patch_side // scale
    fused: List[List[int]] = []
    for row in range(token_side):
        for col in range(token_side):
            fused.append([
                (row * scale + dr) * patch_side + (col * scale + dc)
                for dr in range(scale)
                for dc in range(scale)
            ])

    assert len(fused) == token_side * token_side, "wrong token count"
    for members in fused:
        assert len(members) == scale * scale, "wrong fusion width"
    return fused


def patch_block(
    token_index: int, patch_side: int, scale: int
) -> Tuple[int, int, int, int]:
    """The patch rectangle one token covers, as `(row, col, h, w)`.

    The single-token form of `fuse_map`, for the hover path: lighting
    up one block should not cost building the whole map.
    """
    token_side = patch_side // scale
    assert token_side > 0, "no tokens at this geometry"
    assert 0 <= token_index < token_side * token_side, (
        f"token {token_index} outside a {token_side}-wide grid"
    )

    row = token_index // token_side
    col = token_index % token_side
    return row * scale, col * scale, scale, scale


def geometry(
    encoder: EncoderGeometry, width: int, height: int
) -> ImageGeometry:
    """The three steps composed, for one image and one encoder.

    Width first here and height first inside the steps, which is worth
    stating rather than smoothing over: this is the order a caller
    thinks in, and that is the order the library uses. The conversion
    happens once, on the first call below.
    """
    assert MIN_DIMENSION <= width <= MAX_DIMENSION, (
        f"width {width} outside {MIN_DIMENSION}..{MAX_DIMENSION}"
    )
    assert MIN_DIMENSION <= height <= MAX_DIMENSION, (
        f"height {height} outside {MIN_DIMENSION}..{MAX_DIMENSION}"
    )

    resized_height, resized_width = resize_for_encoder(
        height, width, encoder.longest_edge
    )
    fitted_height, fitted_width = fit_to_whole_tiles(
        resized_height, resized_width, encoder.tile
    )
    rows, cols = split_tiles(
        fitted_height, fitted_width, encoder.tile
    )
    per_tile = encoder.tokens_per_tile
    # Every image gets the whole-image thumbnail on top of its tiles,
    # which is why this is `+ 1` and not `or 1`.
    total = (rows * cols + 1) * per_tile

    assert total >= per_tile, "an image costs at least the thumbnail"
    return ImageGeometry(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        fitted_width=fitted_width,
        fitted_height=fitted_height,
        tile_rows=rows,
        tile_cols=cols,
        tokens_per_tile=per_tile,
        total_tokens=total,
    )
