"""Render the app icon PNG from the same geometry as ``assets/icon.svg``.

The desktop window icon wants a raster PNG for fidelity (some webview
backends render an SVG window icon poorly), but this project's hosts do
not ship an SVG rasterizer. Pillow is already a dependency (the GIF
renderer uses it), so we redraw the icon's simple geometry directly
rather than shell out to ``rsvg-convert``/Inkscape.

Keep this in sync with ``assets/icon.svg``: both draw a dark rounded
tile plus three CP437 diffusion shade blocks (dark -> medium -> light,
i.e. 75% / 50% / 25% dither coverage) filled with a single denoise
gradient on the trio's top-left-to-bottom-right diagonal (dark green ->
bright green, matching the original icon), so both the dither density
and the hue progress. What a passing run proves: ``assets/icon.png``
exists at the target size and matches the SVG's layout. Run:

    .venv/bin/python scripts/render_icon.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "assets" / "icon.png"

# Design geometry, expressed in the SVG's 256x256 user space.
CANVAS_UNITS = 256
TILE_RADIUS_UNITS = 48
BORDER_WIDTH_UNITS = 1.5
# Each block is a vertical rectangle (taller than wide), matching a
# monospace character cell; keep in sync with the rects in icon.svg.
BLOCK_WIDTH_UNITS = 64
BLOCK_HEIGHT_UNITS = 96
BLOCK_TOP_UNITS = 80
BLOCK_LEFT_UNITS: Tuple[int, int, int] = (16, 96, 176)
PIXEL_UNITS = 4  # one dither "pixel"; each block is a grid of these

BG_COLOR = (10, 10, 10, 255)      # #0a0a0a
BORDER_COLOR = (30, 30, 30, 255)  # #1e1e1e

# Denoise gradient across the block trio (matches the original icon):
# dark green at the trio's top-left corner -> bright green at the
# bottom-right corner. The diagonal's vertical component keeps the dense
# block's solid columns from reading as flat color walls.
GRADIENT_DARK = (36, 90, 52)          # #245a34
GRADIENT_BRIGHT = (0, 255, 65)        # #00ff41
GRADIENT_START_UNITS = (16, 80)       # top-left of the leftmost block
GRADIENT_END_UNITS = (240, 176)       # bottom-right of the rightmost block

Shade = Literal["light", "medium", "dark"]
# Left-to-right: densest (resolved) fading to sparsest (noise).
BLOCK_SHADES: Tuple[Shade, Shade, Shade] = ("dark", "medium", "light")

OUTPUT_SIZE_PX = 512
SUPERSAMPLE = 4  # render large, then downscale for smooth tile corners

# The three shades partition a 2x2 cell into 1/2/3 lit quarters.
COVERAGE_QUARTERS = {"light": 1, "medium": 2, "dark": 3}
assert set(COVERAGE_QUARTERS) == {"light", "medium", "dark"}
assert tuple(sorted(COVERAGE_QUARTERS.values())) == (1, 2, 3)


def shade_pixel_filled(shade: Shade, col: int, row: int) -> bool:
    """Whether the dither pixel at grid ``(col, row)`` is lit.

    Mirrors the 2x2 CP437 tiles in ``assets/icon.svg``: light lights one
    of four cells (25%), medium a checkerboard (50%), dark three of four
    (75%).
    """
    assert col >= 0, "column index is non-negative"
    assert row >= 0, "row index is non-negative"
    col_odd = col % 2 == 1
    row_odd = row % 2 == 1
    if shade == "light":
        return (not col_odd) and (not row_odd)
    if shade == "medium":
        return col_odd == row_odd
    if shade == "dark":
        return not (col_odd and row_odd)
    raise ValueError(f"unknown shade: {shade}")


def gradient_color(
    x_px: float,
    y_px: float,
    start_px: Tuple[float, float],
    end_px: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    """Denoise-gradient color at position ``(x_px, y_px)``.

    Projects the point onto the ``start_px -> end_px`` diagonal and
    interpolates dark green -> bright green along it (clamping outside),
    so hue follows a pixel's place across, and down, the trio.
    """
    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    length_sq = dx * dx + dy * dy
    assert length_sq > 0, "gradient endpoints must differ"
    t = ((x_px - start_px[0]) * dx + (y_px - start_px[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    r = round(GRADIENT_DARK[0] + (GRADIENT_BRIGHT[0] - GRADIENT_DARK[0]) * t)
    g = round(GRADIENT_DARK[1] + (GRADIENT_BRIGHT[1] - GRADIENT_DARK[1]) * t)
    b = round(GRADIENT_DARK[2] + (GRADIENT_BRIGHT[2] - GRADIENT_DARK[2]) * t)
    return (r, g, b, 255)


def draw_shade_block(
    draw: ImageDraw.ImageDraw,
    *,
    shade: Shade,
    left_px: float,
    top_px: float,
    block_w_px: float,
    block_h_px: float,
    pixel_px: float,
    gradient_start_px: Tuple[float, float],
    gradient_end_px: Tuple[float, float],
) -> None:
    """Fill one block with its shade's dither, each pixel taking the
    denoise-gradient color at that pixel's center."""
    assert block_w_px > 0, "block width is positive"
    assert block_h_px > 0, "block height is positive"
    assert pixel_px > 0, "pixel size is positive"
    cols = round(block_w_px / pixel_px)
    rows = round(block_h_px / pixel_px)
    assert cols > 0, "at least one dither column per block"
    assert rows > 0, "at least one dither row per block"
    for row in range(rows):
        for col in range(cols):
            if not shade_pixel_filled(shade, col, row):
                continue
            x0 = left_px + col * pixel_px
            y0 = top_px + row * pixel_px
            color = gradient_color(
                x0 + pixel_px / 2,
                y0 + pixel_px / 2,
                gradient_start_px,
                gradient_end_px,
            )
            draw.rectangle(
                (x0, y0, x0 + pixel_px, y0 + pixel_px),
                fill=color,
            )


def render_icon(size_px: int) -> Image.Image:
    """Render the icon at ``size_px`` square, matching ``icon.svg``."""
    assert size_px > 0, "icon size is positive"
    scale = size_px / CANVAS_UNITS
    image = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    radius_px = TILE_RADIUS_UNITS * scale
    draw.rounded_rectangle(
        (0, 0, size_px - 1, size_px - 1),
        radius=radius_px,
        fill=BG_COLOR,
    )
    border_px = max(1, round(BORDER_WIDTH_UNITS * scale))
    inset = border_px / 2
    draw.rounded_rectangle(
        (inset, inset, size_px - 1 - inset, size_px - 1 - inset),
        radius=radius_px - inset,
        outline=BORDER_COLOR,
        width=border_px,
    )

    block_w_px = BLOCK_WIDTH_UNITS * scale
    block_h_px = BLOCK_HEIGHT_UNITS * scale
    top_px = BLOCK_TOP_UNITS * scale
    pixel_px = PIXEL_UNITS * scale
    gradient_start_px = (
        GRADIENT_START_UNITS[0] * scale,
        GRADIENT_START_UNITS[1] * scale,
    )
    gradient_end_px = (
        GRADIENT_END_UNITS[0] * scale,
        GRADIENT_END_UNITS[1] * scale,
    )
    assert len(BLOCK_LEFT_UNITS) == len(BLOCK_SHADES), (
        "one shade per block column"
    )
    for left_units, shade in zip(BLOCK_LEFT_UNITS, BLOCK_SHADES):
        draw_shade_block(
            draw,
            shade=shade,
            left_px=left_units * scale,
            top_px=top_px,
            block_w_px=block_w_px,
            block_h_px=block_h_px,
            pixel_px=pixel_px,
            gradient_start_px=gradient_start_px,
            gradient_end_px=gradient_end_px,
        )
    return image


def main() -> None:
    render_size = OUTPUT_SIZE_PX * SUPERSAMPLE
    hi_res = render_icon(render_size)
    final = hi_res.resize(
        (OUTPUT_SIZE_PX, OUTPUT_SIZE_PX), Image.LANCZOS
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUTPUT_PATH)
    assert OUTPUT_PATH.is_file(), "icon.png was written"
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_SIZE_PX}x{OUTPUT_SIZE_PX})")


if __name__ == "__main__":
    main()
