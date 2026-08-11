"""Render a run's frame texts into an animated GIF.

The GIF is a derivative: pleasant to look at, never the record. It is
written after the run is published, so a failure here costs a preview
rather than the run.

Memory is the reason this file has a budget. Each frame is a 900 by
700 RGB image, 1.89 MB of raw pixels, and the encoder keeps a paletted
copy of every frame besides. Rendering all of a 2,048-frame run, which
the experimental bounds allow, meant multiple gigabytes to produce a
picture. Two changes bound it: frames are rendered one at a time
instead of built into a list, and a long run is sampled down to
``GIF_FRAME_BUDGET`` evenly spaced frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image, ImageDraw, ImageFont

# The most frames any GIF will contain. Chosen against the real
# corpus: 179 of 182 saved runs are at or under it, so in practice
# almost nothing is sampled, and the ones that are were long enough
# that no viewer was watching every frame anyway.
#
# It is the load-bearing bound. Streaming frames into the encoder
# helps, but does not bound anything on its own: Pillow's GIF writer
# collects a paletted copy of every frame before it writes any of
# them, so peak memory grows with frame count no matter how the
# frames arrive. Budget times ~0.63 MB is the ceiling that follows.
GIF_FRAME_BUDGET = 300

GIF_WIDTH = 900
GIF_HEIGHT = 700
GIF_FONT_SIZE = 16
GIF_DURATION_MS = 80

# Beyond this many characters a line is hard-wrapped. Not a measured
# width: the font is monospaced and the frame is fixed, so a column
# count is the honest unit here.
WRAP_COLUMNS = 120
HEADER_LINES_MAX = 6

BACKGROUND = (18, 18, 18)
TEXT_COLOR = (230, 230, 230)
LABEL_COLOR = (180, 180, 180)
DIVIDER_COLOR = (80, 80, 80)

assert GIF_FRAME_BUDGET > 0, "a GIF needs at least one frame"
assert HEADER_LINES_MAX > 0, "the header needs at least one line"


def sample_frame_indices(
    total: int, budget: int = GIF_FRAME_BUDGET
) -> List[int]:
    """Evenly spaced frame indices, first and last always included.

    An even temporal sample rather than a prefix, because a truncated
    run would show the denoising starting and never finishing, which
    is a wrong picture rather than a partial one. The endpoints are
    pinned because they are the two frames anyone looks for: all
    masked, and the finished text.
    """
    assert total > 0, "cannot sample an empty run"
    assert budget > 0, f"budget must be positive, got {budget}"

    if total <= budget:
        return list(range(total))
    if budget == 1:
        return [total - 1]
    step = (total - 1) / (budget - 1)
    return [round(i * step) for i in range(budget)]


def history_to_gif(
    history_texts: List[str],
    out_path: Path,
    *,
    header_text: Optional[str] = None,
    model_label: Optional[str] = None,
    model_type: str = "diffusion",
    width: int = GIF_WIDTH,
    height: int = GIF_HEIGHT,
    font_size: int = GIF_FONT_SIZE,
    duration_ms: int = GIF_DURATION_MS,
    frame_budget: int = GIF_FRAME_BUDGET,
) -> None:
    """Write the run's frames to ``out_path`` as an animated GIF.

    ``model_label`` names the model in the response heading. Every
    GIF used to claim "LLaDA RESPONSE (Diffusion)", including the
    ones produced by the other two models.
    """
    assert len(history_texts) > 0, "a run has at least one frame"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(font_size)
    indices = sample_frame_indices(
        len(history_texts), frame_budget
    )
    heading = _response_heading(
        model_label,
        model_type,
        shown=len(indices),
        total=len(history_texts),
    )

    layout = _Layout(
        font=font,
        width=width,
        height=height,
        line_height=font_size + 4,
        header_text=header_text,
        heading=heading,
    )
    frames = _render_frames(history_texts, indices, layout)

    # The first frame is drawn eagerly because Pillow needs an image
    # to save *from*; the rest arrive from the generator, so only one
    # RGB frame exists at a time rather than all of them at once.
    first = next(frames)
    first.save(
        out_path,
        save_all=True,
        append_images=frames,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


class _Layout:
    """Everything about a frame that does not change between frames.

    Computed once rather than per frame, which matters at 300 of
    them: the header wraps the same way every time.
    """

    def __init__(
        self,
        *,
        font: ImageFont.ImageFont,
        width: int,
        height: int,
        line_height: int,
        header_text: Optional[str],
        heading: str,
    ) -> None:
        self.font = font
        self.width = width
        self.height = height
        self.line_height = line_height
        self.heading = heading
        self.header_lines: List[str] = []
        if header_text:
            self.header_lines = _wrap(header_text)[
                :HEADER_LINES_MAX
            ]


def _render_frames(
    history_texts: List[str],
    indices: List[int],
    layout: _Layout,
) -> Iterator[Image.Image]:
    """One rendered image per sampled frame, lazily.

    A generator so the caller holds one frame at a time. The list it
    replaced held every frame at 1.89 MB each.
    """
    for index in indices:
        yield _render_frame(history_texts[index], layout)


def _render_frame(
    text: str, layout: _Layout
) -> Image.Image:
    image = Image.new(
        "RGB", (layout.width, layout.height), color=BACKGROUND
    )
    draw = ImageDraw.Draw(image)
    y = 10

    if layout.header_lines:
        y = _draw_header(draw, layout, y)

    draw.text(
        (10, y),
        layout.heading,
        font=layout.font,
        fill=LABEL_COLOR,
    )
    y += layout.line_height + 2

    body_lines = _wrap(text)
    remaining = max(
        1, int((layout.height - y - 10) / layout.line_height)
    )
    for line in body_lines[:remaining]:
        draw.text(
            (10, y), line, font=layout.font, fill=TEXT_COLOR
        )
        y += layout.line_height
    return image


def _draw_header(
    draw: ImageDraw.ImageDraw, layout: _Layout, y: int
) -> int:
    """The prompt block and its divider. Returns the new cursor."""
    draw.text(
        (10, y),
        "USER PROMPT:",
        font=layout.font,
        fill=LABEL_COLOR,
    )
    y += layout.line_height
    for line in layout.header_lines:
        draw.text(
            (10, y), line, font=layout.font, fill=TEXT_COLOR
        )
        y += layout.line_height
    y += 4
    draw.line(
        [(10, y), (layout.width - 10, y)],
        fill=DIVIDER_COLOR,
        width=1,
    )
    return y + 10


def _response_heading(
    model_label: Optional[str],
    model_type: str,
    *,
    shown: int,
    total: int,
) -> str:
    """The line above the generated text.

    Says which model produced the run, and says so when the GIF is a
    sample rather than the whole thing. A viewer counting frames
    against the run's step count deserves to know why they disagree.
    """
    name = model_label or "MODEL"
    paradigm = (
        "Autoregressive"
        if model_type == "autoregressive"
        else "Diffusion"
    )
    heading = f"{name} RESPONSE ({paradigm}):"
    if shown < total:
        heading += f"  [{shown} of {total} frames]"
    return heading


def _wrap(text: str) -> List[str]:
    """Hard-wrap at a column count, preserving blank lines."""
    lines: List[str] = []
    for raw_line in text.splitlines() or [""]:
        remainder = raw_line
        while len(remainder) > WRAP_COLUMNS:
            lines.append(remainder[:WRAP_COLUMNS])
            remainder = remainder[WRAP_COLUMNS:]
        lines.append(remainder)
    return lines


def _load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSansMono.ttf", font_size
        )
    except OSError:
        # No monospaced font on this host. The default bitmap font
        # ignores the size, so the frames come out smaller, which is
        # a worse picture rather than no picture.
        return ImageFont.load_default()
