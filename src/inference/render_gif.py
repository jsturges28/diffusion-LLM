from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


def history_to_gif(
    history_texts: List[str],
    out_path: Path,
    *,
    width: int = 900,
    height: int = 700,
    font_size: int = 16,
    duration_ms: int = 80,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try a common mono font on Linux; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    frames: List[Image.Image] = []
    for text in history_texts:
        img = Image.new("RGB", (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(img)

        # Simple wrapping
        lines = []
        for raw_line in text.splitlines() or [""]:
            while len(raw_line) > 120:
                lines.append(raw_line[:120])
                raw_line = raw_line[120:]
            lines.append(raw_line)

        y = 10
        for line in lines[: int(height / (font_size + 4))]:
            draw.text((10, y), line, font=font, fill=(230, 230, 230))
            y += font_size + 4

        frames.append(img)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )