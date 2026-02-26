from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


def history_to_gif(
    history_texts: List[str],
    out_path: Path,
    *,
    header_text: Optional[str] = None,
    width: int = 900,
    height: int = 700,
    font_size: int = 16,
    duration_ms: int = 80,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    frames: List[Image.Image] = []
    line_h = font_size + 4

    for text in history_texts:
        img = Image.new("RGB", (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(img)

        y = 10

        # --- Draw header (prompt) ---
        if header_text:
            draw.text((10, y), "USER PROMPT:", font=font, fill=(180, 180, 180))
            y += line_h

            # crude wrap
            header_lines = []
            for raw_line in header_text.splitlines() or [""]:
                while len(raw_line) > 120:
                    header_lines.append(raw_line[:120])
                    raw_line = raw_line[120:]
                header_lines.append(raw_line)

            for line in header_lines[:6]:  # cap header height
                draw.text((10, y), line, font=font, fill=(230, 230, 230))
                y += line_h

            # divider
            y += 4
            draw.line([(10, y), (width - 10, y)], fill=(80, 80, 80), width=1)
            y += 10

        # --- Draw response heading (always shown) ---
        draw.text((10, y), "LLaDA RESPONSE (Diffusion):", font=font, fill=(180, 180, 180))
        y += line_h + 2

        # --- Draw diffusion text ---
        lines = []
        for raw_line in text.splitlines() or [""]:
            while len(raw_line) > 120:
                lines.append(raw_line[:120])
                raw_line = raw_line[120:]
            lines.append(raw_line)

        max_lines = max(1, int((height - y - 10) / line_h))
        for line in lines[:max_lines]:
            draw.text((10, y), line, font=font, fill=(230, 230, 230))
            y += line_h

        frames.append(img)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
