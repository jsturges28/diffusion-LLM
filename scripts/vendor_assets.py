"""Fetch the third-party browser assets into src/web/static/vendor.

The Analytics page used to load Chart.js, Hammer.js, and the zoom
plugin from a CDN, and every page loaded JetBrains Mono from Google.
A local-first app that loses its saved-run interface when the network
is unfriendly is not local-first, and third-party script running on
the app's own origin can reach the model and deletion APIs. So the
assets are vendored, and this script is how they got here.

Run it only to add or bump an asset, then commit what it wrote:

    .venv/bin/python scripts/vendor_assets.py

It rewrites `vendor/README.md` with the URL, version, byte count, and
SHA-256 of everything it fetched, so a future bump is a diff rather
than an act of faith. That record is also the groundwork `TRUST-03`
wants for model artifacts.

Versions are pinned here, deliberately, rather than read from a
manifest. There are four of them, they change about once a year, and
a pin you have to edit by hand is a pin somebody has to think about.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "src" / "web" / "static" / "vendor"
FONT_DIR = VENDOR_DIR / "fonts"

CHART_VERSION = "4.4.7"
HAMMER_VERSION = "2.0.8"
ZOOM_VERSION = "2.2.0"
# The tag the OFL text is taken from. The font binaries come from
# Google's CDN, which does not publish a version, so this pins the
# license rather than the glyphs.
FONT_LICENSE_TAG = "v2.304"

FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=JetBrains+Mono:wght@300;400;500&display=swap"
)
# Google serves woff2 only to a browser-shaped client; an unadorned
# urllib request gets ttf, which is roughly twice the size.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

TIMEOUT_SECONDS = 30.0


class Fetched(NamedTuple):
    """One vendored file and where it came from."""

    path: Path
    url: str
    size_bytes: int
    sha256: str


def fetch(url: str, *, as_browser: bool = False) -> bytes:
    headers = (
        {"User-Agent": BROWSER_USER_AGENT} if as_browser else {}
    )
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(
        request, timeout=TIMEOUT_SECONDS
    ) as response:
        body = bytes(response.read())
    assert len(body) > 0, f"empty response from {url}"
    return body


def store(dest: Path, url: str, data: bytes) -> Fetched:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return Fetched(
        path=dest,
        url=url,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def script_sources() -> List[Tuple[str, str]]:
    """The (url, filename) pairs for the three chart libraries."""
    npm = "https://cdn.jsdelivr.net/npm"
    return [
        (
            f"{npm}/chart.js@{CHART_VERSION}"
            "/dist/chart.umd.min.js",
            "chart.umd.min.js",
        ),
        (
            f"{npm}/chart.js@{CHART_VERSION}/LICENSE.md",
            "chart.js.LICENSE.md",
        ),
        (
            f"{npm}/hammerjs@{HAMMER_VERSION}/hammer.min.js",
            "hammer.min.js",
        ),
        (
            f"{npm}/hammerjs@{HAMMER_VERSION}/LICENSE.md",
            "hammer.js.LICENSE.md",
        ),
        (
            f"{npm}/chartjs-plugin-zoom@{ZOOM_VERSION}"
            "/dist/chartjs-plugin-zoom.min.js",
            "chartjs-plugin-zoom.min.js",
        ),
        (
            f"{npm}/chartjs-plugin-zoom@{ZOOM_VERSION}"
            "/LICENSE.md",
            "chartjs-plugin-zoom.LICENSE.md",
        ),
    ]


def vendor_scripts() -> List[Fetched]:
    fetched: List[Fetched] = []
    for url, name in script_sources():
        fetched.append(
            store(VENDOR_DIR / name, url, fetch(url))
        )
    return fetched


def parse_font_faces(css: str) -> List[Dict[str, str]]:
    """Pull one record per @font-face block out of Google's CSS.

    Each block arrives preceded by a ``/* subset */`` comment, which
    is the only place the subset name appears, so the blocks are
    split on the comment rather than on the rule.
    """
    faces: List[Dict[str, str]] = []
    for block in css.split("/*"):
        url_match = re.search(
            r"url\((https://[^)]+\.woff2)\)", block
        )
        weight_match = re.search(
            r"font-weight:\s*(\d+)", block
        )
        range_match = re.search(
            r"unicode-range:\s*([^;]+);", block
        )
        if not (url_match and weight_match and range_match):
            continue
        faces.append(
            {
                "subset": block.split("*/")[0].strip(),
                "weight": weight_match.group(1),
                "unicode_range": range_match.group(1).strip(),
                "url": url_match.group(1),
            }
        )
    assert len(faces) > 0, "no @font-face blocks in the CSS"
    return faces


def vendor_fonts() -> List[Fetched]:
    """Download the woff2 files and write the local @font-face CSS.

    JetBrains Mono is a variable font, so Google returns the same
    file for all three weights and varies only the descriptor. The
    files are therefore stored once per subset and shared across the
    weight blocks; downloading per weight wrote three byte-identical
    copies of each, tripling this directory for nothing.
    """
    css = fetch(FONT_CSS_URL, as_browser=True).decode("utf-8")
    faces = parse_font_faces(css)

    fetched: List[Fetched] = []
    by_url: Dict[str, str] = {}
    lines = [_FONT_CSS_HEADER]
    for face in faces:
        url = face["url"]
        if url not in by_url:
            name = f"jetbrains-mono-{face['subset']}.woff2"
            record = store(
                FONT_DIR / name, url, fetch(url, as_browser=True)
            )
            fetched.append(record)
            by_url[url] = name
        lines.append(
            _font_face_rule(
                subset=face["subset"],
                weight=face["weight"],
                unicode_range=face["unicode_range"],
                filename=by_url[url],
            )
        )
    (FONT_DIR / "jetbrains-mono.css").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    license_url = (
        "https://raw.githubusercontent.com/JetBrains"
        f"/JetBrainsMono/{FONT_LICENSE_TAG}/OFL.txt"
    )
    fetched.append(
        store(
            FONT_DIR / "OFL.txt",
            license_url,
            fetch(license_url),
        )
    )
    return fetched


_FONT_CSS_HEADER = """\
/* JetBrains Mono, vendored so every page renders with no
 * third-party request. Generated by scripts/vendor_assets.py from
 * the Google Fonts CSS2 response for weights 300, 400 and 500.
 *
 * The unicode-range blocks are kept as Google wrote them, so a
 * browser still loads only the subsets a page actually needs. The
 * font is variable, so every weight points at the same file per
 * subset and the weight descriptor pins the axis.
 *
 * SIL Open Font License 1.1; see OFL.txt beside this file.
 */
"""


def _font_face_rule(
    *,
    subset: str,
    weight: str,
    unicode_range: str,
    filename: str,
) -> str:
    return (
        f"\n/* {subset} */\n"
        "@font-face {\n"
        "  font-family: 'JetBrains Mono';\n"
        "  font-style: normal;\n"
        f"  font-weight: {weight};\n"
        "  font-display: swap;\n"
        f"  src: url(./{filename}) format('woff2');\n"
        f"  unicode-range: {unicode_range};\n"
        "}\n"
    )


def write_manifest(fetched: List[Fetched]) -> None:
    """Record what was fetched, so a bump is a reviewable diff."""
    rows = [
        "# Vendored browser assets",
        "",
        "Written by `scripts/vendor_assets.py`. Do not edit these",
        "files by hand: rerun the script to bump a version, and",
        "commit the diff.",
        "",
        "They are vendored rather than loaded from a CDN so that",
        "every page works with no outbound network, and so that no",
        "third-party origin gets to run code next to the model and",
        "deletion APIs. See audit finding TRUST-02.",
        "",
        "| File | Bytes | SHA-256 | Source |",
        "|---|---|---|---|",
    ]
    for item in sorted(fetched, key=lambda f: str(f.path)):
        relative = item.path.relative_to(VENDOR_DIR)
        rows.append(
            f"| `{relative}` | {item.size_bytes} |"
            f" `{item.sha256}` | {item.url} |"
        )
    rows.append("")
    (VENDOR_DIR / "README.md").write_text(
        "\n".join(rows), encoding="utf-8"
    )


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    fetched = vendor_scripts() + vendor_fonts()
    write_manifest(fetched)
    total = sum(item.size_bytes for item in fetched)
    print(
        f"vendored {len(fetched)} files,"
        f" {total / 1024:.0f} KiB total"
    )
    for item in sorted(fetched, key=lambda f: str(f.path)):
        print(
            f"  {item.path.relative_to(VENDOR_DIR)}"
            f"  {item.size_bytes} bytes"
        )


if __name__ == "__main__":
    main()
