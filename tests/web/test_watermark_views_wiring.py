"""Static contract for KGW help, legends and non-color evidence.

Strategy: inspect the shipped generator/Analytics markup, shared
stylesheet and durable docs. Passing proves both pages expose the
same three-state legend, excluded evidence has pattern plus outline,
the detector result is announced accessibly, and KGW attribution
does not confuse the later token-specific paper with the origin.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_both_pages_name_every_watermark_legend_state() -> None:
    generator = _text(STATIC / "index.html")
    analytics = _text(STATIC / "analytics.html")

    for page in (generator, analytics):
        assert "favored set" in page
        assert "complement" in page
        assert "excluded from score" in page
        assert "Watermark membership legend" in page


def test_excluded_evidence_uses_pattern_and_outline() -> None:
    stylesheet = _text(STATIC / "style.css")
    start = stylesheet.index(".token-watermark-excluded")
    rule = stylesheet[start : start + 600]

    assert "outline:" in rule
    assert "repeating-linear-gradient" in rule


def test_membership_has_non_color_underline_styles() -> None:
    stylesheet = _text(STATIC / "style.css")
    favored = stylesheet[
        stylesheet.index(".token-watermark-favored") :
    ]
    complement = stylesheet[
        stylesheet.index(".token-watermark-complement") :
    ]

    assert "text-decoration-style: double" in favored[:300]
    assert "text-decoration-style: dotted" in complement[:300]


def test_analytics_crossfade_has_an_accessible_name() -> None:
    analytics = _text(STATIC / "analytics.html")
    start = analytics.index('id="run-blend"')
    control = analytics[start : start + 300]

    assert 'aria-label="Crossfade between the original' in control


def test_threshold_status_does_not_reuse_membership_colors() -> None:
    stylesheet = _text(STATIC / "style.css")
    start = stylesheet.index(".watermark-status-threshold_crossed")
    rule = stylesheet[start : start + 140]

    assert "var(--text-primary)" in rule
    assert "var(--accent)" not in rule
    assert "var(--danger)" not in rule


def test_detector_result_is_live_and_help_is_neutral() -> None:
    generator = _text(STATIC / "index.html")

    assert 'id="watermark-detector-result"' in generator
    assert 'aria-live="polite"' in generator
    assert "without a model forward" in generator
    assert "not an AI or human verdict" in generator
    assert "correctness or confidence" in generator


def test_docs_attribute_kgw_to_its_original_paper() -> None:
    readme = _text(ROOT / "README.md")
    guide = _text(ROOT / "docs" / "GUIDE.md")
    roadmap = _text(ROOT / "docs" / "ROADMAP.md")
    combined = "\n".join((readme, guide, roadmap))

    assert "A Watermark for Large Language Models" in combined
    assert "arXiv:2301.10226" in combined
    assert "Token-Specific Watermarking" in combined
    assert "arXiv:2402.18059" in combined
    assert "not the origin" in combined
    assert "detector overlay remains a later feature" not in combined


def test_pressure_copy_names_all_three_sampling_stages() -> None:
    guide = _text(ROOT / "docs" / "GUIDE.md")
    roadmap = _text(ROOT / "docs" / "ROADMAP.md")
    in_app = _text(STATIC / "index.html")

    for text in (guide, roadmap, in_app):
        assert "Model → KGW → Sampler" in text
    assert "vocabulary share" in guide
    assert "not an exact or calibrated p-value" in guide


def test_distribution_toggle_has_focus_and_touch_targets() -> None:
    stylesheet = _text(STATIC / "style.css")
    start = stylesheet.index(".alt-distribution-toggle button {")
    button_rule = stylesheet[start : start + 300]

    assert "min-height: 24px" in button_rule
    assert (
        ".alt-distribution-toggle button:focus-visible"
        in stylesheet
    )
