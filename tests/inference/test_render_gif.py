"""The GIF is bounded, evenly sampled, and honestly labelled.

Strategy: exercise the real renderer against real Pillow, on runs
short enough to be fast and long enough to cross the budget. The
sampling arithmetic is tested directly, because it is the part that
decides what a viewer sees and an off-by-one at either end silently
drops the frame everybody looks for.

What passing proves. A run longer than the budget produces a GIF with
exactly the budget's frame count, so the encoder's memory has a
ceiling instead of growing with the run: at the 2,048-frame extreme
the experimental bounds allow, the old path needed multiple gigabytes
of raw pixels to draw a picture. The sample is even and keeps both
endpoints, so a long run still shows fully masked at the start and
finished text at the end rather than stopping partway. And the
heading names the model that actually ran, where every GIF used to
claim LLaDA.

Memory itself is not asserted here. Pillow allocates pixels outside
Python's allocator, so `tracemalloc` cannot see them and RSS is too
noisy to gate a suite on; the frame count is the bound, and it is
what these check.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from PIL import Image, ImageSequence

from src.inference.render_gif import (
    GIF_FRAME_BUDGET,
    history_to_gif,
    sample_frame_indices,
)


def _frames(count: int) -> List[str]:
    return [f"frame {i} " + "\u2591" * 20 for i in range(count)]


def _frame_count(path: Path) -> int:
    with Image.open(path) as image:
        return sum(1 for _ in ImageSequence.Iterator(image))


# -- the sample --


def test_a_short_run_keeps_every_frame() -> None:
    """The common case. 179 of the 182 saved runs are under the
    budget, so almost nothing should be sampled at all."""
    assert sample_frame_indices(9, 300) == list(range(9))


def test_a_run_exactly_at_the_budget_is_untouched() -> None:
    """The boundary where sampling begins. Off by one here would
    resample every run at the most common length."""
    assert sample_frame_indices(300, 300) == list(range(300))


def test_one_frame_over_the_budget_is_sampled() -> None:
    indices = sample_frame_indices(301, 300)

    assert len(indices) == 300


def test_the_sample_keeps_both_endpoints() -> None:
    """All masked and finished text are the two frames anyone looks
    for. Losing the last one would make the GIF stop short of the
    answer."""
    indices = sample_frame_indices(2048, 300)

    assert indices[0] == 0
    assert indices[-1] == 2047


def test_the_sample_is_evenly_spaced() -> None:
    """A prefix would show the denoising start and never finish,
    which is a wrong picture rather than a partial one."""
    indices = sample_frame_indices(2048, 300)

    gaps = {
        indices[i + 1] - indices[i]
        for i in range(len(indices) - 1)
    }
    assert max(gaps) - min(gaps) <= 1


def test_the_sample_never_repeats_a_frame() -> None:
    indices = sample_frame_indices(2048, 300)

    assert len(set(indices)) == len(indices)


def test_the_sample_stays_in_range() -> None:
    indices = sample_frame_indices(1000, 300)

    assert min(indices) >= 0
    assert max(indices) < 1000


def test_a_budget_of_one_keeps_the_final_frame() -> None:
    """The degenerate case. If only one frame fits, it is the one
    that shows the answer."""
    assert sample_frame_indices(500, 1) == [499]


def test_an_empty_run_is_a_programmer_error() -> None:
    """A run always has at least one frame, so reaching here means
    something upstream built a bundle it should not have."""
    with pytest.raises(AssertionError):
        sample_frame_indices(0, 300)


def test_a_zero_budget_is_a_programmer_error() -> None:
    with pytest.raises(AssertionError):
        sample_frame_indices(10, 0)


# -- what lands on disk --


def test_a_long_run_is_capped_at_the_budget(
    tmp_path: Path,
) -> None:
    """The bound, measured on the artifact rather than the
    arithmetic. Pillow's GIF writer keeps a paletted copy of every
    frame it is given, so the frame count is the memory ceiling."""
    out = tmp_path / "long.gif"

    history_to_gif(
        _frames(400), out, frame_budget=50, width=200, height=150
    )

    assert _frame_count(out) == 50


def test_a_short_run_is_written_whole(tmp_path: Path) -> None:
    out = tmp_path / "short.gif"

    history_to_gif(_frames(12), out, frame_budget=50)

    assert _frame_count(out) == 12


def test_a_single_frame_run_still_renders(
    tmp_path: Path,
) -> None:
    """One frame is a legal run and Pillow treats a one-frame
    animation as a special case internally."""
    out = tmp_path / "one.gif"

    history_to_gif(["just this"], out)

    assert _frame_count(out) == 1


def test_the_default_budget_applies_without_being_asked(
    tmp_path: Path,
) -> None:
    """The bound has to be the default, not an opt-in the save path
    could forget to pass.

    Drawn small on purpose. This is the only test that renders a
    full budget's worth of frames, and at the real 900 by 700 it
    alone took eight seconds of the suite. Still wide enough for
    each frame's number to be legible, because Pillow merges
    consecutive identical frames and a canvas that clipped the text
    away would undercount.
    """
    out = tmp_path / "default.gif"

    history_to_gif(
        _frames(GIF_FRAME_BUDGET + 40),
        out,
        width=320,
        height=90,
    )

    assert _frame_count(out) == GIF_FRAME_BUDGET


def test_the_parent_directory_is_created(
    tmp_path: Path,
) -> None:
    out = tmp_path / "nested" / "deeper" / "run.gif"

    history_to_gif(_frames(3), out)

    assert out.is_file()


def test_an_awkward_prompt_does_not_break_rendering(
    tmp_path: Path,
) -> None:
    """The header is drawn from user text, so it has to survive
    newlines, very long lines, and unicode."""
    out = tmp_path / "awkward.gif"
    prompt = "line one\n" + ("x" * 500) + "\n\u2591 unicode \u2603"

    history_to_gif(_frames(4), out, header_text=prompt)

    assert _frame_count(out) == 4


def test_an_empty_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        history_to_gif([], tmp_path / "empty.gif")


# -- the label --


def test_the_heading_names_the_model_that_ran() -> None:
    """Every GIF used to say LLaDA, including SmolLM3's and
    DiffusionGemma's."""
    from src.inference.render_gif import _response_heading

    heading = _response_heading(
        "SmolLM3-3B",
        "autoregressive",
        shown=10,
        total=10,
    )

    assert "SmolLM3-3B" in heading
    assert "LLaDA" not in heading


def test_the_heading_names_the_paradigm() -> None:
    from src.inference.render_gif import _response_heading

    diffusion = _response_heading(
        "LLaDA-8B-Instruct", "diffusion", shown=5, total=5
    )
    autoregressive = _response_heading(
        "SmolLM3-3B", "autoregressive", shown=5, total=5
    )

    assert "(Diffusion)" in diffusion
    assert "(Autoregressive)" in autoregressive


def test_the_heading_says_when_frames_were_dropped() -> None:
    """A viewer counting frames against the run's step count
    deserves to know why they disagree."""
    from src.inference.render_gif import _response_heading

    heading = _response_heading(
        "LLaDA-8B-Instruct", "diffusion", shown=300, total=984
    )

    assert "300 of 984" in heading


def test_the_heading_stays_quiet_when_nothing_was_dropped() -> None:
    """The negative space: almost every run takes this path, and a
    frame count on all of them would be noise."""
    from src.inference.render_gif import _response_heading

    heading = _response_heading(
        "LLaDA-8B-Instruct", "diffusion", shown=128, total=128
    )

    assert "of" not in heading.split("(")[0]
    assert "frames" not in heading


def test_an_unnamed_model_does_not_borrow_another_s_name() -> None:
    from src.inference.render_gif import _response_heading

    heading = _response_heading(
        None, "diffusion", shown=5, total=5
    )

    assert "LLaDA" not in heading
    assert "MODEL" in heading
