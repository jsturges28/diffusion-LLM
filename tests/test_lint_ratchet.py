"""Tests that the lint gate cannot be satisfied by accident.

Strategy: drive the comparator with synthetic tallies rather than
running Ruff, so each case states one precise relationship between a
baseline and a current reading. One test at the end runs the real
thing against the committed baseline, which is the only way to catch
a baseline that has drifted from the tree.

The finding this implements asks for three properties, and the third
is the one that shapes the design: the gate must exit zero on the
accepted baseline, fail when a violation is added, and "never pass
because a different finding disappeared". That last clause rules out
comparing totals, because a total accepts a new complexity finding in
a sampler as long as somebody deleted a long line elsewhere. So the
unit of comparison is a (file, rule) pair, and the swap test below is
what proves it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest

from scripts import lint_ratchet

Cell = Tuple[str, str]

# A small stand-in for the real baseline: two files, three rules.
BASELINE: Dict[Cell, int] = {
    ("src/web/server.py", "E501"): 30,
    ("src/web/server.py", "C901"): 1,
    ("src/inference/llada_sampler.py", "E501"): 46,
}


# -- the three properties the finding names --


def test_the_accepted_baseline_passes() -> None:
    assert lint_ratchet.regressions(BASELINE, dict(BASELINE)) == []


def test_one_added_violation_fails() -> None:
    current = dict(BASELINE)
    current[("src/web/server.py", "E501")] = 31

    grown = lint_ratchet.regressions(BASELINE, current)

    assert len(grown) == 1
    cell, allowed, count = grown[0]
    assert cell == ("src/web/server.py", "E501")
    assert (allowed, count) == (30, 31)


def test_a_swap_does_not_pass() -> None:
    """The clause that rules out comparing totals.

    One long line removed from the sampler, one new complexity
    finding added to the server. The total is unchanged at 77, and a
    count-based gate would wave it through.
    """
    current = dict(BASELINE)
    current[("src/inference/llada_sampler.py", "E501")] = 45
    current[("src/web/server.py", "C901")] = 2

    assert sum(current.values()) == sum(BASELINE.values())

    grown = lint_ratchet.regressions(BASELINE, current)

    assert len(grown) == 1
    assert grown[0][0] == ("src/web/server.py", "C901")


# -- the edges around those --


def test_a_brand_new_rule_in_an_existing_file_fails() -> None:
    """Nothing was there before, so anything is an increase."""
    current = dict(BASELINE)
    current[("src/web/server.py", "PLR1702")] = 1

    grown = lint_ratchet.regressions(BASELINE, current)

    assert grown == [
        (("src/web/server.py", "PLR1702"), 0, 1)
    ]


def test_a_brand_new_file_fails() -> None:
    """A file with no recorded debt starts at a ceiling of zero, so
    new code has to be clean. That is the half of the policy that
    stops the baseline growing with the repository."""
    current = dict(BASELINE)
    current[("src/web/brand_new.py", "E501")] = 1

    grown = lint_ratchet.regressions(BASELINE, current)

    assert len(grown) == 1
    assert grown[0][0][0] == "src/web/brand_new.py"


def test_shrinking_is_not_a_regression() -> None:
    current = dict(BASELINE)
    current[("src/web/server.py", "E501")] = 12

    assert lint_ratchet.regressions(BASELINE, current) == []


def test_shrinking_is_reported_so_it_can_be_locked_in() -> None:
    """A ratchet nobody tightens is just a baseline."""
    current = dict(BASELINE)
    current[("src/web/server.py", "E501")] = 12

    shrunk = lint_ratchet.improvements(BASELINE, current)

    assert shrunk == [(("src/web/server.py", "E501"), 30, 12)]


def test_a_fully_cleaned_cell_is_reported() -> None:
    """Dropping out of the current tally entirely still counts as
    the win it is, rather than silently keeping its old ceiling."""
    current = dict(BASELINE)
    del current[("src/web/server.py", "C901")]

    shrunk = lint_ratchet.improvements(BASELINE, current)

    assert (("src/web/server.py", "C901"), 1, 0) in shrunk


def test_regressions_are_ordered_worst_first() -> None:
    current = dict(BASELINE)
    current[("src/web/server.py", "E501")] = 31
    current[("src/inference/llada_sampler.py", "E501")] = 56

    grown = lint_ratchet.regressions(BASELINE, current)

    assert grown[0][0][0] == "src/inference/llada_sampler.py"


# -- reading and writing the baseline file --


def test_a_written_baseline_reads_back_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round trip matters because the file is hand-reviewed in
    diffs; a lossy format would make a review meaningless."""
    monkeypatch.setattr(
        lint_ratchet, "BASELINE_PATH", tmp_path / "baseline.json"
    )

    lint_ratchet.save_baseline(BASELINE)

    assert lint_ratchet.load_baseline() == BASELINE


def test_a_missing_baseline_reads_as_no_accepted_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lint_ratchet, "BASELINE_PATH", tmp_path / "absent.json"
    )

    assert lint_ratchet.load_baseline() == {}


def test_the_written_file_records_a_total_for_humans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate does not read it, but a reviewer wants one number."""
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(lint_ratchet, "BASELINE_PATH", path)

    lint_ratchet.save_baseline(BASELINE)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == sum(BASELINE.values())


# -- the tally, and the real baseline --


def test_findings_are_counted_per_file_and_rule() -> None:
    findings = [
        {"filename": "/repo/src/a.py", "code": "E501"},
        {"filename": "/repo/src/a.py", "code": "E501"},
        {"filename": "/repo/src/a.py", "code": "C901"},
        {"filename": "/repo/src/b.py", "code": "E501"},
    ]

    counts = lint_ratchet.tally(
        [
            {
                "filename": str(
                    lint_ratchet.REPO_ROOT / f["filename"][6:]
                ),
                "code": f["code"],
            }
            for f in findings
        ]
    )

    assert counts[("src/a.py", "E501")] == 2
    assert counts[("src/a.py", "C901")] == 1
    assert counts[("src/b.py", "E501")] == 1


def test_a_finding_with_no_rule_code_is_still_counted() -> None:
    """Ruff reports syntax errors with a null code. Dropping those
    would let a broken file through the gate silently."""
    counts = lint_ratchet.tally(
        [
            {
                "filename": str(
                    lint_ratchet.REPO_ROOT / "src/a.py"
                ),
                "code": None,
            }
        ]
    )

    assert counts[("src/a.py", "UNKNOWN")] == 1


def test_the_committed_baseline_matches_the_tree() -> None:
    """The one test that runs Ruff for real.

    Everything above proves the comparator is correct against
    synthetic data. This proves the recorded ceiling still describes
    this repository, which is the failure the whole gate exists to
    prevent and the one no unit test can see.
    """
    current = lint_ratchet.tally(lint_ratchet.run_ruff())
    baseline = lint_ratchet.load_baseline()

    grown = lint_ratchet.regressions(baseline, current)

    assert grown == [], (
        "lint findings increased; run"
        " .venv/bin/python scripts/lint_ratchet.py for detail"
    )
