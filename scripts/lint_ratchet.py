"""Fail when Ruff findings increase, per file and per rule.

The repository has a real lint debt (140 findings, most of them line
length) that cannot be paid off in one commit without reflowing files
that later audit work is about to rewrite anyway. A remembered total
is not a gate: a new violation hides behind an unrelated removal, and
nobody can tell whether a change made a specific file safer.

So the accepted debt is written down in ``lint_baseline.json`` as a
count per (file, rule) pair, and this script fails when any pair grows
or appears. Counting per pair rather than in total is the whole point.
A single number would happily accept a new complexity finding in the
sampler as long as somebody deleted a long line somewhere else, which
is exactly the trade this is meant to refuse.

Usage::

    .venv/bin/python scripts/lint_ratchet.py       # check
    .venv/bin/python scripts/lint_ratchet.py --update   # ratchet down

Findings only ever have to go down. When they do, the script says so
and ``--update`` records the new, lower ceiling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "lint_baseline.json"

# The same two trees AGENTS.md tells everyone to check. Scripts are
# deliberately outside it, matching the existing habit; widening the
# scope is a decision, not a detail, and it would move the ceiling.
LINT_PATHS = ("src", "tests")

# A (file, rule) pair and how many times it occurs.
Cell = Tuple[str, str]
Counts = Dict[Cell, int]


def run_ruff() -> List[Dict[str, object]]:
    """Ruff's findings for the checked paths, as parsed JSON."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *LINT_PATHS,
            "--output-format",
            "json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Ruff exits non-zero simply because findings exist, so the exit
    # code says nothing here; unparseable output is the real failure.
    try:
        findings = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "could not parse ruff output:\n"
            + (completed.stderr or completed.stdout)
        ) from exc
    assert isinstance(findings, list), "ruff returned a non-list"
    return findings


def tally(findings: List[Dict[str, object]]) -> Counts:
    """Count findings per (repo-relative file, rule code)."""
    counts: Counter[Cell] = Counter()
    for finding in findings:
        filename = str(finding.get("filename", ""))
        code = str(finding.get("code") or "UNKNOWN")
        relative = Path(filename)
        if relative.is_absolute():
            relative = relative.relative_to(REPO_ROOT)
        counts[(relative.as_posix(), code)] += 1
    return dict(counts)


def load_baseline() -> Counts:
    """The accepted debt, or an empty ceiling if none is recorded."""
    if not BASELINE_PATH.is_file():
        return {}
    raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    counts: Counts = {}
    for filename, rules in raw.get("counts", {}).items():
        for code, count in rules.items():
            assert count > 0, f"zero cell recorded: {filename} {code}"
            counts[(filename, code)] = int(count)
    return counts


def save_baseline(counts: Counts) -> None:
    """Record the ceiling, nested by file so diffs stay readable."""
    nested: Dict[str, Dict[str, int]] = {}
    for (filename, code), count in sorted(counts.items()):
        nested.setdefault(filename, {})[code] = count
    payload = {
        "_comment": (
            "Accepted Ruff findings per file and rule. Written by"
            " scripts/lint_ratchet.py --update. These may only go"
            " down; see audit finding QUALITY-02."
        ),
        "total": sum(counts.values()),
        "counts": nested,
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def regressions(
    baseline: Counts, current: Counts
) -> List[Tuple[Cell, int, int]]:
    """Cells that grew or appeared, worst first."""
    grown: List[Tuple[Cell, int, int]] = []
    for cell, count in current.items():
        allowed = baseline.get(cell, 0)
        if count > allowed:
            grown.append((cell, allowed, count))
    grown.sort(key=lambda item: item[2] - item[1], reverse=True)
    return grown


def improvements(
    baseline: Counts, current: Counts
) -> List[Tuple[Cell, int, int]]:
    """Cells that shrank or vanished, biggest win first."""
    shrunk: List[Tuple[Cell, int, int]] = []
    for cell, allowed in baseline.items():
        count = current.get(cell, 0)
        if count < allowed:
            shrunk.append((cell, allowed, count))
    shrunk.sort(key=lambda item: item[1] - item[2], reverse=True)
    return shrunk


def report(
    baseline: Counts, current: Counts
) -> int:
    """Print the verdict and return the process exit code."""
    grown = regressions(baseline, current)
    shrunk = improvements(baseline, current)

    for (filename, code), allowed, count in grown:
        print(
            f"  {filename}: {code} {allowed} -> {count}",
            file=sys.stderr,
        )
    if grown:
        print(
            f"\nlint ratchet: {len(grown)} rule(s) got worse."
            " Fix them, or explain the exception in"
            " IMPLEMENTATION_LEDGER.md and rerun with --update.",
            file=sys.stderr,
        )
        return 1

    total_now = sum(current.values())
    total_was = sum(baseline.values())
    if shrunk:
        for (filename, code), allowed, count in shrunk[:10]:
            print(f"  {filename}: {code} {allowed} -> {count}")
        print(
            f"\nlint ratchet: {total_was} -> {total_now}."
            " Run with --update to lock in the improvement."
        )
        return 0

    print(f"lint ratchet: holding at {total_now} findings.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when Ruff findings increase.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Rewrite the baseline from the current findings."
            " Refuses to raise the ceiling."
        ),
    )
    args = parser.parse_args()

    current = tally(run_ruff())
    baseline = load_baseline()

    if args.update:
        # An absent baseline means "not established yet", not "a
        # ceiling of zero", so the first write is a bootstrap rather
        # than a refused raise.
        established = BASELINE_PATH.is_file()
        grown = regressions(baseline, current) if established else []
        if grown:
            print(
                "lint ratchet: refusing to raise the ceiling."
                " Fix these first:",
                file=sys.stderr,
            )
            for (filename, code), allowed, count in grown:
                print(
                    f"  {filename}: {code} {allowed} -> {count}",
                    file=sys.stderr,
                )
            return 1
        save_baseline(current)
        print(
            "lint ratchet: baseline updated to"
            f" {sum(current.values())} findings."
        )
        return 0

    return report(baseline, current)


if __name__ == "__main__":
    sys.exit(main())
