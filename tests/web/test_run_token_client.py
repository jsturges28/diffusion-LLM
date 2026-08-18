"""The generator keeps hold of which run the worker is serving.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages until `ORG-02` gives them a testable
seam. The worker's half of `LIFE-01` is executed properly in
`tests/backends/test_run_identity.py` and
`tests/backends/test_worker_dispatch.py`; this covers the client half,
which had no test at all and is where the token can quietly go wrong.

What passing proves is that one variable stays in step with the run it
names, across the four things that can move it: a terminal frame
brings a new one, three stateful requests must quote it, a reload must
carry it, and a fresh run must retire it. Miss the last and a token
outlives the state it describes, which is the exact shape of the bug
`ORG-02` exists to prevent and which this file was written after
finding.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "app.js"
)


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- where it comes from --


def test_a_terminal_frame_brings_the_token() -> None:
    """Stamped by the worker on every `done`, including the ones it
    synthesizes for a guided edit, so a resumed run stays namable."""
    region = _region("function handleDone(data)", 1400)

    assert "activeRunToken = data.run_token" in region


def test_only_a_string_is_adopted() -> None:
    """An older worker sends no token, and adopting `undefined` would
    make every later request quote the word undefined."""
    region = _region("function handleDone(data)", 1400)

    assert 'typeof data.run_token === "string"' in region


# -- where it goes --


def test_every_stateful_request_quotes_it() -> None:
    """Resume, substitution and probe are the three the worker checks
    before it reads retained state. A fourth arriving without this is
    the regression to catch."""
    source = _source()

    assert source.count("run_token: activeRunToken") == 3


def test_the_three_are_the_ones_we_think() -> None:
    """Counting alone would pass if one moved to the wrong request."""
    source = _source()

    for request in ('"probe"', '"substitute"', '"resume"'):
        start = source.find("type: " + request)
        assert start != -1, request
        sent = source[start : start + 400]
        assert "run_token: activeRunToken" in sent, request


# -- where it survives --


def test_a_reload_carries_it() -> None:
    """Without this the worker still holds the run, the page still
    shows it, and editing it is refused as stale."""
    region = _region("function saveSessionState()", 2200)

    assert "runToken: activeRunToken" in region


def test_a_restore_defaults_it_to_empty() -> None:
    """Snapshots written before runs had identities have no token, and
    reading `undefined` back would send it to the worker."""
    region = _region("function restoreSessionState()", 3000)
    guarded = 'typeof s.runToken === "string" ? s.runToken : ""'

    assert guarded in region


# -- where it ends --


def test_a_fresh_run_retires_it() -> None:
    """The one this file was written for. `resetRunState` clears the
    rest of what the last run left, and the token was missed when it
    was added among those siblings."""
    region = _region("function resetRunState()", 1600)

    assert 'activeRunToken = ""' in region


def test_it_is_retired_beside_its_siblings() -> None:
    """Not merely present somewhere in the function: next to the other
    facts about the finished run, which is where the next person will
    look when they add the seventh."""
    region = _region("function resetRunState()", 1600)
    provenance = region.find("lastRunProvenance = null")
    token = region.find('activeRunToken = ""')

    assert provenance != -1
    assert token != -1
    assert 0 < token - provenance < 400


def test_nothing_else_writes_the_token() -> None:
    """Four writers, all covered above. A fifth means a path that
    moves the token without the run moving with it."""
    writes = re.findall(r"\bactiveRunToken\s*=(?!=)", _source())

    assert len(writes) == 4
