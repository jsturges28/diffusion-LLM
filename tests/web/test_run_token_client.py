"""The generator keeps hold of which run the worker is serving.

Strategy: inspect `generator_run.js` and its app.js transport wiring.
The controller is driven directly in `generator_run.test.js`; the
worker's half of `LIFE-01` is executed properly in
`tests/backends/test_run_identity.py` and
`tests/backends/test_worker_dispatch.py`; this covers the client half,
which had no test at all and is where the token can quietly go wrong.

What passing proves is that one variable stays in step with the run it
names, across the four things that can move it: a terminal frame
brings a new one, the five stateful requests must quote it, a reload
must carry it, and a fresh run must retire it. Miss the last and a
token outlives the state it describes, which is the exact shape of the
bug `ORG-02` exists to prevent and which this file was written after
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
SNAPSHOT_JS = APP_JS.with_name("run_snapshot.js")
RUN_JS = APP_JS.with_name("generator_run.js")
EDIT_JS = APP_JS.with_name("generator_edit.js")


def _source(path: Path = APP_JS) -> str:
    return path.read_text(encoding="utf-8")


def _region(path: Path, anchor: str, chars: int) -> str:
    source = _source(path)
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {path.name};"
        " update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- where it comes from --


def test_a_terminal_frame_brings_the_token() -> None:
    """Stamped by the worker on every `done`, including the ones it
    synthesizes for a guided edit, so a resumed run stays namable."""
    region = _region(RUN_JS, "function finish(data)", 1200)

    assert "runToken = data.run_token" in region


def test_only_a_string_is_adopted() -> None:
    """An older worker sends no token, and adopting `undefined` would
    make every later request quote the word undefined."""
    region = _region(RUN_JS, "function finish(data)", 1200)

    assert 'typeof data.run_token === "string"' in region


# -- where it goes --


def test_every_stateful_request_quotes_it() -> None:
    """Resume, substitution, probe and rewind are the kinds the worker
    checks before it reads or writes retained state, and a resume is
    sent twice, as an edit and as a Continue. A request arriving
    without this is the regression to catch.

    Rewind joined them when abandoning an edit session turned out to
    leave the worker holding the branch the browser had discarded.
    """
    app = _source()
    edit = _source(EDIT_JS)

    assert edit.count("run.runToken()") == 4
    assert app.count("run_token: generatorRun.runToken()") == 1
    assert app.count("run_token: intent.runToken") == 3


def test_the_five_are_the_ones_we_think() -> None:
    """Counting alone would pass if one moved to the wrong request."""
    source = _source()

    requests = {
        '"probe"': 1,
        '"substitute"': 1,
        '"resume"': 1,
        '"rewind"': 1,
    }
    for request, count in requests.items():
        starts = [
            match.start()
            for match in re.finditer(
                re.escape("type: " + request), source
            )
        ]
        assert len(starts) == count, request
        for start in starts:
            sent = source[start : start + 400]
            token = (
                "run_token: generatorRun.runToken()"
                if request == '"probe"'
                else "run_token: intent.runToken"
            )
            assert token in sent, request

    edit = _source(EDIT_JS)
    assert edit.count("requestResume({") == 2
    assert edit.count("requestSubstitute({") == 1
    assert "requestRewind({ runToken: token })" in edit


# -- where it survives --


def test_a_reload_carries_it() -> None:
    """Without this the worker still holds the run, the page still
    shows it, and editing it is refused as stale."""
    region = _region(RUN_JS, "function sessionRecord()", 1800)

    assert "runToken: runToken" in region


def test_a_restore_defaults_it_to_empty() -> None:
    """Snapshots written before runs had identities have no token, and
    reading `undefined` back would send it to the worker. The default
    is the snapshot codec's; the page applies what it decodes."""
    codec = SNAPSHOT_JS.read_text(encoding="utf-8")
    guarded = (
        'runToken: typeof source.runToken === "string"\n'
        "      ? source.runToken\n"
        '      : "",'
    )

    assert guarded in codec
    assert "runToken = restored.runToken;" in _region(
        RUN_JS, "function applyRestored(restored)", 1200
    )


# -- where it ends --


def test_a_fresh_run_retires_it() -> None:
    """The one this file was written for. `resetRunState` clears the
    rest of what the last run left, and the token was missed when it
    was added among those siblings."""
    region = _region(RUN_JS, "function reset()", 1100)

    assert 'runToken = ""' in region


def test_it_is_retired_beside_its_siblings() -> None:
    """Not merely present somewhere in the function: next to the other
    facts about the finished run, which is where the next person will
    look when they add the seventh."""
    region = _region(RUN_JS, "function reset()", 1100)
    provenance = region.find("provenance = null")
    token = region.find('runToken = ""')

    assert provenance != -1
    assert token != -1
    assert 0 < token - provenance < 400


def test_nothing_else_writes_the_token() -> None:
    """Four writers, all covered above. A fifth means a path that
    moves the token without the run moving with it."""
    writes = re.findall(r"\brunToken\s*=(?!=)", _source(RUN_JS))

    assert len(writes) == 4
