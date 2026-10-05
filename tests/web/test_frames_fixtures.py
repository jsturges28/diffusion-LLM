"""The Analytics adapter's fixtures are what the server sends
(`A2-ORG-04`).

Strategy: save two small runs through the real `/api/save` into a
temporary results directory, as `test_stop_rule.py` does, read each
back through `/api/analytics/runs/{id}/frames`, and require the
committed fixture in `tests/web/static/fixtures/` to equal the
response, its timestamped run id aside. Each run's provenance carries
the signals its model's registry entry declares, which is what a
worker attests on its terminal frame.

The fixtures exist because the browser tests that read a saved run
built their payloads by hand, and a hand-built payload can carry a
field the server never sends: a run's signal manifest once passed in
the browser while Analytics never received it. Passing proves that
the payloads `tests/web/static/overlay_series.test.js` reads are the
server's, in both of the shapes it sends a run in.

When the response changes on purpose, rewrite the fixtures with
`UPDATE_FRAMES_FIXTURES=1 .venv/bin/python -m pytest
tests/web/test_frames_fixtures.py` and review their diff.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
from starlette.testclient import TestClient

from src.backends.registry import REGISTRY
from src.web import server

FIXTURES = Path(__file__).resolve().parent / "static" / "fixtures"
UPDATE_ENV = "UPDATE_FRAMES_FIXTURES"

# Stands in for the run id, which carries the time of the save.
RUN_ID = "fixture"

MASK_TEXT = "\u2591"
MASK_ID = 126336
WORDS: Tuple[str, ...] = (" Yeast", " eats", " sugar")
EDIT_FRAME = 2
EDITED_POSITION = 1
EDITED_WORD = " ate"

# One id per word, so a re-decided position reads as another token
# rather than the same one under a different spelling.
TOKEN_IDS: Dict[str, int] = {
    " Yeast": 100,
    " eats": 101,
    " sugar": 102,
    EDITED_WORD: 103,
}

Record = Dict[str, Any]


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return TestClient(server.app)


def _signals(model: str) -> List[Dict[str, Any]]:
    channels = REGISTRY[model].capabilities.signals
    return [channel.model_dump(mode="json") for channel in channels]


def _record(text: str, ident: int, entropy: float) -> Record:
    masked = text == MASK_TEXT
    return {
        "t": text,
        "m": masked,
        "id": MASK_ID if masked else ident,
        "c": 0.5,
        "e": round(entropy, 4),
    }


def _frame(words: List[str], index: int, lift: float) -> List[Record]:
    """One canvas, its entropy different at every frame and position,
    so a reader of the wrong frame reads a visibly wrong number."""
    return [
        _record(
            word,
            TOKEN_IDS.get(word, MASK_ID),
            0.2 * (index + 1) + lift + position / 100,
        )
        for position, word in enumerate(words)
    ]


def _original_words(index: int) -> List[str]:
    """The pre-edit run, settling one word per frame."""
    return [
        word if position < index else MASK_TEXT
        for position, word in enumerate(WORDS)
    ]


def _edited_words(index: int) -> List[str]:
    """The run after an edit remasked one position at EDIT_FRAME and
    resumed: the same before it, re-decided from it on."""
    if index < EDIT_FRAME:
        return _original_words(index)
    if index == EDIT_FRAME:
        words = _original_words(index)
        words[EDITED_POSITION] = MASK_TEXT
        return words
    words = list(WORDS)
    words[EDITED_POSITION] = EDITED_WORD
    return words


def _llada_edited() -> Dict[str, Any]:
    frames = range(len(WORDS) + 1)
    original = [_frame(_original_words(i), i, 0.0) for i in frames]
    edited = [
        _frame(_edited_words(i), i, 0.0 if i < EDIT_FRAME else 0.5)
        for i in frames
    ]
    texts = ["".join(record["t"] for record in f) for f in edited]
    return {
        "model": "llada",
        "prompt": "explain yeast",
        "params": {},
        "frames": texts,
        "frame_tokens": edited,
        "original_frame_tokens": original,
        "remask_edits": [
            {
                "frame_index": EDIT_FRAME,
                "token_positions": [EDITED_POSITION],
            }
        ],
        "final_text": texts[-1],
        "provenance": {
            "model_id": "llada",
            "device": "cuda",
            "signals": _signals("llada"),
        },
    }


def _smollm3_append() -> Dict[str, Any]:
    positions = [
        _record(word, 200 + position, 1.2 + position / 10)
        for position, word in enumerate(WORDS)
    ]
    positions[0].update({"g": True, "we": False})
    positions[1].update({"g": True, "we": True})
    positions[2].update({"g": False, "we": True})
    p0 = 0.25
    z_score = (1 - 2 * p0) / math.sqrt(2 * p0 * (1 - p0))
    return {
        "model": "smollm3",
        "prompt": "explain yeast",
        "params": {
            "watermark": True,
            "watermark_gamma": 0.25,
            "watermark_delta": 2.0,
            "watermark_z_threshold": 3.5,
        },
        "frame_positions": positions,
        "final_text": "".join(WORDS),
        "provenance": {
            "model_id": "smollm3",
            "device": "cuda",
            "tokenizer": {
                "fingerprint": "ab" * 32,
                "model_vocab_size": 128256,
            },
            "signals": _signals("smollm3"),
            "watermark": {
                "scheme": "kgw",
                "version": 1,
                "key_id": "0123456789abcdef",
                "gamma": 0.25,
                "delta": 2.0,
                "vocab_size": 128256,
                "green_list_size": 32064,
                "tokenizer_fingerprint": "ab" * 32,
                "seeding_contract": "test contract",
                "rng_contract": "test generator",
                "exclusions": [
                    "first output token",
                    "user-forced tokens",
                ],
                "status": "insufficient_evidence",
                "green_count": 1,
                "scored_count": 2,
                "green_rate": 0.5,
                "z_score": z_score,
                "p0": p0,
            },
        },
    }


RUNS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "frames_llada_edited.json": _llada_edited,
    "frames_smollm3_append.json": _smollm3_append,
}


def _frames_response(
    client: TestClient, payload: Dict[str, Any]
) -> Dict[str, Any]:
    saved = client.post("/api/save", json=payload)
    assert saved.status_code == 200, saved.text
    run_id = str(saved.json()["run_id"])
    response = client.get(f"/api/analytics/runs/{run_id}/frames")
    assert response.status_code == 200, response.text
    body: Dict[str, Any] = response.json()
    assert body["run_id"] == run_id
    body["run_id"] = RUN_ID
    return body


@pytest.mark.parametrize("name", sorted(RUNS))
def test_the_fixture_is_what_the_server_sends(
    client: TestClient, name: str
) -> None:
    body = _frames_response(client, RUNS[name]())
    path = FIXTURES / name
    if os.environ.get(UPDATE_ENV) == "1":
        text = json.dumps(body, indent=1, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")

    assert path.is_file(), (
        f"{name} is missing; write it with {UPDATE_ENV}=1"
    )
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == body, (
        f"{name} is no longer what the server sends. If the change is"
        f" deliberate, rewrite it with {UPDATE_ENV}=1 and review the"
        " diff."
    )


def test_the_fixtures_are_the_two_shapes_a_run_arrives_in(
    client: TestClient,
) -> None:
    """The adapter's whole job is to make two shapes one, so a pair of
    fixtures in the same shape would test half of it."""
    snapshot = _frames_response(client, _llada_edited())
    append = _frames_response(client, _smollm3_append())

    assert snapshot["positions"] is None
    assert len(snapshot["frames"]) == len(WORDS) + 1
    assert len(snapshot["original_frames"]) == len(WORDS) + 1
    assert append["frames"] is None
    assert len(append["positions"]) == len(WORDS)
    assert append["watermark"]["record_consistency"] == "consistent"
    assert append["watermark"]["recomputed"]["green_count"] == 1
    assert append["watermark"]["recomputed"]["scored_count"] == 2
    assert append["watermark_display_threshold"] == 3.5
